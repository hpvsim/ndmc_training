"""
Define the calibration class
"""

import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
import hpvsim.misc as hpm
import hpvsim.parameters as hppar
import hpvsim.analysis as hpa
from hpvsim.settings import options as hpo
import hpvsim.plotting as hppl

__all__ = ['MultiCal']


def import_optuna():
    """ A helper function to import Optuna """
    try:
        import optuna as op  # Import here since it's slow
    except ModuleNotFoundError as E:  # pragma: no cover
        errormsg = f'Optuna import failed ({str(E)}), please install first (pip install optuna)'
        raise ModuleNotFoundError(errormsg)
    return op


# noinspection PyTypeChecker
class MultiCal(sc.prettyobj):
    """
    A class to handle joint calibration of multiple HPVsim simulations.
    """

    def __init__(self, sims, datafiles, common_pars=None, unique_pars=None, fit_args=None, par_samplers=None,
                 n_trials=None, n_workers=None, total_trials=None, name=None, db_name=None, load_if_exists=None,
                 keep_db=None, storage=None, rand_seed=None, label=None, die=False, verbose=True, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.json = None
        self.df = None
        self.data = None
        self.par_bounds = None
        self.initial_pars = None
        self.age_results = None
        self.elapsed = None
        self.best_pars = None
        import multiprocessing as mp  # Import here since it's also slow

        # Handle run arguments
        if n_trials is None: n_trials = 20
        if n_workers is None: n_workers = mp.cpu_count()
        if name is None: name = 'hpvsim_calibration'
        if db_name is None: db_name = f'{name}.db'
        if keep_db is None: keep_db = False
        if load_if_exists is None: load_if_exists = False
        if storage is None: storage = f'sqlite:///{db_name}'
        if total_trials is not None: n_trials = int(np.ceil(total_trials / n_workers))
        self.run_args = sc.objdict(n_trials=int(n_trials), n_workers=int(n_workers), name=name, db_name=db_name,
                                   keep_db=keep_db, storage=storage, rand_seed=rand_seed, load_if_exists=load_if_exists)

        # Handle other inputs
        self.label = label
        self.sims = {sim.label: sim for sim in sims}
        self.common_pars = common_pars
        self.unique_pars = unique_pars
        self.fit_args = sc.mergedicts(fit_args)
        self.par_samplers = sc.mergedicts(par_samplers)
        self.die = die
        self.verbose = verbose
        self.calibrated = False
        self.sim_labels = self.sims.keys()

        # Set up target data
        self.target_data = sc.objdict()
        for simname, simdatafiles in datafiles.items():
            self.target_data[simname] = []
            for simdatafile in simdatafiles:
                self.target_data[simname].append(hpm.load_data(simdatafile))

        # Initialize result storage
        sim_results = sc.objdict({sim_label: sc.objdict() for sim_label in self.sim_labels})
        age_results = sc.objdict({sim_label: sc.objdict() for sim_label in self.sim_labels})
        self.result_args = sc.objdict({sim_label: sc.objdict() for sim_label in self.sim_labels})

        for slabel, sim in self.sims.items():
            for targ in self.target_data[slabel]:
                targ_keys = targ.name.unique()
                if len(targ_keys) > 1:
                    errormsg = f'Only support one set of targets per datafile, {len(targ_keys)} provided'
                    raise ValueError(errormsg)
                if 'age' in targ.columns:
                    age_results[slabel][targ_keys[0]] = sc.objdict(
                        datafile=sc.dcp(targ),
                        compute_fit=True,
                    )
                else:
                    sim_results[slabel][targ_keys[0]] = sc.objdict(
                        data=sc.dcp(targ)
                    )

            ar = hpa.age_results(result_args=age_results[slabel])
            sim['analyzers'] += [ar]

            sresults = sim_results[slabel]
            for rkey in sresults.keys():
                if 'weights' not in sresults[rkey].data.columns:
                    sresults[rkey].weights = np.ones(len(sresults[rkey].data))

        self.sim_results = sim_results

        # Store genotype info for each sim - populated during run_trial
        self.ng = sc.objdict()
        self.glabels = sc.objdict()

        # Temporarily store a filename
        self.tmp_filename = 'tmp_calibration_%s_%05i.obj'

        return

    def run_sim(self, sim, calib_pars=None, genotype_pars=None, return_sim=False):
        """ Create and run a simulation """

        new_pars = self.get_full_pars(sim=sim, calib_pars=calib_pars, genotype_pars=genotype_pars)
        sim.update_pars(new_pars)
        sim.timer = sc.timer()

        # Run the sim
        try:
            sim.run()
            sim.timer.toc()
            if return_sim:
                return sim
            else:
                return sim.fit

        except Exception as E:
            if self.die:
                raise E
            else:
                warnmsg = f'Encountered error running sim!\nParameters:\n{new_pars}\nTraceback:\n{sc.traceback()}'
                hpm.warn(warnmsg)
                output = None if return_sim else np.inf
                return output

    @staticmethod
    def get_full_pars(sim=None, calib_pars=None, genotype_pars=None):
        """
        Make a full pardict from the subset of regular sim parameters and genotype parameters used in calibration
        """

        # Prepare the parameters
        if calib_pars is not None:
            new_pars = {}
            for name, par in calib_pars.items():
                if isinstance(par, dict):
                    simpar = sim.pars[name]
                    for parkey, parval in par.items():
                        simpar[parkey] = parval
                    new_pars[name] = simpar
                else:
                    if name in sim.pars:
                        new_pars[name] = par

            if len(new_pars) != len(calib_pars):
                extra = set(calib_pars.keys()) - set(new_pars.keys())
                errormsg = f'The following parameters are not part of the sim, nor is a custom function specified to ' \
                           f'use them: {sc.strjoin(list(extra))} '
                raise ValueError(errormsg)
        else:
            new_pars = {}
        if genotype_pars is not None:
            new_genotype_pars = {}
            for gname, gpars in genotype_pars.items():
                this_genotype_pars = hppar.get_genotype_pars(gname)
                for gpar, gval in gpars.items():
                    if isinstance(gval, dict):
                        for gparkey, gparval in gval.items():
                            this_genotype_pars[gpar][gparkey] = gparval
                    else:
                        this_genotype_pars[gpar] = gval
                new_genotype_pars[gname] = this_genotype_pars

            all_genotype_pars = sc.dcp(sim['genotype_pars'])
            all_genotype_pars.update(new_genotype_pars)
            new_pars['genotype_pars'] = all_genotype_pars

        return new_pars

    def trial_pars_to_sim_pars(self, slabel, trial_pars=None, which_pars=None, return_full=True):
        """
        Create genotype_pars and pars dicts from the trial parameters.
        Note: not used during self.calibrate.
        Args:
            slabel (str): the sim label
            trial_pars (dict): dictionary of parameters from a single trial. If blank, the best parameters will be used
            which_pars (int): which parameters to return
            return_full (bool): whether to return a par dict ready for use in a sim, or the sim/genotype pars separately

        **Example**::

            sim = hpv.Sim(genotypes=[16, 18])
            calib_pars = dict(beta=[0.05, 0.010, 0.20],hpv_control_prob=[.9, 0.5, 1])
            genotype_pars = dict(hpv16=dict(prog_time=[3, 3, 10]))
            calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars
                                datafiles=['test_data/south_africa_hpv_data.xlsx',
                                           'test_data/south_africa_cancer_data.xlsx'],
                                total_trials=10, n_workers=4)
            calib.calibrate()
            new_pars = calib.trial_pars_to_sim_pars() # Returns the best parameters from calibration ready for sim run
            sim.update_pars(new_pars)
            sim.run()
        """

        # Initialize
        calib_pars = {}
        genotype_pars = sc.objdict()
        sim = self.sims[slabel]

        # Deal with trial parameters
        if trial_pars is None:
            try:
                if which_pars is None or which_pars == 0:
                    trial_pars = self.best_pars
                else:
                    ddict = self.df.to_dict(orient='records')[which_pars]
                    trial_pars = {k: v for k, v in ddict.items() if k not in ['index', 'mismatch']}
            except Exception:
                errormsg = 'No trial parameters provided.'
                raise ValueError(errormsg)

        # Handle common parameters
        if self.common_pars is not None:

            # Handle regular sim parameters
            if self.common_pars.get('calib_pars'):
                for name, par in self.common_pars['calib_pars'].items():
                    if isinstance(par, dict):
                        simpar = sim.pars[name]
                        for parkey in par.keys():
                            simpar[parkey] = trial_pars[f'{name}_{parkey}']
                        calib_pars[name] = simpar
                    else:
                        calib_pars[name] = trial_pars[name]

            # Handle genotype parameters
            if self.common_pars.get('genotype_pars'):
                for gname, gpars in self.common_pars['genotype_pars'].items():
                    this_genotype_pars = dict()
                    for gpar, gval in gpars.items():
                        if isinstance(gval, dict):
                            this_genotype_pars[gpar] = dict()
                            for gparkey in gval.keys():
                                this_genotype_pars[gpar][gparkey] = trial_pars[
                                    f'{gname}_{gpar}_{gparkey}']  # Update with values from trial pars
                        else:
                            this_genotype_pars[gpar] = trial_pars[f'{gname}_{gpar}']
                    genotype_pars[gname] = this_genotype_pars

        # Handle unique parameters
        if self.unique_pars is not None and self.unique_pars.get(slabel):
            upars = self.unique_pars[slabel]

            # Handle regular sim parameters
            if upars.get('calib_pars'):
                for name, par in upars['calib_pars'].items():
                    if isinstance(par, dict):
                        simpar = sim.pars[name]
                        for parkey in par.keys():
                            simpar[parkey] = trial_pars[f'{slabel}_{name}_{parkey}']
                        calib_pars[name] = simpar
                    else:
                        calib_pars[name] = trial_pars[f'{slabel}_{name}']

            # Handle genotype parameters
            if upars.get('genotype_pars'):
                for gname, gpars in upars['genotype_pars'].items():
                    this_genotype_pars = dict()
                    # this_genotype_pars = hppar.get_genotype_pars(gname) # Get default values
                    for gpar, gval in gpars.items():
                        if isinstance(gval, dict):
                            this_genotype_pars[gpar] = dict()
                            for gparkey in gval.keys():
                                this_genotype_pars[gpar][gparkey] = trial_pars[
                                    f'{slabel}_{gname}_{gpar}_{gparkey}']  # Update with values from trial pars
                        else:
                            this_genotype_pars[gpar] = trial_pars[f'{slabel}_{gname}_{gpar}']
                    if genotype_pars.get(gname):
                        # noinspection PyTypeChecker
                        genotype_pars[gname] = sc.mergedicts(genotype_pars[gname], this_genotype_pars)
                    else:
                        genotype_pars[gname] = this_genotype_pars

        # Return
        if return_full:
            all_pars = self.get_full_pars(sim=sim, calib_pars=calib_pars, genotype_pars=genotype_pars)
            return all_pars
        else:
            return calib_pars, genotype_pars

    def sim_to_sample_pars(self):
        """ Convert sim pars to sample pars """

        initial_pars = sc.objdict()
        par_bounds = sc.objdict()

        # Convert common pars
        if self.common_pars is not None:
            if self.common_pars.get('calib_pars'):
                for key, val in self.common_pars['calib_pars'].items():
                    if isinstance(val, list):
                        initial_pars[key] = val[0]
                        par_bounds[key] = np.array([val[1], val[2]])
                    elif isinstance(val, dict):
                        for parkey, par_highlowlist in val.items():
                            sampler_key = key + '_' + parkey + '_'
                            initial_pars[sampler_key] = par_highlowlist[0]
                            par_bounds[sampler_key] = np.array([par_highlowlist[1], par_highlowlist[2]])

            # Convert genotype pars
            if self.common_pars.get('genotype_pars'):
                for gname, gpardict in self.common_pars['genotype_pars'].items():
                    for key, val in gpardict.items():
                        if isinstance(val, list):
                            sampler_key = gname + '_' + key
                            initial_pars[sampler_key] = val[0]
                            par_bounds[sampler_key] = np.array([val[1], val[2]])
                        elif isinstance(val, dict):
                            for parkey, par_highlowlist in val.items():
                                sampler_key = gname + '_' + key + '_' + parkey
                                initial_pars[sampler_key] = par_highlowlist[0]
                                par_bounds[sampler_key] = np.array([par_highlowlist[1], par_highlowlist[2]])

        # Convert unique pars
        if self.unique_pars is not None:
            for slabel, upars in self.unique_pars.items():
                if upars.get('calib_pars'):
                    for key, val in upars['calib_pars'].items():
                        sampler_key = slabel + '_' + key
                        if isinstance(val, list):
                            initial_pars[sampler_key] = val[0]
                            par_bounds[sampler_key] = np.array([val[1], val[2]])
                        elif isinstance(val, dict):
                            for parkey, par_highlowlist in val.items():
                                sampler_key = slabel + '_' + key + '_' + parkey
                                initial_pars[sampler_key] = par_highlowlist[0]
                                par_bounds[sampler_key] = np.array([par_highlowlist[1], par_highlowlist[2]])

                if upars.get('genotype_pars'):
                    for gname, gpardict in upars['genotype_pars'].items():
                        for key, val in gpardict.items():
                            sampler_key = slabel + '_' + gname + '_' + key
                            if isinstance(val, list):
                                initial_pars[sampler_key] = val[0]
                                par_bounds[sampler_key] = np.array([val[1], val[2]])
                            elif isinstance(val, dict):
                                for parkey, par_highlowlist in val.items():
                                    sampler_key = slabel + '_' + gname + '_' + key + '_' + parkey
                                    initial_pars[sampler_key] = par_highlowlist[0]
                                    par_bounds[sampler_key] = np.array([par_highlowlist[1], par_highlowlist[2]])

        return initial_pars, par_bounds

    def trial_to_sim_pars(self, pardict=None, trial=None, gname=None, sname=None):
        """
        Take in an optuna trial and sample from pars, after extracting them from the structure they're provided in
        """
        pars = {}

        for key, val in pardict.items():
            if isinstance(val, list):
                low, high = val[1], val[2]
                if len(val) > 3:
                    step = val[3]
                else:
                    step = None
                if key in self.par_samplers:  # If a custom sampler is used, get it now
                    try:
                        sampler_fn = getattr(trial, self.par_samplers[key])
                    except Exception as E:
                        errormsg = 'Requested sampler function not found: it must be a valid attribute of an Optuna ' \
                                   'Trial object '
                        raise AttributeError(errormsg) from E
                else:
                    sampler_fn = trial.suggest_float

                sampler_key = key
                if gname is not None: sampler_key = gname + '_' + sampler_key
                if sname is not None:
                    sampler_key = sname + '_' + sampler_key
                pars[key] = sampler_fn(sampler_key, low, high, step=step)  # Sample from values within this range

            elif isinstance(val, dict):
                sampler_fn = trial.suggest_float
                pars[key] = dict()
                for parkey, par_highlowlist in val.items():

                    sampler_key = key + '_' + parkey
                    if gname is not None: sampler_key = gname + '_' + sampler_key
                    if sname is not None: sampler_key = sname + '_' + sampler_key
                    if isinstance(par_highlowlist, dict):
                        par_highlowlist = par_highlowlist['value']
                        low, high = par_highlowlist[1], par_highlowlist[2]
                        if len(par_highlowlist) > 3:
                            step = par_highlowlist[3]
                        else:
                            step = None
                    elif isinstance(par_highlowlist, list):
                        low, high = par_highlowlist[1], par_highlowlist[2]
                        if len(par_highlowlist) > 3:
                            step = par_highlowlist[3]
                        else:
                            step = None
                    else:
                        raise TypeError()
                    pars[key][parkey] = sampler_fn(sampler_key, low, high, step=step)

        return pars

    def run_trial(self, trial, save=True):
        """ Define the objective for Optuna """

        calib_pars = {}
        genotype_pars = {}

        if self.common_pars is not None:
            if self.common_pars.get('calib_pars'):
                calib_pars = self.trial_to_sim_pars(self.common_pars['calib_pars'], trial)
            if self.common_pars.get('genotype_pars'):
                genotype_pars = {}
                for gname, pardict in self.common_pars['genotype_pars'].items():
                    genotype_pars[gname] = self.trial_to_sim_pars(pardict, trial, gname=gname)

        # Now loop over the sims and add the unique pars for each
        total_fit = 0
        for slabel, sim in self.sims.items():
            if self.unique_pars is not None:
                if self.unique_pars[slabel].get('calib_pars'):
                    # noinspection PyTypeChecker
                    calib_pars = sc.mergedicts(calib_pars,
                                               self.trial_to_sim_pars(self.unique_pars[slabel]['calib_pars'], trial,
                                                                      sname=slabel))
                if self.unique_pars[slabel].get('genotype_pars'):
                    for gname, pardict in self.unique_pars[slabel]['genotype_pars'].items():
                        unique_gpars = self.trial_to_sim_pars(pardict, trial, gname=gname, sname=slabel)
                        if genotype_pars.get(gname):
                            # Some things have already been filled in from the common pars. NB, if something is in
                            # common pars and unique pars, it will be overwritten here with the unique pars values.
                            # TODO add check for this.
                            genotype_pars[gname] = sc.mergedicts(genotype_pars[gname], unique_gpars)
                        else:
                            genotype_pars[gname] = unique_gpars

            sim = self.run_sim(sim.copy(), calib_pars, genotype_pars, return_sim=True)

            # Compute fit for sim results and save sim results
            age_results = sim.get_analyzer('age_results').results
            sim_results = sc.objdict()
            sresults = self.sim_results[slabel]

            # Save color and name properties - to do, this does not have to be part of the trial.
            for rkey in sresults.keys() + age_results.keys():
                self.result_args[slabel][rkey] = sc.objdict()
                self.result_args[slabel][rkey].name = sim.results[rkey].name
                self.result_args[slabel][rkey].color = sim.results[rkey].color

            self.ng[slabel] = sim['n_genotypes']
            self.glabels[slabel] = [g.upper() for g in sim['genotype_map'].values()]

            for rkey in sresults:
                year = sresults[rkey].data.year.unique()[0]
                yind = sc.findinds(sim.results['year'], year)[0]
                if sim.results[rkey][:].ndim == 1:
                    model_output = sim.results[rkey][yind]
                else:
                    model_output = sim.results[rkey][:, yind]
                gofs = hpm.compute_gof(sresults[rkey].data.value, model_output)
                losses = gofs * sresults[rkey].weights
                mismatch = losses.sum()
                sim.fit += mismatch
                sim_results[rkey] = model_output

            # Store results in temporary files
            if save:
                results = dict(sim=sim_results, age=age_results, calib_pars=calib_pars, genotype_pars=genotype_pars,
                               mismatch=total_fit, runtime=sim.timer.timings[0])
                fileslabel = slabel.replace(' ', '_')
                filename = self.tmp_filename % (fileslabel, trial.number)
                sc.save(filename, results)

            # Add this sim fit to total fit
            total_fit += sim.fit

        return total_fit

    def worker(self):
        """ Run a single worker """
        op = import_optuna()
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name)
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials, callbacks=None)  # [tesst]
        return output

    def run_workers(self):
        """ Run multiple workers in parallel """
        if self.run_args.n_workers > 1:  # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else:  # Special case: just run one
            output = [self.worker()]
        return output

    def remove_db(self):
        """
        Remove the database file if keep_db is false and the path exists.
        """
        try:
            op = import_optuna()
            op.delete_study(study_name=self.run_args.name, storage=self.run_args.storage)
            if self.verbose:
                print(f'Deleted study {self.run_args.name} in {self.run_args.storage}')
        except Exception as E:
            print('Could not delete study, skipping...')
            print(str(E))
        if os.path.exists(self.run_args.db_name):
            try:
                sc.rmpath(self.run_args.db_name, die=False)
                if self.verbose:
                    print(f'Removed existing calibration {self.run_args.db_name}')
            except Exception as E:
                print('Could not delete study file, skipping...')
                print(str(E))
        return

    def make_study(self):
        """ Make a study, deleting one if it already exists """
        op = import_optuna()
        if not self.run_args.keep_db:
            self.remove_db()
        if self.run_args.rand_seed is not None:
            sampler = op.samplers.RandomSampler(self.run_args.rand_seed)
            sampler.reseed_rng()
            raise NotImplementedError('Implemented but does not work')
        else:
            sampler = None
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler=sampler,
                                 load_if_exists=self.run_args.load_if_exists)
        return output

    def calibrate(self, common_pars=None, unique_pars=None, load=True, tidyup=True, **kwargs):
        """
        Perform calibration.

        Args:
            common_pars (dict): parameters common to all sims
            unique_pars (dict): parameters unique to each sim
            load (bool): whether to load existing DB
            tidyup (bool): whether to clean up temporary storage files (recommend True)
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        """
        op = import_optuna()

        # Load and validate calibration parameters
        if common_pars is not None:
            self.common_pars = common_pars
        if unique_pars is not None:
            self.unique_pars = unique_pars
        self.run_args.update(kwargs)  # Update optuna settings

        # Run the optimization
        t0 = sc.tic()
        self.make_study()
        self.run_workers()
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name)
        self.best_pars = sc.objdict(study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        # Collect results
        # Replace with something else, this is fragile
        self.age_results = sc.objdict({sim_label: [] for sim_label in self.sim_labels})
        self.sim_results = sc.objdict({sim_label: [] for sim_label in self.sim_labels})

        if load:
            for slabel, sim in self.sims.items():
                print(f'Loading saved results for {slabel}...')
                for trial in study.trials:
                    n = trial.number
                    try:
                        fileslabel = slabel.replace(' ', '_')
                        filename = self.tmp_filename % (fileslabel, trial.number)
                        results = sc.load(filename)
                        self.sim_results[slabel].append(results['sim'])
                        self.age_results[slabel].append(results['age'])
                        if tidyup:
                            try:
                                os.remove(filename)
                                print(f'    Removed temporary file {filename}')
                            except Exception as E:
                                errormsg = f'Could not remove {filename}: {str(E)}'
                                print(errormsg)
                        print(f'  Loaded trial {n}')
                    except Exception as E:
                        errormsg = f'Warning, could not load {slabel} trial {n}: {str(E)}'
                        print(errormsg)

        # Compare the results
        self.initial_pars, self.par_bounds = self.sim_to_sample_pars()
        self.parse_study(study)

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()

        return self

    def parse_study(self, study):
        """ Parse the study into a data frame -- called automatically """
        best = study.best_params
        self.best_pars = best

        print('Making results structure...')
        results = []
        n_trials = len(study.trials)
        failed_trials = []
        for trial in study.trials:
            data = {'index': trial.number, 'mismatch': trial.value}
            for key, val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i, r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    hpm.warn(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.data = data
        self.df = pd.DataFrame.from_dict(data)
        self.df = self.df.sort_values(by=['mismatch'])  # Sort

        return

    def to_json(self, filename=None, indent=2, **kwargs):
        """
        Convert the data to JSON.
        """
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o, :].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key, val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        self.json = json
        if filename:
            return sc.savejson(filename, json, indent=indent, **kwargs)
        else:
            return json

    def plot(self, slabel=None, res_to_plot=None, fig_args=None, axis_args=None, data_args=None, show_args=None,
             do_save=None, fig_path=None, do_show=True, plot_type='sns.boxplot', **kwargs):
        """
        Plot the calibration results

        Args:
            slabel (strong): label of the sim to plot
            res_to_plot (int): number of results to plot. if None, plot them all
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            show_args (dict): arguments for plot showing
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            plot_type (function or string): plot type
            kwargs (dict): passed to ``hpv.options.with_style()``; see that function for choices
        """

        # Import Seaborn here since slow
        if sc.isstring(plot_type) and plot_type.startswith('sns'):
            import seaborn as sns
            if plot_type.split('.')[1] == 'boxplot':
                extra_args = dict(boxprops=dict(alpha=.3), showfliers=False)
            else:
                extra_args = dict()
            plot_func = getattr(sns, plot_type.split('.')[1])
        else:
            plot_func = plot_type
            extra_args = dict()

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12, 8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        show_args = sc.objdict(sc.mergedicts(dict(show=dict(tight=True, maximize=False)), show_args))
        all_args = sc.objdict(sc.mergedicts(fig_args, axis_args, d_args, show_args))

        # Pull out results to use
        age_results = sc.dcp(self.age_results[slabel])
        sim_results = sc.dcp(self.sim_results[slabel])

        # Get rows and columns
        if not len(age_results) and not len(sim_results):
            errormsg = 'Cannot plot since no results were recorded)'
            raise ValueError(errormsg)
        else:
            all_dates = [[date for date in r.keys() if date != 'bins'] for r in age_results[0].values()]
            dates_per_result = [len(date_list) for date_list in all_dates]
            other_results = len(sim_results[0].keys())
            n_plots = sum(dates_per_result) + other_results
            n_rows, n_cols = sc.get_rows_cols(n_plots)

        # Initialize
        fig, axes = pl.subplots(n_rows, n_cols, **fig_args)
        if n_plots > 1:
            for ax in axes.flat[n_plots:]:
                ax.set_visible(False)
            axes = axes.flatten()
        pl.subplots_adjust(**axis_args)

        # Pull out attributes that don't vary by run
        age_labels = sc.objdict()
        age_results_keys = age_results[0].keys()
        sim_results_keys = sim_results[0].keys()
        target_data = self.target_data[slabel]

        for resname, resdict in zip(age_results_keys, age_results[0].values()):
            age_labels[resname] = [str(int(resdict['bins'][i])) + '-' + str(int(resdict['bins'][i + 1])) for i in
                                   range(len(resdict['bins']) - 1)]
            age_labels[resname].append(str(int(resdict['bins'][-1])) + '+')

        # determine how many results to plot
        if res_to_plot is not None:
            index_to_plot = self.df.iloc[0:res_to_plot, 0].values
            age_results = [age_results[i] for i in index_to_plot]
            sim_results = [sim_results[i] for i in index_to_plot]

        # Make the figure
        with hpo.with_style(**kwargs):

            plot_count = 0

            for rn, resname in enumerate(age_results_keys):
                x = np.arange(len(age_labels[resname]))  # the label locations

                for date in all_dates[rn]:

                    # Initialize axis and data storage structures
                    if n_plots > 1:
                        ax = axes[plot_count]
                    else:
                        ax = axes
                    bins = []
                    genotypes = []
                    values = []

                    # Pull out data
                    thisdatadf = target_data[rn][
                        (target_data[rn].year == float(date)) & (target_data[rn].name == resname)]
                    unique_genotypes = thisdatadf.genotype.unique()

                    # Start making plot
                    if 'genotype' in resname:
                        for g in range(self.ng[slabel]):
                            glabel = self.glabels[slabel][g].upper()
                            # Plot data
                            if glabel in unique_genotypes:
                                ydata = np.array(thisdatadf[thisdatadf.genotype == glabel].value)
                                ax.scatter(x, ydata, color=self.result_args[slabel][resname].color[g], marker='s',
                                           label=f'Data - {glabel}')

                            # Construct a dataframe with things in the most logical order for plotting
                            for run_num, run in enumerate(age_results):
                                genotypes += [glabel] * len(x)
                                bins += x.tolist()
                                values += list(run[resname][date][g])

                        # Plot model
                        modeldf = pd.DataFrame({'bins': bins, 'values': values, 'genotypes': genotypes})
                        ax = plot_func(ax=ax, x='bins', y='values', hue="genotypes", data=modeldf, **extra_args)

                    else:
                        # Plot data
                        ydata = np.array(thisdatadf.value)
                        ax.scatter(x, ydata, color='k', marker='s', label='Data')

                        # Construct a dataframe with things in the most logical order for plotting
                        for run_num, run in enumerate(age_results):
                            bins += x.tolist()
                            values += list(run[resname][date])

                        # Plot model
                        modeldf = pd.DataFrame({'bins': bins, 'values': values})
                        ax = plot_func(ax=ax, x='bins', y='values', data=modeldf, color='b', **extra_args)

                    # Set title and labels
                    ax.set_xlabel('Age group')
                    ax.set_title(f'{slabel}, {resname}, {date}')
                    ax.legend()
                    ax.set_xticks(x, age_labels[resname])
                    plot_count += 1

            for rn, resname in enumerate(sim_results_keys):
                ax = axes[plot_count]
                bins = sc.autolist()
                values = sc.autolist()
                thisdatadf = target_data[rn + sum(dates_per_result)][
                    target_data[rn + sum(dates_per_result)].name == resname]
                ydata = np.array(thisdatadf.value)
                x = np.arange(len(ydata))
                ax.scatter(x, ydata, color=pl.cm.Reds(0.95), marker='s', label='Data')

                # Construct a dataframe with things in the most logical order for plotting
                for run_num, run in enumerate(sim_results):
                    bins += x.tolist()
                    if sc.isnumber(run[resname]):
                        values += sc.promotetolist(run[resname])
                    else:
                        values += run[resname].tolist()
                # Plot model
                modeldf = pd.DataFrame({'bins': bins, 'values': values})
                ax = plot_func(ax=ax, x='bins', y='values', data=modeldf, **extra_args)

                # Set title and labels
                date = thisdatadf.year[0]
                # ax.set_xlabel('Genotype')
                ax.set_title(f'{resname}, {date}')
                ax.legend()
                # ax.set_xticks(x, ['16', '18', 'H5', 'OHR'])
                plot_count += 1

        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)
