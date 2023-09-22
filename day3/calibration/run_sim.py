"""
Define the HPVsim simulation objects.
"""

# Turn off multithreading (improves performance in some cases)
import os
os.environ['SCIRIS_NUM_THREADS'] = '1'
import sciris as sc

# Standard imports
import numpy as np
import hpvsim as hpv

# Imports from this repository
import behavior_inputs as bi
import locations as loc


# %% Settings and filepaths
debug = 0  # Run with smaller population sizes and in serial
do_shrink = True  # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


# %% Simulation creation functions
def make_sim(location=None, calib_pars=None, debug=0, analyzers=[], datafile=None, seed=1):
    ''' Define the simulation, including parameters, analyzers, and interventions '''

    pars = dict(
        n_agents=[10e3, 1e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 1980][debug],
        end=2020,
        network='default',
        genotypes=[16, 18, 'hi5', 'ohr'],
        location=location,
        debut=bi.debut[location],
        mixing=bi.mixing[location],
        layer_probs=bi.layer_probs[location],
        partners=bi.partners[location],
        init_hpv_dist=bi.init_genotype_dist[location],
        init_hpv_prev={
            'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
            'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio=100,
        verbose=0.0,
    )

    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    interventions = sc.autolist()

    sim = hpv.Sim(pars=pars, interventions=interventions, analyzers=analyzers, datafile=datafile, rand_seed=seed)

    return sim


# %% Simulation running functions
def run_sim(location=None, analyzers=None, debug=0, seed=0, verbose=0.2, do_save=False, calib_pars=None):

    if analyzers is None:
        analyzers = sc.autolist()
    else:
        analyzers = sc.promotetolist(analyzers)

    # Make sim
    sim = make_sim(
        location=location,
        debug=debug,
        analyzers=analyzers,
        calib_pars=calib_pars
    )
    sim['rand_seed'] = seed
    sim.label = f'{location}--{seed}'

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()

    if do_save:
        sim.save(f'results/{location.replace(" ","_")}.sim')

    return sim


def run_sims(locations=None, verbose=-1, analyzers=None, *args, **kwargs):
    """ Run multiple simulations in parallel """

    kwargs = sc.mergedicts(dict(debug=debug, verbose=verbose, analyzers=analyzers), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(location=locations), kwargs=kwargs, serial=debug, die=True)
    sims = sc.objdict({location: sim for location, sim in zip(locations, simlist)})  # Convert from a list to a dict

    return sims




# %% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    locations = loc.locations
    sims = run_sims(locations=locations)

    T.toc('Done')

