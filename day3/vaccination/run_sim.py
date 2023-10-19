'''
Define the HPVsim simulations that are used as
the basis for the scenarios.

'''

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import pars_data as dp
import pars_scenarios as sp
import utils as ut


#%% Settings and filepaths

# Locations -- comment out a line to not run
locations = [
    'india'
]

# Debug switch
debug = 0 # Run with smaller population sizes and in serial
do_shrink = True # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


#%% Simulation creation functions
def make_sim(location=None, calib_pars=None, debug=0, vx_intv=None, end=None, datafile=None, seed=1):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''
    if end is None:
        end = 2060

    # Parameters
    pars = dict(
        n_agents       = [20e3,1e3][debug],
        dt             = [0.25,1.0][debug],
        start          = [1950,1980][debug],
        end            = end,
        network        = 'default',
        location       = location,
        genotypes      = [16, 18, 'hi5', 'ohr'],
        debut          = dp.debut[location],
        mixing         = dp.mixing[location],
        layer_probs    = dp.layer_probs[location],
        partners       = dp.partners[location],
        dur_pship      = dp.dur_pship[location],
        init_hpv_dist  = dp.init_genotype_dist[location],
        init_hpv_prev  = {
            'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
            'm'             : np.array([ 0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f'             : np.array([ 0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        condoms        = dict(m=0.01, c=0.1, o=0.1),
        eff_condoms    = 0.5,
        ms_agent_ratio = 100,
        verbose        = 0.0,
    )

    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    # Analyzers
    analyzers = sc.autolist()

    # Interventions
    interventions = sc.autolist()
    if vx_intv is not None:
        interventions += vx_intv

    sim = hpv.Sim(pars=pars, analyzers=analyzers, interventions=interventions, datafile=datafile, rand_seed=seed)

    return sim


#%% Simulation running functions

def run_sim(location=None, use_calib_pars=True, vx_intv=None,
            debug=0, seed=0, label=None, meta=None, verbose=0.1, end=None,
            do_save=False, die=False, calib_filestem=''):
    ''' Assemble the parts into a complete sim and run it '''

    # Decide what message to print
    if meta is not None:
        msg = f'Making sim {meta.inds} ({meta.count} of {meta.n_sims}) for {location}'
    else:
        msg = f'Making sim for {location}'
    if debug: msg += ' IN DEBUG MODE'
    print(msg)

    # Make any parameter updates
    if use_calib_pars:
        file = f'{ut.resfolder}/{location}_pars{calib_filestem}.obj'
        try:
            calib_pars = sc.loadobj(file)
        except:
            errormsg = 'Calibration parameters cannot be loaded from disk. Try running load_calib to generate them.'
            raise ValueError(errormsg)

    # Make arguments
    sim = make_sim(location=location, calib_pars=calib_pars, vx_intv=vx_intv, end=end, debug=debug)

   # Store metadata
    sim.meta = sc.objdict()
    if meta is not None:
        sim.meta = meta # Copy over meta info
    else:
        sim.meta = sc.objdict()
    sim.meta.location = location # Store location in an easy-to-access place
    sim['rand_seed'] = seed # Set seed
    sim.label = f'{label}--{location}' # Set label

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()
        
    if do_save:
        sim.save(f'{ut.resfolder}/{location}.sim')
    
    return sim


def run_sims(locations=None, *args, **kwargs):
    ''' Run multiple simulations in parallel '''
    
    kwargs = sc.mergedicts(dict(debug=debug), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(location=locations), kwargs=kwargs, serial=debug, die=True)
    sims = sc.objdict({location:sim for location,sim in zip(locations, simlist)}) # Convert from a list to a dict
    
    return sims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    
    # Run a single sim per location -- usually locally, can be used for sanity checking and debugging

    vx_scen = dict(
        routine_start_year=2020,
        catch_up_year=2025,
        end_year=2030,
        vx_coverage=0.9,
        age_range=(9, 14),
        gender_neutral=True
    )
    screen_scen = {}  # Not varying S&T
    sim0 = run_sim(location='india', end=2030, vx_intv=vx_scen)

    sim0.plot()
    T.toc('Done')

