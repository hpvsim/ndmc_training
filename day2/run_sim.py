"""
Define an HPVsim simulation for India
"""

# Standard imports
import numpy as np
import sciris as sc
import pylab as pl
import hpvsim as hpv

# Imports from this repository
import behavior_inputs as bi

# %% Settings and filepaths

# Debug switch
debug = 0  # Run with smaller population sizes and in serial
do_shrink = True  # Do not keep people when running sims (saves memory)

# Run settings
n_trials    = [100, 2][debug]  # How many trials to run for calibration
n_workers   = [10, 4][debug]    # How many cores to use

# Save settings
do_save = True
save_plots = True

# List of what to run -- uncomment lines to run them
to_run = [
    # 'run_sim',
    'run_calib',
    # 'plot_calib'
]


# %% Simulation creation functions
def make_sim(calib_pars=None, analyzers=[], debug=0, datafile=None, seed=1):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''

    pars = dict(
        n_agents=[10e3, 1e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 1980][debug],
        end=2020,
        network='default',
        genotypes=[16, 18, 'hi5', 'ohr'],
        location='india',
        debut=dict(f=dict(dist='lognormal', par1=14.8, par2=2.),
                   m=dict(dist='lognormal', par1=17.0, par2=2.)),
        mixing=bi.default_mixing,
        layer_probs=bi.default_layer_probs,
        partners=bi.default_partners,
        init_hpv_dist=dict(hpv16=0.4, hpv18=0.15, hi5=0.15, ohr=0.3),
        init_hpv_prev={
            'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
            'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio=100,
        verbose=0.0,
    )

    genotype_pars = {
        16: {
            'sev_fn': dict(form='logf2', k=0.25, x_infl=0, ttc=30)
        }
    }

    pars['genotype_pars'] = dict(
        hpv16=dict(dur_episomal=dict(dist='lognormal', par1=38, par2=1500)),
        hpv18=dict(dur_episomal=dict(dist='lognormal', par1=19, par2=730)),
        hi5=dict(dur_episomal=dict(dist='lognormal', par1=22, par2=2000)),
        ohr=dict(dur_episomal=dict(dist='lognormal', par1=22, par2=2000))
    )

    # If calibration parameters have been supplied, use them here
    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    # Create the sim
    sim = hpv.Sim(pars=pars, datafile=datafile, analyzers=analyzers, rand_seed=seed)

    return sim


# %% Simulation running functions
def run_sim(calib_pars=None, analyzers=None, debug=0, datafile=None, 
            seed=1, verbose=.1, label='', do_save=False):
    # Make sim
    sim = make_sim(
        debug=debug,
        seed=seed,
        datafile=datafile,
        analyzers=analyzers,
        calib_pars=calib_pars
    )
    sim.label = f'Sim {seed} {label}'

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()

    # Optinally save
    if do_save:
        sim.save(f'results/india.sim')

    return sim


def run_calib(n_trials=None, n_workers=None, do_save=True, filestem=''):

    sim = make_sim()
    datafiles = [
        'data/india_hpv_prevalence.csv',
        'data/india_cancer_cases.csv',
        'data/india_cin1_types.csv',
        'data/india_cin3_types.csv',
        'data/india_cancer_types.csv',
    ]

    # Define the calibration parameters
    calib_pars = dict(
        beta=[0.05, 0.02, 0.5, 0.01],
    )
    genotype_pars = dict(
        hpv16=dict(
            transform_prob=[10e-10, 4e-10, 20e-10, 1e-10],
            sev_fn=dict(k=[0.25, 0.15, 0.4, 0.05]),
        ),
        hpv18=dict(
            transform_prob=[6e-10, 4e-10, 10e-10, 1e-10],
            sev_fn=dict(k=[0.2, 0.1, 0.35, 0.05]),
        ),
        hi5=dict(
                transform_prob=[3e-10, 2e-10, 5e-10, 1e-10],
                sev_fn=dict(k=[0.05, 0.04, 0.2, 0.01]),
            ),
        ohr=dict(
            transform_prob=[3e-10, 2e-10, 5e-10, 1e-10],
            sev_fn=dict(k=[0.05, 0.04, 0.2, 0.01]),
        ),
    )

    calib = hpv.Calibration(sim,
        calib_pars=calib_pars,
        genotype_pars=genotype_pars,
        name='india_calib',
        datafiles=datafiles,
        total_trials=n_trials,
        n_workers=n_workers,
    )
    calib.calibrate()
    filename = f'india_calib{filestem}'
    if do_save:
        if do_shrink:
            calib.sim.shrink()
        sc.saveobj(f'results/{filename}.obj', calib)

    print(f'Best pars are {calib.best_pars}')

    return sim, calib


def plot_calib(which_pars=0, save_pars=True, filestem=''):
    filename = f'india_calib{filestem}'
    calib = sc.load(f'results/{filename}.obj')

    fig = calib.plot(res_to_plot=200, plot_type='sns.boxplot', do_save=False)
    fig.tight_layout()
    sc.savefig(f'figures/{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        trial_pars = sc.autolist()
        for i in range(100):
            trial_pars += calib.trial_pars_to_sim_pars(which_pars=i)
        sc.save(f'results/india_pars{filestem}.obj', calib_pars)
        sc.save(f'results/india_pars{filestem}_all.obj', trial_pars)

    return calib


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()  # Start a timer

    if 'run_sim' in to_run:
        sim = run_sim(label='Uncalibrated')  # Run the simulation
        sim.plot(do_show=False)  # Plot the simulation
        try:
            calib_pars = sc.loadobj('results/india_pars.obj')  # Load parameters from a previous calibration
            sim = run_sim(calib_pars=calib_pars, label='Calibrated')  # Run the simulation
            sim.plot(do_show=False)  # Plot the simulation
        except:
            print('Could not plot calibrated version; please run run_calib, then plot_calib')
        pl.show()

    if 'run_calib' in to_run:
        sim, calib = run_calib(n_trials=n_trials, n_workers=n_workers, filestem='', do_save=True)

    if 'plot_calib' in to_run:
        calib = plot_calib(save_pars=True, filestem='')

    T.toc()  # Print out how long the run took
