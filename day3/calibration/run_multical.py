'''
Run a pooled calibration of HPVsim to multiple states to produce estimates
of burden of cervical cancer over 2020-2060.
'''

# Turn off multithreading (improves performance in some cases)
import os
os.environ['SCIRIS_NUM_THREADS'] = '1'
import sciris as sc

# Imports from this repository
import run_sim as rs
import calibration as cal
import locations as loc

# Comment out to not run
to_run = [
    # 'run_calibration',
    'plot_calibration',
]

# Other settings
debug = True  # Smaller runs
do_save = True
locations = loc.locations

# Run settings for calibration (dependent on debug)
n_trials    = [1000, 2][debug]  # How many trials to run for calibration
n_workers   = [40, 1][debug]    # How many cores to use

########################################################################
# Run calibration
########################################################################
def make_unique_priors(locations=None):
    ''' Make priors for the parameters that vary across settings '''

    unique_pars = dict()
    for location in locations:
        unique_pars[location] = dict(
            calib_pars = dict(
                beta=[0.2, 0.1, 0.3, 0.01],
            ),
            genotype_pars = dict(
                hi5=dict(
                    transform_prob=[3e-10, 2e-10, 5e-10, 1e-10],
                    sev_fn=dict(k=[0.05, 0.04, 0.8, 0.01]),
                    rel_beta=[0.75, 0.7, 1.25, 0.05]
                ),
                ohr=dict(
                    transform_prob=[3e-10, 2e-10, 5e-10, 1e-10],
                    sev_fn=dict(k=[0.05, 0.04, 0.8, 0.01]),
                    rel_beta=[0.75, 0.7, 1.25, 0.05]
                ),
            )
        )

    return unique_pars


def make_datafiles(locations):
    ''' Get the relevant datafiles for the selected locations '''
    datafiles = dict()
    for location in locations:
        dflocation = location.replace(' ', '_')
        datafiles[location] = [
            f'data/{dflocation}_cancer_cases.csv',
            f'data/{dflocation}_asr_cancer_incidence.csv',
        ]
    return datafiles


def run_calib(locations=None, n_trials=None, n_workers=None,
              do_plot=False, do_save=True, filestem=''):

    # Define shared calibration parameters - same values used across sims
    common_pars = dict(
        genotype_pars=dict(
            hpv16=dict(
                transform_prob=[10e-10, 4e-10, 20e-10, 1e-10],
                sev_fn=dict(
                    k=[0.25, 0.15, 0.4, 0.05],
                ),
                dur_episomal=dict(
                    par1=[2.5, 1.5, 5, 0.5],
                    par2=[7, 4, 15, 0.5])
            ),
            hpv18=dict(
                transform_prob=[6e-10, 4e-10, 10e-10, 1e-10],
                sev_fn=dict(
                    k=[0.2, 0.1, 0.35, 0.05],
                ),
                dur_episomal=dict(
                    par1=[2.5, 1.5, 3, 0.5],
                    par2=[7, 4, 15, 0.5]),
                rel_beta=[0.75, 0.7, 0.95, 0.05]
            ),
        )
    )

    unique_pars = make_unique_priors(locations)

    sims = []
    for location in locations:
        sim = rs.make_sim(location)
        sim.label = location
        sims.append(sim)

    calib = cal.MultiCal(
        sims,
        common_pars=common_pars,
        unique_pars=unique_pars,
        name=f'multical',
        datafiles=make_datafiles(locations),
        load_if_exists=True,
        db_name='multical0105.db',
        total_trials=n_trials,
        n_workers=n_workers,
        keep_db=False,
    )
    calib.calibrate()

    filename = f'multical{filestem}'
    if do_plot:
        for location in locations:
            calib.plot(slabel=location, do_save=True, fig_path=f'figures/{filename}_{location}.png')
    if do_save:
        sc.saveobj(f'results/{filename}.obj', calib)

    print(f'Best pars are {calib.best_pars}')

    return sims, calib


########################################################################
# Load pre-run calibration
########################################################################
def load_calib(filestem=None, locations=None, do_plot=True, max_to_plot=50, which_pars=0, save_pars=True):

    calib = sc.load(f'results/multical{filestem}.obj')
    res_to_plot = min(max_to_plot, n_trials)

    if save_pars:
        sims = []
        for location in locations:
            pars_file = f'results/{location}_multical{filestem}_pars.obj'
            calib_pars = calib.trial_pars_to_sim_pars(slabel=location, which_pars=which_pars)
            sc.save(pars_file, calib_pars)

    if do_plot:
        for location in locations:
            fig = calib.plot(slabel=location, res_to_plot=res_to_plot, plot_type='sns.boxplot')
            fig.suptitle(f'Calibration results, {location.capitalize()}')
            fig.tight_layout()
            fig.savefig(f'figures/multical{filestem}_{location}.png')


    return calib, sims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    filestem = '_sep22'
    locations = loc.locations

    # Run calibration
    if 'run_calibration' in to_run:
        sims, calib = run_calib(locations=locations, n_trials=n_trials, n_workers=n_workers, do_save=do_save, do_plot=False, filestem=filestem)

    # Load the calibration, plot it, and save the best parameters -- usually locally
    if 'plot_calibration' in to_run:
        calib, sims = load_calib(filestem=filestem, locations=locations, do_plot=True, save_pars=True)

    T.toc('Done')