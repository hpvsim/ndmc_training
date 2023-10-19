'''
Run HPVsim scenarios for NDMC

Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
'''


#%% General settings

# Turn off multithreading (improves performance in some cases)
import os
os.environ['SCIRIS_NUM_THREADS'] = '1'
import sciris as sc

# Standard imports
import hpvsim as hpv

# Imports from this repository
import run_sim as rs

location = 'india'
debug = False
n_seeds = [3, 1][debug]  # How many seeds to use for stochasticity in projections


#%% Functions
def make_vx_scenarios():
    # Construct the scenarios
    vx_scenarios = dict(
        baseline=None,
        male_vx=hpv.routine_vx(
            prob=0.9,
            sex=1,
            start_year=2020,
            product='bivalent',
            age_range=(9, 10),
            label='male vx'
        ),
        female_vx=hpv.routine_vx(
            prob=0.9,
            sex=0,
            start_year=2020,
            product='bivalent',
            age_range=(9, 10),
            label='female vx'
        )
    )
    return vx_scenarios


def make_sims(calib_pars=None):
    """ Set up scenarios to compare algorithms """
    vx_scenarios = make_vx_scenarios()
    sims = sc.autolist()
    for name, intv in vx_scenarios.items():
        for seed in range(n_seeds):
            sims += rs.make_sim(location=location, calib_pars=calib_pars, debug=debug, vx_intv=intv, end=2060, seed=seed)
    return sims


def run_sims(calib_pars=None):
    """ Run the simulations """
    sims = make_sims(calib_pars=calib_pars)
    msim = hpv.parallel(sims)
    msim.compare()
    return msim


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    # Run
    calib_pars = sc.loadobj('results/india_pars.obj')
    msim = run_sims(calib_pars=calib_pars)
    sc.saveobj('vx.msim', msim)

    # # Plot
    # to_plot = [
    #     'asr_cancer_incidence',
    # ]
    # msim.plot(to_plot, color_by_sim=True, max_sims=len(msim))
