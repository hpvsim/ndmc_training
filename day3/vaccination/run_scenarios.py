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
            sims += rs.make_sim(location=location, calib_pars=calib_pars, debug=debug, vx_intv=intv, end=2100, seed=seed)
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
    do_run = True
    do_plot = False
    scen_labels = ['baseline', 'males', 'females']

    # Run
    if do_run:
        calib_pars = sc.loadobj('results/india_pars.obj')
        msim = run_sims(calib_pars=calib_pars)
        mlist = msim.split(chunks=len(scen_labels))
        msim_dict = sc.objdict({scen_labels[i]: mlist[i].reduce(output=True).results for i in range(len(scen_labels))})
        sc.saveobj(f'results/vx.scens', msim_dict)
    else:
        msim_dict = sc.loadobj('results/vx.scens')

    if do_plot:
        import pylab as pl
        colors = sc.gridcolors(3)
        sc.options(fontsize=20)
        fig, ax = pl.subplots(figsize=(18, 14))
        to_plot = 'asr_cancer_incidence'

        for sno, sname, mres in msim_dict.enumitems():
            years = mres.year[70:]
            best = mres[to_plot][70:]
            low = mres[to_plot].low[70:]
            high = mres[to_plot].high[70:]
            ax.plot(years, best, color=colors[sno], label=sname.capitalize())
            ax.fill_between(years, low, high, alpha=0.5, color=colors[sno])

        ax.set_ylim(bottom=0)
        ax.set_title('ASR cancer incidence')
        pl.legend()
        fig.tight_layout()
        fig_name = 'vx_scens.png'
        sc.savefig(fig_name, dpi=100)
        fig.show()
