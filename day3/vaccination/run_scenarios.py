'''
Run HPVsim scenarios for NDMC

Note: requires an HPC to run with debug=False; with debug=True, should take 5-15 min
to run.
'''


#%% General settings

# Standard imports
import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv

# Imports from this repository
import run_sim as rs
import utils as ut
import pars_scenarios as sp

# Comment out to not run
to_run = [
    'run_scenarios',
]

# Comment out locations to not run
locations = [
    'india'
]

debug = 0
n_seeds = [3, 1][debug] # How many seeds to use for stochasticity in projections

label_dict = {
    'No screening': 'no_screening',
    '10% S&T cov': '10snt',
    '30% S&T cov': '30snt',
    '50% S&T cov': '50snt',
    '70% S&T cov': '70snt',
    '90% S&T cov': '90snt',

    '10% VIA S&T cov': '10_via_snt',
    '30% VIA S&T cov': '30_via_snt',
    '50% VIA S&T cov': '50_via_snt',
    '70% VIA S&T cov': '70_via_snt',
    '90% VIA S&T cov': '90_via_snt',

    'No vaccine': 'no_vaccine',
    'Vx, 10% cov, 9-14': '10vx_9_to_14',
    'Vx, 20% cov, 9-14': '20vx_9_to_14',
    'Vx, 30% cov, 9-14': '30vx_9_to_14',
    'Vx, 40% cov, 9-14': '40vx_9_to_14',
    'Vx, 50% cov, 9-14': '50vx_9_to_14',
    'Vx, 60% cov, 9-14': '60vx_9_to_14',
    'Vx, 70% cov, 9-14': '70vx_9_to_14',
    'Vx, 80% cov, 9-14': '80vx_9_to_14',
    'Vx, 90% cov, 9-14': '90vx_9_to_14',
    'Vx, 100% cov, 9-14': '100vx_9_to_14',

    'Vx, 10% cov, 9-18': '10vx_9_to_18',
    'Vx, 20% cov, 9-18': '20vx_9_to_18',
    'Vx, 30% cov, 9-18': '30vx_9_to_18',
    'Vx, 40% cov, 9-18': '40vx_9_to_18',
    'Vx, 50% cov, 9-18': '50vx_9_to_18',
    'Vx, 60% cov, 9-18': '60vx_9_to_18',
    'Vx, 70% cov, 9-18': '70vx_9_to_18',
    'Vx, 80% cov, 9-18': '80vx_9_to_18',
    'Vx, 90% cov, 9-18': '90vx_9_to_18',
    'Vx, 100% cov, 9-18': '100vx_9_to_18',

    'Vx, 10% cov, 9-24': '10vx_9_to_24',
    'Vx, 20% cov, 9-24': '20vx_9_to_24',
    'Vx, 30% cov, 9-24': '30vx_9_to_24',
    'Vx, 40% cov, 9-24': '40vx_9_to_24',
    'Vx, 50% cov, 9-24': '50vx_9_to_24',
    'Vx, 60% cov, 9-24': '60vx_9_to_24',
    'Vx, 70% cov, 9-24': '70vx_9_to_24',
    'Vx, 80% cov, 9-24': '80vx_9_to_24',
    'Vx, 90% cov, 9-24': '90vx_9_to_24',
    'Vx, 100% cov, 9-24': '100vx_9_to_24',

    'Vx, 10% cov, 9-14, gender neutral': '10vx_9_to_14_gender_neutral',
    'Vx, 20% cov, 9-14, gender neutral': '20vx_9_to_14_gender_neutral',
    'Vx, 30% cov, 9-14, gender neutral': '30vx_9_to_14_gender_neutral',
    'Vx, 40% cov, 9-14, gender neutral': '40vx_9_to_14_gender_neutral',
    'Vx, 50% cov, 9-14, gender neutral': '50vx_9_to_14_gender_neutral',
    'Vx, 60% cov, 9-14, gender neutral': '60vx_9_to_14_gender_neutral',
    'Vx, 70% cov, 9-14, gender neutral': '70vx_9_to_14_gender_neutral',
    'Vx, 80% cov, 9-14, gender neutral': '80vx_9_to_14_gender_neutral',
    'Vx, 90% cov, 9-14, gender neutral': '90vx_9_to_14_gender_neutral',
    'Vx, 100% cov, 9-14, gender neutral': '100vx_9_to_14_gender_neutral',

    'Vx, 10% cov, 9-14, gender neutral, 45% eff': '10vx_9_to_14_gender_neutral_male_redux',
    'Vx, 20% cov, 9-14, gender neutral, 45% eff': '20vx_9_to_14_gender_neutral_male_redux',
    'Vx, 30% cov, 9-14, gender neutral, 45% eff': '30vx_9_to_14_gender_neutral_male_redux',
    'Vx, 40% cov, 9-14, gender neutral, 45% eff': '40vx_9_to_14_gender_neutral_male_redux',
    'Vx, 50% cov, 9-14, gender neutral, 45% eff': '50vx_9_to_14_gender_neutral_male_redux',
    'Vx, 60% cov, 9-14, gender neutral, 45% eff': '60vx_9_to_14_gender_neutral_male_redux',
    'Vx, 70% cov, 9-14, gender neutral, 45% eff': '70vx_9_to_14_gender_neutral_male_redux',
    'Vx, 80% cov, 9-14, gender neutral, 45% eff': '80vx_9_to_14_gender_neutral_male_redux',
    'Vx, 90% cov, 9-14, gender neutral, 45% eff': '90vx_9_to_14_gender_neutral_male_redux',
    'Vx, 100% cov, 9-14, gender neutral, 45% eff': '100vx_9_to_14_gender_neutral_male_redux',

    'Vx, 10% cov, 9-14, 9V': '10vx_9_to_14_9v',
    'Vx, 20% cov, 9-14, 9V': '20vx_9_to_14_9v',
    'Vx, 30% cov, 9-14, 9V': '30vx_9_to_14_9v',
    'Vx, 40% cov, 9-14, 9V': '40vx_9_to_14_9v',
    'Vx, 50% cov, 9-14, 9V': '50vx_9_to_14_9v',
    'Vx, 60% cov, 9-14, 9V': '60vx_9_to_14_9v',
    'Vx, 70% cov, 9-14, 9V': '70vx_9_to_14_9v',
    'Vx, 80% cov, 9-14, 9V': '80vx_9_to_14_9v',
    'Vx, 90% cov, 9-14, 9V': '90vx_9_to_14_9v',
    'Vx, 100% cov, 9-14, 9V': '100vx_9_to_14_9v',

    'Vx, 10% cov, 9-18, 9V': '10vx_9_to_18_9v',
    'Vx, 20% cov, 9-18, 9V': '20vx_9_to_18_9v',
    'Vx, 30% cov, 9-18, 9V': '30vx_9_to_18_9v',
    'Vx, 40% cov, 9-18, 9V': '40vx_9_to_18_9v',
    'Vx, 50% cov, 9-18, 9V': '50vx_9_to_18_9v',
    'Vx, 60% cov, 9-18, 9V': '60vx_9_to_18_9v',
    'Vx, 70% cov, 9-18, 9V': '70vx_9_to_18_9v',
    'Vx, 80% cov, 9-18, 9V': '80vx_9_to_18_9v',
    'Vx, 90% cov, 9-18, 9V': '90vx_9_to_18_9v',
    'Vx, 100% cov, 9-18, 9V': '100vx_9_to_18_9v',

    'Vx, 10% cov, 9-24, 9V': '10vx_9_to_24_9v',
    'Vx, 20% cov, 9-24, 9V': '20vx_9_to_24_9v',
    'Vx, 30% cov, 9-24, 9V': '30vx_9_to_24_9v',
    'Vx, 40% cov, 9-24, 9V': '40vx_9_to_24_9v',
    'Vx, 50% cov, 9-24, 9V': '50vx_9_to_24_9v',
    'Vx, 60% cov, 9-24, 9V': '60vx_9_to_24_9v',
    'Vx, 70% cov, 9-24, 9V': '70vx_9_to_24_9v',
    'Vx, 80% cov, 9-24, 9V': '80vx_9_to_24_9v',
    'Vx, 90% cov, 9-24, 9V': '90vx_9_to_24_9v',
    'Vx, 100% cov, 9-24, 9V': '100vx_9_to_24_9v',
}
#%% Functions

def make_msims(sims, use_mean=True, save_msims=False):
    '''
    Utility to take a slice of sims and turn it into a multisim
    '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_sc, i_vx, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except seed matches
        assert i_sc == sim.meta.inds[0]
        assert i_vx == sim.meta.inds[1]
        assert (s == 0) or i_s != sim.meta.inds[2]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_sc, i_vx]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')

    print(f'Processing multisim {msim.meta.vals.values()}...')
    if save_msims:  # Warning, generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{ut.resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim

def run_scens(location=None, screen_intvs=None, vx_intvs=None, # Input data
              calib_filestem='', debug=0, n_seeds=2, verbose=-1# Sim settings
              ):
    '''
    Run all screening/triage product scenarios for a given location
    '''

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(screen_intvs) * len(vx_intvs) * n_seeds

    for i_sc, sc_label, screen_scen_pars in screen_intvs.enumitems():
        for i_vx, vx_label, vx_scen_pars in vx_intvs.enumitems():
            for i_s in range(n_seeds):  # n seeds
                count += 1
                meta = sc.objdict()
                meta.count = count
                meta.n_sims = n_sims
                meta.inds = [i_sc, i_vx, i_s]
                meta.vals = sc.objdict(sc.mergedicts(screen_scen_pars, vx_scen_pars,
                                                     dict(seed=i_s, screen_scen=sc_label, vx_scen=vx_label)))
                ikw.append(
                    sc.objdict(screen_intv=screen_scen_pars, vx_intv=vx_scen_pars, seed=i_s))
                ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(use_calib_pars=True, calib_filestem=calib_filestem, verbose=verbose, debug=debug, location=location)
    all_sims = sc.parallelize(rs.run_sim, iterkwargs=ikw, kwargs=kwargs)

    # Rearrange sims
    sims = np.empty((len(screen_intvs), len(vx_intvs),  n_seeds), dtype=object)

    for sim in all_sims:  # Unflatten array
        i_sc, i_vx, i_s = sim.meta.inds
        sims[i_sc, i_vx, i_s] = sim

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_sc in range(len(screen_intvs)):
        for i_vx in range(len(vx_intvs)):
            sim_seeds = sims[i_sc, i_vx, :].tolist()
            all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    msims = np.empty((len(screen_intvs), len(vx_intvs)), dtype=object)
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    for msim in all_msims:
        i_sc, i_vx = msim.meta.inds
        msims[i_sc, i_vx] = msim
        df = pd.DataFrame()
        df['year']                      = msim.results['year']
        df['cancers']                   = msim.results['cancers'][:] # TODO: process in a loop
        df['cancers_low']               = msim.results['cancers'].low
        df['cancers_high']              = msim.results['cancers'].high
        df['cancer_incidence']          = msim.results['cancer_incidence'][:]
        df['cancer_incidence_high']     = msim.results['cancer_incidence'].high
        df['cancer_incidence_low']      = msim.results['cancer_incidence'].low
        df['asr_cancer_incidence']      = msim.results['asr_cancer_incidence'][:]
        df['asr_cancer_incidence_low']  = msim.results['asr_cancer_incidence'].low
        df['asr_cancer_incidence_high'] = msim.results['asr_cancer_incidence'].high
        df['cancer_deaths']             = msim.results['cancer_deaths'][:]
        df['cancer_deaths_low']         = msim.results['cancer_deaths'].low
        df['cancer_deaths_high']        = msim.results['cancer_deaths'].high
        df['n_vaccinated']              = msim.results['n_vaccinated'][:]
        df['n_vaccinated_low']          = msim.results['n_vaccinated'].low
        df['n_vaccinated_high']         = msim.results['n_vaccinated'].high
        df['location'] = location

        # Store metadata about run #TODO: fix this
        df['vx_scen'] = msim.meta.vals['vx_scen']
        df['screen_scen'] = msim.meta.vals['screen_scen']
        filestem_label= f'{label_dict[msim.meta.vals["vx_scen"]]}_{label_dict[msim.meta.vals["screen_scen"]]}'
        sc.saveobj(f'{ut.resfolder}/{location}_{filestem_label}.obj', df)
        dfs += df

    alldf = pd.concat(dfs)

    return alldf, msims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)

    if 'run_scenarios' in to_run:
        for location in locations:
            # Construct the scenarios
            screen_scens = sc.objdict({
                'No screening': {},
                '70% S&T cov': dict(
                    primary='hpv',
                    screen_coverage=0.7,
                ),
                '70% VIA S&T cov': dict(
                    primary='via',
                    screen_coverage=0.7,
                ),
            })

            vx_scens = sc.objdict({
                'No vaccine': {},

                'Vx, 90% cov, 9-14': dict(
                    vx_coverage=0.9,
                    age_range=(9, 14)
                ),
                'Vx, 90% cov, 9-14, gender neutral': dict(
                    vx_coverage=0.9,
                    age_range=(9, 14),
                    gender_neutral=True
                ),
                'Vx, 90% cov, 9-14, gender neutral, 45% eff': dict(
                    vx_coverage=0.9,
                    age_range=(9, 14),
                    gender_neutral=True,
                    male_eff_redux=True
                ),
                'Vx, 90% cov, 9-18': dict(
                    vx_coverage=0.9,
                    age_range=(9, 18)
                ),
                'Vx, 90% cov, 9-24': dict(
                    vx_coverage=0.9,
                    age_range=(9, 24)
                ),
                'Vx, 90% cov, 9-14, 9V': dict(
                    vx_coverage=0.9,
                    product='nonavalent',
                    age_range=(9, 14)
                ),
                'Vx, 90% cov, 9-18, 9V': dict(
                    vx_coverage=0.9,
                    product='nonavalent',
                    age_range=(9, 18)
                ),
                'Vx, 90% cov, 9-24, 9V': dict(
                    vx_coverage=0.9,
                    product='nonavalent',
                    age_range=(9, 24)
                ),
            })

            alldf, msims = run_scens(screen_intvs=screen_scens, vx_intvs=vx_scens, calib_filestem='',
                                     n_seeds=n_seeds, location=location, debug=debug)

