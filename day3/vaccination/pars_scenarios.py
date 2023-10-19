'''
Define parameters used in scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import pars_data as dp


def get_vx_intvs(routine_start_year=2020, catch_up_year=2020, end_year=2050, vx_coverage=0.9, age_range=(9,14),
                 gender_neutral=False, male_eff_redux=False, product='bivalent'):

    catchup_age = (age_range[0]+1, age_range[1])
    routine_age = (age_range[0], age_range[0]+1)
    sex='m'
    intvs = sc.autolist()

    if male_eff_redux:
        prod = hpv.default_vx(prod_name=product)
        prod.imm_init = dict(dist='beta_mean', par1=0.45, par2=0.025)

        routine_boys_vx = hpv.routine_vx(
            prob=vx_coverage,
            start_year=routine_start_year,
            end_year=end_year,
            product=prod,
            age_range=routine_age,
            sex='m',
            label='Boys routine vx'
        )

        catchup_boys_vx = hpv.campaign_vx(
            prob=vx_coverage,
            years=catch_up_year,
            product=prod,
            age_range=catchup_age,
            sex='m',
            label='Boys catchup vx'
        )
        intvs += [routine_boys_vx, catchup_boys_vx]
    else:
        sex = 'f'  # Set 'sex' to 'f' for female vaccination

    prod = hpv.default_vx(prod_name=product)
    if product == 'bivalent':
        prod.genotype_pars.loc[12, 'rel_imm'] = 0.05

    routine_vx = hpv.routine_vx(
        prob=vx_coverage,
        start_year=routine_start_year,
        end_year=end_year,
        product=prod,
        age_range=routine_age,
        sex=sex,  # Use the 'sex' variable here
        label='Routine vx'
    )

    catchup_vx = hpv.campaign_vx(
        prob=vx_coverage,
        years=catch_up_year,
        product=prod,
        age_range=catchup_age,
        sex=sex,  # Use the 'sex' variable here
        label='Catchup vx'
    )

    intvs += [routine_vx, catchup_vx]

    return intvs




def get_screen_intvs(primary=None, triage=None, screen_coverage=0.7, ltfu=0.3, start_year=2025):
    '''
    Make interventions for screening scenarios

    primary (None or dict): dict of test positivity values for precin, cin1, cin2, cin3 and cancerous
    triage (None or dict): dict of test positivity values for precin, cin1, cin2, cin3 and cancerous
    '''

    # Return empty list if nothing is defined
    if primary is None: return []

    # Create screen products
    if isinstance(primary, str):
        primary_test = hpv.default_dx(prod_name=primary)
    elif isinstance(primary, dict):
        primary_test = make_screen_test(**primary)
    if triage is not None:
        if isinstance(triage, str):
            triage_test = hpv.default_dx(prod_name=triage)
        elif isinstance(triage, dict):
            triage_test = make_screen_test(**triage)

    tx_assigner = make_tx_assigner()

    age_range = [30,50]
    len_age_range = (age_range[1]-age_range[0])/2

    model_annual_screen_prob = 1 - (1 - screen_coverage)**(1/len_age_range)

    # Routine screening
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                  (sim.t > (sim.people.date_screened + 10 / sim['dt']))
    screening = hpv.routine_screening(
        product=primary_test,
        prob=model_annual_screen_prob,
        eligibility=screen_eligible,
        age_range=[30, 50],
        start_year=start_year,
        label='screening'
    )

    if triage is not None:
        # Triage screening
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1 - ltfu,
            annual_prob=False,
            product=triage_test,
            eligibility=screen_positive,
            label='triage'
        )
        triage_positive = lambda sim: sim.get_intervention('triage').outcomes['positive']
        assign_treatment = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product=tx_assigner,
            eligibility=triage_positive,
            label='tx assigner'
        )
    else:
        # Assign treatment
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product=tx_assigner,
            eligibility=screen_positive,
            label='tx assigner'
        )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob=1-ltfu,
        annual_prob=False,
        product='ablation',
        eligibility=ablation_eligible,
        label='ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() +
                                             sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(
        prob=1-ltfu,
        annual_prob=False,
        product='excision',
        eligibility=excision_eligible,
        label='excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_num(
        prob=(1-ltfu)/4,
        annual_prob=False,
        product=hpv.radiation(),
        eligibility=radiation_eligible,
        label='radiation'
    )

    if triage is not None:
        st_intvs = [screening, triage_screening, assign_treatment, ablation, excision, radiation]
    else:
        st_intvs = [screening, triage_screening, ablation, excision, radiation]

    return st_intvs


def make_screen_test(precin=0.25, cin1=0.3, cin2=0.45, cin3=0.45, cancerous=0.6):
    '''
    Make screen product using P(T+| health state) for health states HPV, CIN1, CIN2, CIN3, and cancer
    '''

    basedf = pd.read_csv('dx_pars.csv')
    not_changing_states = ['susceptible', 'latent']
    not_changing = basedf.loc[basedf.state.isin(not_changing_states)].copy()

    new_states = sc.autolist()
    for state, posval in zip(['precin', 'cin1', 'cin2', 'cin3', 'cancerous'],
                             [precin, cin1, cin2, cin3, cancerous]):
        new_pos_vals = basedf.loc[(basedf.state == state) & (basedf.result == 'positive')].copy()
        new_pos_vals.probability = posval
        new_neg_vals = basedf.loc[(basedf.state == state) & (basedf.result == 'negative')].copy()
        new_neg_vals.probability = 1-posval
        new_states += new_pos_vals
        new_states += new_neg_vals
    new_states_df = pd.concat(new_states)

    # Make the screen product
    screen_test = hpv.dx(pd.concat([not_changing, new_states_df]), hierarchy=['positive', 'negative'])
    return screen_test

def make_tx_assigner():
    '''
    Make treatment assigner product
    '''

    basedf = pd.read_csv('tx_assigner_pars.csv')
    # Make the screen product
    screen_test = hpv.dx(basedf, hierarchy=['ablation', 'excision', 'radiation'])
    return screen_test


def get_mv_intvs(dose1=None, dose2=None, campaign_coverage=None, routine_coverage=None,  # Things that must be supplied
                 campaign_years=None, campaign_age=None, dose2_uptake=0.8, intro_year=2030,
                 routine_age=None):  # Can be omitted
    ''' Get mass txvx interventions'''

    # Handle inputs
    if campaign_years is None: campaign_years = [intro_year]
    if campaign_age is None: campaign_age = [25, 50]
    if routine_age is None: routine_age = [25, 26]

    # Eligibility
    first_dose_eligible = lambda sim: (sim.people.txvx_doses == 0)
    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1) & (
            sim.t > (sim.people.date_tx_vaccinated + 0.5 / sim['dt']))

    # Campaign txvx
    campaign_txvx_dose1 = hpv.campaign_txvx(
        prob=campaign_coverage,
        years=campaign_years,
        age_range=campaign_age,
        product=dose1,
        eligibility=first_dose_eligible,
        label='campaign txvx'
    )

    campaign_txvx_dose2 = hpv.campaign_txvx(
        prob=dose2_uptake,
        years=campaign_years,
        age_range=campaign_age,
        product=dose2,
        eligibility=second_dose_eligible,
        label='campaign txvx 2nd dose'
    )

    routine_txvx_dose1 = hpv.routine_txvx(
        prob=routine_coverage,
        start_year=intro_year+1,
        age_range=routine_age,
        eligibility=first_dose_eligible,
        product=dose1,
        label='routine txvx'
    )

    routine_txvx_dose2 = hpv.routine_txvx(
        prob=dose2_uptake,
        start_year=intro_year+1,
        age_range=routine_age,
        product=dose2,
        eligibility=second_dose_eligible,
        label='routine txvx 2nd dose'
    )

    mv_intvs = [campaign_txvx_dose1, campaign_txvx_dose2, routine_txvx_dose1, routine_txvx_dose2]

    return mv_intvs


def get_tnv_intvs(dose1=None, dose2=None, campaign_coverage=None, routine_coverage=None,  # Things that must be supplied
                  test_product=None, campaign_years=None, campaign_age=None, dose2_uptake=0.8, intro_year=2030,
                  routine_age=None):  # Can be omitted
    ''' Get test & txvx interventions'''

    # Handle inputs
    if campaign_years is None: campaign_years = [intro_year]
    if campaign_age is None: campaign_age = [25, 50]
    if routine_age is None: routine_age = [25, 26]
    if test_product is None: test_product = 'hpv'

    # Run a one-time campaign to test & vaccinate everyone aged 25-50
    test_eligible = lambda sim: (sim.people.txvx_doses == 0)
    txvx_campaign_testing = hpv.campaign_screening(
        product=test_product,
        prob=campaign_coverage,
        eligibility=test_eligible,
        age_range=campaign_age,
        years=campaign_years,
        label='txvx_campaign_testing'
    )

    # In addition, run routine vaccination of everyone aged 25
    test_eligible = lambda sim: (sim.people.txvx_doses == 0)
    txvx_routine_testing = hpv.routine_screening(
        product=test_product,
        prob=routine_coverage,
        eligibility=test_eligible,
        age_range=routine_age,
        start_year=intro_year,
        label='txvx_routine_testing'
    )

    screened_pos = lambda sim: list(set(sim.get_intervention('txvx_routine_testing').outcomes['positive'].tolist()
                                        + sim.get_intervention('txvx_campaign_testing').outcomes['positive'].tolist()))
    deliver_txvx = hpv.linked_txvx(
        prob=1.0,
        product=dose1,
        eligibility=screened_pos,
        label='routine txvx'
    )

    second_dose_eligible = lambda sim: (sim.people.txvx_doses == 1) & (
                sim.t > (sim.people.date_tx_vaccinated + 0.5 / sim['dt']))
    txvx_dose2 = hpv.linked_txvx(
        prob=dose2_uptake,
        annual_prob=False,
        product=dose2,
        eligibility=second_dose_eligible,
        label='routine txvx 2nd dose'
    )

    tnv_intvs = [txvx_campaign_testing, txvx_routine_testing, deliver_txvx, txvx_dose2]

    return tnv_intvs


def get_txvx_intvs(use_case=None, indication=None, low_eff=None, high_eff=None, txvx_prods=None, intro_year=2030,
                   paired_px=False, genotypes=None, campaign_age=(25,50), routine_age=(25,26), campaign_coverage=0.7,
                   routine_coverage=0.7, dose2_uptake=0.8):
    ''' Get txvx interventions '''

    if indication is not None:
        txvx_prods = make_txvx_indication(indication)
        dose1 = txvx_prods[0]
        dose2 = txvx_prods[1]
    elif low_eff is not None:
        txvx_prods, _ = make_txvx_prods(low_eff, high_eff, genotypes)
        dose1 = txvx_prods[0]
        dose2 = txvx_prods[1]
    elif txvx_prods is not None:
        dose1 = txvx_prods[0]
        dose2 = txvx_prods[1]
    else:
        dose1 = 'txvx1'
        dose2 = 'txvx2'

    common_args = dict(dose1=dose1, dose2=dose2, campaign_age=campaign_age, dose2_uptake=dose2_uptake,
                       campaign_coverage=campaign_coverage, routine_coverage=routine_coverage,
                       intro_year=intro_year, routine_age=routine_age)
    if use_case == 'mass_vaccination':
        intvs = get_mv_intvs(**common_args)
    elif use_case == 'test_and_vaccinate':
        intvs = get_tnv_intvs(**common_args)

    if paired_px:
        px_eligible = lambda sim: (sim.people.txvx_doses == 2) & (sim.t == sim.people.date_tx_vaccinated)
        paired_vx = hpv.routine_vx(
            prob=dose2_uptake,
            eligibility=px_eligible,
            product='bivalent',
            age_range=[25, 50],
            label='Paired Px'
        )
        intvs += [paired_vx]

    return intvs


def make_txvx_prods(lo_eff=None, hi_eff=None, genotypes=None, first_dose_redux=0.5):
    ''' Get txvx indication parameters '''

    basedf = pd.read_csv('txvx_pars.csv')
    lo_grade = ['precin', 'cin1']  # No lesions or low grade lesions
    hi_grade = ['cin2', 'cin3']  # High grade lesions
    no_grade = ['latent', 'cancerous']  # All other states - txvx not effective

    if genotypes is None:
        genotypes=['hpv16']

    # Randomly perturb the efficacy values
    if lo_eff is None: lo_eff = np.random.uniform(0, 1)
    if hi_eff is None: hi_eff = np.random.uniform(0, 1)
    txv1_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx1')].copy()
    txv1_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx1')].copy()
    txv1_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx1')].copy()
    txv1_lo.efficacy = lo_eff*first_dose_redux
    txv1_hi.efficacy = hi_eff*first_dose_redux

    txv1_not_covered = basedf.loc[~(basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx1')].copy()
    txv2_not_covered = basedf.loc[~(basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx2')].copy()

    # Make the assumption that the dose efficacy is independent
    txv2_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx2')].copy()
    txv2_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx2')].copy()
    txv2_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.genotype.isin(genotypes)) & (basedf.name == 'txvx2')].copy()
    txv2_lo.efficacy = lo_eff
    txv2_hi.efficacy = hi_eff

    # Make the products
    txvx1 = hpv.tx(pd.concat([txv1_lo, txv1_hi, txv1_no, txv1_not_covered]))
    txvx2 = hpv.tx(pd.concat([txv2_lo, txv2_hi, txv2_no, txv2_not_covered]))

    txvx_prods = [txvx1, txvx2]
    eff_vals = sc.objdict(
        lo_eff=lo_eff,
        hi_eff=hi_eff
    )

    return txvx_prods, eff_vals


def make_txvx_indication(indication='lesion_regression'):
    ''' Get txvx indication parameters '''

    basedf = pd.read_csv('txvx_pars.csv')
    lo_grade = ['precin', 'cin1']  # No lesions or low grade lesions
    hi_grade = ['cin2', 'cin3']  # High grade lesions
    no_grade = ['latent', 'cancerous']  # All other states - txvx not effective

    # Randomly perturb the efficacy values

    txv1_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.name == 'txvx1')].copy()
    txv1_lo.efficacy = 0.01
    txv1_hi.efficacy = 0.01

    txv2_lo = basedf.loc[(basedf.state.isin(lo_grade)) & (basedf.name == 'txvx2')].copy()
    txv2_hi = basedf.loc[(basedf.state.isin(hi_grade)) & (basedf.name == 'txvx2')].copy()
    txv2_no = basedf.loc[(basedf.state.isin(no_grade)) & (basedf.name == 'txvx2')].copy()

    if indication == 'lesion_regression':
        txv2_lo.efficacy = 0.5
        txv2_hi.efficacy = 0.9
    elif indication == 'virologic_clearance':
        txv2_lo.efficacy = 0.9
        txv2_hi.efficacy = 0.5

    # Make the products
    txvx1 = hpv.tx(pd.concat([txv1_lo, txv1_hi, txv1_no]))
    txvx2 = hpv.tx(pd.concat([txv2_lo, txv2_hi, txv2_no]))

    txvx_prods = [txvx1, txvx2]

    return txvx_prods