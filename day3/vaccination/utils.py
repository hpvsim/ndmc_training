'''
Utilities for HPVsim NDMC analyses
'''

import sciris as sc
import numpy as np
import pandas as pd

import pylab as pl
import seaborn as sns
import hpvsim.plotting as hppl
import matplotlib.ticker as ticker
import hpvsim as hpv


resfolder = 'results'
figfolder = 'figures'
datafolder = 'data'

########################################################################
#%% Other utils
########################################################################
def make_msims(sims, use_mean=True, save_msims=False):
    '''
    Utility to take a slice of sims and turn it into a multisim
    '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_sc, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except seed matches
        assert i_sc == sim.meta.inds[0]
        assert (s == 0) or i_s != sim.meta.inds[1]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_sc]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')

    print(f'Processing multisim {msim.meta.vals.values()}...')
    if save_msims:  # Warning, generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim

def make_msims_sweeps(sims, use_mean=True, save_msims=False):
    ''' Take a slice of sims and turn it into a multisim '''
    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_txs, draw, i_s = sims[0].meta.inds
    for s,sim in enumerate(sims): # Check that everything except seed matches
        assert i_txs == sim.meta.inds[0]
        assert draw == sim.meta.inds[1]
        assert (s==0) or i_s != sim.meta.inds[2]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_txs, draw]
    msim.meta.eff_vals = sc.dcp(sims[0].meta.eff_vals)
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')
    print(f'Processing multisim {msim.meta.vals.values()}...')

    if save_msims: # Generates a lot of files!
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{resfolder}/final_msim{id_str}.msim'
        msim.save(msimfile)

    return msim

def set_font(size=None, font='Libertinus Sans'):
    ''' Set a custom font '''
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return

def plot_residual_burden_vx(location=None, vx_scens=None, screen_scens=None,
                            label_dict=None):
    '''
    Plot the residual burden of HPV under different vx scenarios
    '''
    set_font(20)
    all_vxs = ['No vaccine'] + vx_scens
    alldfs = sc.autolist()
    for screen_scen_label in screen_scens:
        for vx_scen_label in all_vxs:
            if vx_scen_label == 'No vaccine':
                filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}'
                alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                alldfs += alldf
            else:
                filestem_label = f'{label_dict[vx_scen_label]}_{label_dict[screen_scen_label]}'
                alldf = sc.loadobj(f'{resfolder}/{location}_{filestem_label}.obj')
                alldfs += alldf

    bigdf = pd.concat(alldfs)

    colors = sc.gridcolors(20)
    for ir, (res, reslabel) in enumerate({'cancers': 'Cervical cancer cases',
                                          'cancer_deaths': 'Cervical cancer deaths',
                                          'asr_cancer_incidence': 'Age standardized cervical cancer incidence rate (per 100,000)', }.items()):
        fig, ax = pl.subplots(figsize=(18, 14))
        for screen_scen_label in screen_scens:
            # first plot not vaccine:
            df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == 'No vaccine')].groupby('year')[
                [f'{res}', f'{res}_low', f'{res}_high']].sum()
            years = np.array(df.index)[70:111]
            best = np.array(df[res])[70:111]
            label = f'No vaccine'
            novx_handle, = ax.plot(years, best, color=colors[0], label=label)

            vx_handles = []
            for vn, vx_scen_label in enumerate(vx_scens):
                df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)].groupby(
                    'year')[
                    [f'{res}', f'{res}_low', f'{res}_high']].sum()
                years = np.array(df.index)[70:111]
                best = np.array(df[res])[70:111]
                if res == 'cancers':
                    print(f'{vx_scen_label}: {np.sum(best)} cancers')
                label = f'{vx_scen_label}'
                vx_handle, = ax.plot(years, best, color=colors[vn + 1], label=label)
                vx_handles.append(vx_handle)

        if res == 'asr_cancer_incidence':
            ax.plot(years, np.full(len(years), fill_value=4), linestyle='dashed')
        ax.set_ylim(bottom=0)

        # ax.legend([novx_handle, vx_handles[0], vx_handles[1], vx_handles[2], vx_handles[3], vx_handles[4]],
        #           ['None', '10%', '30%', '50%', '70%', '90%', ], loc='best', title='Vaccine coverage')
        sc.SIticks(ax)
        ax.set_ylabel(f'{reslabel}')
        ax.set_title(f'{reslabel} in {location.capitalize()}')
        fig.tight_layout()
        fig_name = f'{figfolder}/{res}_residual_burden_vx_{location}.png'
        sc.savefig(fig_name, dpi=100)
        fig.show()

    return