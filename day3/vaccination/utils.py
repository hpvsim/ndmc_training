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
