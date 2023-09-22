"""
Plot implied natural history.
"""
import hpvsim as hpv
import hpvsim.utils as hpu
import hpvsim.parameters as hppar
import pylab as pl
import pandas as pd
from scipy.stats import lognorm, norm
import numpy as np
import sciris as sc
import utils as ut
import seaborn as sns

import run_sim as rs


# %% Functions
def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")
def plot_calib(calib, res_to_plot=100):

    ut.set_font(size=16)
    fig = pl.figure(layout="tight", figsize=(10, 7))
    prev_col = '#5f5cd2'
    canc_col = '#c1981d'
    ms = 80
    gen_cols = sc.gridcolors(4)

    # Make 2 rows, with 2 panels in the top row and 3 in the bottom
    gs0 = fig.add_gridspec(2, 1)
    gs00 = gs0[0].subgridspec(1, 2)
    gs01 = gs0[1].subgridspec(1, 3)

    # Pull out the analyzer and sim results
    index_to_plot = calib.df.iloc[0:res_to_plot, 0].values
    analyzer_results = [calib.analyzer_results[i] for i in index_to_plot]
    sim_results = [calib.sim_results[i] for i in index_to_plot]

    ###############
    # Panel A: HPV prevalence by age
    ###############
    ax = fig.add_subplot(gs00[0])

    # Extract data
    datadf = calib.target_data[0]
    age_labels = ['15-25', '25-34', '35-44', '45-54', '55-64', '65+']
    x = np.arange(len(age_labels))
    best = datadf.value.values

    # Pull out lower and upper bounds from Figure 54 here: https://hpvcentre.net/statistics/reports/IND.pdf
    lowererr = np.array([0.025, 0.015, 0.02 , 0.025, 0.08 , 0.08 ])
    uppererr = np.array([0.02 , 0.01 , 0.015, 0.03 , 0.09 , 0.08 ])
    err = [lowererr, uppererr]

    # Extract model results
    bins = []
    values = []
    for run_num, run in enumerate(analyzer_results):
        bins += x.tolist()
        values += list(run['hpv_prevalence'][2020])
    modeldf = pd.DataFrame({'bins': bins, 'values': values})

    # Plot model
    sns.lineplot(ax=ax, x='bins', y='values', data=modeldf, color=prev_col, errorbar=('pi', 95))
    # Plot data
    ax.errorbar(x, best, yerr=err, ls='none', marker='d', markersize=ms/10, color='k')

    # Axis sttings
    ax.set_ylim([0,0.25])
    ax.set_xticks(x, age_labels)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('HPV prevalence by age, 2020')

    ###############
    # Panel B: Cancers by age
    ###############
    ax = fig.add_subplot(gs00[1])

    # Data
    datadf = calib.target_data[1]
    # age_labels = ['0-14', '15-20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
    #               '65-69', '70-74', '75-79', '80-84', '85+']
    # age_labels = ['', '15-20', '', '25-29', '', '35-39', '', '45-49', '', '55-59', '', '65-69', '', '75-79', '', '85+']
    age_labels = ['0', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85']
    x = np.arange(len(age_labels))
    best = datadf.value.values

    # Extract model results
    bins = []
    values = []
    for run_num, run in enumerate(analyzer_results):
        bins += x.tolist()
        values += list(run['cancers'][2020])
    modeldf = pd.DataFrame({'bins': bins, 'values': values})

    sns.lineplot(ax=ax, x='bins', y='values', data=modeldf, color=canc_col, errorbar=('pi', 95))
    ax.scatter(x, best, marker='d', s=ms, color='k')

    ax.set_ylim([0,30_000])
    ax.set_xticks(x, age_labels)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Cancers by age, 2020')

    # CINS and cancer by genotype
    rkeys = ['cin1_genotype_dist', 'cin3_genotype_dist', 'cancerous_genotype_dist']
    rlabels = ['CIN1s by genotype', 'CIN3s by genotype', 'Cancers by genotype']
    for ai, rkey in enumerate(rkeys):
        ax = fig.add_subplot(gs01[ai])

        # Plot data
        datadf = calib.target_data[ai+2]
        ydata = datadf.value.values
        x = np.arange(len(ydata))

        # Extract model results
        bins = []
        values = []
        for run_num, run in enumerate(sim_results):
            bins += x.tolist()
            if sc.isnumber(run[rkey]):
                values += sc.promotetolist(run[rkey])
            else:
                values += run[rkey].tolist()
        modeldf = pd.DataFrame({'bins': bins, 'values': values})

        # Plot model
        sns.boxplot(ax=ax, x='bins', y='values', data=modeldf, palette=gen_cols, showfliers=False)
        ax.scatter(x, ydata, color='k', marker='d', s=ms)

        ax.set_ylim([0,1])
        ax.set_xticks(np.arange(4), calib.glabels)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(rlabels[ai])

    pl.savefig(f"figures/figS1.png", dpi=300)
    pl.show()

    return

# %% Run as a script
if __name__ == '__main__':

    location = 'india'
    calib = sc.loadobj(f'results/{location}_calib.obj')
    plot_calib(calib)

    print('Done.') 
