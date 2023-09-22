"""
Compilation of sexual behavior data and assumptions
"""


#%% Initialization

import numpy as np
import locations as loc

# Initialize objects with per-country results
layer_probs = dict()
mixing = dict()
partners = dict()
debut = dict()
init_genotype_dist = dict()


#%% LAYER PROBS
default_layer_probs = dict(
    m=np.array([
        # Share of females (row 1) and males (row 2) of each age who are married
        [0, 5,  10,    15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],  # Age bracket
        [0, 0, 0.05, 0.25, 0.70, 0.90, 0.95, 0.70, 0.75, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],  # Share f
        [0, 0, 0.01, 0.01, 0.10, 0.50, 0.60, 0.70, 0.70, 0.70, 0.70, 0.80, 0.70, 0.60, 0.50, 0.60]]  # Share m
    ),
    c=np.array([
        # Share of females (row 1) and males (row 2) of each age having casual relationships
        [0, 5,   10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],  # Age bracket
        [0, 0, 0.10, 0.70, 0.80, 0.60, 0.60, 0.50, 0.20, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # Share f
        [0, 0, 0.05, 0.70, 0.80, 0.60, 0.60, 0.50, 0.50, 0.40, 0.30, 0.10, 0.05, 0.01, 0.01, 0.01]],  # Share m
    ),
    o=np.array([
        # Share of females (row 1) and males (row 2) of each age having one-off relationships
        [0, 5,   10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],  # Age bracket
        [0, 0, 0.01, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # Share f
        [0, 0, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]],  # Share m
    ),
)

for location in loc.locations:
    layer_probs[location] = default_layer_probs


#%% PARTNERS
default_partners = dict(
        m=dict(dist='poisson', par1=0.1),
        c=dict(dist='poisson', par1=0.5),
        o=dict(dist='poisson', par1=0.0),
)

for location in loc.locations:
    partners[location] = default_partners


#%% MIXING
default_mixing_all = np.array([
    #       0,  5, 10, 15, 20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75
    [0,     0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [5,     0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [10,    0,  0,  1,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [15,    0,  0,  1,  1,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [20,    0,  0, .5,  1,  1, .01,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [25,    0,  0,  0, .5,  1,   1, .01,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [30,    0,  0,  0,  0, .5,   1,   1, .01,   0,   0,   0,   0,   0,   0,   0,   0],
    [35,    0,  0,  0,  0, .1,  .5,   1,   1, .01,   0,   0,   0,   0,   0,   0,   0],
    [40,    0,  0,  0,  0,  0,  .1,  .5,   1,   1, .01,   0,   0,   0,   0,   0,   0],
    [45,    0,  0,  0,  0,  0,   0,  .1,  .5,   1,   1, .01,   0,   0,   0,   0,   0],
    [50,    0,  0,  0,  0,  0,   0,   0,  .1,  .5,   1,   1,  .01,  0,   0,   0,   0],
    [55,    0,  0,  0,  0,  0,   0,   0,   0,  .1,  .5,   1,   1, .01,   0,   0,   0],
    [60,    0,  0,  0,  0,  0,   0,   0,   0,   0,  .1,  .5,   1,   1, .01,   0,   0],
    [65,    0,  0,  0,  0,  0,   0,   0,   0,   0,   0,  .1,  .5,   1,   1, .01,   0],
    [70,    0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,  .1,  .5,   1,   1, .01],
    [75,    0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,  .1,  .5,   1,   1],
])

default_mixing = dict()
for k in ['m','c','o']: default_mixing[k] = default_mixing_all

default_mixing = dict()
for k in ['m', 'c', 'o']: default_mixing[k] = default_mixing_all
for location in loc.locations:
    mixing[location] = default_mixing

# Debut
default_debut = dict(
    f=dict(dist='lognorm', par1=18, par2=2),
    m=dict(dist='lognorm', par1=19.5, par2=2),
)
for location in loc.locations:
    debut[location] = default_debut


# Initial genotype distribution
default_init_genotype_dist = dict(hpv16=0.4, hpv18=0.25, hi5=0.25, ohr=.1)
for location in loc.locations:
    init_genotype_dist[location] = default_init_genotype_dist
