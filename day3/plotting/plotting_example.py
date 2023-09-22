"""
Illustrate manual plotting and analyzers
"""

import hpvsim as hpv
import pylab as pl # Equivalent to "import matplotlib.pyplot as plt"

debug = False

pars = dict(
    n_agents = [20e3, 2e3][debug],
    location = 'india',
)


sim = hpv.Sim(pars)
sim.run()

pl.plot(sim.results.year, sim.results.asr_cancer_incidence)