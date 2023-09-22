# Vaccination and screening

This folder contains the template scripts for doing vaccination and screening analyses in HPVsim. The structure is as follows:

- `dx_pars.csv` contains data on the properties of different diagnostics.
- `pars_data.py` contains sexual behavior data (mixing matrices and age of debut).
- `pars_scenarios.py` defines the different screening and vaccination interventions, and assembles them into scenarios.
- `run_scenarios.py` actually runs the scenarios.
- `run_sim.py` runs a single sim.
- `tx_assigner_pars.csv` describes how treatment is assigned.
- `utils.py` contains extra Python functions to help run.