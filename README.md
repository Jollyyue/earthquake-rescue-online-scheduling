# Earthquake Rescue Online Scheduling

This repository provides the Python source code supporting the computational
experiments and sensitivity analyses reported in the manuscript:

**"Earthquake Emergency Response under Uncertain Scenarios:
Online Decision-making for Rescue Team Scheduling and Allocation"**

## Code files

The source code is provided in the `src/` directory:

- `main_simulation.py`  
  Main simulation framework used for the comparative experiments, including
  Greedy, Rollout-Time, Rollout-Benefit, Genetic Algorithm (GA), and
  Ant Colony Optimization (ACO).

- `sensitivity_analysis.py`  
  Code for the one-factor-at-a-time sensitivity analysis.

- `sensitivity_plots.py`  
  Code used to generate the corresponding sensitivity-analysis figures.

- `sobol_analysis.py`  
  Code for the Sobol global sensitivity analysis.

- `validate_adapter.py`  
  Validation script for checking the consistency of the simulation adapter
  used in the Sobol analysis.

## Reproducibility

The repository is provided to support the computational reproducibility of
the associated manuscript. Model definitions, parameter settings, algorithm
configurations, and experimental procedures are described in the manuscript
and its appendices.

The source code includes the computational procedures used for the main
simulation experiments and sensitivity analyses.

## Software

The computational experiments were implemented in Python. Detailed computing
environment and software information are reported in the associated manuscript.

## Citation

If using this code, please cite the associated article. Citation information
will be updated after publication.
