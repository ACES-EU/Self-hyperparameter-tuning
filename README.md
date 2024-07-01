# Self-hyperparameter-tuning
SUPSI - WP4

## Requirements

- [Python 3.8]
- [Python libraries listed in the scripts to be run]
- [MATLAB R2022a]

## Deploy
### Instructions for benchmark problems
1. In the same directory where all the files of this library are stored, the following directory should also be created: "Data/<problem_name>/". One different subfolder must be created for each different benchmark problem that it is of interest to solve.
2. Run Python script 'main_benchmarks.py' to launch the D-GLIS algorithm. The name of the benchmark problem to be solved must be specified in 'main_benchmarks.py'. The problem can be one of the existing ones or can be defined in the separate script 'utilities.py', according to the syntax which can be inferred from existing examples.
3. Run Python script 'plot_solutions_benchmarks.py' to print/plot results and additional information.

### Instructions for distributed MPC problem
1. In the same directory where all the files of this library are stored, the following directory should also be created: "Data/mpc/".
2. Specify local objectives for each agant in the MATLAB script 'benchmark_MPC_calibration.m'.
3. Run Python script 'main_MPC.py' to launch the D-GLIS algorithm for solving the distributed MPC calibration problem.
4. Run Python script 'plot_solutions_MPC.py' to print/plot results and additional information.

## To Read
- https://ieeexplore.ieee.org/abstract/document/10107979
