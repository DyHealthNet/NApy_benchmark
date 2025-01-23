# NApy: Benchmarking and Test Implementations

This repository contains all benchmark implementations, results and plots for the manuscript "NApy: Efficient Statistics in Python for Large-Scale
Heterogeneous Data with Enhanced Support for Missing Data".

In order to run the benchmarks and tests described below, you need to have NApy installed and set up as described in our [main repository](https://github.com/DyHealthNet/NApy).

# Benchmarking

All benchmark implementations and results presented in our paper can be found in the `benchmark/` directory. Competitor implementations and our (parallelized) Python baseline implementation can be found in the subdirectory `benchmark/competitors/`, wrapper functions for NApy's functions to be called by benchmarking scripts can be found under `benchmark/nanpy_wrapper/`. The actual benchmarking scripts for running memory, runtime and CHRIS benchmark are located directly in `benchmark/`. Results of our runtime and memory analyses can be found in the respective subdirectories under `benchmark/results/`. R scripts for generating the plots included in our manuscript are available under `benchmark/R_plotting/` while the generated plots for our paper can be found under `benchmark/plots/`. We offer a conda environment supporting the necessary python packages used in the benchmarking scripts in the `benchmark/` directory.
    

# Testing

We implemented several unit tests in python to ensure the correctness of the computed test statistic, P-values and effect sizes. All test statistics and associated P-values are compared against Python's Scipy library and against one implementation from a corresponding R library. Effect sizes are compared against one corresponding implementation in R. The test script `tests/test_napy.py` can simply be run via

```python
python tests/test_napy.py
```

In order to be able to run the test script, you need to have the python-R bridge-package `rpy2` (https://rpy2.github.io/) installed, as well as the common python libraries `numpy`, `scipy`, and `pandas`. We offer a conda environment supporting the necessary python packages in the `tests/` directory.

Note that in order for the R library functions to work, you need to have R (>=4.4.1) installed in combination with the packages `effectsize`, `Hmisc`, `lsr` and `rstatix`.
