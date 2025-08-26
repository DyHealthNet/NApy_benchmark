import numpy as np
import nanpy
import pandas as pd
from argparse import ArgumentParser
from simulator import simulate_cont_data
from simulator import simulate_cat_data
from logger import add_time_to_results
from nanpy_wrapper.anova_wrapper import nanpy_anova_wrapper, nanpy_anova_wrapper_numba
from nanpy_wrapper.kw_wrapper import nanpy_kruskal_wallis_wrapper, nanpy_kruskal_wallis_wrapper_numba
from competitors.scipy_anova import calculate_scipy_anova
from competitors.scipy_kruskal import calculate_scipy_kruskal
import memray
import os

if __name__ == "__main__":

    # Add argument parser for benchmarking parameters.
    parser = ArgumentParser()
    parser.add_argument("--features", nargs="+", type=int,
                        help="Numbers of features (rows) in both the continuous and binary dataset.")
    parser.add_argument("--samples", nargs="+", type=int, help="Numbers of samples (columns) in the dataset.")
    parser.add_argument("--na_ratios", nargs="+", type=float,
                        help="Ratio of introduced NAs in each feature (row) of the dataset.")
    parser.add_argument("--threads", nargs="+", type=int, help="Number of threads to use in parallel computations.")
    parser.add_argument("--num_runs", type=int, help="Number of sequential runs for stabilizing benchmark.")
    parser.add_argument("--dir_name", type=str, help="Name of resulting output file.")
    parser.add_argument("--method", type=str,
                        help="Correlation method to be used. Choose between 'anova', 'kw', or 'all'.")

    # Set benchmark parameters.
    args = parser.parse_args()
    thread_list = args.threads
    feature_list = args.features
    sample_list = args.samples
    na_ratios = args.na_ratios
    method = args.method

    NUM_RUNS = args.num_runs
    NA_VALUE = -99
    NUM_CATEGORIES = 4

    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)
    file_dir = args.dir_name

    cat_data = np.array([[0, 0, 0, 1, 2, 2]])
    cont_data = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1, 2, 2, 2]])
    _, _ = nanpy_anova_wrapper_numba(cat_data, cont_data, NA_VALUE, 1)
    _, _ = calculate_scipy_anova(cat_data, cont_data, 2)
    _, _ = calculate_scipy_kruskal(cat_data, cont_data, 2)

    # Loop thru all possible combinations of benchmark parameters.
    for na_ratio in na_ratios:
        print(f'Setting NA ratio = {na_ratio}...')
        for nr_features in feature_list:
            print(f'Setting num_features = {nr_features}...')
            for nr_samples in sample_list:
                print(f'Setting num_samples = {nr_samples}...')
                for i in range(NUM_RUNS):
                    # Simulate datasets.
                    cont_data = simulate_cont_data(nr_features, nr_samples, na_ratio, np.nan)
                    cat_data = simulate_cat_data(nr_features, nr_samples, na_ratio, np.nan, NUM_CATEGORIES)
                    cont_data_np = cont_data.copy()
                    cont_nan_mask = np.isnan(cont_data_np)
                    cont_data_np[cont_nan_mask] = NA_VALUE
                    cat_data_np = cat_data.copy()
                    cat_nan_mask = np.isnan(cat_data_np)
                    cat_data_np[cat_nan_mask] = NA_VALUE

                    if method == 'anova' or method == 'all':
                        for nr_threads in thread_list:
                            # Run nanpy anova.
                            with memray.Tracker(file_dir + f'/nanpy_anova_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin', follow_fork=True, native_traces=False):
                                res, nanpy_time = nanpy_anova_wrapper(cat_data_np, cont_data_np, NA_VALUE, nr_threads)

                            # Run numba anova.
                            with memray.Tracker(file_dir + f'/numba_anova_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin', follow_fork=True, native_traces=False):
                                res, numba_time = nanpy_anova_wrapper_numba(cat_data_np, cont_data_np, NA_VALUE, nr_threads)

                            # Run scipy anova.
                            with memray.Tracker(file_dir + f'/scipypar_anova_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin', follow_fork=True, native_traces=False):
                                res, sc_time = calculate_scipy_anova(cat_data, cont_data, nr_threads)

                    if method == 'kw' or method == 'all':
                        for nr_threads in thread_list:
                            # Run nanpy Kruskal-Wallis test.
                            with memray.Tracker(file_dir + f'/nanpy_kruskal_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin', follow_fork=True, native_traces=False):
                                res, nanpy_time = nanpy_kruskal_wallis_wrapper(cat_data_np, cont_data_np, NA_VALUE,
                                                                           nr_threads)

                            with memray.Tracker(file_dir + f'/numba_kruskal_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin', follow_fork=True, native_traces=False):
                                res, nanpy_time = nanpy_kruskal_wallis_wrapper_numba(cat_data_np, cont_data_np, NA_VALUE,
                                                                           nr_threads)

                            # Run scipy Kruskal-Wallis test.
                            with memray.Tracker(file_dir + f'/scipypar_kruskal_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin', follow_fork=True, native_traces=False):
                                res, sc_time = calculate_scipy_kruskal(cat_data, cont_data, nr_threads)

    print("============= FINISHED ============")
