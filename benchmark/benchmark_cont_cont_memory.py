import numpy as np
import nanpy
import pandas as pd
from argparse import ArgumentParser
from simulator import simulate_cont_data
from competitors.pandas_corr import calculate_pandas_corr_matrix
from competitors.pingouin_corr import calculate_pingouin_correlation
from competitors.scipy_spearman import calculate_scipy_spearman
from competitors.scipy_pearson import calculate_scipy_pearson_par
from competitors.pandas_pearson_par import calculate_pandas_pearson_par
from competitors.pandas_spearman_par import calculate_pandas_spearman_par
from competitors.scipy_spearman_par import calculate_scipy_spearman_par
from nanpy_wrapper.pearson_wrapper import nanpy_pearson_wrapper, nanpy_pearson_numba_wrapper
from nanpy_wrapper.spearman_wrapper import nanpy_spearman_wrapper, nanpy_spearman_numba_wrapper
import memray
import os

if __name__ == "__main__":

    # Add argument parser for benchmarking parameters.
    parser = ArgumentParser()
    parser.add_argument("--features", nargs="+", type=int, help="Numbers of features (rows) in the dataset.")
    parser.add_argument("--samples", nargs="+", type=int, help="Numbers of samples (columns) in the dataset.")
    parser.add_argument("--na_ratios", nargs="+", type=float,
                        help="Ratio of introduced NAs in each feature (row) of the dataset.")
    parser.add_argument("--threads", nargs="+", type=int, help="Number of threads to use in parallel computations.")
    parser.add_argument("--num_runs", type=int, help="Number of sequential runs for stabilizing benchmark.")
    parser.add_argument("--dir_name", type=str, help="Name of resulting output directory.")
    parser.add_argument("--corr_method", type=str,
                        help="Correlation method to be used. Choose between 'pearson', 'spearman', or 'all'.")

    # Set benchmark parameters.
    args = parser.parse_args()
    thread_list = args.threads
    feature_list = args.features
    sample_list = args.samples
    na_ratios = args.na_ratios
    corr_method = args.corr_method

    NUM_RUNS = args.num_runs
    NA_VALUE = -99
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)
    file_dir = args.dir_name

    ex_data = np.random.rand(3, 5)
    _ = nanpy_pearson_numba_wrapper(ex_data, NA_VALUE, 2)
    _ = nanpy_spearman_numba_wrapper(ex_data, NA_VALUE, 2)

    # Loop thru all possible combinations of benchmark parameters.
    for na_ratio in na_ratios:
        print(f'Setting NA ratio = {na_ratio}...')
        for nr_features in feature_list:
            print(f'Setting num_features = {nr_features}...')
            for nr_samples in sample_list:
                print(f'Setting num_samples = {nr_samples}...')
                for i in range(NUM_RUNS):
                    # Simulate dataset.
                    data = simulate_cont_data(nr_features, nr_samples, na_ratio, np.nan)
                    data_pd = pd.DataFrame(data).T
                    data_np = data.copy()
                    nan_mask = np.isnan(data_np)
                    data_np[nan_mask] = NA_VALUE

                    # Perform benchmarks for Pearson correlation.
                    if corr_method == 'pearson' or corr_method == 'all':
                        # Run pandas correlation computation.
                        with memray.Tracker(file_dir + f'/pandas_pearson_{nr_features}_{nr_samples}_{na_ratio}_1_{i}.bin',
                                            native_traces=False, follow_fork=True):
                            pandas_res, pandas_time = calculate_pandas_corr_matrix(data_pd, method='pearson')

                        with memray.Tracker(file_dir + f'/pingouin_pearson_{nr_features}_{nr_samples}_{na_ratio}_1_{i}.bin',
                                            native_traces=False, follow_fork=True):
                            ping_res, ping_time = calculate_pingouin_correlation(data_pd, method='pearson')

                        # Run nanpy Pearson Correlation.
                        for nr_threads in thread_list:
                            with memray.Tracker(
                                    file_dir + f'/nanpy_pearson_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',
                                    native_traces=False, follow_fork=True):
                                res, nanpy_time = nanpy_pearson_wrapper(data_np, nan_value=NA_VALUE, threads=nr_threads)

                            with memray.Tracker(file_dir + f'/numba_pearson_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',
                                                native_traces=False, follow_fork=True):
                                res, numba_time = nanpy_pearson_numba_wrapper(data_np, nan_value=NA_VALUE,
                                                                              threads=nr_threads)

                            with memray.Tracker(file_dir + f'/scipypar_pearson_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',
                                                native_traces=False, follow_fork=True):
                                res, scipy_time = calculate_scipy_pearson_par(data_np, num_threads=nr_threads)

                            with memray.Tracker(file_dir + f'/pandaspar_pearson_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',
                                                native_traces=False, follow_fork=True):
                                res, pandas_time = calculate_pandas_pearson_par(data_np, num_threads=nr_threads)


                    if corr_method == 'spearman' or corr_method == 'all':
                        with memray.Tracker(file_dir + f'/scipy_spearman_{nr_features}_{nr_samples}_{na_ratio}_1_{i}.bin',native_traces=False, follow_fork=True):
                            sc_res, scipy_time = calculate_scipy_spearman(data)

                        # Run pandas correlation computation.
                        with memray.Tracker(file_dir + f'/pandas_spearman_{nr_features}_{nr_samples}_{na_ratio}_1_{i}.bin', native_traces=False, follow_fork=True):
                            pandas_res, pandas_time = calculate_pandas_corr_matrix(data_pd, method='spearman')

                        with memray.Tracker(file_dir + f'/pingouin_spearman_{nr_features}_{nr_samples}_{na_ratio}_1_{i}.bin',native_traces=False, follow_fork=True):
                            ping_res, ping_time = calculate_pingouin_correlation(data_pd, method='spearman')

                        # Run nanpy Spearman Correlation.
                        for nr_threads in thread_list:
                            with memray.Tracker(file_dir + f'/nanpy_spearman_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',native_traces=False, follow_fork=True):
                                nanpy_res, nanpy_time = nanpy_spearman_wrapper(data_np, nan_value=NA_VALUE,
                                                                               threads=nr_threads)
                            with memray.Tracker(
                                        file_dir + f'/numba_spearman_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',native_traces=False, follow_fork=True):
                                res, numba_time = nanpy_spearman_numba_wrapper(data_np, nan_value=NA_VALUE,
                                                                           threads=nr_threads)
                            with memray.Tracker(file_dir + f'/scipypar_spearman_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',native_traces=False, follow_fork=True):
                                res, sc_time = calculate_scipy_spearman_par(data_np, num_threads=nr_threads)

                            with memray.Tracker(file_dir + f'/pandaspar_spearman_{nr_features}_{nr_samples}_{na_ratio}_{nr_threads}_{i}.bin',native_traces=False, follow_fork=True):
                                res, pandas_time = calculate_pandas_spearman_par(data_np, num_threads=nr_threads)

    print("============= FINISHED ============")

