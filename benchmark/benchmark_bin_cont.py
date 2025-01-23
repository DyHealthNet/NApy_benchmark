import numpy as np
from pandas.core.config_init import use_numba_cb

import nanpy
import pandas as pd
from argparse import ArgumentParser
from simulator import simulate_cont_data
from simulator import simulate_cat_data
from logger import add_time_to_results
from nanpy_wrapper.ttest_wrapper import nanpy_ttest_wrapper, nanpy_ttest_wrapper_numba
from nanpy_wrapper.mwu_wrapper import nanpy_mwu_wrapper
from nanpy_wrapper.mwu_wrapper import nanpy_mwu_wrapper_numba
from competitors.scipy_ttest import calculate_scipy_ttest
from competitors.scipy_mwu import calculate_scipy_mwu

if __name__=="__main__":
    
    # Add argument parser for benchmarking parameters.
    parser = ArgumentParser()
    parser.add_argument("--features", nargs = "+", type = int, help="Numbers of features (rows) in both the continuous and binary dataset.")
    parser.add_argument("--samples", nargs = "+", type=int, help="Numbers of samples (columns) in the dataset.")
    parser.add_argument("--na_ratios", nargs = "+", type=float, help="Ratio of introduced NAs in each feature (row) of the dataset.")
    parser.add_argument("--threads", nargs="+", type=int, help="Number of threads to use in parallel computations.")
    parser.add_argument("--num_runs", type=int, help="Number of sequential runs for stabilizing benchmark.")
    parser.add_argument("--file_name", type = str, help = "Name of resulting output file.")
    parser.add_argument("--method", type=str, help="Correlation method to be used. Choose between 'ttest', 'mwu', or 'all'.")
    parser.add_argument("--save", action='store_true', help="Set intermediate storing of results to true.")
    
    # Set benchmark parameters.
    args = parser.parse_args()
    thread_list = args.threads
    feature_list = args.features
    sample_list = args.samples
    na_ratios = args.na_ratios
    
    file_name = args.file_name
    method = args.method
    save_intermediate = args.save
    
    NUM_RUNS = args.num_runs
    NA_VALUE = -99

    results = {'samples': [], 'features': [], 'threads': [], 'na_ratio': [], 'library': [], 'method': [], 'time': []}
    bin_data = np.array([[0,0,0,1,1,1], [1,0,1,0,1,0]])
    cont_data = np.array([[2,3,4,5,6,1], [2,2,2,7,7,7]])
    _, _ = nanpy_ttest_wrapper_numba(bin_data, cont_data, NA_VALUE, 1)
    _, _ = nanpy_mwu_wrapper_numba(bin_data, cont_data, NA_VALUE, 1)
    
    # Loop thru all possible combinations of benchmark parameters.
    for na_ratio in na_ratios:
        print(f'Setting NA ratio = {na_ratio}...')
        for nr_features in feature_list:
            print(f'Setting num_features = {nr_features}...')
            for nr_samples in sample_list:
                print(f'Setting num_samples = {nr_samples}...')
                for _ in range(NUM_RUNS):
                    # Simulate datasets.
                    cont_data = simulate_cont_data(nr_features, nr_samples, na_ratio, np.nan)
                    bin_data = simulate_cat_data(nr_features, nr_samples, na_ratio, np.nan, 2)
                    cont_data_np = cont_data.copy()
                    cont_nan_mask = np.isnan(cont_data_np)
                    cont_data_np[cont_nan_mask] = NA_VALUE
                    bin_data_np = bin_data.copy()
                    bin_nan_mask = np.isnan(bin_data_np)
                    bin_data_np[bin_nan_mask] = NA_VALUE
                    if method == 'ttest' or method == 'all':
                        for nr_threads in thread_list:
                            # Run nanpy ttest.
                            res, nanpy_time = nanpy_ttest_wrapper(bin_data_np, cont_data_np, NA_VALUE, nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, nanpy_time, "ttest", "nanpy")

                            # Run numba ttest.
                            res, numba_time = nanpy_ttest_wrapper_numba(bin_data_np, cont_data_np, NA_VALUE, nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          numba_time, "ttest", "numba")
                            
                            # Run scipy ttest.
                            res, sc_time = calculate_scipy_ttest(bin_data, cont_data, nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, sc_time, "ttest", "scipypar")
                    
                    if method == 'mwu' or method == 'all':
                        for nr_threads in thread_list:
                            # Run nanpy MWU test.
                            res, nanpy_time = nanpy_mwu_wrapper(bin_data_np, cont_data_np, NA_VALUE, nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, nanpy_time, "mwu", "nanpy")

                            res, numba_time = nanpy_mwu_wrapper_numba(bin_data_np, cont_data_np, NA_VALUE, nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          numba_time, "mwu", "numba")
                            
                            # Run scipy MWU test.
                            res, sc_time = calculate_scipy_mwu(bin_data, cont_data, nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, sc_time, "mwu", "scipypar")
                            
            if save_intermediate: 
                df_columns = ['samples', 'features', 'threads', 'na_ratio', 'library', 'method', 'time']
                results_df = pd.DataFrame(results, columns=df_columns)
                results_df.to_csv(file_name, index=False) 
    
    print("============= FINISHED ============")
    df_columns = ['samples', 'features', 'threads', 'na_ratio', 'library', 'method', 'time']
    results_df = pd.DataFrame(results, columns=df_columns)
    results_df.to_csv(file_name, index=False)   
