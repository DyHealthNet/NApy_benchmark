import numpy as np
import nanpy
import pandas as pd
from argparse import ArgumentParser
from simulator import simulate_cat_data
from logger import add_time_to_results
from nanpy_wrapper.chi2_wrapper import nanpy_chi2_wrapper, nanpy_chi2_numba_wrapper
from competitors.pingouin_chi2  import calculate_pingouin_chi2_par
from competitors.scipy_chi2 import calculate_scipy_chi2

if __name__=="__main__":
    
    # Add argument parser for benchmarking parameters.
    parser = ArgumentParser()
    parser.add_argument("--features", nargs = "+", type = int, help="Numbers of features (rows) in the dataset.")
    parser.add_argument("--samples", nargs = "+", type=int, help="Numbers of samples (columns) in the dataset.")
    parser.add_argument("--na_ratios", nargs = "+", type=float, help="Ratio of introduced NAs in each feature (row) of the dataset.")
    parser.add_argument("--threads", nargs="+", type=int, help="Number of threads to use in parallel computations.")
    parser.add_argument("--num_runs", type=int, help="Number of sequential runs for stabilizing benchmark.")
    parser.add_argument("--file_name", type = str, help = "Name of resulting output file.")
    parser.add_argument("--save", action='store_true', help="Set intermediate storing of results to true.")
    
    # Set benchmark parameters.
    args = parser.parse_args()
    thread_list = args.threads
    feature_list = args.features
    sample_list = args.samples
    na_ratios = args.na_ratios

    file_name = args.file_name
    save_intermediate = args.save
    
    NUM_RUNS = args.num_runs
    NA_VALUE = -99
    NUM_CATEGORIES = 4

    results = {'samples': [], 'features': [], 'threads': [], 'na_ratio': [], 'library': [], 'method': [], 'time': []}
    # Compile numba code in advance.
    test_data = np.array([[2,1,1,0,0,2], [1,1,1,0,0,0], [3,2,1,0,1,2]])
    _, _ = nanpy_chi2_numba_wrapper(test_data, nan_value=NA_VALUE, threads=2)
    
    # Loop thru all possible combinations of benchmark parameters.
    for na_ratio in na_ratios:
        print(f'Setting NA ratio = {na_ratio}...')
        for nr_features in feature_list:
            print(f'Setting num_features = {nr_features}...')
            for nr_samples in sample_list:
                for _ in range(NUM_RUNS):
                    print(f'Setting num_samples = {nr_samples}...')
                    # Simulate dataset.
                    data = simulate_cat_data(nr_features, nr_samples, na_ratio, np.nan, NUM_CATEGORIES)
                    data_np = data.copy()
                    nan_mask = np.isnan(data_np)
                    data_np[nan_mask] = NA_VALUE
                    for nr_threads in thread_list:
                        # Run nanpy chi2 test.
                        res, nanpy_time = nanpy_chi2_wrapper(data_np, NA_VALUE, nr_threads)
                        results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, nanpy_time, "chi2", "nanpy")

                        # Run nanpy-numba chi2 test.
                        res_numba, numba_time = nanpy_chi2_numba_wrapper(data_np, NA_VALUE, nr_threads)
                        results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                      numba_time, "chi2", "numba")
                        
                        # Run python implementation of chi-squared test.
                        res, sc_time = calculate_scipy_chi2(data, nr_threads)
                        results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, sc_time, "chi2", "scipypar")

                        # res, ping_time = calculate_pingouin_chi2_par(data_np, num_threads=nr_threads)
                        # results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, ping_time,
                        #                              "chi2", "pingouinpar")
            
            if save_intermediate: 
                df_columns = ['samples', 'features', 'threads', 'na_ratio', 'library', 'method', 'time']
                results_df = pd.DataFrame(results, columns=df_columns)
                results_df.to_csv(file_name, index=False) 
    
    print("============= FINISHED ============")
    df_columns = ['samples', 'features', 'threads', 'na_ratio', 'library', 'method', 'time']
    results_df = pd.DataFrame(results, columns=df_columns)
    results_df.to_csv(file_name, index=False)
