import numpy as np
import pandas as pd
from argparse import ArgumentParser
from simulator import simulate_cont_data
from competitors.pandas_corr import calculate_pandas_corr_matrix
from competitors.pingouin_corr import calculate_pingouin_correlation
from competitors.scipy_spearman import calculate_scipy_spearman
from competitors.scipy_pearson import calculate_scipy_pearson_par
from competitors.scipy_spearman_par import calculate_scipy_spearman_par
from competitors.pandas_spearman_par import calculate_pandas_spearman_par
from competitors.pandas_pearson_par import calculate_pandas_pearson_par
from nanpy_wrapper.pearson_wrapper import nanpy_pearson_wrapper, nanpy_pearson_numba_wrapper
from nanpy_wrapper.spearman_wrapper import nanpy_spearman_wrapper, nanpy_spearman_numba_wrapper
from logger import add_time_to_results

if __name__=="__main__":
    
    # Add argument parser for benchmarking parameters.
    parser = ArgumentParser()
    parser.add_argument("--features", nargs = "+", type = int, help="Numbers of features (rows) in the dataset.")
    parser.add_argument("--samples", nargs = "+", type=int, help="Numbers of samples (columns) in the dataset.")
    parser.add_argument("--na_ratios", nargs = "+", type=float, help="Ratio of introduced NAs in each feature (row) of the dataset.")
    parser.add_argument("--threads", nargs="+", type=int, help="Number of threads to use in parallel computations.")
    parser.add_argument("--num_runs", type=int, help="Number of sequential runs for stabilizing benchmark.")
    parser.add_argument("--file_name", type = str, help = "Name of resulting output file.")
    parser.add_argument("--corr_method", type=str, help="Correlation method to be used. Choose between 'pearson', 'spearman', or 'all'.")
    parser.add_argument("--save", action='store_true', help="Set intermediate storing of results to true.")
    
    # Set benchmark parameters.
    args = parser.parse_args()
    thread_list = args.threads
    feature_list = args.features
    sample_list = args.samples
    na_ratios = args.na_ratios
    
    file_name = args.file_name
    corr_method = args.corr_method
    save_intermediate = args.save
    
    NUM_RUNS = args.num_runs
    NA_VALUE = -99

    results = {'samples': [], 'features': [], 'threads': [], 'na_ratio': [], 'library': [], 'method': [], 'time': []}
    test_data = np.random.rand(5,5)
    _ = nanpy_pearson_numba_wrapper(test_data, nan_value=NA_VALUE, threads=2)
    _ = nanpy_spearman_numba_wrapper(test_data, nan_value=NA_VALUE, threads=2)

    # Loop thru all possible combinations of benchmark parameters.
    for na_ratio in na_ratios:
        print(f'Setting NA ratio = {na_ratio}...')
        for nr_features in feature_list:
            print(f'Setting num_features = {nr_features}...')
            for nr_samples in sample_list:
                for _ in range(NUM_RUNS):
                    print(f'Setting num_samples = {nr_samples}...')
                    # Simulate dataset.
                    data = simulate_cont_data(nr_features, nr_samples, na_ratio, np.nan)
                    data_pd = pd.DataFrame(data).T
                    data_np = data.copy()
                    nan_mask = np.isnan(data_np)
                    data_np[nan_mask] = NA_VALUE
                    # Perform benchmarks for Pearson correlation.
                    if corr_method == 'pearson' or corr_method == 'all':
                        # Run pandas correlation computation.
                        pandas_res, pandas_time = calculate_pandas_corr_matrix(data_pd, method='pearson')
                        results = add_time_to_results(results, nr_samples, nr_features, na_ratio, 1, pandas_time, "pearson", "pandas")

                        # Run pingouin correlation computation.
                        ping_res, ping_time = calculate_pingouin_correlation(data_pd, method='pearson')
                        print(ping_res)
                        results = add_time_to_results(results, nr_samples, nr_features, na_ratio, 1, ping_time, "pearson", "pingouin")
                        
                        # Run parallelized implementations and competitors.
                        for nr_threads in thread_list:
                            res, nanpy_time = nanpy_pearson_wrapper(data_np, nan_value=NA_VALUE, threads=nr_threads)
                            # print("Nanpy result: ", res[0])
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, nanpy_time, "pearson", "nanpy")

                            res, numba_time = nanpy_pearson_numba_wrapper(data_np, nan_value=NA_VALUE, threads=nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          numba_time, "pearson", "numba")

                            res, scipy_time = calculate_scipy_pearson_par(data_np, num_threads=nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          scipy_time, "pearson", "scipypar")

                            res, pandas_time = calculate_pandas_pearson_par(data_np, num_threads=nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          pandas_time, "pearson", "pandaspar")

                    if corr_method == 'spearman' or corr_method == 'all':  
                        # Compute scipy spearman correlation.
                        sc_res, scipy_time = calculate_scipy_spearman(data)
                        add_time_to_results(results, nr_samples, nr_features, na_ratio, 1, scipy_time, "spearman", "scipy")
                        
                        # Run pandas correlation computation.
                        pandas_res, pandas_time = calculate_pandas_corr_matrix(data_pd, method='spearman')
                        results = add_time_to_results(results, nr_samples, nr_features, na_ratio, 1, pandas_time, "spearman", "pandas")
                        
                        # Run pingouin Spearman correlation.
                        ping_res, ping_time = calculate_pingouin_correlation(data_pd, method='spearman')
                        print(ping_res)
                        results = add_time_to_results(results, nr_samples, nr_features, na_ratio, 1, ping_time, "spearman", "pingouin")
                         
                        # Run nanpy Spearman Correlation.
                        for nr_threads in thread_list:
                            nanpy_res, nanpy_time = nanpy_spearman_wrapper(data_np, nan_value=NA_VALUE, threads=nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads, nanpy_time, "spearman", "nanpy")

                            numba_res, numba_time = nanpy_spearman_numba_wrapper(data_np, nan_value=NA_VALUE, threads=nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          numba_time, "spearman", "numba")

                            res, sc_time = calculate_scipy_spearman_par(data_np, num_threads=nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          sc_time, "spearman", "scipypar")

                            res, pandas_time = calculate_pandas_spearman_par(data_np, num_threads=nr_threads)
                            results = add_time_to_results(results, nr_samples, nr_features, na_ratio, nr_threads,
                                                          pandas_time, "spearman", "pandaspar")
            
            if save_intermediate: 
                df_columns = ['samples', 'features', 'threads', 'na_ratio', 'library', 'method', 'time']
                results_df = pd.DataFrame(results, columns=df_columns)
                results_df.to_csv(file_name, index=False) 
    
    print("============= FINISHED ============")
    df_columns = ['samples', 'features', 'threads', 'na_ratio', 'library', 'method', 'time']
    results_df = pd.DataFrame(results, columns=df_columns)
    results_df.to_csv(file_name, index=False)   
