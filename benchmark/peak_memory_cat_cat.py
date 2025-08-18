import subprocess
import shlex
import pandas as pd
import numpy as np
import os
from simulator import simulate_cat_data


def run_benchmark_subprocess(name, args):
    print("Running ", name)
    proc = subprocess.run(
        shlex.join(["python", "peak_memory_cat_cat_runner.py"] + args),
        capture_output=True,
        text=True,
        shell=True
    )
    # Transform input arguments into dictionary for saving in CSV file.
    output_dict = {args[i][2:]: args[i + 1] for i in range(0, len(args), 2)}

    # Read peak memory from file.
    with open("process_cat_memory.txt", 'r') as f:
        peak_mem = float(f.read())
    output_dict['memory'] = peak_mem

    # Delete memory result file.
    if os.path.exists("process_cat_memory.txt"):
        os.remove("process_cat_memory.txt")

    print(output_dict)
    return output_dict


def record_results(total_results: dict, run_results: dict, num_run: int):
    total_results['num_run'].append(num_run)
    total_results['num_threads'].append(run_results['threads'])
    total_results['tool'].append(run_results['tool'])
    total_results['test'].append(run_results['measure'])
    total_results['peak_memory'].append(run_results['memory'])
    return total_results


if __name__ == "__main__":
    # Input parameters.
    NUM_FEATURES = 100
    NUM_SAMPLES = 100
    THREAD_LIST = [1, 4]
    NA_RATIO = 0.1
    NUM_RUNS = 3

    NA_VALUE = -99.0
    OUT_DIR = "peak_cat_cat_results/"
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {'num_run': [],
                   'num_threads': [],
                   'tool': [],
                   'test': [],
                   'peak_memory': []
                   }

    # Repeat simulation and memory benchmarking as often as desired.
    for i in range(NUM_RUNS):
        # Simulate datasets for NApy and pandas format.
        data_nan = simulate_cat_data(NUM_FEATURES, NUM_SAMPLES, NA_RATIO, np.nan, max_categories=4)
        data_pd = pd.DataFrame(data_nan).T
        data_napy = data_nan.copy()
        nan_mask = np.isnan(data_napy)
        data_napy[nan_mask] = NA_VALUE

        # Drop simulated data to files in oder to be readable by subprocess.
        data_pd_path = os.path.abspath(os.path.join(OUT_DIR, f'data_pandas_run_{i}.csv'))
        data_pd.to_csv(data_pd_path, index=True)

        data_napy_path = os.path.abspath(os.path.join(OUT_DIR, f'data_napy_run_{i}.npy'))
        np.save(data_napy_path, data_napy)

        data_nan_path = os.path.abspath(os.path.join(OUT_DIR, f'data_nan_run_{i}.npy'))
        np.save(data_nan_path, data_nan)

        for num_threads in THREAD_LIST:

            # Pearson Correlation methods.
            run_results = run_benchmark_subprocess(f'Chi2 NApyNumba NUM={i}',
                                                   ["--threads", f'{num_threads}', "--file", str(data_napy_path),
                                                    "--tool", "napy_numba", '--measure', "chi2"])
            all_results = record_results(all_results, run_results, num_run=i)
            run_results = run_benchmark_subprocess(f'Chi2 NApyCPP NUM={i}',
                                                   ["--threads", f'{num_threads}', "--file", str(data_napy_path),
                                                    "--tool", "napy_cpp", '--measure', "chi2"])
            all_results = record_results(all_results, run_results, num_run=i)
            run_results = run_benchmark_subprocess(f'Chi2 Scipy NUM={i}',
                                                   ["--threads", f'{num_threads}', "--file", str(data_nan_path),
                                                    "--tool", "scipy_par", '--measure', "chi2"])
            all_results = record_results(all_results, run_results, num_run=i)


    # Save dictionary into CSV file.
    results_df = pd.DataFrame(all_results)
    results_df['num_samples'] = NUM_SAMPLES
    results_df['num_features'] = NUM_FEATURES
    results_df['na_ratio'] = NA_RATIO
    results_df.to_csv('peak_memory_cat_cat_results.tsv', sep='\t')








