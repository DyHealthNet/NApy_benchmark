import subprocess
import shlex
import pandas as pd
import numpy as np
import os
from simulator import simulate_cont_data, simulate_cat_data


def run_benchmark_subprocess(name, args):
    print("Running ", name)
    proc = subprocess.run(
        shlex.join(["python", "peak_memory_bin_cont_runner.py"] + args),
        capture_output=True,
        text=True,
        shell=True
    )
    # Transform input arguments into dictionary for saving in CSV file.
    output_dict = {args[i][2:]: args[i + 1] for i in range(0, len(args), 2)}

    # Read peak memory from file.
    with open("process_bin_cont_memory.txt", 'r') as f:
        peak_mem = float(f.read())
    output_dict['memory'] = peak_mem

    # Delete memory result file.
    if os.path.exists("process_bin_cont_memory.txt"):
        os.remove("process_bin_cont_memory.txt")

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
    OUT_DIR = "peak_bin_cont_results/"
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
        cont_data_nan = simulate_cont_data(NUM_FEATURES, NUM_SAMPLES, NA_RATIO, np.nan)
        bin_data_nan = simulate_cat_data(NUM_FEATURES, NUM_SAMPLES, NA_RATIO, np.nan, max_categories=2)
        cont_data_pd = pd.DataFrame(cont_data_nan).T
        bin_data_pd = pd.DataFrame(bin_data_nan).T
        cont_data_napy = cont_data_nan.copy()
        bin_data_napy = bin_data_nan.copy()
        cont_nan_mask = np.isnan(cont_data_napy)
        bin_nan_mask = np.isnan(bin_data_napy)
        cont_data_napy[cont_nan_mask] = NA_VALUE
        bin_data_napy[bin_nan_mask] = NA_VALUE

        # Drop simulated data to files in oder to be readable by subprocess.
        cont_data_pd_path = os.path.abspath(os.path.join(OUT_DIR, f'cont_data_pandas_run_{i}.csv'))
        cont_data_pd.to_csv(cont_data_pd_path, index=True)
        bin_data_pd_path = os.path.abspath(os.path.join(OUT_DIR, f'bin_data_pandas_run_{i}.csv'))
        bin_data_pd.to_csv(bin_data_pd_path, index=True)

        cont_data_napy_path = os.path.abspath(os.path.join(OUT_DIR, f'cont_data_napy_run_{i}.npy'))
        np.save(cont_data_napy_path, cont_data_napy)
        bin_data_napy_path = os.path.abspath(os.path.join(OUT_DIR, f'bin_data_napy_run_{i}.npy'))
        np.save(bin_data_napy_path, bin_data_napy)

        cont_data_nan_path = os.path.abspath(os.path.join(OUT_DIR, f'cont_data_nan_run_{i}.npy'))
        np.save(cont_data_nan_path, cont_data_nan)
        bin_data_nan_path = os.path.abspath(os.path.join(OUT_DIR, f'bin_data_nan_run_{i}.npy'))
        np.save(bin_data_nan_path, bin_data_nan)


        for num_threads in THREAD_LIST:
            # TTest methods.
            run_results = run_benchmark_subprocess(f'Ttest NApyNumba NUM={i}',
                                                   ["--threads", f'{num_threads}', "--cont_file", str(cont_data_napy_path),
                                                    "--tool", "napy_numba", '--measure', "ttest",
                                                    "--bin_file", str(bin_data_napy_path)])
            all_results = record_results(all_results, run_results, num_run=i)
            run_results = run_benchmark_subprocess(f'Ttest NApyCPP NUM={i}',
                                                   ["--threads", f'{num_threads}', "--cont_file", str(cont_data_napy_path),
                                                    "--tool", "napy_cpp", '--measure', "ttest",
                                                    "--bin_file", str(bin_data_napy_path)])
            all_results = record_results(all_results, run_results, num_run=i)
            run_results = run_benchmark_subprocess(f'Ttest ScipyPar NUM={i}',
                                                   ["--threads", f'{num_threads}', "--cont_file", str(cont_data_nan_path),
                                                    "--tool", "scipy_par", '--measure', "ttest",
                                                    "--bin_file", str(bin_data_nan_path)])
            all_results = record_results(all_results, run_results, num_run=i)

            # MWU methods.
            run_results = run_benchmark_subprocess(f'MWU NApyNumba NUM={i}',
                                                   ["--threads", f'{num_threads}', "--cont_file", str(cont_data_napy_path),
                                                    "--tool", "napy_numba", "--bin_file", str(bin_data_napy_path),
                                                    '--measure', "mwu"])
            all_results = record_results(all_results, run_results, num_run=i)
            run_results = run_benchmark_subprocess(f'MWU NApyCPP NUM={i}',
                                                   ["--threads", f'{num_threads}', "--cont_file", str(cont_data_napy_path),
                                                    "--tool", "napy_cpp", '--measure',
                                                    "mwu", "--bin_file", str(bin_data_napy_path)])
            all_results = record_results(all_results, run_results, num_run=i)
            run_results = run_benchmark_subprocess(f'MWU ScipyPar NUM={i}',
                                                   ["--threads", f'{num_threads}', "--cont_file", str(cont_data_nan_path),
                                                    "--tool", "scipy_par",
                                                    '--measure', "mwu", "--bin_file", str(bin_data_nan_path)])
            all_results = record_results(all_results, run_results, num_run=i)

    # Save dictionary into CSV file.
    results_df = pd.DataFrame(all_results)
    results_df['num_samples'] = NUM_SAMPLES
    results_df['num_features'] = NUM_FEATURES
    results_df['na_ratio'] = NA_RATIO
    results_df.to_csv('peak_memory_bin_cont_results.tsv', sep='\t')








