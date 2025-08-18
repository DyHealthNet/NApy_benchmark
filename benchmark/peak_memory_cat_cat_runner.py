import numpy as np
import itertools
import scipy as sc
from joblib import Parallel, delayed
import psutil, os, time
from threading import Thread, Event
import argparse
import napy


def mem_usage_all():
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / (1024 ** 2)  # MB


def chi2_on_rows(row1, row2):
    # Take non-NAN subarrays.
    mask1 = np.isnan(row1)
    mask2 = np.isnan(row2)
    joined_mask = ~(mask1 | mask2)

    subrow1 = row1[joined_mask]
    subrow2 = row2[joined_mask]

    # Compute contigency table.
    sc_result = sc.stats.contingency.crosstab(subrow1, subrow2)
    cont_table = sc_result.count

    # Compute chi2 test on given contingency table.
    try:
        chi2_res = sc.stats.chi2_contingency(cont_table, correction=False)
        return (chi2_res[0], chi2_res[1])
    except Exception as e:
        return (-999.0, -999.0)

def calculate_scipy_chi2(data: np.ndarray, num_threads: int):
    # Generate all pairs of rows, i.e. variables in the data.
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    row_pairs = itertools.combinations(data, 2)

    results = Parallel(n_jobs=num_threads)(delayed(chi2_on_rows)(row1, row2) for row1, row2 in row_pairs)
    results = list(results)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)

def nanpy_chi2_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    out_dict = napy.chi_squared(data, nan_value=nan_value, axis=0, threads=threads, use_numba=False,
                                return_types=['chi2', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)

def nanpy_chi2_numba_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):

    napy.chi_squared(np.eye(5), nan_value=nan_value, axis=0, threads=threads, use_numba=True,
                     return_types=['chi2', 'p_unadjusted'])
    time.sleep(2.0)

    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    out_dict = napy.chi_squared(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True,
                                return_types=['chi2', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int)
    parser.add_argument("--file", type=str)
    parser.add_argument("--tool", type=str)
    parser.add_argument("--measure", type=str)
    args = parser.parse_args()

    if args.measure == "chi2":
        if args.tool == "napy_numba":
            data = np.load(args.file)
            peak_mem = nanpy_chi2_numba_wrapper(data=data, threads=args.threads)
        elif args.tool == "napy_cpp":
            data = np.load(args.file)
            peak_mem = nanpy_chi2_wrapper(data=data, threads=args.threads)
        elif args.tool == "scipy_par":
            data = np.load(args.file)
            peak_mem = calculate_scipy_chi2(data=data, num_threads=args.threads)
        else:
            raise ValueError(f'Unknown tool: {args.tool}')

    else:
        raise ValueError(f'Unknown measure: {args.measure}')

    # Drop peak memory value into file.
    with open("process_cat_memory.txt", 'w') as f:
        f.write(f"{peak_mem}\n")

