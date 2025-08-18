import numpy as np
import itertools
import scipy as sc
from joblib import Parallel, delayed
import psutil, os, time
from threading import Thread, Event
import argparse
import napy
import pandas as pd
import pingouin as pg

def mem_usage_all():
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / (1024 ** 2)  # MB

def calculate_pandas_corr_matrix(data : pd.DataFrame, method : str):
    """
    Compute full correlation matrix for row-wise correlations of pandas dataframe (default is column-wise).
    """
    print("Running pandas corr matrix computation...")
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    corr_mat = data.corr(method=method)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)

def calculate_pingouin_correlation(data : pd.DataFrame, method : str):
    """
    Compute pairwise, row-wise Pearson Correlation on given pandas DataFrame.
    """
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    res = pg.rcorr(data, method=method)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def calculate_scipy_spearman(data : np.ndarray):
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    res = sc.stats.spearmanr(data, axis=1, nan_policy='omit')

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def nanpy_pearson_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):
    """
    Wrapper around nanpy Pearson Correlation for timing computation.
    """
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    res = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=False, return_types=['r2', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def nanpy_pearson_numba_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):

    res = napy.pearsonr(compile_input, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['r2', 'p_unadjusted'])
    time.sleep(2.0)

    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    res = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['r2', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def pearson_on_rows(row1, row2):
    # Take non-NAN subarrays.
    mask1 = np.isnan(row1)
    mask2 = np.isnan(row2)
    joined_mask = ~(mask1 | mask2)

    sub1 = row1[joined_mask]
    sub2 = row2[joined_mask]

    try:
        res = sc.stats.pearsonr(sub1, sub2)
        return (res[0], res[1])
    except Exception as e:
        return (np.nan, np.nan)

def calculate_scipy_pearson_par(data: np.ndarray, num_threads: int):
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    row_pairs = itertools.combinations(data, 2)
    results = Parallel(n_jobs=num_threads)(delayed(pearson_on_rows)(row1, row2) for row1, row2 in row_pairs)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def pearson_on_rows_pandas(row1, row2):
    # Take non-NAN subarrays.
    df = pd.DataFrame(columns=[0,1])
    df[0] = row1
    df[1] = row2
    res_mat = df.corr(method='pearson')
    return (res_mat.iloc[0,1])

def calculate_pandas_pearson_par(data: np.ndarray, num_threads: int):
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
    results = Parallel(n_jobs=num_threads)(delayed(pearson_on_rows_pandas)(row1, row2) for row1, row2 in row_pairs)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def nanpy_spearman_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):
    """
    Wrapper around nanpy Spearman Correlation for timing computation.
    """
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    res = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=False,
                         return_types=['rho', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def nanpy_spearman_numba_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):
    
    res = napy.spearmanr(np.eye(5), nan_value=nan_value, axis=0, threads=threads, use_numba=True,
                         return_types=['rho', 'p_unadjusted'])
    time.sleep(2.0)
    
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    res = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True,
                         return_types=['rho', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def spearman_on_rows_scipy(row1, row2):
    # Take non-NAN subarrays.
    res = sc.stats.spearmanr(row1, row2, nan_policy='omit')
    return (res[0], res[1])

def calculate_scipy_spearman_par(data: np.ndarray, num_threads: int):
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
    results = Parallel(n_jobs=num_threads)(delayed(spearman_on_rows_scipy)(row1, row2) for row1, row2 in row_pairs)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def spearman_on_rows(row1, row2):
    # Take non-NAN subarrays.
    df = pd.DataFrame(columns=[0,1])
    df[0] = row1
    df[1] = row2
    res_mat = df.corr(method='spearman')
    return (res_mat.iloc[0,1])

def calculate_pandas_spearman_par(data: np.ndarray, num_threads: int):

    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    row_pairs = itertools.combinations(data, 2)
    results = Parallel(n_jobs=num_threads)(delayed(spearman_on_rows)(row1, row2) for row1, row2 in row_pairs)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def run_empty_subprocess():
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    time.sleep(2.0)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

def thread_sleep():
    time.sleep(1.0)

def run_sleeping_pool(threads : int):
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    results = Parallel(n_jobs=threads)(delayed(thread_sleep)() for _ in range(threads))

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return max(mem_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int)
    parser.add_argument("--file", type=str)
    parser.add_argument("--tool", type=str)
    parser.add_argument("--measure", type=str)
    args = parser.parse_args()

    if args.measure == "pearson":
        if args.tool == "pandas":
            data = pd.read_csv(args.file, index_col=0)
            peak_mem = 0.0
            peak_mem = calculate_pandas_corr_matrix(data=data, method='pearson')
        elif args.tool == "pingouin":
            data = pd.read_csv(args.file, index_col=0)
            peak_mem = calculate_pingouin_correlation(data=data, method='pearson')
        elif args.tool == "napy_numba":
            data = np.load(args.file)
            peak_mem = nanpy_pearson_numba_wrapper(data=data, threads=args.threads)
        elif args.tool == "napy_cpp":
            data = np.load(args.file)
            peak_mem = nanpy_pearson_wrapper(data=data, threads=args.threads)
        elif args.tool == "scipy_par":
            data = np.load(args.file)
            peak_mem = calculate_scipy_pearson_par(data=data, num_threads=args.threads)
        elif args.tool == "pandas_par":
            data = np.load(args.file)
            peak_mem = calculate_pandas_pearson_par(data=data, num_threads=args.threads)
        else:
            raise ValueError(f'Unknown tool: {args.tool}')

    elif args.measure == "spearman":
        if args.tool == "scipy":
            data = np.load(args.file)
            peak_mem = calculate_scipy_spearman(data)
        elif args.tool == "pandas":
            data = pd.read_csv(args.file, index_col=0)
            peak_mem = calculate_pandas_corr_matrix(data=data, method='spearman')
        elif args.tool == "pingouin":
            data = pd.read_csv(args.file, index_col=0)
            peak_mem = calculate_pingouin_correlation(data=data, method='spearman')
        elif args.tool == "napy_numba":
            data = np.load(args.file)
            peak_mem = nanpy_spearman_numba_wrapper(data, threads=args.threads)
        elif args.tool == "napy_cpp":
            data = np.load(args.file)
            peak_mem = nanpy_spearman_wrapper(data, threads=args.threads)
        elif args.tool == "scipy_par":
            data = np.load(args.file)
            peak_mem = calculate_scipy_spearman_par(data, num_threads=args.threads)
        elif args.tool == "pandas_par":
            data = np.load(args.file)
            peak_mem = calculate_pandas_spearman_par(data, num_threads=args.threads)
        else:
            raise ValueError(f'Unknown tool: {args.tool}')

    elif args.measure == "empty":
        if args.threads == 1:
            peak_mem = run_empty_subprocess()
        else:
            peak_mem = run_sleeping_pool(threads=args.threads)
    else:
        raise ValueError(f'Unknown measure: {args.measure}')

    # Drop peak memory value into file.
    with open("process_memory.txt", 'w') as f:
        f.write(f"{peak_mem}\n")

