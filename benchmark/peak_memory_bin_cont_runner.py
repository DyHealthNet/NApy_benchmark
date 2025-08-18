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

def nanpy_ttest_wrapper(bin_data : np.ndarray, cont_data : np.ndarray, threads : int, nan_value : float = -99.0):
    """
    Wrapper around nanpy Student's t-test for timing computation.
    """
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    out = napy.ttest(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, use_numba=False, return_types=['t', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)

def nanpy_ttest_wrapper_numba(bin_data : np.ndarray, cont_data : np.ndarray, threads : int, nan_value : float = -99.0):

    out = napy.ttest(np.eye(5), np.eye(5), nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['t', 'p_unadjusted'])
    time.sleep(2.0)

    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    out = napy.ttest(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['t', 'p_unadjusted'])

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)


def ttest_on_rows(cat, cont):
    # Take non-NAN subarrays.
    mask1 = np.isnan(cat)
    mask2 = np.isnan(cont)
    joined_mask = ~(mask1 | mask2)

    sub_cat = cat[joined_mask]
    sub_cont = cont[joined_mask]

    # Compute list of continuous values for each category.
    zipped = np.column_stack((sub_cat, sub_cont))
    zipped = zipped[zipped[:, 0].argsort()]
    category_lists = np.split(zipped[:, 1], np.unique(zipped[:, 0], return_index=True)[1][1:])

    # Compute ttest using scipy.
    try:
        ttest_res = sc.stats.ttest_ind(category_lists[0], category_lists[1])
        return (ttest_res[0], ttest_res[1])
    except Exception as e:
        return (np.nan, np.nan)

def calculate_scipy_ttest(bin_data: np.ndarray, cont_data: np.ndarray, num_threads: int):
    # Generate all pairs of rows, i.e. variables in the data.
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    row_pairs = itertools.product(bin_data, cont_data)

    results = Parallel(n_jobs=num_threads)(delayed(ttest_on_rows)(cat, cont) for cat, cont in row_pairs)
    out = list(results)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)

def nanpy_mwu_wrapper(bin_data : np.ndarray, cont_data : np.ndarray, threads : int, nan_value : float = -99.0):
    """
        Wrapper around nanpy's MWU test for timing computation.
    """
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    out = napy.mwu(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads,
                   return_types=['U', 'p_unadjusted'], use_numba=False)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)

def nanpy_mwu_wrapper_numba(bin_data : np.ndarray, cont_data : np.ndarray, threads : int, nan_value : float = -99.0):
    """
        Wrapper around numba's MWU test for timing computation.
    """
    napy.mwu(np.eye(5), np.eye(5), nan_value=nan_value, axis=0, threads=threads,
             return_types=['U', 'p_unadjusted'], use_numba=True)
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    out = napy.mwu(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads,
                   return_types=['U', 'p_unadjusted'], use_numba=True)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)


def mwu_on_rows(cat, cont):
    # Take non-NAN subarrays.
    mask1 = np.isnan(cat)
    mask2 = np.isnan(cont)
    joined_mask = ~(mask1 | mask2)

    sub_cat = cat[joined_mask]
    sub_cont = cont[joined_mask]

    # Compute list of continuous values for each category.
    zipped = np.column_stack((sub_cat, sub_cont))
    zipped = zipped[zipped[:, 0].argsort()]
    category_lists = np.split(zipped[:, 1], np.unique(zipped[:, 0], return_index=True)[1][1:])

    # Compute MWU using scipy.
    try:
        ttest_res = sc.stats.mannwhitneyu(category_lists[0], category_lists[1], use_continuity=False)
        return (ttest_res[0], ttest_res[1])
    except Exception as e:
        return (np.nan, np.nan)

def calculate_scipy_mwu(bin_data: np.ndarray, cont_data: np.ndarray, num_threads: int):
    # Generate all pairs of rows, i.e. variables in the data.
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    row_pairs = itertools.product(bin_data, cont_data)

    results = Parallel(n_jobs=num_threads)(delayed(mwu_on_rows)(cat, cont) for cat, cont in row_pairs)
    out = list(results)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}", flush=True)
    return max(mem_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cont_file", type=str)
    parser.add_argument("--bin_file", type=str)
    parser.add_argument("--tool", type=str)
    parser.add_argument("--measure", type=str)
    args = parser.parse_args()


    if args.measure == "ttest":
        if args.tool == "napy_numba":
            cont_data = np.load(args.cont_file)
            bin_data = np.load(args.bin_file)
            peak_mem = nanpy_ttest_wrapper_numba(cont_data=cont_data, bin_data=bin_data, threads=args.threads)
        elif args.tool == "napy_cpp":
            cont_data = np.load(args.cont_file)
            bin_data = np.load(args.bin_file)
            peak_mem = nanpy_ttest_wrapper(cont_data=cont_data, bin_data=bin_data, threads=args.threads)
        elif args.tool == "scipy_par":
            cont_data = np.load(args.cont_file)
            bin_data = np.load(args.bin_file)
            peak_mem = calculate_scipy_ttest(cont_data=cont_data, bin_data=bin_data, num_threads=args.threads)
        else:
            raise ValueError(f'Unknown tool: {args.tool}')

    elif args.measure == "mwu":
        if args.tool == "napy_numba":
            cont_data = np.load(args.cont_file)
            bin_data = np.load(args.bin_file)
            peak_mem = nanpy_mwu_wrapper_numba(cont_data=cont_data, bin_data=bin_data, threads=args.threads)
        elif args.tool == "napy_cpp":
            cont_data = np.load(args.cont_file)
            bin_data = np.load(args.bin_file)
            peak_mem = nanpy_mwu_wrapper(cont_data=cont_data, bin_data=bin_data, threads=args.threads)
        elif args.tool == "scipy_par":
            cont_data = np.load(args.cont_file)
            bin_data = np.load(args.bin_file)
            peak_mem = calculate_scipy_mwu(cont_data=cont_data, bin_data=bin_data, num_threads=args.threads)
        else:
            raise ValueError(f'Unknown tool: {args.tool}')
    else:
        raise ValueError(f'Unknown measure: {args.measure}')

    # Drop peak memory value into file.
    with open("process_bin_cont_memory.txt", 'w') as f:
        f.write(f"{peak_mem}\n")

