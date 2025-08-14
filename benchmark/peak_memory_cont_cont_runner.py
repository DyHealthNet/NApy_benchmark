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

def anova_on_rows(cat, cont):
    mask1 = np.isnan(cat)
    mask2 = np.isnan(cont)
    joined_mask = ~(mask1 | mask2)

    sub_cat = cat[joined_mask]
    sub_cont = cont[joined_mask]

    zipped = np.column_stack((sub_cat, sub_cont))
    zipped = zipped[zipped[:, 0].argsort()]
    category_lists = np.split(
        zipped[:, 1],
        np.unique(zipped[:, 0], return_index=True)[1][1:]
    )

    try:
        res = sc.stats.f_oneway(*category_lists)
        return (res[0], res[1])
    except Exception:
        return (-999.0, -999.0)

def mem_usage_all():
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / (1024 ** 2)  # MB

def calculate_scipy_anova(bin_data: np.ndarray, cont_data: np.ndarray, num_threads: int):
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    row_pairs = itertools.product(bin_data, cont_data)
    results = Parallel(n_jobs=num_threads)(
        delayed(anova_on_rows)(cat, cont) for cat, cont in row_pairs
    )

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return results

def calculate_napy_anova(bin_data: np.ndarray, cont_data: np.ndarray, num_threads: int):
    mem_log = []
    stop_event = Event()

    def monitor():
        while not stop_event.is_set():
            mem_log.append(mem_usage_all())
            time.sleep(0.1)

    monitor_thread = Thread(target=monitor)
    monitor_thread.start()

    res = napy.anova(cat_data=bin_data, cont_data=cont_data, threads=num_threads)

    stop_event.set()
    monitor_thread.join()

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return res

def calculate_pandas_corr_matrix(data : pd.DataFrame, method : str):
    """
    Compute full correlation matrix for row-wise correlations of pandas dataframe (default is column-wise).
    """
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

    print(f"PEAK_MEMORY_MB: {max(mem_log):.2f}")
    return corr_mat

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
    return res

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
    return res

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
    return res

def nanpy_pearson_numba_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):

    #res = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['r2', 'p_unadjusted'])

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
    return res

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
    return list(results)

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
    return list(results)

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
    return res

def nanpy_spearman_numba_wrapper(data : np.ndarray, threads : int, nan_value : float = -99.0):
    #res = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True,
    #                     return_types=['rho', 'p_unadjusted'])
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
    return res

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
    return list(results)

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
    return list(results)

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
            result = calculate_pandas_corr_matrix(data=data, method='pearson')
        elif args.tool == "pingouin":
            data = pd.read_csv(args.file, index_col=0)
            result = calculate_pingouin_correlation(data=data, method='pearson')
        elif args.tool == "napy_numba":
            data = np.load(args.file)
            result = nanpy_pearson_numba_wrapper(data=data, threads=args.threads)
        elif args.tool == "napy_cpp":
            data = np.load(args.file)
            result = nanpy_pearson_wrapper(data=data, threads=args.threads)
        elif args.tool == "scipy_par":
            data = np.load(args.file)
            result = calculate_scipy_pearson_par(data=data, num_threads=args.threads)
        elif args.tool == "pandas_par":
            data = np.load(args.file)
            result = calculate_pandas_pearson_par(data=data, num_threads=args.threads)
        else:
            raise ValueError(f'Unknown tool: {args.tool}')

    elif args.measure == "spearman":
        if args.tool == "scipy":
            data = np.load(args.file)
            result = calculate_scipy_spearman(data)
        elif args.tool == "pandas":
            data = pd.read_csv(args.file, index_col=0)
            result = calculate_pandas_corr_matrix(data=data, method='spearman')
        elif args.tool == "pingouin":
            data = pd.read_csv(args.file, index_col=0)
            result = calculate_pingouin_correlation(data=data, method='spearman')
        elif args.tool == "napy_numba":
            data = np.load(args.file)
            result = nanpy_spearman_numba_wrapper(data, threads=args.threads)
        elif args.tool == "napy_cpp":
            data = np.load(args.file)
            result = nanpy_spearman_wrapper(data, threads=args.threads)
        elif args.tool == "scipy_par":
            data = np.load(args.file)
            result = calculate_scipy_spearman_par(data, num_threads=args.threads)
        elif args.tool == "pandas_par":
            data = np.load(args.file)
            result = calculate_pandas_spearman_par(data, num_threads=args.threads)
        else:
            raise ValueError(f'Unknown tool: {args.tool}')

    else:
        raise ValueError(f'Unknown measure: {args.measure}')


