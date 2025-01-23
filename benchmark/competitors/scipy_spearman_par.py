import numpy as np
import scipy as sc
import itertools
from joblib import Parallel, delayed
from timer_wrapper import timeit

def spearman_on_rows(row1, row2):
    # Take non-NAN subarrays.
    res = sc.stats.spearmanr(row1, row2, nan_policy='omit')
    return (res[0], res[1])

@timeit
def calculate_scipy_spearman_par(data: np.ndarray, num_threads: int):
    # Generate all pairs of rows, i.e. variables in the data.
    row_pairs = itertools.combinations(data, 2)
    results = Parallel(n_jobs=num_threads)(delayed(spearman_on_rows)(row1, row2) for row1, row2 in row_pairs)
    return list(results)