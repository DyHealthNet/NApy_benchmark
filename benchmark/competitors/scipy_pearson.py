import numpy as np
import scipy as sc
import itertools
from joblib import Parallel, delayed
from timer_wrapper import timeit

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

@timeit
def calculate_scipy_pearson_par(data: np.ndarray, num_threads: int):
    # Generate all pairs of rows, i.e. variables in the data.
    row_pairs = itertools.combinations(data, 2)
    results = Parallel(n_jobs=num_threads)(delayed(pearson_on_rows)(row1, row2) for row1, row2 in row_pairs)
    return list(results)