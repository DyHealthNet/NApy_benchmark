import numpy as np
import scipy as sc
import itertools
from joblib import Parallel, delayed
from timer_wrapper import timeit

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
    
@timeit
def calculate_scipy_ttest(bin_data : np.ndarray, cont_data : np.ndarray, num_threads : int):
    # Generate all pairs of rows, i.e. variables in the data.
    row_pairs = itertools.product(bin_data, cont_data)

    results = Parallel(n_jobs=num_threads)(delayed(ttest_on_rows)(cat, cont) for cat, cont in row_pairs)
    return list(results)