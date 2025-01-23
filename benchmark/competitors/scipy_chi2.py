import numpy as np
import scipy as sc
import itertools
from joblib import Parallel, delayed
from timer_wrapper import timeit

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
    
    
@timeit
def calculate_scipy_chi2(data : np.ndarray, num_threads : int):
    # Generate all pairs of rows, i.e. variables in the data.
    row_pairs = itertools.combinations(data, 2)
    
    results = Parallel(n_jobs=num_threads)(delayed(chi2_on_rows)(row1, row2) for row1, row2 in row_pairs)
    results = list(results)
    return results 
    