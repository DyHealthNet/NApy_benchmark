import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from timer_wrapper import timeit
import pingouin as pg

def chi2_on_rows(row1, row2):
    # Take non-NAN subarrays.
    mask1 = np.isnan(row1)
    mask2 = np.isnan(row2)
    joined_mask = ~(mask1 | mask2)
    
    subrow1 = row1[joined_mask]
    subrow2 = row2[joined_mask]
    
    data_df = pd.DataFrame(columns=[0,1])
    data_df[0] = subrow1.tolist()
    data_df[1] = subrow2.tolist()

    # print(data_df)
    res = pg.chi2_independence(data_df, x=0, y=1, correction=False)
    return (res[2].iloc[0,2], res[2].iloc[0,4])
    
@timeit
def calculate_pingouin_chi2_par(data : np.ndarray, num_threads : int):
    results = list(results)
    # Count how often pair (-999.0, -999.0) occurs.
    num_exceptions = results.count((-999.0, -999.0))
    print("    Number of total feature pairs: ", len(results))
    print("    Number of exception pairs: ", num_exceptions)
    return results
    