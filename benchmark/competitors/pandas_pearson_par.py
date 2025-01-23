import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from timer_wrapper import timeit

def pearson_on_rows(row1, row2):
    # Take non-NAN subarrays.
    df = pd.DataFrame(columns=[0,1])
    df[0] = row1
    df[1] = row2
    res_mat = df.corr(method='pearson')
    return (res_mat.iloc[0,1])

@timeit
def calculate_pandas_pearson_par(data: np.ndarray, num_threads: int):
    # Generate all pairs of rows, i.e. variables in the data.
    row_pairs = itertools.combinations(data, 2)
    results = Parallel(n_jobs=num_threads)(delayed(pearson_on_rows)(row1, row2) for row1, row2 in row_pairs)
    return list(results)