import pandas as pd
from timer_wrapper import timeit

@timeit
def calculate_pandas_corr_matrix(data : pd.DataFrame, method : str):
    """
    Compute full correlation matrix for row-wise correlations of pandas dataframe (default is column-wise).
    """
    corr_mat = data.corr(method=method)
    return corr_mat
    