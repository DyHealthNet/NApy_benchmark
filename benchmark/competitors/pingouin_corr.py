import pingouin as pg
import pandas as pd
from timer_wrapper import timeit

@timeit
def calculate_pingouin_correlation(data : pd.DataFrame, method : str):
    """
    Compute pairwise, row-wise Pearson Correlation on given pandas DataFrame.
    """
    res = pg.rcorr(data, method=method)
    return res

