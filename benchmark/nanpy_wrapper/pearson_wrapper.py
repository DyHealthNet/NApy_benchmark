import napy
import numpy as np
from timer_wrapper import timeit

@timeit
def nanpy_pearson_wrapper(data : np.ndarray, nan_value : float, threads : int):
    """
    Wrapper around nanpy Pearson Correlation for timing computation.
    """
    res = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=False, return_types=['r2', 'p_unadjusted'])
    return res

@timeit
def nanpy_pearson_numba_wrapper(data : np.ndarray, nan_value : float, threads : int):
    res = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['r2', 'p_unadjusted'])
    return res
