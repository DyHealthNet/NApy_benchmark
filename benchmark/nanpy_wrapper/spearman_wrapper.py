import napy
import numpy as np
from timer_wrapper import timeit

@timeit
def nanpy_spearman_wrapper(data : np.ndarray, nan_value : float, threads : int):
    """
    Wrapper around nanpy Spearman Correlation for timing computation.
    """
    res = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=False, return_types=['rho', 'p_unadjusted'])
    return res

@timeit
def nanpy_spearman_numba_wrapper(data : np.ndarray, nan_value : float, threads : int):
    res = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['rho', 'p_unadjusted'])
    return res
