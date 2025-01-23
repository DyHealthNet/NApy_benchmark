import napy
import numpy as np
from timer_wrapper import timeit

@timeit
def nanpy_ttest_wrapper(bin_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    """
    Wrapper around nanpy Student's t-test for timing computation.
    """
    out = napy.ttest(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, use_numba=False, return_types=['t', 'p_unadjusted'])
    return out

@timeit
def nanpy_ttest_wrapper_numba(bin_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    out = napy.ttest(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['t', 'p_unadjusted'])
    return out
