import napy
import numpy as np
from timer_wrapper import timeit

@timeit
def nanpy_chi2_wrapper(data : np.ndarray, nan_value : float, threads : int):
    out_dict = napy.chi_squared(data, nan_value=nan_value, axis=0, threads=threads, use_numba=False, return_types=['chi2', 'p_unadjusted'])
    return out_dict

@timeit
def nanpy_chi2_numba_wrapper(data : np.ndarray, nan_value : float, threads : int):
    out_dict = napy.chi_squared(data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['chi2', 'p_unadjusted'])
    return out_dict
