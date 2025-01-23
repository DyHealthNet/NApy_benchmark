import napy
import numpy as np
from timer_wrapper import timeit

@timeit
def nanpy_mwu_wrapper(bin_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    """
        Wrapper around nanpy's MWU test for timing computation.
    """
    out = napy.mwu(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, return_types=['U', 'p_unadjusted'], use_numba=False)
    return out

@timeit
def nanpy_mwu_wrapper_numba(bin_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    """
        Wrapper around numba's MWU test for timing computation.
    """
    out = napy.mwu(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, return_types=['U', 'p_unadjusted'], use_numba=True)
    return out