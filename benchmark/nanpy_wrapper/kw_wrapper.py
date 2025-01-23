import napy
import numpy as np
from timer_wrapper import timeit

@timeit
def nanpy_kruskal_wallis_wrapper(bin_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    """
        Wrapper around nanpy's Kruskal-Wallis test for timing computation.
    """
    out_dict = napy.kruskal_wallis(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, return_types=['H', 'p_unadjusted'], use_numba=False)
    return out_dict

@timeit
def nanpy_kruskal_wallis_wrapper_numba(bin_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    """
        Wrapper around nanpy's Kruskal-Wallis test for timing computation.
    """
    out_dict = napy.kruskal_wallis(bin_data, cont_data, nan_value=nan_value, axis=0, threads=threads, return_types=['H', 'p_unadjusted'], use_numba=True)
    return out_dict