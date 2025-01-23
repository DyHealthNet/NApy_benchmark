import napy
import numpy as np
from timer_wrapper import timeit

@timeit
def nanpy_anova_wrapper(cat_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    """
        Wrapper around nanpy's ANOVA for timing computation.
    """
    out_dict = napy.anova(cat_data, cont_data, nan_value=nan_value, axis=0, threads=threads, use_numba=False, return_types=['F', 'p_unadjusted'])
    return out_dict

@timeit
def nanpy_anova_wrapper_numba(cat_data : np.ndarray, cont_data : np.ndarray, nan_value : float, threads : int):
    out_dict = napy.anova(cat_data, cont_data, nan_value=nan_value, axis=0, threads=threads, use_numba=True, return_types=['F', 'p_unadjusted'])
    return out_dict
