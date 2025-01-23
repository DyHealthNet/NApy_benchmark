import scipy as sc
import numpy as np
from timer_wrapper import timeit

@timeit
def calculate_scipy_spearman(data : np.ndarray):
    res = sc.stats.spearmanr(data, axis=1, nan_policy='omit')
    return res
