import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func( *args, **kwargs )
        end = time.perf_counter()
        elapsed = end - start
        print( f'{func.__name__} took: {elapsed:.6f} seconds' )
        return result, elapsed
    return wrapper
