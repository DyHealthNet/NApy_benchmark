def add_time_to_results(results : dict, nr_samples : int, nr_features : int, na_ratio : float, 
                        nr_threads : int, time : float, method : str, library : str):
    """
        Stores timing result in result dictionary.
    """
    results['samples'].append(nr_samples)
    results['features'].append(nr_features)
    results['threads'].append(nr_threads)
    results['na_ratio'].append(na_ratio)
    results['library'].append(library)
    results['method'].append(method)
    results['time'].append(time)
    return results
