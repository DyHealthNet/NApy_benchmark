import numpy as np

def simulate_cont_data(num_features, num_samples, na_ratio, na_value):
    """
    Simulates given numpy data matrix with num_features as rows, num_samples as columns and
    a per-feature missing value ratio of na_ratio filled with na_value.
    """
    shape = (num_features, num_samples)
    data_np = np.random.rand(*shape)
    num_nas = int(num_samples*na_ratio)
    for row in data_np:
        indices = np.random.choice(num_samples, num_nas, replace=False)
        row[indices] = na_value
    return data_np

def simulate_cat_data(num_variables, num_samples, na_ratio, na_value, max_categories):
    """
    Simulates categorical numpy data matrix with num_variables as rows, num_samples as columns
    and a per-variable missing value ration of na_ratio filled with na_value.
    """
    # Initialize an empty matrix
    N = max_categories-1
    shape = (num_variables, num_samples)
    matrix = np.zeros(shape, dtype=float)
    num_nas = int(num_samples*na_ratio)
    
    for i in range(num_variables):
        # Start with a permutation of [0, 1, ..., N] for each row
        row = np.random.permutation(N + 1)
        
        # If the row has more columns than N+1, fill the remaining columns with random values from [0, N]
        if num_samples > N + 1:
            extra_nans = [na_value]*num_nas
            extra_values = np.random.randint(0, N + 1, size=num_samples-(N+1)-num_nas)
            row = np.concatenate([row, extra_values, extra_nans])
        
        # Shuffle the row to randomize the order
        np.random.shuffle(row)
        
        # Assign the row to the matrix
        matrix[i, :] = row[:num_samples]
    
    return matrix
    