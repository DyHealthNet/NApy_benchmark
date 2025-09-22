import numpy as np
from missmecha.generator import MissMechaGenerator
import pandas as pd

def simulate_cont_data(num_features, num_samples, na_ratio, na_value, mode='mcar'):
    """
    Simulates given numpy data matrix with num_features as rows, num_samples as columns and
    a per-feature missing value ratio of na_ratio filled with na_value.
    """
    if mode=='mcar':
        shape = (num_features, num_samples)
        data_np = np.random.rand(*shape)
        num_nas = int(num_samples*na_ratio)
        for row in data_np:
            indices = np.random.choice(num_samples, num_nas, replace=False)
            row[indices] = na_value
        return data_np
    elif mode=='mnar':
        # MissMecha simulates NAs column-wise.
        shape = (num_samples, num_features)
        data_full = np.random.rand(*shape)
        generator = MissMechaGenerator(mechanism="mnar", mechanism_type=1, missing_rate=na_ratio)
        data_na = generator.fit_transform(data_full)
        data_na = data_na.T.copy()
        result = np.nan_to_num(data_na, nan=na_value)
        return result
    elif mode == "mar":
        # MissMecha simulates NAs column-wise.
        shape = (num_samples, num_features)
        data_full = np.random.rand(*shape)
        generator = MissMechaGenerator(mechanism="mar", mechanism_type=1, missing_rate=na_ratio)
        data_na = generator.fit_transform(data_full)
        data_na = data_na.T.copy()
        result = np.nan_to_num(data_na, nan=na_value)
        return result
    else:
        raise ValueError(f"Unknown NA mode: {mode}")

def simulate_cat_data(num_variables, num_samples, na_ratio, na_value, max_categories, mode='mcar'):
    """
    Simulates categorical numpy data matrix with num_variables as rows, num_samples as columns
    and a per-variable missing value ration of na_ratio filled with na_value.
    """
    if mode == 'mcar':
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

    elif mode == 'mnar':
        rng = np.random.default_rng()
        arr = rng.integers(0, max_categories, size=(num_samples, num_variables))
        column_names = [f"Col{i}" for i in range(num_variables)]
        df = pd.DataFrame(arr, columns=column_names)
        generator = MissMechaGenerator(mechanism="mnar", mechanism_type=1, missing_rate=na_ratio, cat_cols=column_names)
        df_na = generator.fit_transform(df)
        numpy_na = df_na.to_numpy().T.copy()
        result = np.nan_to_num(numpy_na, nan=na_value)
        return result

    elif mode == "mar":
        rng = np.random.default_rng()
        arr = rng.integers(0, max_categories, size=(num_samples, num_variables))
        column_names = [f"Col{i}" for i in range(num_variables)]
        df = pd.DataFrame(arr, columns=column_names)
        generator = MissMechaGenerator(mechanism="mar", mechanism_type=1, missing_rate=na_ratio, cat_cols=column_names)
        df_na = generator.fit_transform(df)
        numpy_na = df_na.to_numpy().T.copy()
        result = np.nan_to_num(numpy_na, nan=na_value)
        return result

    else:
        raise ValueError(f"Unknown NA mode for missMecha: {mode}")

    