import numpy as np
import pandas as pd
import napy
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_correlated_pairs(N, M, rho=0.9, random_state=None):
    """
    Simulate N pairs of highly correlated variables with M samples.

    Parameters:
        N (int): Number of pairs.
        M (int): Number of samples.
        rho (float): Correlation coefficient (between -1 and 1).
        random_state (int or None): Seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated data.
    """
    np.random.seed(random_state)
    data = {}

    # Covariance matrix for two variables with correlation rho
    cov = [[1, rho], [rho, 1]]

    for i in range(1, N+1):
        # Generate M samples for each pair
        samples = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=M)
        data[f"x{i}"] = samples[:, 0]
        data[f"y{i}"] = samples[:, 1]

    return pd.DataFrame(data)

def simulate_mcar(arr1, arr2, ratio, random_state=None):
    """
    Insert NaNs into two numpy arrays at random positions with a given ratio.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Input 1D arrays.
    ratio : float
        Fraction of entries to be replaced with NaN (between 0 and 1).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    arr1_new, arr2_new : np.ndarray
        Arrays with NaNs inserted.
    """
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("Both arrays must be 1D.")
    if not (0 <= ratio <= 1):
        raise ValueError("ratio must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    arr1_new = arr1.astype(float, copy=True)
    arr2_new = arr2.astype(float, copy=True)

    # number of NaNs to insert
    n1 = int(len(arr1) * ratio)
    n2 = int(len(arr2) * ratio)

    # choose random indices for NaNs
    idx1 = rng.choice(len(arr1), size=n1, replace=False)
    idx2 = rng.choice(len(arr2), size=n2, replace=False)

    arr1_new[idx1] = np.nan
    arr2_new[idx2] = np.nan

    return arr1_new, arr2_new

def simulate_mar(arr1, arr2, ratio, random_state=None):
    """
    Insert NaNs into two numpy arrays under a MAR mechanism,
    ensuring each array has exactly ratio * len(arr) NaNs.

    Missingness in arr1 depends on arr2, and missingness in arr2 depends on arr1.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Input 1D arrays (must have the same length).
    ratio : float
        Fraction of entries to be replaced with NaN in each array (between 0 and 1).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    arr1_new, arr2_new : np.ndarray
        Arrays with NaNs inserted under MAR.
    """
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("Both arrays must be 1D.")
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length.")
    if not (0 <= ratio <= 1):
        raise ValueError("ratio must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    arr1_new = arr1.astype(float, copy=True)
    arr2_new = arr2.astype(float, copy=True)

    n_missing = int(len(arr1) * ratio)

    # --- Missingness for arr1 (depends on arr2 values) ---
    p1 = (arr2 - np.nanmin(arr2)) / (np.nanmax(arr2) - np.nanmin(arr2) + 1e-9)
    p1 /= p1.sum()
    idx1 = rng.choice(len(arr1), size=n_missing, replace=False, p=p1)
    arr1_new[idx1] = np.nan

    # --- Missingness for arr2 (depends on arr1 values) ---
    p2 = 1 - (arr1 - np.nanmin(arr1)) / (np.nanmax(arr1) - np.nanmin(arr1) + 1e-9)
    p2 /= p2.sum()
    idx2 = rng.choice(len(arr2), size=n_missing, replace=False, p=p2)
    arr2_new[idx2] = np.nan

    return arr1_new, arr2_new

def save_results(x_data : int,
                 y_data : int,
                 na_ratio : float,
                 test : str,
                 mechanism : str,
                 pvalue : float,
                 effect_size : float,
                 results_dict : dict):
    results_dict['x_data'].append(str(x_data))
    results_dict['y_data'].append(str(y_data))
    results_dict['na_ratio'].append(na_ratio)
    results_dict['test'].append(test)
    results_dict['mechanism'].append(mechanism)
    results_dict['pvalue'].append(pvalue)
    results_dict['effect_size'].append(effect_size)
    return results_dict

def simulate_mnar(arr1, arr2, ratio, random_state=None):
    """
    Insert NaNs into two numpy arrays under a MNAR mechanism,
    ensuring each array has exactly ratio * len(arr) NaNs.

    Missingness in arr1 depends on arr1, and missingness in arr2 depends on arr2.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Input 1D arrays (must have the same length).
    ratio : float
        Fraction of entries to be replaced with NaN in each array (between 0 and 1).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    arr1_new, arr2_new : np.ndarray
        Arrays with NaNs inserted under MNAR.
    """
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("Both arrays must be 1D.")
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length.")
    if not (0 <= ratio <= 1):
        raise ValueError("ratio must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    arr1_new = arr1.astype(float, copy=True)
    arr2_new = arr2.astype(float, copy=True)

    n_missing = int(len(arr1) * ratio)

    def safe_probs(x, invert=False):
        """Compute safe probability distribution from array values."""
        if np.all(np.isnan(x)):
            return np.ones(len(x)) / len(x)
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if np.isclose(x_max, x_min):
            return np.ones(len(x)) / len(x)
        p = (x - x_min) / (x_max - x_min)
        if invert:
            p = 1 - p
        p = np.clip(p, 0, None)
        p_sum = p.sum()
        return p / p_sum if p_sum > 0 else np.ones(len(x)) / len(x)

    # --- Missingness for arr1 (depends on arr1 itself) ---
    p1 = safe_probs(arr1_new, invert=False)  # bias: larger arr1 values more likely missing
    idx1 = rng.choice(len(arr1), size=n_missing, replace=False, p=p1)
    arr1_new[idx1] = np.nan

    # --- Missingness for arr2 (depends on arr2 itself) ---
    p2 = safe_probs(arr2_new, invert=True)  # bias: smaller arr2 values more likely missing
    idx2 = rng.choice(len(arr2), size=n_missing, replace=False, p=p2)
    arr2_new[idx2] = np.nan

    return arr1_new, arr2_new


def simulate_high_pearson(n=100, noise=0.05, random_state=None):
    """
    Simulate a pair of variables with high Pearson correlation.

    Parameters
    ----------
    n : int
        Number of samples.
    noise : float
        Strength of noise (0 = perfect linear relation).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    x, y : np.ndarray
        Two 1D arrays with high Pearson correlation.
    corr : float
        Pearson correlation coefficient.
    """
    rng = np.random.default_rng(random_state)

    # Base variable
    x = rng.normal(size=n)

    # Linear relationship with Gaussian noise
    y = 2 * x + noise * rng.normal(size=n)

    return x, y


def simulate_high_spearman(n=100, noise=0.05, random_state=None):
    """
    Simulate a pair of variables with high Spearman rank correlation.

    Parameters
    ----------
    n : int
        Number of samples.
    noise : float
        Strength of noise (0 = perfect monotonic relation).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    x, y : np.ndarray
        Two 1D arrays with high Spearman correlation.
    corr : float
        Spearman correlation coefficient.
    """
    rng = np.random.default_rng(random_state)

    # Base variable
    x = rng.normal(size=n)

    # Sort to impose strong order
    x_sorted = np.sort(x)

    # Create y as a monotonic transformation of x, plus noise
    y = np.tanh(x_sorted) + noise * rng.normal(size=n)

    return x_sorted, y

if __name__ == "__main__":
    NUM_PAIRS = 100   # number of pairs
    NUM_SAMPLES = 1000 # samples per pair

    NA_RATIOS =[0.0, 0.1, 0.2, 0.3, 0.4]

    result_dict = {'x_data' : [], 'y_data': [], 'na_ratio' : [], 'test' : [], 'mechanism' : [], 'pvalue' : [], 'effect_size' : []}

    for num_pair in range(NUM_PAIRS):
        # Simulate given ratio into both variables of all variable pairs.
        pearson_x_sim, pearson_y_sim = simulate_high_pearson(n=NUM_SAMPLES, noise=1.5, random_state=num_pair)
        spearman_x_sim, spearman_y_sim = simulate_high_spearman(n=NUM_SAMPLES, noise=0.5, random_state=num_pair)
        print(f"Prcessing pair {num_pair}...")
        for na_ratio in NA_RATIOS:
            # Simulate MCAR missingness.
            pearson_x_mcar, pearson_y_mcar = simulate_mcar(pearson_x_sim.copy(), pearson_y_sim.copy(), na_ratio, random_state=num_pair)
            spearman_x_mcar, spearman_y_mcar = simulate_mcar(spearman_x_sim.copy(), spearman_y_sim.copy(), na_ratio, random_state=num_pair)

            # Simulate MAR missingness.
            pearson_x_mar, pearson_y_mar = simulate_mar(pearson_x_sim.copy(), pearson_y_sim.copy(), na_ratio, random_state=num_pair)
            spearman_x_mar, spearman_y_mar = simulate_mar(spearman_x_sim.copy(), spearman_y_sim.copy(), na_ratio, random_state=num_pair)

            # Simulate MNAR missingness.
            pearson_x_mnar, pearson_y_mnar = simulate_mnar(pearson_x_sim.copy(), pearson_y_sim.copy(), na_ratio,
                                                        random_state=num_pair)
            spearman_x_mnar, spearman_y_mnar = simulate_mnar(spearman_x_sim.copy(), spearman_y_sim.copy(), na_ratio,
                                                          random_state=num_pair)

            # Put into NApy format.
            pearson_mcar_stacked = np.array([pearson_x_mcar, pearson_y_mcar], dtype=np.float64)
            pearson_mcar_napy = np.nan_to_num(pearson_mcar_stacked, nan=-99999)

            pearson_mar_stacked = np.array([pearson_x_mar, pearson_y_mar], dtype=np.float64)
            pearson_mar_napy = np.nan_to_num(pearson_mar_stacked, nan=-99999)

            pearson_mnar_stacked = np.array([pearson_x_mnar, pearson_y_mnar], dtype=np.float64)
            pearson_mnar_napy = np.nan_to_num(pearson_mnar_stacked, nan=-99999)

            spearman_mcar_stacked = np.array([spearman_x_mcar, spearman_y_mcar], dtype=np.float64)
            spearman_mcar_napy = np.nan_to_num(spearman_mcar_stacked, nan=-99999)

            spearman_mar_stacked = np.array([spearman_x_mar, spearman_y_mar], dtype=np.float64)
            spearman_mar_napy = np.nan_to_num(spearman_mar_stacked, nan=-99999)

            spearman_mnar_stacked = np.array([spearman_x_mnar, spearman_y_mnar], dtype=np.float64)
            spearman_mnar_napy = np.nan_to_num(spearman_mnar_stacked, nan=-99999)

            # Compute Pearson and Spearman correlation.
            pearson_mcar_res = napy.pearsonr(pearson_mcar_napy, nan_value=-99999)
            spearman_mcar_res = napy.spearmanr(spearman_mcar_napy, nan_value=-99999)

            pearson_mar_res = napy.pearsonr(pearson_mar_napy, nan_value=-99999)
            spearman_mar_res = napy.spearmanr(spearman_mar_napy, nan_value=-99999)

            pearson_mnar_res = napy.pearsonr(pearson_mnar_napy, nan_value=-99999)
            spearman_mnar_res = napy.spearmanr(spearman_mnar_napy, nan_value=-99999)

            pearson_mcar_corr = pearson_mcar_res['r2'][0,1]
            pearson_mcar_pval = pearson_mcar_res['p_unadjusted'][0,1]
            spearman_mcar_corr = spearman_mcar_res['rho'][0,1]
            spearman_mcar_pval = spearman_mcar_res['p_unadjusted'][0,1]

            pearson_mar_corr = pearson_mar_res['r2'][0, 1]
            pearson_mar_pval = pearson_mar_res['p_unadjusted'][0, 1]
            spearman_mar_corr = spearman_mar_res['rho'][0, 1]
            spearman_mar_pval = spearman_mar_res['p_unadjusted'][0, 1]

            pearson_mnar_corr = pearson_mnar_res['r2'][0, 1]
            pearson_mnar_pval = pearson_mnar_res['p_unadjusted'][0, 1]
            spearman_mnar_corr = spearman_mnar_res['rho'][0, 1]
            spearman_mnar_pval = spearman_mnar_res['p_unadjusted'][0, 1]

            result_dict = save_results(x_data=num_pair,
                         y_data=num_pair,
                         na_ratio=na_ratio,
                         test='pearson',
                         mechanism='MCAR',
                         pvalue=pearson_mcar_pval,
                         effect_size=pearson_mcar_corr,
                         results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='pearson',
                                       mechanism='MAR',
                                       pvalue=pearson_mar_pval,
                                       effect_size=pearson_mar_corr,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='pearson',
                                       mechanism='MNAR',
                                       pvalue=pearson_mnar_pval,
                                       effect_size=pearson_mnar_corr,
                                       results_dict=result_dict)

            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='spearman',
                                       mechanism='MCAR',
                                       pvalue=spearman_mcar_pval,
                                       effect_size=spearman_mcar_corr,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='spearman',
                                       mechanism='MAR',
                                       pvalue=spearman_mar_pval,
                                       effect_size=spearman_mar_corr,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='spearman',
                                       mechanism='MNAR',
                                       pvalue=spearman_mnar_pval,
                                       effect_size=spearman_mnar_corr,
                                       results_dict=result_dict)

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("stability_analysis_cont_cont.csv", index=False)
    quit()
    df = result_df[result_df['test']=='pearson']

    df["neglog10_pvalue"] = -np.log10(df["pvalue"])

    # Plot
    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(
        data=df,
        x="na_ratio",
        y="effect_size",
        hue='mechanism',
        inner="box",  # shows median + IQR inside
        cut=0
    )

    #plt.title("-log10(p-value) vs Missing Data Ratio")
    ax.legend(title="Mechanism")
    plt.ylabel("Pearson R")
    #plt.ylabel("Pearson R")
    plt.xlabel("Simulated NA ratio")
    plt.title("Pearson Correlation - Effect Size Stability")
    plt.tight_layout()
    plt.savefig('pearson_effect_size.png')
    plt.show()

