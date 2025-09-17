from random import random

import numpy as np
import pandas as pd
import napy
import matplotlib.pyplot as plt
import seaborn as sns
from missmecha import MissMechaGenerator

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

def simulate_missmecha(arr1 : np.ndarray, arr2 : np.ndarray, missing_mode : str, na_ratio : float, random_state : int):
    # Put input arrays into two-column matrix.
    input_matrix = np.column_stack((arr1, arr2))
    generator = MissMechaGenerator(mechanism=missing_mode, mechanism_type=1, missing_rate=na_ratio, seed=random_state)
    simulated_matrix = generator.fit_transform(input_matrix)
    arr1_simulated = simulated_matrix[:, 0].copy()
    arr2_simulated = simulated_matrix[:, 1].copy()
    return arr1_simulated, arr2_simulated

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
            pearson_x_mcar, pearson_y_mcar = simulate_missmecha(pearson_x_sim.copy(),
                                                                pearson_y_sim.copy(),
                                                                missing_mode="mcar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)
            spearman_x_mcar, spearman_y_mcar = simulate_missmecha(spearman_x_sim.copy(),
                                                                spearman_y_sim.copy(),
                                                                missing_mode="mcar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)

            # Simulate MAR missingness.
            pearson_x_mar, pearson_y_mar = simulate_missmecha(pearson_x_sim.copy(),
                                                                pearson_y_sim.copy(),
                                                                missing_mode="mar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)
            spearman_x_mar, spearman_y_mar = simulate_missmecha(spearman_x_sim.copy(),
                                                                spearman_y_sim.copy(),
                                                                missing_mode="mar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)

            # Simulate MNAR missingness.
            pearson_x_mnar, pearson_y_mnar = simulate_missmecha(pearson_x_sim.copy(),
                                                                pearson_y_sim.copy(),
                                                                missing_mode="mnar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)
            spearman_x_mnar, spearman_y_mnar = simulate_missmecha(spearman_x_sim.copy(),
                                                                spearman_y_sim.copy(),
                                                                missing_mode="mnar",
                                                                na_ratio=na_ratio,
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
        y="neglog10_pvalue",
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
    plt.savefig('pearson_pvalue.png')
    plt.show()

