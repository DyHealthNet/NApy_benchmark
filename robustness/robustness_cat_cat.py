import numpy as np
import pandas as pd
import napy
import matplotlib.pyplot as plt
import seaborn as sns
from missmecha import MissMechaGenerator
from math import sqrt

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

def make_joint_prob_matrix(r, V_target):
    """
    Construct r x r joint probability matrix with uniform marginals
    and (approximately) population Cramer's V == V_target.
    """
    # delta derived from algebra (uniform marginal case)
    delta = V_target * (r - 1) / (r**2)

    p_diag = 1.0 / (r**2) + delta
    p_off  = 1.0 / (r**2) - delta / (r - 1)

    if p_off < -1e-12:
        raise ValueError(f"Requested V={V_target} too large for r={r} (off-diagonal negative).")

    P = np.full((r, r), p_off)
    np.fill_diagonal(P, p_diag)

    # numeric fix (tiny negative rounding)
    P[P < 0] = 0.0
    P = P / P.sum()   # re-normalize to ensure sums to 1 (safety)
    return P

def sample_categorical_pair(r, V_target, n, labels=None, random_state=None):
    """
    Return two arrays (size n) of categorical values (0..r-1 or provided labels)
    whose population joint distribution has approx Cramer's V = V_target.
    """
    rng = np.random.default_rng(random_state)
    P = make_joint_prob_matrix(r, V_target)
    probs = P.ravel()
    counts = rng.multinomial(n, probs)
    counts = counts.reshape((r, r))

    # expand counts to sample arrays
    xs = []
    ys = []
    for i in range(r):
        for j in range(r):
            c = counts[i, j]
            if c:
                xs.extend([i] * c)
                ys.extend([j] * c)
    xs = np.array(xs)
    ys = np.array(ys)

    # optionally map to labels
    if labels is not None:
        labels = list(labels)
        xs = np.array([labels[i] for i in xs])
        ys = np.array([labels[j] for j in ys])

    # shuffle to randomize order
    perm = rng.permutation(n)
    return xs[perm], ys[perm], P, counts


def simulate_missmecha(arr1 : np.ndarray, arr2 : np.ndarray, missing_mode : str, na_ratio : float, random_state : int):
    # Put input arrays into two-column matrix.
    input_df = pd.DataFrame({"cat1" : arr1, "cat2" : arr2})
    generator = MissMechaGenerator(mechanism=missing_mode, mechanism_type=1, missing_rate=na_ratio, seed=random_state,
                                   cat_cols=["cat1", "cat2"])
    simulated_matrix = generator.fit_transform(input_df)
    arr1_simulated = simulated_matrix.to_numpy()[:, 0].copy()
    arr2_simulated = simulated_matrix.to_numpy()[:, 1].copy()
    return arr1_simulated, arr2_simulated

if __name__ == "__main__":
    NUM_PAIRS = 100   # number of pairs
    NUM_SAMPLES = 1000 # samples per pair
    NUM_CATEGORIES = 4

    NA_RATIOS =[0.0, 0.1, 0.2, 0.3, 0.4]

    result_dict = {'x_data' : [], 'y_data': [], 'na_ratio' : [], 'test' : [], 'mechanism' : [], 'pvalue' : [], 'effect_size' : []}

    for num_pair in range(NUM_PAIRS):
        # Simulate given ratio into both variables of all variable pairs.
        chi2_x_sim, chi2_y_sim, _, _ = sample_categorical_pair(r=NUM_CATEGORIES, V_target=0.4, n=NUM_SAMPLES, random_state=num_pair)
        print(f"Prcessing pair {num_pair}...")
        for na_ratio in NA_RATIOS:
            # Simulate MCAR missingness.
            chi2_x_mcar, chi2_y_mcar = simulate_missmecha(chi2_x_sim.copy(),
                                                                chi2_y_sim.copy(),
                                                                missing_mode="mcar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)

            # Simulate MAR missingness.
            chi2_x_mar, chi2_y_mar = simulate_missmecha(chi2_x_sim.copy(),
                                                                chi2_y_sim.copy(),
                                                                missing_mode="mar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)

            # Simulate MNAR missingness.
            chi2_x_mnar, chi2_y_mnar = simulate_missmecha(chi2_x_sim.copy(),
                                                                chi2_y_sim.copy(),
                                                                missing_mode="mnar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)

            # Put into NApy format.
            chi2_mcar_stacked = np.array([chi2_x_mcar, chi2_y_mcar], dtype=np.float64)
            chi2_mcar_napy = np.nan_to_num(chi2_mcar_stacked, nan=-99999)

            chi2_mar_stacked = np.array([chi2_x_mar, chi2_y_mar], dtype=np.float64)
            chi2_mar_napy = np.nan_to_num(chi2_mar_stacked, nan=-99999)

            chi2_mnar_stacked = np.array([chi2_x_mnar, chi2_y_mnar], dtype=np.float64)
            chi2_mnar_napy = np.nan_to_num(chi2_mnar_stacked, nan=-99999)


            # Compute Pearson and Spearman correlation.
            chi2_mcar_res = napy.chi_squared(chi2_mcar_napy, nan_value=-99999)

            chi2_mar_res = napy.chi_squared(chi2_mar_napy, nan_value=-99999)

            chi2_mnar_res = napy.chi_squared(chi2_mnar_napy, nan_value=-99999)

            chi2_mcar_eff = chi2_mcar_res['cramers_v'][0,1]
            chi2_mcar_pval = chi2_mcar_res['p_unadjusted'][0,1]

            chi2_mar_eff = chi2_mar_res['cramers_v'][0, 1]
            chi2_mar_pval = chi2_mar_res['p_unadjusted'][0, 1]

            chi2_mnar_eff = chi2_mnar_res['cramers_v'][0, 1]
            chi2_mnar_pval = chi2_mnar_res['p_unadjusted'][0, 1]

            result_dict = save_results(x_data=num_pair,
                         y_data=num_pair,
                         na_ratio=na_ratio,
                         test='chi2',
                         mechanism='MCAR',
                         pvalue=chi2_mcar_pval,
                         effect_size=chi2_mcar_eff,
                         results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='chi2',
                                       mechanism='MAR',
                                       pvalue=chi2_mar_pval,
                                       effect_size=chi2_mar_eff,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='chi2',
                                       mechanism='MNAR',
                                       pvalue=chi2_mnar_pval,
                                       effect_size=chi2_mnar_eff,
                                       results_dict=result_dict)


    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("stability_analysis_cat_cat.csv", index=False)

    df = result_df[result_df['test']=='chi2']

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
    plt.ylabel("Chi")
    #plt.ylabel("Pearson R")
    plt.xlabel("Simulated NA ratio")
    #plt.title("Pearson Correlation - Effect Size Stability")
    plt.tight_layout()
    plt.savefig('chi2_pvalue.png')
    plt.show()

