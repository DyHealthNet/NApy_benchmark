import numpy as np
import pandas as pd
import napy
import matplotlib.pyplot as plt
import seaborn as sns
from missmecha import MissMechaGenerator
from scipy.stats import mannwhitneyu, norm
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

def simulate_cohens_d(      n,
                            prop=0.5,
                            d=0.5,
                            sigma=1.0,
                            mu0=0.0,
                            random_state=None):
    """
    Simulate a binary group indicator (0/1) and a numeric outcome such that
    the true population Cohen's d between the groups is (approximately) `d`,
    assuming equal group standard deviations = sigma.

    Parameters
    ----------
    n : int
        Total sample size.
    prop : float in (0,1)
        Proportion of samples in group 1. group0 size = n0 = n - n1.
    d : float
        Desired Cohen's d (mean1 - mean0) / sigma.
    sigma : float
        Common standard deviation for both groups.
    mu0 : float
        Mean of group 0. Group 1 mean will be mu0 + d*sigma.
    random_state : int or None
        RNG seed.

    Returns
    -------
    groups : ndarray (n,) of 0/1
    values : ndarray (n,) numeric
    """
    rng = np.random.default_rng(random_state)
    n1 = int(round(prop * n))
    n0 = n - n1
    mu1 = mu0 + d * sigma

    x0 = rng.normal(loc=mu0, scale=sigma, size=n0)
    x1 = rng.normal(loc=mu1, scale=sigma, size=n1)

    groups = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    values = np.concatenate([x0, x1])

    # shuffle
    perm = rng.permutation(n)
    perm_groups = groups[perm]
    out_groups = 1-perm_groups
    return out_groups, values[perm]


def mannwhitney_r_from_counts(U, n0, n1):
    """Compute Z and r from U (no tie correction)."""
    mean_U = n0 * n1 / 2.0
    var_U = n0 * n1 * (n0 + n1 + 1) / 12.0
    z = (U - mean_U) / sqrt(var_U)
    r = z / sqrt(n0 + n1)
    return z, r

def estimate_mean_r_for_delta(delta, n0, n1, sigma=1.0, reps=200, random_state=None):
    """
    Monte Carlo estimate of mean r (Z/sqrt(N)) for Mann-Whitney when group1 ~ N(delta, sigma)
    and group0 ~ N(0, sigma).
    """
    rng = np.random.default_rng(random_state)
    N = n0 + n1
    r_vals = []
    for rep in range(reps):
        x0 = rng.normal(loc=0.0, scale=sigma, size=n0)
        x1 = rng.normal(loc=delta, scale=sigma, size=n1)
        # mannwhitneyu returns U of the first sample vs second sample by default (use x1,x0 so U relates to x1 > x0)
        Ustat, p = mannwhitneyu(x1, x0, alternative='two-sided')
        z, r = mannwhitney_r_from_counts(Ustat, n0, n1)
        r_vals.append(r)
    return float(np.mean(r_vals)), float(np.std(r_vals))

def find_delta_for_target_r(target_r, n0, n1, sigma=1.0, reps=400, tol_r=1e-3,
                            delta_bounds=(-5.0, 5.0), random_state=None):
    """
    Find delta (mean shift between groups) so that mean Monte-Carlo r ≈ target_r.
    Uses bisection on delta_bounds. Returns (delta, est_r, est_r_sd).
    """
    lo, hi = delta_bounds
    rng = np.random.default_rng(random_state)
    # Evaluate endpoints
    r_lo, sd_lo = estimate_mean_r_for_delta(lo, n0, n1, sigma=sigma, reps=reps//4, random_state=rng.bit_generator)
    r_hi, sd_hi = estimate_mean_r_for_delta(hi, n0, n1, sigma=sigma, reps=reps//4, random_state=rng.bit_generator)
    # If target is outside bracket, warn but proceed
    if not (r_lo <= target_r <= r_hi or r_hi <= target_r <= r_lo):
        # Still attempt search, but user should pick wider bounds
        pass

    # Bisection
    max_iter = 30
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid, sd_mid = estimate_mean_r_for_delta(mid, n0, n1, sigma=sigma, reps=reps, random_state=rng.bit_generator)
        # debug print can be enabled if needed:
        # print(f"iter {i}: delta={mid:.4f}, r={r_mid:.4f}")
        if abs(r_mid - target_r) <= tol_r:
            return mid, r_mid, sd_mid
        # decide side: assume monotonic (increasing delta increases r)
        if r_mid < target_r:
            lo = mid
        else:
            hi = mid
    # return last estimate if not converged
    return mid, r_mid, sd_mid

def simulate_target_mannwhitney_r(n, prop=0.5, target_r=0.2,
                                  sigma=1.0, reps_for_search=400,
                                  random_state=None):
    """
    High-level function:
    - finds delta so that expected r (Z/sqrt(N)) from Mann-Whitney is ≈ target_r
    - returns one simulated dataset at that delta and diagnostics.
    """
    rng = np.random.default_rng(random_state)
    n1 = int(round(prop * n))
    n0 = n - n1
    if n0 <= 0 or n1 <= 0:
        raise ValueError("Invalid group sizes from n and prop.")

    # find delta
    delta, est_r, est_r_sd = find_delta_for_target_r(
        target_r, n0, n1, sigma=sigma, reps=reps_for_search,
        delta_bounds=(-6.0, 6.0), random_state=rng.bit_generator
    )

    # generate final sample with that delta
    x0 = rng.normal(loc=0.0, scale=sigma, size=n0)
    x1 = rng.normal(loc=delta, scale=sigma, size=n1)
    groups = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    values = np.concatenate([x0, x1])
    perm = rng.permutation(n)
    groups = groups[perm]
    #groups = 1.0-groups
    values = values[perm]

    return groups, values


def simulate_missmecha(bin : np.ndarray, cont : np.ndarray, missing_mode : str, na_ratio : float, random_state : int):
    # Put input arrays into two-column matrix.
    input_df = pd.DataFrame({"bin" : bin, "cont" : cont})
    generator = MissMechaGenerator(mechanism=missing_mode, mechanism_type=1, missing_rate=na_ratio, seed=random_state,
                                   cat_cols=["bin"])
    simulated_matrix = generator.fit_transform(input_df)
    arr1_simulated = simulated_matrix.to_numpy()[:, 0].copy()
    arr2_simulated = simulated_matrix.to_numpy()[:, 1].copy()
    return arr1_simulated, arr2_simulated

if __name__ == "__main__":
    NUM_PAIRS = 100   # number of pairs
    NUM_SAMPLES = 1000 # samples per pair

    NA_RATIOS =[0.0, 0.1, 0.2, 0.3, 0.4]

    result_dict = {'x_data' : [], 'y_data': [], 'na_ratio' : [], 'test' : [], 'mechanism' : [], 'pvalue' : [], 'effect_size' : []}

    for num_pair in range(NUM_PAIRS):
        # Simulate given ratio into both variables of all variable pairs.
        ttest_bin_sim, ttest_cont_sim = simulate_cohens_d(n=NUM_SAMPLES, prop=0.5, d=1.5, random_state=num_pair)
        mwu_bin_sim, mwu_cont_sim = simulate_target_mannwhitney_r(n=NUM_SAMPLES, target_r=0.3, random_state=num_pair)
        print(f"Prcessing pair {num_pair}...")
        for na_ratio in NA_RATIOS:
            # Simulate MCAR missingness.
            ttest_bin_mcar, ttest_cont_mcar = simulate_missmecha(ttest_bin_sim.copy(),
                                                                ttest_cont_sim.copy(),
                                                                missing_mode="mcar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)
            mwu_bin_mcar, mwu_cont_mcar = simulate_missmecha(mwu_bin_sim.copy(),
                                                                mwu_cont_sim.copy(),
                                                                missing_mode="mcar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)

            # Simulate MAR missingness.
            ttest_bin_mar, ttest_cont_mar = simulate_missmecha(ttest_bin_sim.copy(),
                                                                 ttest_cont_sim.copy(),
                                                                 missing_mode="mar",
                                                                 na_ratio=na_ratio,
                                                                 random_state=num_pair)
            mwu_bin_mar, mwu_cont_mar = simulate_missmecha(mwu_bin_sim.copy(),
                                                             mwu_cont_sim.copy(),
                                                             missing_mode="mar",
                                                             na_ratio=na_ratio,
                                                             random_state=num_pair)

            # Simulate MNAR missingness.
            ttest_bin_mnar, ttest_cont_mnar = simulate_missmecha(ttest_bin_sim.copy(),
                                                                 ttest_cont_sim.copy(),
                                                                 missing_mode="mnar",
                                                                 na_ratio=na_ratio,
                                                                 random_state=num_pair)
            mwu_bin_mnar, mwu_cont_mnar = simulate_missmecha(mwu_bin_sim.copy(),
                                                             mwu_cont_sim.copy(),
                                                             missing_mode="mnar",
                                                             na_ratio=na_ratio,
                                                             random_state=num_pair)

            # Put into NApy format.
            ttest_mcar_stacked_bin = np.array([ttest_bin_mcar], dtype=np.float64)
            ttest_mcar_napy_bin = np.nan_to_num(ttest_mcar_stacked_bin, nan=-99999)
            ttest_mcar_stacked_cont = np.array([ttest_cont_mcar], dtype=np.float64)
            ttest_mcar_napy_cont = np.nan_to_num(ttest_mcar_stacked_cont, nan=-99999)

            ttest_mar_stacked_bin = np.array([ttest_bin_mar], dtype=np.float64)
            ttest_mar_napy_bin = np.nan_to_num(ttest_mar_stacked_bin, nan=-99999)
            ttest_mar_stacked_cont = np.array([ttest_cont_mar], dtype=np.float64)
            ttest_mar_napy_cont = np.nan_to_num(ttest_mar_stacked_cont, nan=-99999)

            ttest_mnar_stacked_bin = np.array([ttest_bin_mnar], dtype=np.float64)
            ttest_mnar_napy_bin = np.nan_to_num(ttest_mnar_stacked_bin, nan=-99999)
            ttest_mnar_stacked_cont = np.array([ttest_cont_mnar], dtype=np.float64)
            ttest_mnar_napy_cont = np.nan_to_num(ttest_mnar_stacked_cont, nan=-99999)

            mwu_mcar_stacked_bin = np.array([mwu_bin_mcar], dtype=np.float64)
            mwu_mcar_napy_bin = np.nan_to_num(mwu_mcar_stacked_bin, nan=-99999)
            mwu_mcar_stacked_cont = np.array([mwu_cont_mcar], dtype=np.float64)
            mwu_mcar_napy_cont = np.nan_to_num(mwu_mcar_stacked_cont, nan=-99999)

            mwu_mar_stacked_bin = np.array([mwu_bin_mar], dtype=np.float64)
            mwu_mar_napy_bin = np.nan_to_num(mwu_mar_stacked_bin, nan=-99999)
            mwu_mar_stacked_cont = np.array([mwu_cont_mar], dtype=np.float64)
            mwu_mar_napy_cont = np.nan_to_num(mwu_mar_stacked_cont, nan=-99999)

            mwu_mnar_stacked_bin = np.array([mwu_bin_mnar], dtype=np.float64)
            mwu_mnar_napy_bin = np.nan_to_num(mwu_mnar_stacked_bin, nan=-99999)
            mwu_mnar_stacked_cont = np.array([mwu_cont_mnar], dtype=np.float64)
            mwu_mnar_napy_cont = np.nan_to_num(mwu_mnar_stacked_cont, nan=-99999)

            # Compute Pearson and Spearman correlation.
            ttest_mcar_res = napy.ttest(bin_data=ttest_mcar_napy_bin, cont_data=ttest_mcar_napy_cont, nan_value=-99999)
            mwu_mcar_res = napy.mwu(bin_data=mwu_mcar_napy_bin, cont_data=mwu_mcar_napy_cont, nan_value=-99999)

            ttest_mar_res = napy.ttest(bin_data=ttest_mar_napy_bin, cont_data=ttest_mar_napy_cont, nan_value=-99999)
            mwu_mar_res = napy.mwu(bin_data=mwu_mar_napy_bin, cont_data=mwu_mar_napy_cont, nan_value=-99999)

            ttest_mnar_res = napy.ttest(bin_data=ttest_mnar_napy_bin, cont_data=ttest_mnar_napy_cont, nan_value=-99999)
            mwu_mnar_res = napy.mwu(bin_data=mwu_mnar_napy_bin, cont_data=mwu_mnar_napy_cont, nan_value=-99999)

            ttest_mcar_eff = ttest_mcar_res['cohens_d'][0,0]
            ttest_mcar_pval = ttest_mcar_res['p_unadjusted'][0,0]
            mwu_mcar_eff = mwu_mcar_res['r'][0,0]
            mwu_mcar_pval = mwu_mcar_res['p_unadjusted'][0,0]

            ttest_mar_eff = ttest_mar_res['cohens_d'][0, 0]
            ttest_mar_pval = ttest_mar_res['p_unadjusted'][0, 0]
            mwu_mar_eff = mwu_mar_res['r'][0, 0]
            mwu_mar_pval = mwu_mar_res['p_unadjusted'][0, 0]

            ttest_mnar_eff = ttest_mnar_res['cohens_d'][0, 0]
            ttest_mnar_pval = ttest_mnar_res['p_unadjusted'][0, 0]
            mwu_mnar_eff = mwu_mnar_res['r'][0, 0]
            mwu_mnar_pval = mwu_mnar_res['p_unadjusted'][0, 0]

            result_dict = save_results(x_data=num_pair,
                         y_data=num_pair,
                         na_ratio=na_ratio,
                         test='ttest',
                         mechanism='MCAR',
                         pvalue=ttest_mcar_pval,
                         effect_size=ttest_mcar_eff,
                         results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='ttest',
                                       mechanism='MAR',
                                       pvalue=ttest_mar_pval,
                                       effect_size=ttest_mar_eff,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='ttest',
                                       mechanism='MNAR',
                                       pvalue=ttest_mnar_pval,
                                       effect_size=ttest_mnar_eff,
                                       results_dict=result_dict)

            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='mwu',
                                       mechanism='MCAR',
                                       pvalue=mwu_mcar_pval,
                                       effect_size=mwu_mcar_eff,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='mwu',
                                       mechanism='MAR',
                                       pvalue=mwu_mar_pval,
                                       effect_size=mwu_mar_eff,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='mwu',
                                       mechanism='MNAR',
                                       pvalue=mwu_mnar_pval,
                                       effect_size=mwu_mnar_eff,
                                       results_dict=result_dict)

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("stability_analysis_bin_cont.csv", index=False)
    df = result_df[result_df['test']=='mwu']

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
    plt.ylabel("Effect Size")
    #plt.ylabel("Pearson R")
    plt.xlabel("Simulated NA ratio")
    #plt.title("Pearson Correlation - Effect Size Stability")
    plt.tight_layout()
    #plt.savefig('pearson_pvalue.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(
        data=df,
        x="na_ratio",
        y="neglog10_pvalue",
        hue='mechanism',
        inner="box",  # shows median + IQR inside
        cut=0
    )

    # plt.title("-log10(p-value) vs Missing Data Ratio")
    ax.legend(title="Mechanism")
    plt.ylabel("Pvalue")
    # plt.ylabel("Pearson R")
    plt.xlabel("Simulated NA ratio")
    #plt.title("Pearson Correlation - Effect Size Stability")
    plt.tight_layout()
    #plt.savefig('pearson_pvalue.png')
    plt.show()

