import numpy as np
import pandas as pd
import napy
import matplotlib.pyplot as plt
import seaborn as sns
from missmecha import MissMechaGenerator
from scipy.stats import rankdata

def simulate_oneway_for_eta2(k,
                            n,
                            eta2_target,
                            sigma=1.0,
                            grand_mean=0.0,
                            random_state=None):
    """
    Simulate a categorical (k levels) and numeric variable (n per group)
    so that population partial eta squared for one-way ANOVA ~= eta2_target.

    Parameters
    ----------
    k : int
        Number of groups (levels).
    n : int
        Sample size per group (equal-n design). For unequal n, see notes.
    eta2_target : float in (0,1)
        Desired partial eta squared.
    sigma : float
        Within-group standard deviation.
    grand_mean : float
        Overall mean (center) for group means.
    random_state : int or None
        RNG seed.

    Returns
    -------
    dict with keys:
      'groups' : array of group labels (0..k-1)
      'values' : numeric outcomes
      'group_means' : the true group means used
      'f' : Cohen's f (population)
      'eta2_target' : requested eta2
      'observed_eta2' : observed partial eta squared from the drawn sample
      'F', 'p' : ANOVA F-statistic and p-value
      'SS_between', 'SS_within' : sums of squares from the sample
    """
    if not (0.0 < eta2_target < 1.0):
        raise ValueError("eta2_target must be between 0 and 1 (exclusive).")
    if k < 2:
        raise ValueError("k must be >= 2.")
    if n < 1:
        raise ValueError("n must be >= 1.")

    rng = np.random.default_rng(random_state)

    # Convert eta2 -> Cohen's f and then obtain tau (SD of group means)
    f = np.sqrt(eta2_target / (1.0 - eta2_target))
    tau = f * sigma   # desired SD of group means

    # Build a base sequence of k values centered at 0 and scale to have unit SD
    base = np.arange(k, dtype=float)  # 0,1,2,...,k-1
    base = base - base.mean()         # centered
    base_sd = base.std(ddof=0)
    if base_sd == 0:
        # only possible if k==1 which we prevented
        raise RuntimeError("unexpected base_sd==0")
    unit_sequence = base / base_sd   # now sd(unit_sequence) == 1

    # set group means so that SD(group_means) == tau and centered at grand_mean
    group_means = grand_mean + tau * unit_sequence

    # draw observations
    values = []
    groups = []
    for i, mu in enumerate(group_means):
        xi = rng.normal(loc=mu, scale=sigma, size=n)
        values.append(xi)
        groups.append(np.full(n, i, dtype=int))
    values = np.concatenate(values)
    groups = np.concatenate(groups)

    # compute ANOVA sums of squares (one-way)
    N = len(values)
    overall_mean = values.mean()
    # SS_between = sum_{i} n * (mean_i - overall_mean)^2
    means_per_group = np.array([values[groups == i].mean() for i in range(k)])
    SS_between = float(np.sum(n * (means_per_group - overall_mean) ** 2))
    # SS_within = sum_{i} sum_{j in group i} (x_ij - mean_i)^2
    SS_within = float(np.sum([(values[groups == i] - means_per_group[i]) ** 2 for i in range(k)]))
    df_between = k - 1
    df_within = N - k
    MS_between = SS_between / df_between
    MS_within = SS_within / df_within

    return groups, values

def kruskal_H_and_eta2(groups, values):
    """
    Compute Kruskal-Wallis H with tie correction, and eta2_H = (H - k + 1)/(N - k).
    groups: array-like of integer group labels 0..k-1
    values: numeric outcomes
    returns: H, eta2_H
    """
    groups = np.asarray(groups)
    values = np.asarray(values)
    N = len(values)
    labels, idx = np.unique(groups, return_inverse=True)
    k = len(labels)
    # ranks
    ranks = rankdata(values, method="average")
    # sum of ranks per group and sizes
    R = np.zeros(k, dtype=float)
    n = np.zeros(k, dtype=int)
    for i in range(k):
        mask = (idx == i)
        n[i] = mask.sum()
        R[i] = ranks[mask].sum()
    if np.any(n == 0):
        return np.nan, np.nan

    # uncorrected H
    H_uncorr = (12.0 / (N * (N + 1))) * np.sum((R ** 2) / n) - 3.0 * (N + 1)

    # tie correction factor:
    # counts of equal values
    _, counts = np.unique(values, return_counts=True)
    if len(counts) > 0:
        tie_term = np.sum(counts * (counts ** 2 - 1))
        # correction denominator:
        denom = N ** 3 - N
        if denom != 0:
            C = 1.0 - (tie_term / denom)
            if C <= 0:
                # degenerate tie situation
                H = H_uncorr
            else:
                H = H_uncorr / C
        else:
            H = H_uncorr
    else:
        H = H_uncorr

    # eta2_H (sometimes called epsilon-squared); ensure we don't return negative due to sampling noise
    eta2 = (H - k + 1) / (N - k) if (N - k) > 0 else np.nan
    if not np.isnan(eta2) and eta2 < 0:
        eta2 = 0.0
    return H, eta2

def _make_group_means(k, spread):
    """
    Build k group means centered on 0 whose SD == spread.
    A simple symmetric construction: use 0..k-1 centered and scale.
    """
    base = np.arange(k, dtype=float)
    base = base - base.mean()
    base_sd = base.std(ddof=0)
    if base_sd == 0:
        return np.full(k, 0.0)
    return (base / base_sd) * spread

def estimate_eta2_for_spread(spread, k, n_per_group, sigma=1.0, reps=200, rng=None):
    """
    Monte-Carlo estimate of mean eta2_H when group means have SD = spread,
    groups ~ N(mu_i, sigma), equal n_per_group.
    """
    if rng is None:
        rng = np.random.default_rng()
    k = int(k)
    n_per_group = int(n_per_group)
    eta_vals = []
    for _ in range(reps):
        mus = _make_group_means(k, spread)
        values = []
        groups = []
        for i, mu in enumerate(mus):
            xi = rng.normal(loc=mu, scale=sigma, size=n_per_group)
            values.append(xi)
            groups.append(np.full(n_per_group, i, dtype=int))
        values = np.concatenate(values)
        groups = np.concatenate(groups)
        _, eta2 = kruskal_H_and_eta2(groups, values)
        if not np.isnan(eta2):
            eta_vals.append(eta2)
    if len(eta_vals) == 0:
        return np.nan, np.nan
    return float(np.mean(eta_vals)), float(np.std(eta_vals))

def find_spread_for_target_eta2(target_eta2, k, n_per_group,
                                sigma=1.0, reps=300, tol=1e-3,
                                spread_bounds=(0.0, 6.0), rng=None, max_iter=30):
    """
    Bisection search over 'spread' (SD of group means) to match expected eta2_H ≈ target_eta2.
    Returns (spread, est_eta2, est_eta2_sd).
    """
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = spread_bounds
    r_lo, sd_lo = estimate_eta2_for_spread(lo, k, n_per_group, sigma=sigma, reps=max(10, reps//4), rng=rng)
    r_hi, sd_hi = estimate_eta2_for_spread(hi, k, n_per_group, sigma=sigma, reps=max(10, reps//4), rng=rng)
    # If target outside bracket, we still attempt search but warn (return closest endpoint if not possible)
    if not (r_lo <= target_eta2 <= r_hi or r_hi <= target_eta2 <= r_lo):
        # no bracket; continue but user may need to extend bounds
        pass

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid, sd_mid = estimate_eta2_for_spread(mid, k, n_per_group, sigma=sigma, reps=reps, rng=rng)
        # stop if close
        if not np.isnan(r_mid) and abs(r_mid - target_eta2) <= tol:
            return mid, r_mid, sd_mid
        # monotonic assumption: larger spread -> larger eta2 (true for symmetric shifts)
        if r_mid < target_eta2:
            lo = mid
        else:
            hi = mid
    # fallback: return last mid
    return mid, r_mid, sd_mid

def simulate_kruskal_with_target_eta2(k, n_per_group, target_eta2,
                                      sigma=1.0, reps_for_search=300,
                                      rng_seed=None, verbose=False):
    """
    High-level function.
    - Finds spread so expected eta2_H ≈ target_eta2 (Monte-Carlo + bisection).
    - Returns one simulated dataset using that spread and diagnostics.
    """
    rng = np.random.default_rng(rng_seed)
    spread, est_eta2, est_sd = find_spread_for_target_eta2(
        target_eta2, k, n_per_group, sigma=sigma, reps=reps_for_search, rng=rng
    )

    # generate one final dataset at found spread
    mus = _make_group_means(k, spread)
    values = []
    groups = []
    for i, mu in enumerate(mus):
        xi = rng.normal(loc=mu, scale=sigma, size=n_per_group)
        values.append(xi)
        groups.append(np.full(n_per_group, i, dtype=int))
    values = np.concatenate(values)
    groups = np.concatenate(groups)

    H, observed_eta2 = kruskal_H_and_eta2(groups, values)
    if verbose:
        print(f"spread={spread:.4f}, estimated eta2={est_eta2:.4f}±{est_sd:.4f}, observed eta2={observed_eta2:.4f}")
    return groups, values

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
    NUM_PAIRS = 100  # number of pairs
    NUM_SAMPLES = 1000 # samples per pair

    NA_RATIOS =[0.0, 0.1, 0.2, 0.3, 0.4]

    result_dict = {'x_data' : [], 'y_data': [], 'na_ratio' : [], 'test' : [], 'mechanism' : [], 'pvalue' : [], 'effect_size' : []}

    for num_pair in range(NUM_PAIRS):
        # Simulate given ratio into both variables of all variable pairs.
        anova_cat_sim, anova_cont_sim = simulate_oneway_for_eta2(k=4, n=250, eta2_target=0.5, random_state=num_pair)
        kruskal_cat_sim, kruskal_cont_sim = simulate_kruskal_with_target_eta2(k=4, n_per_group=250, target_eta2=0.05, rng_seed=num_pair)
        print(f"Prcessing pair {num_pair}...")
        for na_ratio in NA_RATIOS:
            # Simulate MCAR missingness.
            anova_cat_mcar, anova_cont_mcar = simulate_missmecha(anova_cat_sim.copy(),
                                                                anova_cont_sim.copy(),
                                                                missing_mode="mcar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)
            kruskal_cat_mcar, kruskal_cont_mcar = simulate_missmecha(kruskal_cat_sim.copy(),
                                                                kruskal_cont_sim.copy(),
                                                                missing_mode="mcar",
                                                                na_ratio=na_ratio,
                                                                random_state=num_pair)

            # Simulate MAR missingness.
            anova_cat_mar, anova_cont_mar = simulate_missmecha(anova_cat_sim.copy(),
                                                                 anova_cont_sim.copy(),
                                                                 missing_mode="mar",
                                                                 na_ratio=na_ratio,
                                                                 random_state=num_pair)
            kruskal_cat_mar, kruskal_cont_mar = simulate_missmecha(kruskal_cat_sim.copy(),
                                                             kruskal_cont_sim.copy(),
                                                             missing_mode="mar",
                                                             na_ratio=na_ratio,
                                                             random_state=num_pair)

            # Simulate MNAR missingness.
            anova_cat_mnar, anova_cont_mnar = simulate_missmecha(anova_cat_sim.copy(),
                                                                 anova_cont_sim.copy(),
                                                                 missing_mode="mnar",
                                                                 na_ratio=na_ratio,
                                                                 random_state=num_pair)
            kruskal_cat_mnar, kruskal_cont_mnar = simulate_missmecha(kruskal_cat_sim.copy(),
                                                             kruskal_cont_sim.copy(),
                                                             missing_mode="mnar",
                                                             na_ratio=na_ratio,
                                                             random_state=num_pair)

            # Put into NApy format.
            anova_mcar_stacked_cat = np.array([anova_cat_mcar], dtype=np.float64)
            anova_mcar_napy_cat = np.nan_to_num(anova_mcar_stacked_cat, nan=-99999)
            anova_mcar_stacked_cont = np.array([anova_cont_mcar], dtype=np.float64)
            anova_mcar_napy_cont = np.nan_to_num(anova_mcar_stacked_cont, nan=-99999)

            anova_mar_stacked_cat = np.array([anova_cat_mar], dtype=np.float64)
            anova_mar_napy_cat = np.nan_to_num(anova_mar_stacked_cat, nan=-99999)
            anova_mar_stacked_cont = np.array([anova_cont_mar], dtype=np.float64)
            anova_mar_napy_cont = np.nan_to_num(anova_mar_stacked_cont, nan=-99999)

            anova_mnar_stacked_cat = np.array([anova_cat_mnar], dtype=np.float64)
            anova_mnar_napy_cat = np.nan_to_num(anova_mnar_stacked_cat, nan=-99999)
            anova_mnar_stacked_cont = np.array([anova_cont_mnar], dtype=np.float64)
            anova_mnar_napy_cont = np.nan_to_num(anova_mnar_stacked_cont, nan=-99999)

            kruskal_mcar_stacked_cat = np.array([kruskal_cat_mcar], dtype=np.float64)
            kruskal_mcar_napy_cat = np.nan_to_num(kruskal_mcar_stacked_cat, nan=-99999)
            kruskal_mcar_stacked_cont = np.array([kruskal_cont_mcar], dtype=np.float64)
            kruskal_mcar_napy_cont = np.nan_to_num(kruskal_mcar_stacked_cont, nan=-99999)

            kruskal_mar_stacked_cat = np.array([kruskal_cat_mar], dtype=np.float64)
            kruskal_mar_napy_cat = np.nan_to_num(kruskal_mar_stacked_cat, nan=-99999)
            kruskal_mar_stacked_cont = np.array([kruskal_cont_mar], dtype=np.float64)
            kruskal_mar_napy_cont = np.nan_to_num(kruskal_mar_stacked_cont, nan=-99999)

            kruskal_mnar_stacked_cat = np.array([kruskal_cat_mnar], dtype=np.float64)
            kruskal_mnar_napy_cat = np.nan_to_num(kruskal_mnar_stacked_cat, nan=-99999)
            kruskal_mnar_stacked_cont = np.array([kruskal_cont_mnar], dtype=np.float64)
            kruskal_mnar_napy_cont = np.nan_to_num(kruskal_mnar_stacked_cont, nan=-99999)

            # Compute Pearson and Spearman correlation.
            anova_mcar_res = napy.anova(cat_data=anova_mcar_napy_cat, cont_data=anova_mcar_napy_cont, nan_value=-99999)
            kruskal_mcar_res = napy.kruskal_wallis(cat_data=kruskal_mcar_napy_cat, cont_data=kruskal_mcar_napy_cont, nan_value=-99999)

            anova_mar_res = napy.anova(cat_data=anova_mar_napy_cat, cont_data=anova_mar_napy_cont, nan_value=-99999)
            kruskal_mar_res = napy.kruskal_wallis(cat_data=kruskal_mar_napy_cat, cont_data=kruskal_mar_napy_cont, nan_value=-99999)

            anova_mnar_res = napy.anova(cat_data=anova_mnar_napy_cat, cont_data=anova_mnar_napy_cont, nan_value=-99999)
            kruskal_mnar_res = napy.kruskal_wallis(cat_data=kruskal_mnar_napy_cat, cont_data=kruskal_mnar_napy_cont, nan_value=-99999)

            anova_mcar_eff = anova_mcar_res['np2'][0,0]
            anova_mcar_pval = anova_mcar_res['p_unadjusted'][0,0]
            kruskal_mcar_eff = kruskal_mcar_res['eta2'][0,0]
            kruskal_mcar_pval = kruskal_mcar_res['p_unadjusted'][0,0]

            anova_mar_eff = anova_mar_res['np2'][0, 0]
            anova_mar_pval = anova_mar_res['p_unadjusted'][0, 0]
            kruskal_mar_eff = kruskal_mar_res['eta2'][0, 0]
            kruskal_mar_pval = kruskal_mar_res['p_unadjusted'][0, 0]

            anova_mnar_eff = anova_mnar_res['np2'][0, 0]
            anova_mnar_pval = anova_mnar_res['p_unadjusted'][0, 0]
            kruskal_mnar_eff = kruskal_mnar_res['eta2'][0, 0]
            kruskal_mnar_pval = kruskal_mnar_res['p_unadjusted'][0, 0]

            result_dict = save_results(x_data=num_pair,
                         y_data=num_pair,
                         na_ratio=na_ratio,
                         test='anova',
                         mechanism='MCAR',
                         pvalue=anova_mcar_pval,
                         effect_size=anova_mcar_eff,
                         results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='anova',
                                       mechanism='MAR',
                                       pvalue=anova_mar_pval,
                                       effect_size=anova_mar_eff,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='anova',
                                       mechanism='MNAR',
                                       pvalue=anova_mnar_pval,
                                       effect_size=anova_mnar_eff,
                                       results_dict=result_dict)

            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='kruskal',
                                       mechanism='MCAR',
                                       pvalue=kruskal_mcar_pval,
                                       effect_size=kruskal_mcar_eff,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='kruskal',
                                       mechanism='MAR',
                                       pvalue=kruskal_mar_pval,
                                       effect_size=kruskal_mar_eff,
                                       results_dict=result_dict)
            result_dict = save_results(x_data=num_pair,
                                       y_data=num_pair,
                                       na_ratio=na_ratio,
                                       test='kruskal',
                                       mechanism='MNAR',
                                       pvalue=kruskal_mnar_pval,
                                       effect_size=kruskal_mnar_eff,
                                       results_dict=result_dict)

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("stability_analysis_cat_cont.csv", index=False)
    df = result_df[result_df['test']=='kruskal']

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

