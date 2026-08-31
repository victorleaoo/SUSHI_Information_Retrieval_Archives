#!/usr/bin/env python3
"""
Four-way ANOVA for the SUSHI training-set experiments (EVIA '26).

Design, following Ferro & Sanderson (SIGIR '19) with training sets in the role of shards:

    topic (tau)  x  training set (gamma)  x  query type (q)  x  search fields (phi)

Model M7:  y = mu + tau + gamma + q + phi
                + tau*gamma + tau*q + tau*phi + gamma*q + gamma*phi + q*phi
                + error            (the four-way interaction IS the error term)

Sums of squares are computed in closed form from marginal means. That is exact for a
balanced design with one observation per cell, and avoids building a 24,300 x 1,878
design matrix. If your design is NOT balanced, use statsmodels with Type III SS instead
(see --check-statsmodels and the guide, section 5.3).

Usage
-----
    python sushi_anova.py --demo
    python sushi_anova.py --input scores.csv --outdir results/
    python sushi_anova.py --input scores.csv --outdir results_nofolder/ --drop-fields "--F-"

Input CSV schema (24,300 rows, no missing cells):
    topic,training_set,query,fields,ndcg
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import warnings

import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning
from scipy.stats import f as f_dist
from scipy.stats import studentized_range

# studentized_range integrates numerically and warns harmlessly at very large df_error.
warnings.filterwarnings("ignore", category=IntegrationWarning)

ALPHA = 0.05
FACTORS = ["topic", "training_set", "query", "fields"]
SYMBOLS = {"topic": "tau", "training_set": "gamma", "query": "q", "fields": "phi"}

# The fourth factor is always carried in the "fields" column, but it need not BE search
# fields (see build_grid.py). These are overwritten by --factor4-label / --factor4-symbol.
F4_LABEL = "Search fields"
F4_SYMBOL = "phi"

# Preferred display order for levels (anything not listed is appended, sorted).
LEVEL_ORDER = {
    "query": ["T--", "TD-", "TDN"],
    "fields": ["T---", "-O--", "--F-", "---S", "TOFS"],
}


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_long(path: str, drop_fields=None, drop_queries=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FACTORS + ["ndcg"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if drop_fields:
        df = df[~df["fields"].isin(drop_fields)]
    if drop_queries:
        df = df[~df["query"].isin(drop_queries)]
    for c in FACTORS:
        df[c] = df[c].astype(str)
    if df["ndcg"].isna().any():
        raise ValueError("ndcg contains NaN; every cell must have a score (use 0.0, not blank)")
    return df.reset_index(drop=True)


def to_array(df: pd.DataFrame, value: str = "ndcg"):
    """Reshape the long frame into a dense 4-D array [topic, training_set, query, fields]."""
    levels = {}
    for fac in FACTORS:
        obs = set(df[fac].unique())
        if fac in LEVEL_ORDER:
            ordered = [v for v in LEVEL_ORDER[fac] if v in obs]
            ordered += sorted(obs - set(ordered))
        else:
            ordered = sorted(obs, key=_natural_key)
        levels[fac] = ordered

    idx = {f: {v: i for i, v in enumerate(levels[f])} for f in FACTORS}
    shape = tuple(len(levels[f]) for f in FACTORS)

    expected = int(np.prod(shape))
    if len(df) != expected:
        raise ValueError(
            f"design is not complete: got {len(df)} rows, expected {expected} "
            f"({' x '.join(f'{f}={n}' for f, n in zip(FACTORS, shape))}). "
            "Use statsmodels with Type III SS for an unbalanced design."
        )

    Y = np.full(shape, np.nan)
    pos = tuple(df[f].map(idx[f]).to_numpy() for f in FACTORS)
    Y[pos] = df[value].to_numpy()
    if np.isnan(Y).any():
        n_missing = int(np.isnan(Y).sum())
        raise ValueError(f"{n_missing} empty cells (duplicate rows?); design must be balanced")
    return Y, levels


def _natural_key(s: str):
    return (len(s), s) if not s.lstrip("-").isdigit() else (0, f"{int(s):012d}")


# --------------------------------------------------------------------------------------
# Core ANOVA engine (balanced, one observation per cell)
# --------------------------------------------------------------------------------------

def _marginal(Y: np.ndarray, keep: tuple) -> np.ndarray:
    """Mean of Y over all axes NOT in `keep`, with keepdims=True."""
    drop = tuple(a for a in range(Y.ndim) if a not in keep)
    return Y.mean(axis=drop, keepdims=True) if drop else Y


def _effect(Y: np.ndarray, term: tuple) -> np.ndarray:
    """Inclusion-exclusion estimate of the effect of `term` (a tuple of axis indices).

    e.g. term=(0,)    ->  m_0 - m
         term=(0, 1)  ->  m_01 - m_0 - m_1 + m
    """
    out = 0.0
    k = len(term)
    for r in range(k + 1):
        sign = (-1) ** (k - r)
        for sub in itertools.combinations(term, r):
            out = out + sign * _marginal(Y, sub)
    return out


def ss_of(Y: np.ndarray, term: tuple) -> float:
    """Sum of squares for a term, summed over all N cells (broadcasting the effect)."""
    e = _effect(Y, term)
    return float(np.sum(np.broadcast_to(e, Y.shape) ** 2))


def df_of(shape, term: tuple) -> int:
    d = 1
    for a in term:
        d *= shape[a] - 1
    return d


def omega_sq(df_fact: int, F: float, N: int) -> float:
    """Unbiased effect size, Olejnik & Algina; Ferro & Sanderson eq. (6). Negatives -> 0."""
    num = df_fact * (F - 1.0)
    val = num / (num + N)
    return max(val, 0.0)


def anova(Y: np.ndarray, terms, names=None) -> pd.DataFrame:
    """Fit a balanced ANOVA with the given list of terms (tuples of axis indices)."""
    N = Y.size
    shape = Y.shape
    grand = Y.mean()
    ss_total = float(np.sum((Y - grand) ** 2))

    rows = []
    ss_model, df_model = 0.0, 0
    for t in terms:
        ss = ss_of(Y, t)
        df = df_of(shape, t)
        ss_model += ss
        df_model += df
        rows.append({"term": t, "SS": ss, "DF": df})

    df_err = N - 1 - df_model
    ss_err = ss_total - ss_model
    if df_err <= 0:
        raise ValueError("no residual degrees of freedom left; remove a term")
    ms_err = ss_err / df_err

    for r in rows:
        r["MS"] = r["SS"] / r["DF"]
        r["F"] = r["MS"] / ms_err
        r["p"] = float(f_dist.sf(r["F"], r["DF"], df_err))
        r["omega2"] = omega_sq(r["DF"], r["F"], N)

    out = pd.DataFrame(rows)
    if names is not None:
        out.insert(0, "Source", [_term_name(t, names) for t in out["term"]])
    out = out.drop(columns=["term"])

    out.loc[len(out)] = _blank_row(out, "Error", ss_err, df_err, ms_err)
    out.loc[len(out)] = _blank_row(out, "Total", ss_total, N - 1, np.nan)

    out.attrs["ms_error"] = ms_err
    out.attrs["df_error"] = df_err
    out.attrs["N"] = N
    out.attrs["ss_total"] = ss_total
    return out


def _blank_row(df, label, ss, dfree, ms):
    row = {c: np.nan for c in df.columns}
    if "Source" in df.columns:
        row["Source"] = label
    row["SS"], row["DF"], row["MS"] = ss, dfree, ms
    return row


def _term_name(term, names):
    return " x ".join(names[a] for a in term)


def residuals(Y: np.ndarray, terms):
    fitted = np.broadcast_to(Y.mean(), Y.shape).astype(float).copy()
    for t in terms:
        fitted = fitted + np.broadcast_to(_effect(Y, t), Y.shape)
    return Y - fitted, fitted


# --------------------------------------------------------------------------------------
# Tukey HSD
# --------------------------------------------------------------------------------------

def tukey(cell_means: np.ndarray, labels, n_per_mean: int, ms_err: float,
          df_err: int, alpha: float = ALPHA):
    """Tukey HSD over the levels of one factor.

    Returns (summary DataFrame with marginal means and half-width CIs,
             pairwise DataFrame).
    Ferro & Sanderson eq. (1) for the test, eq. (3) for the interval.
    """
    k = len(labels)
    se = np.sqrt(ms_err / n_per_mean)
    q_crit = float(studentized_range.ppf(1 - alpha, k, df_err))
    half = 0.5 * q_crit * se

    summary = pd.DataFrame({
        "level": labels,
        "marginal_mean": cell_means,
        "ci_low": cell_means - half,
        "ci_high": cell_means + half,
        "ci_halfwidth": half,
    }).sort_values("marginal_mean", ascending=False).reset_index(drop=True)

    pairs = []
    for a, b in itertools.combinations(range(k), 2):
        diff = cell_means[a] - cell_means[b]
        stat = abs(diff) / se
        p = float(studentized_range.sf(stat, k, df_err))
        pairs.append({
            "level_a": labels[a], "level_b": labels[b],
            "diff": diff, "q_stat": stat, "p": p,
            "significant": p < alpha,
        })
    pairs = pd.DataFrame(pairs).sort_values("p").reset_index(drop=True)

    summary.attrs["q_crit"] = q_crit
    summary.attrs["se"] = se
    summary.attrs["n_significant"] = int(pairs["significant"].sum())
    summary.attrs["n_pairs"] = len(pairs)
    return summary, pairs


# --------------------------------------------------------------------------------------
# Model ladder (Ferro & Sanderson MD1..MD6 analogues, plus our M7)
# --------------------------------------------------------------------------------------

LADDER = [
    ("L1", "tau + alpha (single training set)",        None),
    ("L2", "tau + alpha",                              [(0,), (2,)]),
    ("L3", "tau + alpha + tau*alpha",                  [(0,), (2,), (0, 2)]),
    ("L4", "tau + alpha + gamma + tau*alpha",          [(0,), (1,), (2,), (0, 2)]),
    ("L5", "L4 + alpha*gamma",                         [(0,), (1,), (2,), (0, 2), (1, 2)]),
    ("L6", "L5 + tau*gamma  (= F&S MD6)",              [(0,), (1,), (2,), (0, 2), (1, 2), (0, 1)]),
]


def model_ladder(Y4: np.ndarray, levels, alpha: float = ALPHA,
                 single_training_index: int = 0) -> pd.DataFrame:
    """Build the F&S-style ladder. `alpha` here is the significance level, not the
    system factor; the system factor is the flattened (query x fields) dimension."""
    T, G, Q, F = Y4.shape
    R = Q * F                                    # number of "systems"
    Y3 = Y4.reshape(T, G, R)                     # topic x training set x system
    sys_labels = [f"{q}.{f}" for q in levels["query"] for f in levels["fields"]]
    n_pairs = R * (R - 1) // 2

    rows = []

    for name, desc, terms in LADDER:
        if terms is None:
            # L1: one training set only, one score per (topic, system).
            Y1 = Y3[:, single_training_index, :]
            tbl = anova(Y1, [(0,), (1,)], names=["tau", "alpha"])
            ms, dfe = tbl.attrs["ms_error"], tbl.attrs["df_error"]
            n_per_mean = T                       # F&S note: S = 1 so denominator is T
            om = float(tbl.loc[tbl["Source"] == "alpha", "omega2"].iloc[0])
            means = Y1.mean(axis=0)
        else:
            tbl = anova(Y3, terms, names=["tau", "gamma", "alpha"])
            ms, dfe = tbl.attrs["ms_error"], tbl.attrs["df_error"]
            n_per_mean = T * G
            om = float(tbl.loc[tbl["Source"] == "alpha", "omega2"].iloc[0])
            means = Y3.mean(axis=(0, 1))

        summ, prs = tukey(means, sys_labels, n_per_mean, ms, dfe, alpha)
        rows.append({
            "model": name, "terms": desc,
            "df_error": dfe, "MS_error": ms,
            "omega2_system": om,
            "tukey_halfwidth": float(summ.attrs["se"] * 0.5 * summ.attrs["q_crit"]),
            "sig_pairs": int(prs["significant"].sum()),
            "total_pairs": n_pairs,
        })

    # L7: the full four-way factorial with all two-way interactions.
    terms7 = TERMS_M7
    tbl7 = anova(Y4, terms7, names=["tau", "gamma", "q", F4_SYMBOL])
    ms, dfe = tbl7.attrs["ms_error"], tbl7.attrs["df_error"]
    means = Y4.mean(axis=(0, 1)).reshape(-1)
    summ, prs = tukey(means, sys_labels, T * G, ms, dfe, alpha)
    om_q = float(tbl7.loc[tbl7["Source"] == "q", "omega2"].iloc[0])
    om_f = float(tbl7.loc[tbl7["Source"] == F4_SYMBOL, "omega2"].iloc[0])
    rows.append({
        "model": "L7", "terms": "full 4-way, all two-way interactions",
        "df_error": dfe, "MS_error": ms,
        "omega2_system": np.nan,
        "omega2_query": om_q, "omega2_fields": om_f,
        "tukey_halfwidth": float(summ.attrs["se"] * 0.5 * summ.attrs["q_crit"]),
        "sig_pairs": int(prs["significant"].sum()),
        "total_pairs": n_pairs,
    })

    out = pd.DataFrame(rows)
    base = out.loc[out["model"] == "L1"].iloc[0]
    out["MS_error_pct_vs_L1"] = 100.0 * (out["MS_error"] - base["MS_error"]) / base["MS_error"]
    out["sig_pairs_pct_vs_L1"] = 100.0 * (out["sig_pairs"] - base["sig_pairs"]) / max(base["sig_pairs"], 1)
    return out


TERMS_M7 = [(0,), (1,), (2,), (3,),
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

TERMS_M8 = TERMS_M7 + [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------

def to_latex_anova(tbl: pd.DataFrame, caption: str, label: str) -> str:
    def fmt(v, nd=4):
        if isinstance(v, float) and np.isnan(v):
            return ""
        return f"{v:.{nd}f}"

    def fmt_p(v):
        if isinstance(v, float) and np.isnan(v):
            return ""
        return "$<10^{-4}$" if v < 1e-4 else f"{v:.4f}"

    lines = [
        r"\begin{table}", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Source & SS & DF & MS & $F$ & $p$ & $\hat{\omega}^2$ \\", r"\midrule",
    ]
    for _, r in tbl.iterrows():
        if r["Source"] in ("Error", "Total"):
            lines.append(r"\midrule")
        lines.append(
            f"{_tex_escape(r['Source'])} & {fmt(r['SS'])} & {int(r['DF'])} & "
            f"{fmt(r['MS'], 6)} & {fmt(r['F'], 2)} & {fmt_p(r['p'])} & {fmt(r['omega2'], 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


_GREEK = {"tau": r"$\tau$", "gamma": r"$\gamma$", "q": r"$q$",
          "phi": r"$\varphi$", "alpha": r"$\alpha$", "sigma": r"$\sigma$",
          "psi": r"$\psi$", "delta": r"$\delta$"}


def _tex_escape(s: str) -> str:
    parts = [p.strip() for p in s.split(" x ")]
    parts = [_GREEK.get(p, p.replace("_", r"\_")) for p in parts]
    return r" $\times$ ".join(parts)


def save(df: pd.DataFrame, outdir: str, name: str):
    path = os.path.join(outdir, name)
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def diagnostics_plot(resid: np.ndarray, fitted: np.ndarray, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("  (matplotlib unavailable, skipping diagnostics plot)")
        return
    r = resid.ravel()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    stats.probplot(r, dist="norm", plot=ax[0])
    ax[0].set_title("Normal Q-Q of residuals")
    idx = np.random.default_rng(0).choice(r.size, size=min(5000, r.size), replace=False)
    ax[1].scatter(fitted.ravel()[idx], r[idx], s=3, alpha=0.25)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("fitted"); ax[1].set_ylabel("residual")
    ax[1].set_title("Residuals vs fitted (5k sample)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


def tukey_plot(summaries: dict, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (title, s) in zip(axes, summaries.items()):
        y = np.arange(len(s))[::-1]
        ax.errorbar(s["marginal_mean"], y,
                    xerr=s["ci_halfwidth"], fmt="o", capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(s["level"])
        ax.set_xlabel("nDCG@5 marginal mean")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, outdir: str, alpha: float = ALPHA,
                 do_ladder: bool = True, do_threeway: bool = True):
    os.makedirs(outdir, exist_ok=True)
    Y, levels = to_array(df)
    T, G, Q, F = Y.shape
    names = ["tau", "gamma", "q", F4_SYMBOL]
    print(f"\nDesign: {T} topics x {G} training sets x {Q} query types x "
          f"{F} {F4_LABEL} levels = {Y.size} observations")
    print(f"Levels: query={levels['query']}  {F4_LABEL}={levels['fields']}")

    # --- primary model M7 ---------------------------------------------------------
    tbl = anova(Y, TERMS_M7, names=names)
    ms_err, df_err = tbl.attrs["ms_error"], tbl.attrs["df_error"]
    print(f"\n=== M7: main effects + all two-way interactions "
          f"(MS_error={ms_err:.6f}, df_error={df_err}) ===")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    save(tbl, outdir, "anova_table.csv")
    with open(os.path.join(outdir, "anova_table.tex"), "w") as fh:
        fh.write(to_latex_anova(
            tbl,
            "Four-way ANOVA on nDCG@5. Effect sizes $\\hat{\\omega}^2$ follow "
            "Olejnik and Algina; $\\ge 0.14$ is a large effect.",
            "tab:anova"))
    print(f"  wrote {os.path.join(outdir, 'anova_table.tex')}")

    # --- three-way robustness -----------------------------------------------------
    if do_threeway:
        tbl8 = anova(Y, TERMS_M8, names=names)
        save(tbl8, outdir, "anova_table_threeway.csv")
        comp = tbl.merge(tbl8[["Source", "omega2"]], on="Source",
                         suffixes=("_M7", "_M8"), how="inner")
        comp["abs_delta"] = (comp["omega2_M7"] - comp["omega2_M8"]).abs()
        print(f"\nM8 (with three-way terms): MS_error={tbl8.attrs['ms_error']:.6f}, "
              f"df_error={tbl8.attrs['df_error']}. "
              f"Max |delta omega^2| vs M7 = {comp['abs_delta'].max():.4f}")
        save(comp, outdir, "omega2_M7_vs_M8.csv")

    # --- Tukey on the interpretable factors ---------------------------------------
    summaries = {}
    for axis, fac, label in [(2, "query", "Query type"), (3, "fields", F4_LABEL)]:
        others = tuple(a for a in range(4) if a != axis)
        means = Y.mean(axis=others)
        n_per_mean = Y.size // Y.shape[axis]
        summ, prs = tukey(means, levels[fac], n_per_mean, ms_err, df_err, alpha)
        print(f"\n=== Tukey HSD: {label} "
              f"(n per mean = {n_per_mean}, q_crit = {summ.attrs['q_crit']:.3f}, "
              f"half-width = {summ['ci_halfwidth'].iloc[0]:.4f}) ===")
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"  {summ.attrs['n_significant']}/{summ.attrs['n_pairs']} pairs significant "
              f"at alpha={alpha}")
        save(summ, outdir, f"tukey_{fac}.csv")
        save(prs, outdir, f"tukey_{fac}_pairs.csv")
        summaries[label] = summ

    tukey_plot(summaries, os.path.join(outdir, "tukey_marginal_means.png"))

    # --- all 18 system combinations -----------------------------------------------
    sys_labels = [f"{q}.{f}" for q in levels["query"] for f in levels["fields"]]
    sys_means = Y.mean(axis=(0, 1)).reshape(-1)
    summ_s, prs_s = tukey(sys_means, sys_labels, T * G, ms_err, df_err, alpha)
    print(f"\n=== Tukey HSD across all {len(sys_labels)} (query, fields) systems: "
          f"{summ_s.attrs['n_significant']}/{summ_s.attrs['n_pairs']} pairs significant ===")
    save(summ_s, outdir, "tukey_systems.csv")
    save(prs_s, outdir, "tukey_systems_pairs.csv")

    cell = pd.DataFrame(Y.mean(axis=(0, 1)), index=levels["query"], columns=levels["fields"])
    cell.index.name = "query"
    print("\nCell means (query x fields):")
    print(cell.to_string(float_format=lambda v: f"{v:.4f}"))
    cell.to_csv(os.path.join(outdir, "interaction_query_fields.csv"))

    # --- diagnostics --------------------------------------------------------------
    resid, fitted = residuals(Y, TERMS_M7)
    diagnostics_plot(resid, fitted, os.path.join(outdir, "diagnostics.png"))

    # --- arcsine-transform robustness ---------------------------------------------
    Yt = np.arcsin(np.sqrt(np.clip(Y, 0.0, 1.0)))
    tbl_t = anova(Yt, TERMS_M7, names=names)
    save(tbl_t, outdir, "anova_table_arcsine.csv")
    cmp_t = tbl[["Source", "omega2"]].merge(
        tbl_t[["Source", "omega2"]], on="Source", suffixes=("_raw", "_arcsine"))
    print("\nArcsine-transform robustness check (omega^2):")
    print(cmp_t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    save(cmp_t, outdir, "omega2_raw_vs_arcsine.csv")

    # --- model ladder --------------------------------------------------------------
    if do_ladder:
        lad = model_ladder(Y, levels, alpha)
        print("\n=== Model ladder (Ferro & Sanderson MD1..MD6 analogues + our L7) ===")
        print(lad.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        save(lad, outdir, "ladder.csv")

    return tbl


# --------------------------------------------------------------------------------------
# Demo / self-test
# --------------------------------------------------------------------------------------

def make_demo(seed: int = 0) -> pd.DataFrame:
    """Synthetic data with known effects, to validate the pipeline before real runs exist.

    Ground truth built in:
      - large topic effect
      - small training-set main effect
      - LARGE topic x training-set interaction  (the 'unlucky sample' effect)
      - large field effect, small query effect
    """
    rng = np.random.default_rng(seed)
    topics = [f"T{i:03d}" for i in range(45)]
    tsets = [str(i) for i in range(1, 31)]
    queries = LEVEL_ORDER["query"]
    fields = LEVEL_ORDER["fields"]

    tau = rng.normal(0, 0.10, len(topics))
    gam = rng.normal(0, 0.01, len(tsets))
    q_eff = np.array([0.010, -0.005, -0.005])
    f_eff = np.array([-0.03, -0.005, -0.10, -0.01, 0.02, 0.03])
    tg = rng.normal(0, 0.06, (len(topics), len(tsets)))     # big interaction
    tf = rng.normal(0, 0.02, (len(topics), len(fields)))

    rows = []
    for i, t in enumerate(topics):
        for j, g in enumerate(tsets):
            for k, q in enumerate(queries):
                for l, f in enumerate(fields):
                    y = (0.21 + tau[i] + gam[j] + q_eff[k] + f_eff[l]
                         + tg[i, j] + tf[i, l] + rng.normal(0, 0.06))
                    rows.append((t, g, q, f, float(np.clip(y, 0.0, 1.0))))
    return pd.DataFrame(rows, columns=FACTORS + ["ndcg"])


def check_statsmodels(df: pd.DataFrame, n_topics: int = 6, n_tsets: int = 5):
    """Verify the closed-form SS against statsmodels OLS on a small subsample."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed; skipping cross-check")
        return
    tk = sorted(df["topic"].unique())[:n_topics]
    gk = sorted(df["training_set"].unique())[:n_tsets]
    sub = df[df["topic"].isin(tk) & df["training_set"].isin(gk)].copy()
    Y, levels = to_array(sub)
    mine = anova(Y, TERMS_M7, names=["tau", "gamma", "q", F4_SYMBOL])

    m = smf.ols(
        "ndcg ~ C(topic)*C(training_set) + C(topic)*C(query) + C(topic)*C(fields)"
        " + C(training_set)*C(query) + C(training_set)*C(fields) + C(query)*C(fields)",
        data=sub).fit()
    sm_tbl = sm.stats.anova_lm(m, typ=2)
    mine_ss = mine.set_index("Source")["SS"]
    print("\nCross-check against statsmodels (Type II SS) on a subsample:")
    print(f"  closed-form MS_error = {mine.attrs['ms_error']:.8f}")
    print(f"  statsmodels MS_error = {sm_tbl.loc['Residual', 'sum_sq'] / m.df_resid:.8f}")
    print(f"  closed-form SS(topic) = {mine_ss['tau']:.8f}")
    print(f"  statsmodels SS(topic) = {sm_tbl.loc['C(topic)', 'sum_sq']:.8f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="long-format CSV with per-topic scores")
    ap.add_argument("--outdir", default="anova_results")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--drop-fields", nargs="*", default=None,
                    help="field levels to exclude, e.g. --drop-fields '--F-'")
    ap.add_argument("--drop-queries", nargs="*", default=None)
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--no-threeway", action="store_true")
    ap.add_argument("--check-statsmodels", action="store_true")
    ap.add_argument("--factor4-label", default="Search fields",
                    help="display name of the fourth factor (e.g. 'Training doc scoring')")
    ap.add_argument("--factor4-symbol", default="phi",
                    help="short symbol for the fourth factor in tables (e.g. 'sigma')")
    args = ap.parse_args()

    global F4_LABEL, F4_SYMBOL
    F4_LABEL, F4_SYMBOL = args.factor4_label, args.factor4_symbol

    if args.demo:
        df = make_demo()
        print("Running on SYNTHETIC demo data (known ground truth: large topic effect, "
              "large topic x training-set interaction, large field effect, small query effect).")
        if args.drop_fields:
            df = df[~df["fields"].isin(args.drop_fields)]
    elif args.input:
        df = load_long(args.input, args.drop_fields, args.drop_queries)
    else:
        ap.error("give --input or --demo")

    run_analysis(df, args.outdir, args.alpha,
                 do_ladder=not args.no_ladder, do_threeway=not args.no_threeway)

    if args.check_statsmodels:
        check_statsmodels(df)


if __name__ == "__main__":
    sys.exit(main())
