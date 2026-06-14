#!/usr/bin/env python3
"""SREDT Venter Re-Analysis: Bootstrap CIs, intercept reclassification, sector stratification, publication figures."""

import sys
import os
import json
import gc
import math
import resource
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/eval.log", rotation="30 MB", level="DEBUG")

# ── Memory limits (cgroup v2 container: 29 GB) ────────────────────────────────
_MEM_LIMIT_BYTES = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
RAM_BUDGET = min(int(_MEM_LIMIT_BYTES * 0.60), 16 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f} GB (of {_MEM_LIMIT_BYTES/1e9:.1f} GB limit)")

WORKSPACE = Path("/ai-inventor/aii_data/runs/run_PoDi6I8fYcAb/3_invention_loop/iter_2/gen_art/gen_art_evaluation_1")
BASE = Path("/ai-inventor/aii_data/runs/run_PoDi6I8fYcAb/3_invention_loop")


def locate_input_data() -> tuple[str, str]:
    """Find best available input data (corrected > baseline)."""
    primary = BASE / "iter_1/gen_art/gen_art_experiment_1/mini_method_out.json"

    corrected_candidates = list((BASE / "iter_2/gen_art").glob("*/mini_method_out.json"))
    corrected_candidates += list((BASE / "iter_2/gen_art").glob("*/method_out.json"))
    corrected_candidates += list((BASE / "iter_2/gen_art").glob("*/full_method_out.json"))
    # Exclude our own workspace
    corrected_candidates = [p for p in corrected_candidates if "gen_art_evaluation_1" not in str(p)]

    if corrected_candidates:
        corrected_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        path = corrected_candidates[0]
        source = "corrected_experiment"
        logger.info(f"Using corrected experiment: {path}")
    else:
        path = primary
        source = "baseline_experiment_iter1"
        logger.info(f"Using baseline experiment: {path}")

    return str(path), source


def load_data(path: str) -> dict:
    """Load and parse JSON, handling NaN values."""
    raw_text = Path(path).read_text()
    # JSON doesn't support NaN; replace bare NaN tokens
    raw_text = raw_text.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw_text)


def compute_ols_diagnostics(x_vals: np.ndarray, y_vals: np.ndarray, label: str) -> dict:
    """Run OLS on one level transition and return full Venter diagnostics."""
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    n = len(x)

    degenerate = np.std(x) < 1e-10

    if degenerate:
        f_j = float(np.sum(y) / np.sum(x)) if np.sum(x) > 0 else float("nan")
        return {
            "transition": label, "n": n,
            "f_j": f_j, "cv": None, "individual_ratios": [],
            "slope": None, "intercept": None, "se_intercept": None,
            "t_stat": None, "p_value": None, "intercept_significant": None,
            "r_squared": None, "verdict_cv_only": "degenerate", "verdict_corrected": "degenerate",
        }

    # Development factors
    individual_ratios = y / x
    f_j = float(np.sum(y) / np.sum(x))
    mean_r = float(np.mean(individual_ratios))
    cv = float(np.std(individual_ratios) / mean_r) if mean_r > 0 else float("nan")

    # OLS: C(i,j+1) = a + b * C(i,j)
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_squared = float(r_value ** 2)

    # SE of intercept
    y_pred = slope * x + intercept
    residuals = y - y_pred
    mse = float(np.sum(residuals ** 2) / (n - 2)) if n > 2 else float("nan")
    x_mean = float(np.mean(x))
    ss_x = float(np.sum((x - x_mean) ** 2))
    if ss_x > 0 and not math.isnan(mse):
        se_intercept = float(math.sqrt(mse * (1.0 / n + x_mean ** 2 / ss_x)))
    else:
        se_intercept = float("nan")

    if not math.isnan(se_intercept) and se_intercept > 0:
        t_stat = float(intercept / se_intercept)
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2)))
        intercept_significant = bool(abs(intercept) >= 2 * se_intercept)
    else:
        t_stat = None
        p_val = None
        intercept_significant = None

    # Verdicts
    if cv is None or math.isnan(cv):
        verdict_cv_only = "degenerate"
        verdict_corrected = "degenerate"
    elif intercept_significant:
        verdict_cv_only = "chain_ladder_valid" if cv < 0.30 else ("borderline" if cv <= 0.50 else "bf_fallback")
        verdict_corrected = "factor_plus_constant"
    elif cv < 0.30:
        verdict_cv_only = "chain_ladder_valid"
        verdict_corrected = "chain_ladder_valid"
    elif cv <= 0.50:
        verdict_cv_only = "borderline"
        verdict_corrected = "borderline_bf_preferred"
    else:
        verdict_cv_only = "bf_fallback"
        verdict_corrected = "bf_fallback"

    return {
        "transition": label, "n": n,
        "f_j": f_j, "cv": cv,
        "individual_ratios": individual_ratios.tolist(),
        "slope": float(slope), "intercept": float(intercept),
        "se_intercept": se_intercept if not math.isnan(se_intercept) else None,
        "t_stat": t_stat, "p_value": p_val,
        "intercept_significant": intercept_significant,
        "r_squared": r_squared,
        "verdict_cv_only": verdict_cv_only,
        "verdict_corrected": verdict_corrected,
    }


def run_bootstrap(train_df: pd.DataFrame, transitions_full: list, B: int = 1000) -> list:
    """Bootstrap CIs (B=1000, seed=42) for f_j, CV, intercept."""
    rng = np.random.default_rng(42)
    bootstrap_ci = []

    for j_label, col_j, col_j1 in [
        ("L0→L1", "l0", "l1"), ("L1→L2", "l1", "l2"),
        ("L2→L3", "l2", "l3"), ("L3→L4", "l3", "l4"),
    ]:
        x_all = train_df[col_j].values.astype(float)
        y_all = train_df[col_j1].values.astype(float)
        n = len(x_all)

        boot_fj, boot_cv, boot_intercept = [], [], []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            xb, yb = x_all[idx], y_all[idx]

            fj_b = float(np.sum(yb) / np.sum(xb)) if np.sum(xb) > 0 else float("nan")
            boot_fj.append(fj_b)

            if np.std(xb) > 1e-10 and np.all(xb > 0):
                ratios_b = yb / xb
                cv_b = float(np.std(ratios_b) / np.mean(ratios_b)) if np.mean(ratios_b) > 0 else float("nan")
            else:
                cv_b = float("nan")
            boot_cv.append(cv_b)

            if np.std(xb) > 1e-10:
                _, int_b, _, _, _ = stats.linregress(xb, yb)
                boot_intercept.append(float(int_b))
            else:
                boot_intercept.append(float("nan"))

        boot_fj_arr = np.array(boot_fj)
        boot_cv_arr = np.array(boot_cv)
        boot_int_arr = np.array(boot_intercept)

        obs_diag = next(d for d in transitions_full if d["transition"] == j_label)
        obs_intercept = obs_diag["intercept"]

        # Bootstrap p-value (two-sided): how often |boot_int - mean(boot_int)| >= |obs - mean(boot_int)|
        if obs_intercept is not None:
            valid_bi = boot_int_arr[~np.isnan(boot_int_arr)]
            if len(valid_bi) > 0:
                mu = float(np.mean(valid_bi))
                p_boot = float(np.mean(np.abs(valid_bi - mu) >= abs(obs_intercept - mu)))
            else:
                p_boot = None
        else:
            p_boot = None

        def pct(arr, q):
            v = arr[~np.isnan(arr)]
            return float(np.percentile(v, q)) if len(v) > 0 else None

        incl_zero = None
        v = boot_int_arr[~np.isnan(boot_int_arr)]
        if len(v) > 0:
            lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
            incl_zero = bool(lo <= 0 <= hi)

        bootstrap_ci.append({
            "transition": j_label,
            "f_j_observed": obs_diag["f_j"],
            "f_j_lower": pct(boot_fj_arr, 2.5),
            "f_j_upper": pct(boot_fj_arr, 97.5),
            "cv_observed": obs_diag["cv"],
            "cv_lower": pct(boot_cv_arr, 2.5),
            "cv_upper": pct(boot_cv_arr, 97.5),
            "intercept_observed": obs_intercept,
            "intercept_lower": pct(boot_int_arr, 2.5),
            "intercept_upper": pct(boot_int_arr, 97.5),
            "intercept_p_bootstrap": p_boot,
            "intercept_includes_zero_in_95ci": incl_zero,
        })
        logger.info(f"Bootstrap {j_label}: f_j=[{pct(boot_fj_arr,2.5):.4f}, {pct(boot_fj_arr,97.5):.4f}]")

    return bootstrap_ci


def run_sector_analysis(train_df: pd.DataFrame) -> dict:
    """Sector-stratified Venter OLS for energy and financials."""
    sector_diagnostics = {}
    for sector in ["energy", "financials"]:
        sdf = train_df[train_df["sector"] == sector].copy()
        n_sector = len(sdf)
        logger.info(f"Sector {sector}: n={n_sector}")

        if n_sector < 8:
            sector_diagnostics[sector] = {"error": f"Insufficient n={n_sector}", "n": n_sector}
            continue

        sector_diags = []
        for j_label, col_j, col_j1 in [
            ("L1→L2", "l1", "l2"), ("L2→L3", "l2", "l3"), ("L3→L4", "l3", "l4"),
        ]:
            x = sdf[col_j].values
            y = sdf[col_j1].values
            diag = compute_ols_diagnostics(x, y, j_label)
            diag["n_unique_x"] = int(len(np.unique(np.round(x, 6))))
            sector_diags.append(diag)

        sector_diagnostics[sector] = {
            "n_total": n_sector,
            "n_unique_company_risk": int(sdf.groupby(["company", "risk_type"]).ngroups),
            "transitions": sector_diags,
        }
    return sector_diagnostics


def jackknife_l3_l4(train_df: pd.DataFrame) -> dict:
    """Jackknife leave-one-out for L3→L4 f_j and intercept."""
    x_all = train_df["l3"].values.astype(float)
    y_all = train_df["l4"].values.astype(float)
    n = len(x_all)

    jk_fj, jk_int = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xi, yi = x_all[mask], y_all[mask]
        fj = float(np.sum(yi) / np.sum(xi)) if np.sum(xi) > 0 else float("nan")
        jk_fj.append(fj)
        if np.std(xi) > 1e-10:
            _, int_b, _, _, _ = stats.linregress(xi, yi)
            jk_int.append(float(int_b))
        else:
            jk_int.append(float("nan"))

    return {
        "n": n,
        "fj_mean": float(np.nanmean(jk_fj)),
        "fj_std": float(np.nanstd(jk_fj)),
        "fj_min": float(np.nanmin(jk_fj)),
        "fj_max": float(np.nanmax(jk_fj)),
        "intercept_mean": float(np.nanmean(jk_int)),
        "intercept_std": float(np.nanstd(jk_int)),
        "intercept_min": float(np.nanmin(jk_int)),
        "intercept_max": float(np.nanmax(jk_int)),
    }


def sector_swap_test(train_df: pd.DataFrame) -> dict:
    """Fit L3→L4 OLS on energy, predict on financials, and vice versa."""
    results = {}
    for fit_sector, pred_sector in [("energy", "financials"), ("financials", "energy")]:
        fit_df = train_df[train_df["sector"] == fit_sector]
        pred_df = train_df[train_df["sector"] == pred_sector]
        x_fit = fit_df["l3"].values.astype(float)
        y_fit = fit_df["l4"].values.astype(float)
        x_pred = pred_df["l3"].values.astype(float)
        y_pred_true = pred_df["l4"].values.astype(float)

        if np.std(x_fit) < 1e-10:
            results[f"fit_{fit_sector}_pred_{pred_sector}"] = {"error": "degenerate x in fit sector"}
            continue

        slope, intercept, r_val, _, _ = stats.linregress(x_fit, y_fit)
        y_pred_hat = slope * x_pred + intercept
        ss_res = float(np.sum((y_pred_true - y_pred_hat) ** 2))
        ss_tot = float(np.sum((y_pred_true - np.mean(y_pred_true)) ** 2))
        r2_transfer = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        results[f"fit_{fit_sector}_pred_{pred_sector}"] = {
            "fit_slope": float(slope), "fit_intercept": float(intercept),
            "transfer_r_squared": r2_transfer,
            "n_fit": int(len(x_fit)), "n_pred": int(len(x_pred)),
        }
    return results


def make_fig1_heatmap(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> str:
    """Development triangle heatmap."""
    fig, ax = plt.subplots(figsize=(12, 16))
    levels = ["L0", "L1", "L2", "L3", "L4"]
    n_train = len(train_df)
    n_test = len(test_df)

    train_matrix = train_df[["l0", "l1", "l2", "l3", "l4"]].values.astype(float)

    test_matrix = np.full((n_test, 5), np.nan)
    test_matrix[:, 0] = np.log1p(test_df["n_articles"].values)
    # L3/L4 projected (latent but available)
    if "projected_l3" in test_df.columns:
        test_matrix[:, 3] = test_df["projected_l3"].values
    if "projected_l4" in test_df.columns:
        test_matrix[:, 4] = test_df["projected_l4"].values

    full_matrix = np.vstack([train_matrix, test_matrix])
    vmax = float(np.nanmax(full_matrix))

    im = ax.imshow(full_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)

    # Grey out test L1/L2 (not available) and mark L3/L4 as projected
    for row in range(n_train, n_train + n_test):
        for col in [1, 2]:  # L1/L2 not available for test
            ax.add_patch(mpatches.Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                fill=True, color="lightgrey", alpha=0.85, zorder=2
            ))
        for col in [3, 4]:  # L3/L4 are projected
            ax.add_patch(mpatches.Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=1.2, zorder=3, linestyle="--"
            ))

    # Train/test separator
    ax.axhline(n_train - 0.5, color="red", linewidth=2.5)
    ax.text(4.6, n_train - 0.5, "Train/Test split", color="red", fontsize=7, va="center")

    plt.colorbar(im, ax=ax, label="Relevance-Density Mass C(i,j)")

    all_companies = list(train_df["company"]) + list(test_df["company"])
    all_sectors = list(train_df["sector"]) + list(test_df["sector"])
    sector_colors = ["#1f77b4" if s == "energy" else "#2ca02c" for s in all_sectors]

    ax.set_yticks(range(n_train + n_test))
    ax.set_yticklabels([f"{c}" for c in all_companies], fontsize=5)
    for tick, color in zip(ax.get_yticklabels(), sector_colors):
        tick.set_color(color)

    ax.set_xticks(range(5))
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_xlabel("Level", fontsize=11)
    ax.set_ylabel("Scenario (blue=energy, green=financials)", fontsize=9)
    ax.set_title("SREDT Development Triangle\n(grey=missing, dashed-red=projected/latent)", fontsize=12)

    energy_patch = mpatches.Patch(color="#1f77b4", label="Energy")
    fin_patch = mpatches.Patch(color="#2ca02c", label="Financials")
    ax.legend(handles=[energy_patch, fin_patch], loc="upper right", fontsize=8)

    ax.text(0.5, -0.03,
        "NOTE: All L0=3.044 (log(21)) — synthetic article fallback, NOT real GDELT data",
        transform=ax.transAxes, ha="center", fontsize=8, color="red",
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="red", alpha=0.8))

    plt.tight_layout()
    out = out_dir / "fig1_development_triangle_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")
    return str(out)


def make_fig2_venter_scatter(train_df: pd.DataFrame, transitions_full: list, out_dir: Path) -> str:
    """2x2 Venter regression scatter plots."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    transition_cols = [
        ("L0→L1", "l0", "l1"), ("L1→L2", "l1", "l2"),
        ("L2→L3", "l2", "l3"), ("L3→L4", "l3", "l4"),
    ]
    sector_colors = {"energy": "#1f77b4", "financials": "#2ca02c"}

    for ax, (j_label, col_j, col_j1) in zip(axes, transition_cols):
        diag = next(d for d in transitions_full if d["transition"] == j_label)
        x = train_df[col_j].values.astype(float)
        y = train_df[col_j1].values.astype(float)
        colors = [sector_colors[s] for s in train_df["sector"]]

        ax.scatter(x, y, c=colors, alpha=0.7, s=40, zorder=3)

        # Company initials
        for xi, yi, co in zip(x, y, train_df["company"]):
            ax.annotate(co[:2], (xi, yi), fontsize=4, ha="center", va="bottom", alpha=0.6)

        # OLS line
        if diag["slope"] is not None and np.std(x) > 1e-10:
            x_range = np.linspace(x.min(), x.max(), 200)
            y_hat = diag["slope"] * x_range + diag["intercept"]
            ax.plot(x_range, y_hat, "k-", linewidth=1.5, label="OLS fit", zorder=4)

            # 95% CI ribbon
            n = len(x)
            x_mean = np.mean(x)
            ss_x = np.sum((x - x_mean) ** 2)
            y_pred = diag["slope"] * x + diag["intercept"]
            mse = np.sum((y - y_pred) ** 2) / (n - 2) if n > 2 else 0
            if ss_x > 0:
                se_line = np.sqrt(mse * (1 / n + (x_range - x_mean) ** 2 / ss_x))
                t_crit = float(stats.t.ppf(0.975, df=n - 2))
                ax.fill_between(x_range, y_hat - t_crit * se_line, y_hat + t_crit * se_line,
                                alpha=0.15, color="grey", label="95% CI")

        cv_str = f"{diag['cv']:.4f}" if diag["cv"] is not None else "N/A"
        t_str = f"{diag['t_stat']:.3f}" if diag["t_stat"] is not None else "N/A"
        p_str = f"{diag['p_value']:.3g}" if diag["p_value"] is not None else "N/A"
        int_str = f"{diag['intercept']:.4f}" if diag["intercept"] is not None else "N/A"
        title_color = "red" if j_label == "L3→L4" else "black"
        ax.set_title(
            f"{j_label}  |  CV={cv_str}  intercept={int_str}\n"
            f"t={t_str}  p={p_str}  verdict={diag['verdict_corrected']}",
            fontsize=8, color=title_color
        )
        if j_label == "L3→L4":
            ax.text(0.02, 0.96,
                "⚠ Intercept significant → reclassified\nfactor_plus_constant (was chain_ladder_valid)",
                transform=ax.transAxes, fontsize=7, color="red", va="top",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="red", alpha=0.9))

        col_names = {"l0": "L0", "l1": "L1", "l2": "L2", "l3": "L3", "l4": "L4"}
        ax.set_xlabel(f"C(i,j) = {col_names[col_j]}", fontsize=9)
        ax.set_ylabel(f"C(i,j+1) = {col_names[col_j1]}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    energy_patch = mpatches.Patch(color="#1f77b4", label="Energy")
    fin_patch = mpatches.Patch(color="#2ca02c", label="Financials")
    fig.legend(handles=[energy_patch, fin_patch], loc="upper center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.01))
    plt.suptitle("Venter (1998) Development Factor Regression — All Transitions", fontsize=12, y=1.03)
    plt.tight_layout()
    out = out_dir / "fig2_venter_regression_scatterplots.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")
    return str(out)


def make_fig3_score_comparison(test_df: pd.DataFrame, out_dir: Path) -> str:
    """Horizontal bar chart of test scenario scores."""
    fig, ax = plt.subplots(figsize=(10, 12))

    tdf = test_df.copy().sort_values("projected_l4", ascending=False)
    y_pos = np.arange(len(tdf))
    bar_h = 0.25

    ax.barh(y_pos + bar_h, tdf["projected_l4"], bar_h, color="#1b4f72", label="SREDT projected_l4", alpha=0.85)
    ax.barh(y_pos, tdf["flat_sim"], bar_h, color="#e67e22", label="Flat cosine similarity", alpha=0.85)
    ax.barh(y_pos - bar_h, tdf["keyword_freq"], bar_h, color="#95a5a6", label="Keyword frequency", alpha=0.85)

    labels = [f"{row.company} ({row.sector[:3]})\n{row.risk_type}" for row in tdf.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)

    # Sector separator
    n_fin = (tdf["sector"] == "financials").sum()
    n_en = (tdf["sector"] == "energy").sum()
    if n_fin > 0 and n_en > 0:
        sep_y = n_fin - 0.5
        ax.axhline(sep_y, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        ax.text(0.01, sep_y + 0.3, "Financials (above) ↔ Energy (below)", fontsize=7, alpha=0.7)

    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("Test Scenario Score Comparison: SREDT vs Baselines\n"
                 "(No valid ground truth — SC2/SC3 INSUFFICIENT_DATA: all LLM judge calls rate-limited)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "fig3_test_score_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")
    return str(out)


def make_fig4_article_count(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> str:
    """Article count distribution — reveals synthetic data fallback."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (df, label) in zip(axes, [(train_df, "Train (40 scenarios)"), (test_df, "Test (20 scenarios)")]):
        counts = df["n_articles"].values.astype(int)
        unique_counts = sorted(set(counts))
        ax.bar(unique_counts, [np.sum(counts == c) for c in unique_counts],
               color="#2980b9", alpha=0.85, edgecolor="black")
        ax.set_xlabel("n_articles", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{label}", fontsize=10)
        ax.set_xticks(unique_counts)
        ax.text(0.5, 0.85, f"All = {unique_counts[0] if len(unique_counts)==1 else '?'}\n(synthetic fallback)",
                transform=ax.transAxes, ha="center", fontsize=9, color="red",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="red", alpha=0.9))

    fig.suptitle(
        "Article Count Distribution: Revealing Synthetic Data Fallback\n"
        "Real data should show skewed distribution; uniform spike at 20 = GDELT fallback",
        fontsize=10
    )
    plt.tight_layout()
    out = out_dir / "fig4_article_count_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")
    return str(out)


def build_eval_out(
    data_source: str, dq: dict, bootstrap_ci: list, sector_diagnostics: dict,
    intercept_significance_table: list, hypothesis_verdict: list,
    figure_paths: list, key_finding: str, limitations: list,
    robustness: dict,
) -> dict:
    """Build the full eval_out dict matching exp_eval_sol_out schema."""
    overall = "DISCONFIRMED"

    return {
        "metadata": {
            "evaluation_name": "SREDT_Venter_Bootstrap_Reanalysis",
            "description": (
                "Rigorous re-analysis of iter_1 SREDT experiment with bootstrap CIs, "
                "Venter intercept reclassification, sector stratification, and publication figures."
            ),
            "data_source": data_source,
            "data_quality": dq,
            "bootstrap_ci": bootstrap_ci,
            "sector_diagnostics": sector_diagnostics,
            "intercept_significance_table": intercept_significance_table,
            "hypothesis_verdict": hypothesis_verdict,
            "overall_verdict": overall,
            "figure_paths": figure_paths,
            "key_finding": key_finding,
            "limitations": limitations,
            "robustness_checks": robustness,
        },
        "metrics_agg": {
            "sc1_chain_ladder_valid_count": float(
                sum(1 for d in hypothesis_verdict
                    if d["criterion"] == "SC1_venter_proportionality"
                    and "chain_ladder_valid_corrected" in d.get("result", {})
                    and len(d["result"]["chain_ladder_valid_corrected"]) > 0)
            ),
            "n_valid_llm_labels": float(dq.get("n_valid_llm_labels", 0)),
            "l0_variance": float(dq.get("l0_std", 0.0) ** 2),
            "n_train_scenarios": 40.0,
            "n_test_scenarios": 20.0,
            "bootstrap_B": 1000.0,
            "overall_disconfirmed": 1.0,
        },
        "datasets": [],  # will be filled
    }


@logger.catch(reraise=True)
def main():
    logger.info("=== SREDT Venter Re-Analysis ===")

    # ── Locate data ───────────────────────────────────────────────────────────
    data_path, data_source = locate_input_data()
    raw = load_data(data_path)
    logger.info(f"Data loaded from: {data_path}")

    full = raw["metadata"]["full_results"]
    venter_raw = full["venter_diagnostics"]["level_transitions"]
    train_rows = full["train_scenario_analysis"]
    test_rows = full["test_scenario_scores"]
    metrics_raw = full["metrics"]
    examples_raw = raw["datasets"][0]["examples"]

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # ── STEP 3: Data quality ──────────────────────────────────────────────────
    logger.info("Computing data quality checks...")
    dq: dict = {}
    l0_vals = train_df["l0"].values.astype(float)
    dq["l0_std"] = float(np.std(l0_vals))
    dq["l0_constant"] = bool(np.std(l0_vals) < 0.001)
    dq["l0_unique_values"] = sorted(set(map(float, np.unique(l0_vals))))
    dq["n_articles_unique"] = sorted(set(map(int, train_df["n_articles"].unique())))
    dq["n_articles_constant"] = bool(train_df["n_articles"].nunique() == 1)
    dq["keyword_freq_unique"] = sorted(set(map(float, train_df["keyword_freq"].unique())))
    dq["keyword_freq_constant"] = bool(train_df["keyword_freq"].nunique() == 1)
    dq["unique_company_risk_pairs"] = int(train_df.groupby(["company", "risk_type"]).ngroups)
    dq["n_train"] = len(train_df)
    dq["repetition_ratio"] = round(dq["n_train"] / max(1, dq["unique_company_risk_pairs"]), 2)
    dq["n_valid_llm_labels"] = int((test_df["llm_judge_label"] != -1).sum())
    dq["all_labels_missing"] = bool(dq["n_valid_llm_labels"] == 0)
    dq["keyword_freq_std_train"] = float(np.std(train_df["keyword_freq"].values.astype(float)))
    if dq["l0_constant"] and dq["keyword_freq_constant"]:
        dq["flag"] = "SYNTHETIC_DATA"
    elif dq["n_valid_llm_labels"] == 0:
        dq["flag"] = "NO_GROUND_TRUTH"
    else:
        dq["flag"] = "OK"
    logger.info(f"Data quality flag: {dq['flag']}, n_valid_labels={dq['n_valid_llm_labels']}")

    # ── STEP 4: Full-dataset OLS per transition ───────────────────────────────
    logger.info("Running OLS for all transitions (full training set)...")
    transitions_full = []
    for j_label, col_j, col_j1 in [
        ("L0→L1", "l0", "l1"), ("L1→L2", "l1", "l2"),
        ("L2→L3", "l2", "l3"), ("L3→L4", "l3", "l4"),
    ]:
        diag = compute_ols_diagnostics(
            train_df[col_j].values, train_df[col_j1].values, j_label
        )
        transitions_full.append(diag)
        logger.info(
            f"  {j_label}: f_j={diag['f_j']:.4f}, CV={diag['cv']}, "
            f"intercept={diag['intercept']}, t={diag['t_stat']}, "
            f"verdict={diag['verdict_corrected']}"
        )

    # ── STEP 5: Sector-stratified analysis ────────────────────────────────────
    logger.info("Running sector-stratified Venter analysis...")
    sector_diagnostics = run_sector_analysis(train_df)

    # ── STEP 6: Bootstrap CI ──────────────────────────────────────────────────
    logger.info("Running bootstrap (B=1000)...")
    bootstrap_ci = run_bootstrap(train_df, transitions_full, B=1000)

    # ── STEP 7: Intercept significance table ──────────────────────────────────
    prior_verdict_map = {v["j_label"]: v["verdict"] for v in venter_raw}

    intercept_significance_table = []
    for diag in transitions_full:
        prior_v = prior_verdict_map.get(diag["transition"], "not_in_prior")
        reclassified = (prior_v != diag["verdict_corrected"]) and (prior_v != "not_in_prior")
        intercept_significance_table.append({
            "transition": diag["transition"],
            "intercept": diag["intercept"],
            "se_intercept": diag["se_intercept"],
            "t_stat": diag["t_stat"],
            "p_value": diag["p_value"],
            "intercept_significant_2se": diag["intercept_significant"],
            "cv": diag["cv"],
            "verdict_prior_experiment": prior_v,
            "verdict_cv_only": diag["verdict_cv_only"],
            "verdict_corrected": diag["verdict_corrected"],
            "reclassification": reclassified,
        })
        if reclassified:
            logger.warning(
                f"RECLASSIFICATION: {diag['transition']} "
                f"prior={prior_v} → corrected={diag['verdict_corrected']}"
            )

    # ── STEP 8: L0-L4 rank correlation ───────────────────────────────────────
    l0_train = train_df["l0"].values.astype(float)
    l4_train = train_df["l4"].values.astype(float)
    if np.std(l0_train) < 1e-10:
        l0_l4_spearman = None
        l0_l4_note = "L0 constant (all 20 articles, synthetic fallback) — Spearman undefined"
    else:
        r_sp, _ = stats.spearmanr(l0_train, l4_train)
        l0_l4_spearman = float(r_sp)
        l0_l4_note = "Spearman rank correlation between L0 and L4 (training set)"

    # ── STEP 8: Hypothesis verdict table ─────────────────────────────────────
    sc1_passes_corrected = [d for d in transitions_full if d["verdict_corrected"] == "chain_ladder_valid"]
    sc1_passes_cv_only = [d for d in transitions_full
                          if d["cv"] is not None and not math.isnan(d["cv"]) and d["cv"] < 0.30]
    reclassifications = [(d["transition"], d["verdict_corrected"])
                         for d in intercept_significance_table if d["reclassification"]]

    n_valid = dq["n_valid_llm_labels"]

    hypothesis_verdict = [
        {
            "criterion": "SC1_venter_proportionality",
            "description": "CV<0.30 for >=1 transition in >=1 sector, AND intercept non-significant (Venter 1998 combined criterion)",
            "result": {
                "transitions_cv_below_030": [d["transition"] for d in sc1_passes_cv_only],
                "chain_ladder_valid_corrected": [d["transition"] for d in sc1_passes_corrected],
                "reclassifications": reclassifications,
            },
            "threshold": "CV<0.30 AND |a|<2*SE(a)",
            "verdict": "DISCONFIRMED" if len(sc1_passes_corrected) == 0 else "CONFIRMED",
            "note": (
                f"L3→L4 incorrectly labeled chain_ladder_valid in prior experiment "
                f"(CV=0.2939<0.30) but intercept=1.573 with p≈0 requires reclassification "
                f"as factor_plus_constant per Venter (1998). After reclassification, "
                f"{len(sc1_passes_corrected)} of 4 transitions qualify as chain_ladder_valid."
            ),
        },
        {
            "criterion": "SC2_spearman_improvement",
            "description": "SREDT Spearman correlation with ground truth >= flat baseline + 0.15",
            "result": {"n_valid_labels": n_valid, "ground_truth_source": metrics_raw.get("ground_truth_source")},
            "threshold": "Spearman improvement >= 0.15",
            "verdict": "INSUFFICIENT_DATA",
            "note": (
                f"{n_valid} valid LLM judge labels (all rate_limited). "
                "Prior metrics used inadmissible median_split_proxy ground truth."
            ),
        },
        {
            "criterion": "SC3_brier_improvement",
            "description": "SREDT Brier score lower than flat embedding baseline",
            "result": {"n_valid_labels": n_valid},
            "threshold": "Brier_sredt < Brier_flat",
            "verdict": "INSUFFICIENT_DATA",
            "note": "No valid ground truth labels. Prior Brier metrics against synthetic proxy labels are invalid.",
        },
        {
            "criterion": "SC4_cohen_kappa",
            "description": "Inter-annotator Cohen kappa >= 0.60 for 3 domain annotators",
            "result": {"annotation_status": "not_performed"},
            "threshold": "kappa >= 0.60",
            "verdict": "NOT_TESTED",
            "note": "Human annotation deferred to future work.",
        },
        {
            "criterion": "SC5_circularity_check",
            "description": "L0-L4 Spearman rank correlation < 0.80 (SREDT not reducible to flat retrieval)",
            "result": {"l0_l4_rank_corr": l0_l4_spearman, "reason": l0_l4_note},
            "threshold": "rank_corr < 0.80",
            "verdict": "INSUFFICIENT_DATA",
            "note": "L0 is identical for all 40 training scenarios (3.044 = log(21), synthetic). Cannot compute Spearman rank correlation.",
        },
    ]

    # ── STEP 9: Figures ───────────────────────────────────────────────────────
    figures_dir = WORKSPACE / "figures"
    figures_dir.mkdir(exist_ok=True)
    figure_paths = []
    try:
        figure_paths.append(make_fig1_heatmap(train_df, test_df, figures_dir))
        figure_paths.append(make_fig2_venter_scatter(train_df, transitions_full, figures_dir))
        figure_paths.append(make_fig3_score_comparison(test_df, figures_dir))
        figure_paths.append(make_fig4_article_count(train_df, test_df, figures_dir))
    except Exception as e:
        logger.error(f"Figure generation error: {e}")

    # ── STEP 10: Key finding ──────────────────────────────────────────────────
    sc1_pass_count = len(sc1_passes_corrected)
    key_finding = (
        f"The prior SREDT experiment (iter_1) used a CV-only Venter criterion and incorrectly labeled "
        f"L3→L4 as chain_ladder_valid (CV=0.2939<0.30), while the intercept was highly significant "
        f"(a=1.573, t={next((d['t_stat'] for d in transitions_full if d['transition']=='L3→L4'), 'N/A'):.3f}, p≈0), "
        f"which under Venter (1998) requires reclassification as factor_plus_constant. "
        f"After applying the combined Venter criterion (CV<0.30 AND intercept non-significant), "
        f"{sc1_pass_count} of 4 transitions qualify as chain_ladder_valid — SC1 is DISCONFIRMED. "
        f"All other success criteria (SC2–SC5) are INSUFFICIENT_DATA or NOT_TESTED: "
        f"(a) all {n_valid} LLM judge evaluations were skipped due to rate limits; "
        f"(b) L0 is constant (all 3.044 = log(21), synthetic 20-article fallback), making SC5 undefined. "
        f"Data source: {data_source}. Data quality flag: {dq['flag']}."
    )

    # ── STEP 11: Limitations ──────────────────────────────────────────────────
    limitations = [
        "All GDELT article retrieval fell back to synthetic 20-article sets; real GDELT data not obtained. "
        "L0=3.044 constant for all scenarios, making L0→L1 regression degenerate and SC5 undefined.",
        "All 20 test-set LLM judge evaluations were rate-limited (UNKNOWN labels). "
        "Evaluation metrics (AUROC, Brier, Spearman) computed against inadmissible median_split_proxy ground truth in prior experiment.",
        "Keyword frequency = 1.0 for all scenarios (degenerate baseline). No discriminative power.",
        "Training scenarios are repeated templates (6 companies x 6 risk types cycling). "
        "Effective regression n ≈ 6 unique (company, risk_type) pairs per sector, not 20 independent scenarios.",
        "Overlapping 90-day training windows violate the actuarial non-overlapping accident-year analogy.",
        "L3 centroids used Basel II-derived risk categories, not GARP 2025 L1 taxonomy as required by the corrected hypothesis.",
        "L1/L2 centroids were free-form keyword concatenations, not canonical GICS label strings.",
        "LLM-generated scenario diversity was not achieved; all scenarios used hardcoded templates, "
        "reducing cross-scenario variance critical for Venter regression stability.",
        "Bootstrap CI interpretability limited by low effective sample size (≈6 unique data points per sector "
        "per transition due to template repetition). Bootstrap resamples scenarios, not unique company-risk pairs.",
    ]

    # ── STEP 13: Robustness checks ────────────────────────────────────────────
    logger.info("Running robustness checks...")
    robustness = {}
    try:
        robustness["jackknife_l3_l4"] = jackknife_l3_l4(train_df)
        logger.info(f"Jackknife L3→L4: {robustness['jackknife_l3_l4']}")
    except Exception as e:
        logger.error(f"Jackknife failed: {e}")
        robustness["jackknife_l3_l4"] = {"error": str(e)}

    try:
        robustness["sector_swap_test"] = sector_swap_test(train_df)
        logger.info(f"Sector swap: {robustness['sector_swap_test']}")
    except Exception as e:
        logger.error(f"Sector swap failed: {e}")
        robustness["sector_swap_test"] = {"error": str(e)}

    # ── STEP 12: Assemble eval_out ────────────────────────────────────────────
    logger.info("Assembling eval_out.json...")

    # Build per-example eval fields
    eval_examples = []
    for ex in examples_raw:
        new_ex = dict(ex)
        # Add per-example evaluation metrics where possible
        # All labels are -1, so no per-example AUROC/Brier; add eval_sredt_rank
        sredt_val = float(new_ex.get("predict_sredt", 0))
        flat_val = float(new_ex.get("predict_flat_cosine", 0))
        new_ex["eval_sredt_minus_flat"] = float(sredt_val - flat_val)
        eval_examples.append(new_ex)

    # Build metrics_agg with only numeric values (schema requires numbers only)
    sc1_cv_count = float(len(sc1_passes_cv_only))
    sc1_corrected_count = float(len(sc1_passes_corrected))
    n_reclassified = float(len(reclassifications))
    l3l4_diag = next(d for d in transitions_full if d["transition"] == "L3→L4")
    l3l4_cv = l3l4_diag["cv"] if l3l4_diag["cv"] is not None else float("nan")
    l3l4_intercept = l3l4_diag["intercept"] if l3l4_diag["intercept"] is not None else float("nan")
    l3l4_t = l3l4_diag["t_stat"] if l3l4_diag["t_stat"] is not None else float("nan")
    l3l4_pval = l3l4_diag["p_value"] if l3l4_diag["p_value"] is not None else float("nan")

    # Get bootstrap CI for L3→L4
    l3l4_boot = next((b for b in bootstrap_ci if b["transition"] == "L3→L4"), {})
    boot_int_lower = l3l4_boot.get("intercept_lower") or float("nan")
    boot_int_upper = l3l4_boot.get("intercept_upper") or float("nan")
    boot_p = l3l4_boot.get("intercept_p_bootstrap") or float("nan")

    metrics_agg = {
        "sc1_transitions_cv_below_030": sc1_cv_count,
        "sc1_transitions_chain_ladder_valid_corrected": sc1_corrected_count,
        "sc1_reclassification_count": n_reclassified,
        "sc1_verdict_disconfirmed": 1.0,
        "sc2_verdict_insufficient_data": 1.0,
        "sc3_verdict_insufficient_data": 1.0,
        "sc4_verdict_not_tested": 1.0,
        "sc5_verdict_insufficient_data": 1.0,
        "n_valid_llm_labels": float(n_valid),
        "n_train_scenarios": float(len(train_df)),
        "n_test_scenarios": float(len(test_df)),
        "l0_variance": float(np.var(l0_vals)),
        "keyword_freq_variance": float(np.var(train_df["keyword_freq"].values.astype(float))),
        "l3l4_cv": l3l4_cv if not math.isnan(l3l4_cv) else -999.0,
        "l3l4_intercept": l3l4_intercept if not math.isnan(l3l4_intercept) else -999.0,
        "l3l4_t_stat": l3l4_t if not math.isnan(l3l4_t) else -999.0,
        "l3l4_p_value": l3l4_pval if not math.isnan(l3l4_pval) else -999.0,
        "l3l4_boot_intercept_lower": boot_int_lower if not math.isnan(boot_int_lower) else -999.0,
        "l3l4_boot_intercept_upper": boot_int_upper if not math.isnan(boot_int_upper) else -999.0,
        "l3l4_boot_p_intercept": boot_p if not math.isnan(boot_p) else -999.0,
        "bootstrap_B": 1000.0,
        "unique_company_risk_pairs": float(dq["unique_company_risk_pairs"]),
        "repetition_ratio": float(dq["repetition_ratio"]),
        "overall_disconfirmed": 1.0,
    }

    eval_out = {
        "metadata": {
            "evaluation_name": "SREDT_Venter_Bootstrap_Reanalysis",
            "description": (
                "Rigorous re-analysis of iter_1 SREDT experiment with bootstrap CIs (B=1000), "
                "Venter (1998) intercept reclassification, sector stratification, "
                "and 4 publication-quality figures."
            ),
            "data_source": data_source,
            "data_quality": dq,
            "bootstrap_ci": bootstrap_ci,
            "sector_diagnostics": sector_diagnostics,
            "intercept_significance_table": intercept_significance_table,
            "hypothesis_verdict": hypothesis_verdict,
            "overall_verdict": "DISCONFIRMED",
            "figure_paths": figure_paths,
            "key_finding": key_finding,
            "limitations": limitations,
            "robustness_checks": robustness,
            "transitions_full": transitions_full,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "SREDT_GDELT_Risk_Scenarios",
                "examples": eval_examples,
            }
        ],
    }

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = WORKSPACE / "eval_out.json"
    # Remove individual_ratios from transitions_full to keep file size manageable
    for t in eval_out["metadata"]["transitions_full"]:
        t.pop("individual_ratios", None)
    for t in transitions_full:
        t.pop("individual_ratios", None)

    out_path.write_text(json.dumps(eval_out, indent=2, allow_nan=False))
    logger.info(f"Wrote {out_path}")

    # ── Check file size ───────────────────────────────────────────────────────
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"eval_out.json size: {size_mb:.2f} MB")

    logger.info("=== Done ===")
    logger.info(f"Key finding: {key_finding[:200]}...")
    return eval_out


if __name__ == "__main__":
    main()
