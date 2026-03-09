#!/usr/bin/env python3
"""Cross-Dataset JRN Consolidation: Unified Statistical Analysis & Hypothesis Scorecard.

Consolidates results from 4 iteration-2 experiments into a unified cross-dataset
statistical report with 7 analyses (A-G) and a hypothesis scorecard.
"""

import json
import math
import os
import resource
import sys
import warnings
from pathlib import Path

import diptest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.stats import gaussian_kde, shapiro, spearmanr, kendalltau
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Logging ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource Limits ──────────────────────────────────────────────────────
_container_ram = None
for _p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
    try:
        _v = Path(_p).read_text().strip()
        if _v != "max" and int(_v) < 1_000_000_000_000:
            _container_ram = int(_v)
            break
    except (FileNotFoundError, ValueError):
        pass
RAM_BUDGET = int((_container_ram or 29_000_000_000) * 0.5)  # 50% of available
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_2/gen_art")
EXP1_PATH = BASE / "exp_id1_it2__opus" / "full_method_out.json"
EXP2_PATH = BASE / "exp_id2_it2__opus" / "full_method_out.json"
EXP3_PATH = BASE / "exp_id3_it2__opus" / "preview_method_out.json"
EXP4_PATH = BASE / "exp_id4_it2__opus" / "full_method_out.json"

FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# STEP 0: Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_experiments() -> dict:
    """Load all 4 experiment outputs."""
    logger.info("Loading experiment data...")

    exp1_raw = json.loads(EXP1_PATH.read_text())
    exp2_raw = json.loads(EXP2_PATH.read_text())
    exp3_raw = json.loads(EXP3_PATH.read_text())
    exp4_raw = json.loads(EXP4_PATH.read_text())

    logger.info(f"Exp1: {len(exp1_raw['datasets'][0]['examples'])} examples")
    logger.info(f"Exp2: {len(exp2_raw['datasets'][0]['examples'])} examples")
    logger.info(f"Exp3: metadata loaded (preview)")
    logger.info(f"Exp4: {len(exp4_raw['datasets'][0]['examples'])} examples")

    return {"exp1": exp1_raw, "exp2": exp2_raw, "exp3": exp3_raw, "exp4": exp4_raw}


def parse_exp1(exp1_raw: dict) -> pd.DataFrame:
    """Parse exp1 examples into DataFrame with JRN, task, join info."""
    rows = []
    for ex in exp1_raw["datasets"][0]["examples"]:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        # Skip summary rows
        if "join_idx" not in inp:
            continue
        row = {
            "join_idx": inp["join_idx"],
            "source_table": inp.get("source_table", ""),
            "target_table": inp.get("target_table", ""),
            "task_name": inp.get("task_name", ex.get("metadata_task_name", "")),
            "connectivity": inp.get("connectivity", ex.get("metadata_connectivity", "")),
            "fanout_mean": inp.get("fanout_mean", 0),
            "jrn": out.get("jrn", ex.get("metadata_jrn", 0)),
            "agg_sensitivity": out.get("agg_sensitivity", ex.get("metadata_agg_sensitivity", 0)),
        }
        # per-strategy performances
        if "strategy_perfs" in out:
            for strat, val in out["strategy_perfs"].items():
                row[f"perf_{strat}"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info(f"Exp1 parsed: {len(df)} data rows, {df['join_idx'].nunique()} joins, {df['task_name'].nunique()} tasks")
    return df


def parse_exp2(exp2_raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse exp2 into phase1 (individual JRN) and phase3 (chains) DataFrames."""
    phase1_rows = []
    phase3_rows = []
    for ex in exp2_raw["datasets"][0]["examples"]:
        phase = ex.get("metadata_phase", "")
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        pred_baseline = json.loads(ex.get("predict_baseline", "{}")) if isinstance(ex.get("predict_baseline"), str) else ex.get("predict_baseline", {})
        pred_method = json.loads(ex.get("predict_our_method", "{}")) if isinstance(ex.get("predict_our_method"), str) else ex.get("predict_our_method", {})

        if phase == "phase1":
            row = {
                "join": ex.get("metadata_join", inp.get("join", "")),
                "task": ex.get("metadata_task", inp.get("task", "")),
                "jrn": out.get("JRN", 0),
                "M_base": out.get("M_base", 0),
                "M_join": out.get("M_join", 0),
                "heuristic_jrn": pred_baseline.get("heuristic_jrn", 0),
            }
            phase1_rows.append(row)
        elif phase == "phase3":
            # predicted_chain_jrn is in predict_our_method, not output
            actual_jrn = out.get("actual_chain_jrn", 0)
            predicted_jrn = pred_method.get("predicted_chain_jrn", 0)
            deviation = out.get("deviation_from_predicted", pred_method.get("deviation", 0))
            chain_joins = inp.get("chain", inp.get("joins", inp.get("chain_joins", [])))
            if isinstance(chain_joins, str):
                chain_joins = [chain_joins]
            chain_depth = inp.get("depth", len(chain_joins) if chain_joins else 2)
            row = {
                "chain": str(chain_joins),
                "task": ex.get("metadata_task", inp.get("task", "")),
                "actual_chain_jrn": actual_jrn,
                "predicted_chain_jrn": predicted_jrn,
                "deviation": deviation,
                "chain_depth": chain_depth,
                "joins_in_chain": chain_joins,
            }
            phase3_rows.append(row)

    df1 = pd.DataFrame(phase1_rows)
    df3 = pd.DataFrame(phase3_rows)
    logger.info(f"Exp2 phase1: {len(df1)} rows, phase3: {len(df3)} chains")
    return df1, df3


def parse_exp3(exp3_raw: dict) -> dict:
    """Extract exp3 metadata (global JRN, bucket JRN, training-free baselines)."""
    meta = exp3_raw.get("metadata", {})
    result = {
        "global_jrn": meta.get("global_jrn_results", {}),
        "bucket_jrn": meta.get("bucket_jrn_results", {}),
        "training_free": meta.get("training_free_baselines", {}),
        "analysis": meta.get("analysis", {}),
    }
    logger.info(f"Exp3: {len(result['global_jrn'])} global JRN, {len(result['bucket_jrn'])} bucket entries")
    return result


def parse_exp4(exp4_raw: dict) -> pd.DataFrame:
    """Parse exp4 into DataFrame with GT JRN, probe JRN, and proxy values."""
    rows = []
    for ex in exp4_raw["datasets"][0]["examples"]:
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
        # Skip summary rows
        if inp.get("type") == "spearman_correlation_summary" or "join" not in inp:
            continue
        row = {
            "join_idx": ex.get("metadata_join_idx", 0),
            "join_name": inp.get("join", ""),
            "task_name": ex.get("metadata_task_name", inp.get("task", "")),
            "task_type": ex.get("metadata_task_type", inp.get("task_type", "")),
            "source_table": ex.get("metadata_source_table", ""),
            "target_table": ex.get("metadata_target_table", ""),
            "jrn_gt_mean": out.get("jrn_gt_mean", 0),
            "jrn_gt_std": out.get("jrn_gt_std", 0),
            "M_base_gt": out.get("M_base_gt", 0),
            "M_join_gt": out.get("M_join_gt", 0),
            "predict_jrn_probe_gbm": float(ex.get("predict_jrn_probe_gbm", 0)),
            "predict_jrn_probe_mlp": float(ex.get("predict_jrn_probe_mlp", 0)),
            "proxy_fanout": float(ex.get("predict_proxy_fanout", 0)),
            "proxy_correlation": float(ex.get("predict_proxy_correlation", 0)),
            "proxy_MI": float(ex.get("predict_proxy_MI", 0)),
            "proxy_entropy_reduction": float(ex.get("predict_proxy_entropy_reduction", 0)),
            "proxy_homophily": float(ex.get("predict_proxy_homophily", 0)),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info(f"Exp4 parsed: {len(df)} join-task pairs, {df['join_idx'].nunique()} unique joins")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Analysis A: JRN Distribution & Threshold Testing
# ═══════════════════════════════════════════════════════════════════════

def analysis_A(df_exp1: pd.DataFrame, df_exp2_p1: pd.DataFrame,
               exp3_data: dict, df_exp4: pd.DataFrame) -> dict:
    """JRN Distribution & Threshold Testing."""
    logger.info("=== Analysis A: JRN Distribution & Threshold Testing ===")

    # Collect all JRN values
    jrn_exp1 = df_exp1["jrn"].values  # MLP probe JRN
    jrn_exp2 = df_exp2_p1["jrn"].values  # MLP probe JRN
    jrn_exp3_global = []
    for key, val in exp3_data["global_jrn"].items():
        jrn_exp3_global.append(val["JRN"])
    jrn_exp3_global = np.array(jrn_exp3_global)

    # Bucket-level JRN from exp3
    jrn_exp3_bucket = []
    for key, val in exp3_data["bucket_jrn"].items():
        jrn_exp3_bucket.append(val["JRN"])
    jrn_exp3_bucket = np.array(jrn_exp3_bucket)

    jrn_exp4_gt = df_exp4["jrn_gt_mean"].values
    jrn_exp4_gbm = df_exp4["predict_jrn_probe_gbm"].values
    jrn_exp4_mlp = df_exp4["predict_jrn_probe_mlp"].values

    # Pool MLP-probe JRN (exp1 + exp2 + exp3 global)
    pooled_mlp = np.concatenate([jrn_exp1, jrn_exp2, jrn_exp3_global])
    logger.info(f"Pooled MLP-probe JRN: {len(pooled_mlp)} values")
    logger.info(f"GT JRN (exp4): {len(jrn_exp4_gt)} values")

    def descriptive_stats(arr: np.ndarray, name: str) -> dict:
        return {
            "name": name,
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "skewness": float(stats.skew(arr)) if len(arr) > 2 else 0.0,
            "kurtosis": float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0,
        }

    per_dataset_stats = {
        "exp1_rel_f1_mlp": descriptive_stats(jrn_exp1, "rel-f1 MLP probe"),
        "exp2_rel_stack_mlp": descriptive_stats(jrn_exp2, "rel-stack MLP probe"),
        "exp3_rel_hm_global": descriptive_stats(jrn_exp3_global, "rel-hm global"),
        "exp3_rel_hm_bucket": descriptive_stats(jrn_exp3_bucket, "rel-hm bucket-level"),
        "exp4_rel_f1_gt": descriptive_stats(jrn_exp4_gt, "rel-f1 ground-truth"),
        "exp4_rel_f1_gbm": descriptive_stats(jrn_exp4_gbm, "rel-f1 GBM probe"),
        "exp4_rel_f1_mlp": descriptive_stats(jrn_exp4_mlp, "rel-f1 MLP probe (exp4)"),
    }
    pooled_stats = descriptive_stats(pooled_mlp, "pooled MLP probe")

    # KDE density estimate
    kde_modes = []
    try:
        kde = gaussian_kde(pooled_mlp, bw_method="silverman")
        x_grid = np.linspace(pooled_mlp.min() - 0.5, pooled_mlp.max() + 0.5, 500)
        y_kde = kde(x_grid)
        # Find modes (local maxima)
        for i in range(1, len(y_kde) - 1):
            if y_kde[i] > y_kde[i - 1] and y_kde[i] > y_kde[i + 1]:
                kde_modes.append(float(x_grid[i]))
        logger.info(f"KDE modes: {kde_modes}")
    except Exception:
        logger.exception("KDE failed")
        x_grid = np.array([])
        y_kde = np.array([])

    # Hartigan's dip test
    dip_pooled = {"statistic": 0.0, "p_value": 1.0}
    dip_gt = {"statistic": 0.0, "p_value": 1.0}
    try:
        dip_stat, dip_p = diptest.diptest(pooled_mlp)
        dip_pooled = {"statistic": float(dip_stat), "p_value": float(dip_p)}
        logger.info(f"Dip test (pooled MLP): stat={dip_stat:.4f}, p={dip_p:.4f}")
    except Exception:
        logger.exception("Dip test failed on pooled MLP")
    try:
        dip_stat_gt, dip_p_gt = diptest.diptest(jrn_exp4_gt)
        dip_gt = {"statistic": float(dip_stat_gt), "p_value": float(dip_p_gt)}
        logger.info(f"Dip test (GT JRN): stat={dip_stat_gt:.4f}, p={dip_p_gt:.4f}")
    except Exception:
        logger.exception("Dip test failed on GT JRN")

    # Fraction near threshold [0.95, 1.05]
    near_threshold_mlp = np.sum((pooled_mlp >= 0.95) & (pooled_mlp <= 1.05)) / len(pooled_mlp)
    near_threshold_gt = np.sum((jrn_exp4_gt >= 0.95) & (jrn_exp4_gt <= 1.05)) / len(jrn_exp4_gt) if len(jrn_exp4_gt) > 0 else 0.0
    logger.info(f"Fraction near threshold: MLP={near_threshold_mlp:.3f}, GT={near_threshold_gt:.3f}")

    # Per-dataset JRN range
    per_dataset_range = {}
    for name, arr in [("exp1_rel_f1", jrn_exp1), ("exp2_rel_stack", jrn_exp2),
                       ("exp3_global", jrn_exp3_global), ("exp3_bucket", jrn_exp3_bucket),
                       ("exp4_gt", jrn_exp4_gt)]:
        per_dataset_range[name] = {
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "has_below_threshold": bool(np.any(arr < 1.0)),
            "has_above_threshold": bool(np.any(arr > 1.0)),
        }

    # ── Plots ──
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of pooled MLP JRN
        axes[0].hist(pooled_mlp, bins=20, alpha=0.7, color="steelblue", edgecolor="black", label=f"MLP probe (n={len(pooled_mlp)})")
        axes[0].axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="JRN=1 threshold")
        axes[0].set_xlabel("JRN")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Pooled MLP-Probe JRN Distribution")
        axes[0].legend()

        # Histogram of GT JRN
        axes[1].hist(jrn_exp4_gt, bins=15, alpha=0.7, color="darkorange", edgecolor="black", label=f"GT JRN (n={len(jrn_exp4_gt)})")
        axes[1].axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="JRN=1 threshold")
        axes[1].set_xlabel("JRN")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Ground-Truth JRN Distribution (rel-f1)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / "A_jrn_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # KDE plot
        if len(x_grid) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_grid, y_kde, "b-", linewidth=2, label="KDE (Silverman)")
            ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="JRN=1")
            for mode in kde_modes:
                ax.axvline(x=mode, color="green", linestyle=":", alpha=0.7)
                ax.annotate(f"mode={mode:.2f}", xy=(mode, kde(np.array([mode]))[0]),
                           fontsize=8, ha="center", va="bottom")
            ax.set_xlabel("JRN")
            ax.set_ylabel("Density")
            ax.set_title("KDE of Pooled MLP-Probe JRN")
            ax.legend()
            plt.tight_layout()
            plt.savefig(str(FIGURES_DIR / "A_jrn_kde.png"), dpi=150, bbox_inches="tight")
            plt.close()
    except Exception:
        logger.exception("Plot generation failed for Analysis A")

    total_jrn_measurements = len(pooled_mlp) + len(jrn_exp3_bucket) + len(jrn_exp4_gt) + len(jrn_exp4_gbm) + len(jrn_exp4_mlp)

    return {
        "pooled_stats": pooled_stats,
        "per_dataset_stats": per_dataset_stats,
        "dip_test_pooled_mlp": dip_pooled,
        "dip_test_gt_jrn": dip_gt,
        "fraction_near_threshold_mlp": float(near_threshold_mlp),
        "fraction_near_threshold_gt": float(near_threshold_gt),
        "kde_modes": [float(m) for m in kde_modes],
        "per_dataset_range": per_dataset_range,
        "total_jrn_measurements": int(total_jrn_measurements),
    }


# ═══════════════════════════════════════════════════════════════════════
# Analysis B: Task Stability (Kendall's W Concordance)
# ═══════════════════════════════════════════════════════════════════════

def analysis_B(df_exp1: pd.DataFrame, df_exp2_p1: pd.DataFrame) -> dict:
    """Task Stability via Kendall's W concordance."""
    logger.info("=== Analysis B: Task Stability (Kendall's W) ===")

    # ── Exp1 (rel-f1): Build join × task JRN matrix ──
    # Average JRN across connectivity patterns for each (join_idx, task) pair
    exp1_avg = df_exp1.groupby(["join_idx", "task_name"])["jrn"].mean().reset_index()
    pivot1 = exp1_avg.pivot(index="join_idx", columns="task_name", values="jrn")
    # Drop joins/tasks with too many missing values
    pivot1_clean = pivot1.dropna(thresh=2, axis=0).dropna(thresh=2, axis=1)
    # For Kendall's W, we need complete cases - fill remaining NaN with column median
    pivot1_complete = pivot1_clean.fillna(pivot1_clean.median())

    n_joins_1 = len(pivot1_complete)
    n_tasks_1 = len(pivot1_complete.columns)
    logger.info(f"Exp1 Kendall's W matrix: {n_joins_1} joins × {n_tasks_1} tasks")

    kendall_w_exp1 = compute_kendalls_W(pivot1_complete)

    # ── Exp2 (rel-stack): user-entity joins × user tasks ──
    # Filter to user-entity joins (those with user tasks)
    user_tasks = ["user-engagement", "user-badge"]
    exp2_user = df_exp2_p1[df_exp2_p1["task"].isin(user_tasks)]
    pivot2 = exp2_user.pivot(index="join", columns="task", values="jrn")
    pivot2_clean = pivot2.dropna()

    n_joins_2 = len(pivot2_clean)
    n_tasks_2 = len(pivot2_clean.columns)
    logger.info(f"Exp2 Kendall's W matrix: {n_joins_2} joins × {n_tasks_2} tasks")

    kendall_w_exp2 = compute_kendalls_W(pivot2_clean)

    # ── Spearman rank correlation between task pairs (exp1) ──
    task_pair_corrs = {}
    tasks = list(pivot1_complete.columns)
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            rho, pval = spearmanr(pivot1_complete[t1], pivot1_complete[t2])
            key = f"{t1} vs {t2}"
            task_pair_corrs[key] = {"spearman_rho": float(rho), "p_value": float(pval)}

    # Interpretation
    w1 = kendall_w_exp1.get("W", 0)
    if w1 > 0.7:
        interp1 = "Strong concordance — JRN is join-intrinsic"
    elif w1 > 0.3:
        interp1 = "Moderate concordance — JRN partially task-dependent"
    else:
        interp1 = "Poor concordance — JRN is task-specific"

    w2 = kendall_w_exp2.get("W", 0)
    if w2 > 0.7:
        interp2 = "Strong concordance"
    elif w2 > 0.3:
        interp2 = "Moderate concordance"
    else:
        interp2 = "Poor concordance"

    return {
        "kendalls_W_rel_f1": {**kendall_w_exp1, "n_joins": n_joins_1, "n_tasks": n_tasks_1, "interpretation": interp1},
        "kendalls_W_rel_stack": {**kendall_w_exp2, "n_joins": n_joins_2, "n_tasks": n_tasks_2, "interpretation": interp2},
        "task_pair_correlations_rel_f1": task_pair_corrs,
    }


def compute_kendalls_W(pivot_df: pd.DataFrame) -> dict:
    """Compute Kendall's coefficient of concordance W."""
    if pivot_df.empty or len(pivot_df.columns) < 2 or len(pivot_df) < 2:
        return {"W": 0.0, "chi2": 0.0, "p_value": 1.0}

    # Each column (task) is a "judge", each row (join) is an "item"
    # Rank each column
    ranked = pivot_df.rank(axis=0)
    k = len(ranked.columns)  # number of judges (tasks)
    n = len(ranked)  # number of items (joins)

    # Sum of ranks for each item
    R = ranked.sum(axis=1)
    R_mean = R.mean()
    S = ((R - R_mean) ** 2).sum()

    # W = 12S / (k²(n³ - n))
    W = 12 * S / (k ** 2 * (n ** 3 - n))

    # Chi-squared approximation
    chi2 = k * (n - 1) * W
    df = n - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)

    logger.info(f"Kendall's W={W:.4f}, chi2={chi2:.4f}, p={p_value:.4f}")
    return {"W": float(W), "chi2": float(chi2), "p_value": float(p_value)}


# ═══════════════════════════════════════════════════════════════════════
# Analysis C: Probe Type Comparison (MLP vs GBM)
# ═══════════════════════════════════════════════════════════════════════

def analysis_C(df_exp4: pd.DataFrame, df_exp1: pd.DataFrame) -> dict:
    """MLP vs GBM probe comparison."""
    logger.info("=== Analysis C: Probe Type Comparison (MLP vs GBM) ===")

    gt = df_exp4["jrn_gt_mean"].values
    gbm = df_exp4["predict_jrn_probe_gbm"].values
    mlp = df_exp4["predict_jrn_probe_mlp"].values

    # 1. Spearman ρ (GBM vs GT)
    rho_gbm, p_gbm = spearmanr(gbm, gt)
    ci_gbm = fisher_z_ci(rho_gbm, len(gt))
    logger.info(f"GBM vs GT: ρ={rho_gbm:.4f}, p={p_gbm:.6f}, CI={ci_gbm}")

    # 2. Spearman ρ (MLP vs GT)
    rho_mlp, p_mlp = spearmanr(mlp, gt)
    ci_mlp = fisher_z_ci(rho_mlp, len(gt))
    logger.info(f"MLP vs GT: ρ={rho_mlp:.4f}, p={p_mlp:.6f}, CI={ci_mlp}")

    # 4. MLP range analysis
    mlp_range = {"min": float(np.min(mlp)), "max": float(np.max(mlp)),
                  "range": float(np.max(mlp) - np.min(mlp)),
                  "std": float(np.std(mlp, ddof=1)),
                  "cv": float(np.std(mlp, ddof=1) / np.mean(mlp)) if np.mean(mlp) != 0 else 0}
    gbm_range = {"min": float(np.min(gbm)), "max": float(np.max(gbm)),
                  "range": float(np.max(gbm) - np.min(gbm)),
                  "std": float(np.std(gbm, ddof=1)),
                  "cv": float(np.std(gbm, ddof=1) / np.mean(gbm)) if np.mean(gbm) != 0 else 0}
    gt_range = {"min": float(np.min(gt)), "max": float(np.max(gt)),
                 "range": float(np.max(gt) - np.min(gt)),
                 "std": float(np.std(gt, ddof=1)),
                 "cv": float(np.std(gt, ddof=1) / np.mean(gt)) if np.mean(gt) != 0 else 0}

    # 5. Task-stratified analysis
    task_stratified = {}
    for task_name, group in df_exp4.groupby("task_name"):
        if len(group) < 3:
            continue
        rho_g, p_g = spearmanr(group["predict_jrn_probe_gbm"], group["jrn_gt_mean"])
        rho_m, p_m = spearmanr(group["predict_jrn_probe_mlp"], group["jrn_gt_mean"])
        task_stratified[task_name] = {
            "n": int(len(group)),
            "gbm_spearman_rho": float(rho_g), "gbm_p_value": float(p_g),
            "mlp_spearman_rho": float(rho_m), "mlp_p_value": float(p_m),
        }

    # 6. Rank displacement analysis
    rank_gt = stats.rankdata(gt)
    rank_gbm = stats.rankdata(gbm)
    rank_mlp = stats.rankdata(mlp)
    disp_gbm = np.abs(rank_gbm - rank_gt)
    disp_mlp = np.abs(rank_mlp - rank_gt)

    rank_displacements = []
    for i in range(len(df_exp4)):
        rank_displacements.append({
            "join": df_exp4.iloc[i]["join_name"],
            "task": df_exp4.iloc[i]["task_name"],
            "gt_rank": int(rank_gt[i]),
            "gbm_rank": int(rank_gbm[i]),
            "mlp_rank": int(rank_mlp[i]),
            "gbm_displacement": int(disp_gbm[i]),
            "mlp_displacement": int(disp_mlp[i]),
        })

    mean_disp_gbm = float(np.mean(disp_gbm))
    mean_disp_mlp = float(np.mean(disp_mlp))

    # 7. Cross-check with exp1 MLP JRN
    cross_config = {}
    try:
        # Match on join_idx and task_name
        exp1_avg = df_exp1.groupby(["join_idx", "task_name"])["jrn"].mean().reset_index()
        merged = pd.merge(
            df_exp4[["join_idx", "task_name", "predict_jrn_probe_mlp"]].rename(columns={"predict_jrn_probe_mlp": "mlp_exp4"}),
            exp1_avg[["join_idx", "task_name", "jrn"]].rename(columns={"jrn": "mlp_exp1"}),
            on=["join_idx", "task_name"],
            how="inner"
        )
        if len(merged) >= 3:
            rho_cross, p_cross = spearmanr(merged["mlp_exp1"], merged["mlp_exp4"])
            cross_config = {
                "n_overlapping": int(len(merged)),
                "spearman_rho": float(rho_cross),
                "p_value": float(p_cross),
                "exp1_config": "hidden=32, epochs=10",
                "exp4_config": "hidden=[32,16], epochs=50",
            }
            logger.info(f"Cross-config MLP stability: ρ={rho_cross:.4f} on {len(merged)} overlapping pairs")
        else:
            cross_config = {"n_overlapping": int(len(merged)), "note": "Too few overlapping pairs"}
    except Exception:
        logger.exception("Cross-config MLP comparison failed")
        cross_config = {"error": "comparison failed"}

    # ── Scatter plots ──
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(gt, gbm, c="forestgreen", alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        lims = [min(gt.min(), gbm.min()) - 0.05, max(gt.max(), gbm.max()) + 0.05]
        axes[0].plot(lims, lims, "r--", alpha=0.5, label="Identity")
        axes[0].set_xlabel("Ground-Truth JRN")
        axes[0].set_ylabel("GBM Probe JRN")
        axes[0].set_title(f"GBM Probe vs GT (ρ={rho_gbm:.3f})")
        axes[0].legend()

        axes[1].scatter(gt, mlp, c="darkorange", alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        lims2 = [min(gt.min(), mlp.min()) - 0.05, max(gt.max(), mlp.max()) + 0.05]
        axes[1].plot([gt.min(), gt.max()], [gt.min(), gt.max()], "r--", alpha=0.5, label="Identity")
        axes[1].set_xlabel("Ground-Truth JRN")
        axes[1].set_ylabel("MLP Probe JRN")
        axes[1].set_title(f"MLP Probe vs GT (ρ={rho_mlp:.3f})")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / "C_probe_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        logger.exception("Scatter plot failed for Analysis C")

    return {
        "gbm_vs_gt_spearman": {"rho": float(rho_gbm), "p_value": float(p_gbm), "ci_95": ci_gbm},
        "mlp_vs_gt_spearman": {"rho": float(rho_mlp), "p_value": float(p_mlp), "ci_95": ci_mlp},
        "mlp_range_analysis": mlp_range,
        "gbm_range_analysis": gbm_range,
        "gt_range_analysis": gt_range,
        "task_stratified": task_stratified,
        "rank_displacement_summary": {
            "mean_gbm_displacement": mean_disp_gbm,
            "mean_mlp_displacement": mean_disp_mlp,
            "max_gbm_displacement": int(np.max(disp_gbm)),
            "max_mlp_displacement": int(np.max(disp_mlp)),
        },
        "rank_displacements_top5_worst_mlp": sorted(rank_displacements, key=lambda x: -x["mlp_displacement"])[:5],
        "cross_config_stability": cross_config,
    }


def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> list:
    """95% CI for Spearman ρ via Fisher z-transform."""
    if n < 4:
        return [float(rho), float(rho)]
    z = np.arctanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return [float(lo), float(hi)]


# ═══════════════════════════════════════════════════════════════════════
# Analysis D: Training-Free Proxy Generalization
# ═══════════════════════════════════════════════════════════════════════

def analysis_D(df_exp4: pd.DataFrame, df_exp2_p1: pd.DataFrame,
               exp3_data: dict) -> dict:
    """Training-free proxy generalization analysis."""
    logger.info("=== Analysis D: Training-Free Proxy Generalization ===")

    # 1. Reproduce exp4 correlation matrix
    predictors = ["proxy_fanout", "proxy_correlation", "proxy_MI",
                   "proxy_entropy_reduction", "proxy_homophily",
                   "predict_jrn_probe_gbm", "predict_jrn_probe_mlp"]
    gt = df_exp4["jrn_gt_mean"].values

    proxy_correlations = {}
    for pred_col in predictors:
        vals = df_exp4[pred_col].values
        rho, pval = spearmanr(vals, gt)
        proxy_correlations[pred_col] = {"spearman_rho": float(rho), "p_value": float(pval)}
        logger.info(f"  {pred_col}: ρ={rho:.4f}, p={pval:.4f}")

    # 2. Fan-out vs MLP-probe JRN on rel-stack
    fanout_stack_corr = {"note": "not enough data"}
    try:
        if "heuristic_jrn" in df_exp2_p1.columns and len(df_exp2_p1) >= 3:
            rho_fan, p_fan = spearmanr(df_exp2_p1["heuristic_jrn"], df_exp2_p1["jrn"])
            fanout_stack_corr = {
                "spearman_rho": float(rho_fan),
                "p_value": float(p_fan),
                "n": int(len(df_exp2_p1)),
                "note": "heuristic JRN (fan-out-based) vs MLP-probe JRN",
            }
            logger.info(f"Fan-out heuristic vs MLP JRN on rel-stack: ρ={rho_fan:.4f}")
    except Exception:
        logger.exception("Fan-out vs MLP on rel-stack failed")

    # 3. Fan-out vs probe JRN on rel-hm (from exp3 analysis)
    fanout_hm_corr = {}
    try:
        analysis = exp3_data.get("analysis", {})
        jrn_decorr = analysis.get("jrn_decorrelation_from_fanout", {})
        for task_key, data in jrn_decorr.items():
            fanout_hm_corr[task_key] = {
                "spearman_rho": float(data.get("spearman_jrn_vs_fanout", 0)),
                "p_value": float(data.get("p_value", 1)),
                "n_buckets": int(data.get("n_buckets", 0)),
            }
    except Exception:
        logger.exception("Fan-out vs JRN on rel-hm failed")

    # 4. Cross-dataset fan-out consistency
    cross_dataset_fanout = {
        "rel_f1_fanout_vs_gt": proxy_correlations.get("proxy_fanout", {}),
        "rel_stack_heuristic_vs_mlp": fanout_stack_corr,
        "rel_hm_user_churn": fanout_hm_corr.get("user-churn", {}),
        "rel_hm_item_sales": fanout_hm_corr.get("item-sales", {}),
    }

    # 5. MI ratio comparison
    training_free = exp3_data.get("training_free", {})
    mi_comparison = {}
    if training_free:
        mi_uc = training_free.get("user-churn", {}).get("MI_ratio", 0)
        mi_is = training_free.get("item-sales", {}).get("MI_ratio", 0)
        global_jrn = exp3_data.get("global_jrn", {})
        jrn_uc = global_jrn.get("user-churn_J1_txn_to_customer", {}).get("JRN", 0)
        jrn_is = global_jrn.get("item-sales_J2_txn_to_article", {}).get("JRN", 0)
        mi_comparison = {
            "user_churn": {"MI_ratio": float(mi_uc), "JRN": float(jrn_uc)},
            "item_sales": {"MI_ratio": float(mi_is), "JRN": float(jrn_is)},
            "higher_MI_higher_JRN": bool(mi_uc > mi_is and jrn_uc > jrn_is),
            "note": "Higher MI_ratio corresponds to higher JRN for this pair",
        }

    # 6. Training-free proxy ranking
    proxy_ranking = []
    proxy_names = ["proxy_fanout", "proxy_correlation", "proxy_MI",
                    "proxy_entropy_reduction", "proxy_homophily"]
    for p in proxy_names:
        rho_val = abs(proxy_correlations.get(p, {}).get("spearman_rho", 0))
        proxy_ranking.append({"proxy": p, "abs_rho_rel_f1": rho_val})
    proxy_ranking.sort(key=lambda x: -x["abs_rho_rel_f1"])

    return {
        "rel_f1_proxy_correlations": proxy_correlations,
        "rel_stack_fanout_vs_mlp": fanout_stack_corr,
        "rel_hm_fanout_vs_jrn": fanout_hm_corr,
        "cross_dataset_fanout_summary": cross_dataset_fanout,
        "mi_ratio_comparison": mi_comparison,
        "proxy_ranking": proxy_ranking,
    }


# ═══════════════════════════════════════════════════════════════════════
# Analysis E: Compounding Robustness
# ═══════════════════════════════════════════════════════════════════════

def analysis_E(df_exp2_p3: pd.DataFrame) -> dict:
    """Compounding robustness analysis."""
    logger.info("=== Analysis E: Compounding Robustness ===")

    if len(df_exp2_p3) < 2:
        logger.warning("Too few chain measurements for compounding analysis")
        return {"error": "insufficient data", "n_chains": int(len(df_exp2_p3))}

    actual = df_exp2_p3["actual_chain_jrn"].values
    predicted = df_exp2_p3["predicted_chain_jrn"].values
    n = len(actual)

    # 1. Bootstrap CI on R²
    r2_orig = r2_score(actual, predicted)
    logger.info(f"Original R² = {r2_orig:.4f}")

    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    r2_boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a_b = actual[idx]
        p_b = predicted[idx]
        # R² can be undefined if all same value
        if np.std(a_b) > 0:
            r2_boot.append(r2_score(a_b, p_b))
    r2_boot = np.array(r2_boot)
    bootstrap_r2 = {
        "original": float(r2_orig),
        "mean": float(np.mean(r2_boot)),
        "ci_95": [float(np.percentile(r2_boot, 2.5)), float(np.percentile(r2_boot, 97.5))],
        "n_bootstrap": int(len(r2_boot)),
    }
    logger.info(f"Bootstrap R² = {np.mean(r2_boot):.4f} [{np.percentile(r2_boot, 2.5):.4f}, {np.percentile(r2_boot, 97.5):.4f}]")

    # 2. Systematic deviation by chain length
    by_chain_length = {}
    if "chain_depth" in df_exp2_p3.columns:
        for depth, group in df_exp2_p3.groupby("chain_depth"):
            deviations = np.abs(group["actual_chain_jrn"].values - group["predicted_chain_jrn"].values)
            by_chain_length[str(depth)] = {
                "n": int(len(group)),
                "mean_abs_deviation": float(np.mean(deviations)),
                "std_deviation": float(np.std(deviations, ddof=1)) if len(deviations) > 1 else 0.0,
            }

    # 3. Residual analysis
    residuals = actual - predicted
    residual_analysis = {}

    # 3a. Normality (Shapiro-Wilk)
    if len(residuals) >= 3:
        sw_stat, sw_p = shapiro(residuals)
        residual_analysis["shapiro_wilk"] = {"statistic": float(sw_stat), "p_value": float(sw_p),
                                               "is_normal": bool(sw_p > 0.05)}

    # 3b. Heteroscedasticity
    abs_res = np.abs(residuals)
    if len(abs_res) >= 3:
        rho_het, p_het = spearmanr(predicted, abs_res)
        residual_analysis["heteroscedasticity"] = {
            "spearman_rho_abs_residual_vs_predicted": float(rho_het),
            "p_value": float(p_het),
            "significant": bool(p_het < 0.05),
        }

    # 3c. Systematic bias (mean residual ≠ 0)
    mean_res = np.mean(residuals)
    if len(residuals) >= 2:
        t_stat, t_p = stats.ttest_1samp(residuals, 0)
        residual_analysis["systematic_bias"] = {
            "mean_residual": float(mean_res),
            "t_statistic": float(t_stat),
            "p_value": float(t_p),
            "has_bias": bool(t_p < 0.05),
        }

    # 4. Hub-node involvement
    hub_node_analysis = {}
    try:
        if "joins_in_chain" in df_exp2_p3.columns:
            has_posts_hub = []
            for _, row in df_exp2_p3.iterrows():
                joins = row.get("joins_in_chain", [])
                if isinstance(joins, str):
                    joins = [joins]
                involves_posts = any("posts" in str(j).lower() or "Posts" in str(j) for j in joins)
                has_posts_hub.append(involves_posts)
            df_exp2_p3 = df_exp2_p3.copy()
            df_exp2_p3["has_posts_hub"] = has_posts_hub

            hub_devs = np.abs(df_exp2_p3[df_exp2_p3["has_posts_hub"]]["actual_chain_jrn"].values -
                              df_exp2_p3[df_exp2_p3["has_posts_hub"]]["predicted_chain_jrn"].values)
            non_hub_devs = np.abs(df_exp2_p3[~df_exp2_p3["has_posts_hub"]]["actual_chain_jrn"].values -
                                   df_exp2_p3[~df_exp2_p3["has_posts_hub"]]["predicted_chain_jrn"].values)
            hub_node_analysis = {
                "n_with_hub": int(sum(has_posts_hub)),
                "n_without_hub": int(sum(not x for x in has_posts_hub)),
                "mean_deviation_with_hub": float(np.mean(hub_devs)) if len(hub_devs) > 0 else 0.0,
                "mean_deviation_without_hub": float(np.mean(non_hub_devs)) if len(non_hub_devs) > 0 else 0.0,
            }
    except Exception:
        logger.exception("Hub-node analysis failed")

    # 5. LOO cross-validation R²
    loo_predictions = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        # Simple: use slope from remaining points
        x_train = predicted[mask]
        y_train = actual[mask]
        if np.std(x_train) > 0:
            slope = np.cov(x_train, y_train)[0, 1] / np.var(x_train)
            intercept = np.mean(y_train) - slope * np.mean(x_train)
            loo_predictions[i] = slope * predicted[i] + intercept
        else:
            loo_predictions[i] = np.mean(y_train)

    loo_r2 = float(r2_score(actual, loo_predictions)) if np.std(actual) > 0 else 0.0
    logger.info(f"LOO R² = {loo_r2:.4f}")

    # ── Scatter plot ──
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(predicted, actual, c="royalblue", s=80, edgecolors="black", linewidth=0.5, alpha=0.8)
        lims = [min(predicted.min(), actual.min()) - 0.05, max(predicted.max(), actual.max()) + 0.05]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Identity (perfect compounding)")
        ax.set_xlabel("Predicted Chain JRN (product of individual JRNs)")
        ax.set_ylabel("Actual Chain JRN")
        ax.set_title(f"Multiplicative Compounding (R²={r2_orig:.3f})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / "E_compounding.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        logger.exception("Scatter plot failed for Analysis E")

    return {
        "n_chains": int(n),
        "bootstrap_r2": bootstrap_r2,
        "by_chain_length": by_chain_length,
        "residual_analysis": residual_analysis,
        "hub_node_analysis": hub_node_analysis,
        "loo_r2": float(loo_r2),
    }


# ═══════════════════════════════════════════════════════════════════════
# Analysis F: Metric Consistency Audit
# ═══════════════════════════════════════════════════════════════════════

def analysis_F(exp3_data: dict) -> dict:
    """Metric direction audit across experiments."""
    logger.info("=== Analysis F: Metric Consistency Audit ===")

    metric_inventory = [
        {
            "experiment": "exp1 (rel-f1)",
            "dataset": "rel-f1",
            "metric": "MAE for regression, R² for classification (via MLP probe)",
            "direction": "lower=better for MAE",
            "jrn_formula": "enriched_perf / baseline_perf",
            "correction_applied": "None documented — JRN>1 when enriched > baseline",
            "potential_issue": "For MAE tasks, JRN>1 means HIGHER error = WORSE. But exp1 uses a custom metric that may already account for this.",
        },
        {
            "experiment": "exp2 (rel-stack)",
            "dataset": "rel-stack",
            "metric": "neg_MAE",
            "direction": "higher=better (negated MAE)",
            "jrn_formula": "M_join / M_base (on neg_MAE)",
            "correction_applied": "YES — uses neg_MAE so JRN>1 = improvement",
            "potential_issue": "None — properly corrected",
        },
        {
            "experiment": "exp3 (rel-hm)",
            "dataset": "rel-hm",
            "metric": "AUROC for user-churn (higher=better), MAE for item-sales (lower=better)",
            "direction": "mixed",
            "jrn_formula": "join_metric / base_metric",
            "correction_applied": "None — raw ratio used",
            "potential_issue": "item-sales: base_MAE=8.94, join_MAE=5.59 → JRN=0.625. Lower MAE is BETTER, so join actually HELPS. But JRN<1 is reported as 'join hurts'. THIS IS A METRIC DIRECTION BUG.",
        },
        {
            "experiment": "exp4 (rel-f1)",
            "dataset": "rel-f1",
            "metric": "Ground-truth model performance (various)",
            "direction": "task-dependent",
            "jrn_formula": "M_join_gt / M_base_gt with direction correction",
            "correction_applied": "YES — explicit correction for metric direction",
            "potential_issue": "None — properly handled",
        },
    ]

    # Corrections needed
    corrections_needed = []
    corrected_values = {}

    # exp3 item-sales correction
    global_jrn = exp3_data.get("global_jrn", {})
    item_sales = global_jrn.get("item-sales_J2_txn_to_article", {})
    base_mae = item_sales.get("base_metric_mean", 8.943371)
    join_mae = item_sales.get("join_metric_mean", 5.59212)
    reported_jrn = item_sales.get("JRN", 0.625281)

    # For MAE (lower=better): corrected_JRN = base/join (so JRN>1 when join is better)
    corrected_jrn = base_mae / join_mae if join_mae != 0 else 0
    corrections_needed.append({
        "experiment": "exp3",
        "task": "item-sales",
        "join": "J2_txn_to_article",
        "reported_jrn": float(reported_jrn),
        "corrected_jrn": float(corrected_jrn),
        "reason": "MAE is lower=better; JRN = base_MAE/join_MAE to make JRN>1 = improvement",
        "base_metric": float(base_mae),
        "join_metric": float(join_mae),
        "original_interpretation": "join hurts (JRN < 1)",
        "corrected_interpretation": f"join HELPS (corrected JRN = {corrected_jrn:.3f} > 1)",
    })

    corrected_values["item_sales_corrected_jrn"] = float(corrected_jrn)

    # Also correct bucket-level JRN for item-sales
    bucket_corrections = {}
    bucket_jrn = exp3_data.get("bucket_jrn", {})
    for key, val in bucket_jrn.items():
        if val.get("task") == "item-sales":
            b_base = val.get("base_metric_mean", 0)
            b_join = val.get("join_metric_mean", 0)
            b_reported = val.get("JRN", 0)
            b_corrected = b_base / b_join if b_join != 0 else 0
            bucket_corrections[key] = {
                "reported_jrn": float(b_reported),
                "corrected_jrn": float(b_corrected),
                "base_mae": float(b_base),
                "join_mae": float(b_join),
            }
    corrected_values["item_sales_bucket_corrections"] = bucket_corrections

    # Count affected JRN values
    n_corrected = 1 + len(bucket_corrections)  # 1 global + bucket-level

    # Impact assessment
    impact = (
        f"Found {n_corrected} JRN values with metric direction issue (all in exp3 item-sales). "
        f"The item-sales global JRN changes from {reported_jrn:.3f} (interpreted as 'join hurts') "
        f"to {corrected_jrn:.3f} (join helps — MAE decreased from {base_mae:.2f} to {join_mae:.2f}). "
        f"This REVERSES the conclusion for item-sales: the join is beneficial, not harmful. "
        f"Both joins in rel-hm now show JRN>1 (user-churn=3.83, item-sales corrected={corrected_jrn:.2f}), "
        f"which changes the narrative from 'mixed results' to 'consistent improvement'."
    )

    return {
        "metric_inventory": metric_inventory,
        "corrections_needed": corrections_needed,
        "corrected_values": corrected_values,
        "n_values_affected": int(n_corrected),
        "impact_on_conclusions": impact,
    }


# ═══════════════════════════════════════════════════════════════════════
# Analysis G: Hypothesis Scorecard
# ═══════════════════════════════════════════════════════════════════════

def analysis_G(results_A: dict, results_B: dict, results_C: dict,
               results_D: dict, results_E: dict, results_F: dict) -> list:
    """Generate hypothesis scorecard."""
    logger.info("=== Analysis G: Hypothesis Scorecard ===")

    scorecard = []

    # 1. JRN is a meaningful per-join metric
    gbm_rho = results_C["gbm_vs_gt_spearman"]["rho"]
    mlp_rho = results_C["mlp_vs_gt_spearman"]["rho"]
    fanout_rho = results_D["rel_f1_proxy_correlations"].get("proxy_fanout", {}).get("spearman_rho", 0)

    if gbm_rho > 0.8:
        rating1 = "Moderate Support"
    elif gbm_rho > 0.6:
        rating1 = "Moderate Support"
    else:
        rating1 = "Inconclusive"

    scorecard.append({
        "prediction": "JRN is a meaningful per-join metric (probe-GT rho > 0.6)",
        "rating": rating1,
        "evidence_summary": (
            f"GBM probe achieves ρ={gbm_rho:.3f} (strong), but MLP probe fails with ρ={mlp_rho:.3f}. "
            f"This shows JRN CAN be measured accurately but is method-dependent, not universal. "
            f"Fan-out heuristic ρ={fanout_rho:.3f} provides a simpler alternative."
        ),
        "key_numbers": {
            "gbm_probe_rho": float(gbm_rho),
            "mlp_probe_rho": float(mlp_rho),
            "fanout_rho": float(fanout_rho),
        },
        "caveats": [
            "Only tested on rel-f1 dataset",
            "MLP probe (the originally proposed method) fails completely",
            "GBM success may depend on specific hyperparameters",
        ],
        "suggested_next_steps": [
            "Test GBM probes on rel-stack and rel-hm",
            "Ablation study on GBM hyperparameters",
            "Investigate why MLP fails (convergence, architecture, data)",
        ],
    })

    # 2. Inverted-U relationship
    scorecard.append({
        "prediction": "Inverted-U: aggregation innovation impact peaks near JRN≈1",
        "rating": "Inconclusive",
        "evidence_summary": (
            "Exp1 quadratic regression β₂=-0.82 with p=0.071 (marginal). "
            "Peak at JRN≈1.13. Only tested on one dataset (rel-f1) with MLP probes "
            "which have low correlation with ground truth (ρ=-0.067). "
            "The marginal p-value combined with unreliable MLP probes makes this inconclusive."
        ),
        "key_numbers": {
            "beta_2": -0.82,
            "p_value": 0.071,
            "peak_jrn": 1.13,
        },
        "caveats": [
            "Marginal p-value (0.071 > 0.05)",
            "Only one dataset",
            "MLP probes unreliable (ρ=-0.067 with GT)",
            "Quadratic fit may be spurious",
        ],
        "suggested_next_steps": [
            "Repeat with GBM-probe JRN values",
            "Test on rel-stack and rel-hm",
            "Use ground-truth JRN from exp4 instead of MLP probe",
        ],
    })

    # 3. Multiplicative compounding
    r2_val = results_E.get("bootstrap_r2", {}).get("original", 0)
    r2_ci = results_E.get("bootstrap_r2", {}).get("ci_95", [0, 0])
    loo = results_E.get("loo_r2", 0)
    n_chains = results_E.get("n_chains", 0)

    if r2_val > 0.7 and r2_ci[0] > 0.5:
        rating3 = "Moderate Support"
    elif r2_val > 0.5:
        rating3 = "Moderate Support"
    else:
        rating3 = "Inconclusive"

    scorecard.append({
        "prediction": "Multiplicative compounding holds (R² > 0.5)",
        "rating": rating3,
        "evidence_summary": (
            f"R²={r2_val:.3f} on {n_chains} chains from rel-stack. "
            f"Bootstrap 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]. "
            f"LOO-CV R²={loo:.3f}. Strong R² but small sample and single dataset."
        ),
        "key_numbers": {
            "r_squared": float(r2_val),
            "bootstrap_ci_lower": float(r2_ci[0]),
            "bootstrap_ci_upper": float(r2_ci[1]),
            "loo_r2": float(loo),
            "n_chains": int(n_chains),
        },
        "caveats": [
            f"Only {n_chains} chains (small sample)",
            "Single dataset (rel-stack)",
            "Independence assumption may be violated for hub-node chains",
        ],
        "suggested_next_steps": [
            "Test compounding on rel-f1 multi-hop joins",
            "Test on rel-hm chain paths",
            "Increase chain diversity and depth",
        ],
    })

    # 4. JRN is join-intrinsic (stable across tasks)
    w_f1 = results_B["kendalls_W_rel_f1"].get("W", 0)
    w_stack = results_B["kendalls_W_rel_stack"].get("W", 0)
    p_f1 = results_B["kendalls_W_rel_f1"].get("p_value", 1)
    p_stack = results_B["kendalls_W_rel_stack"].get("p_value", 1)

    # Weight by dataset reliability: rel-f1 (13 joins × 5 tasks) vs rel-stack (5 × 2)
    # Require both datasets to show concordance, or significance on at least one
    if w_f1 > 0.7 and w_stack > 0.7:
        rating4 = "Strong Support"
    elif w_f1 > 0.5 and w_stack > 0.5:
        rating4 = "Moderate Support"
    elif (w_f1 > 0.5 and p_f1 < 0.05) or (w_stack > 0.5 and p_stack < 0.05):
        rating4 = "Moderate Support"
    elif w_f1 > 0.3 or w_stack > 0.3:
        rating4 = "Inconclusive"
    else:
        rating4 = "Moderate Disconfirmation"

    scorecard.append({
        "prediction": "JRN is join-intrinsic (stable across tasks)",
        "rating": rating4,
        "evidence_summary": (
            f"Kendall's W on rel-f1: W={w_f1:.3f} (p={p_f1:.4f}). "
            f"Kendall's W on rel-stack: W={w_stack:.3f} (p={p_stack:.4f}). "
            f"{results_B['kendalls_W_rel_f1'].get('interpretation', 'Unknown')}."
        ),
        "key_numbers": {
            "kendalls_W_rel_f1": float(w_f1),
            "kendalls_W_rel_stack": float(w_stack),
            "p_value_rel_f1": float(p_f1),
            "p_value_rel_stack": float(p_stack),
        },
        "caveats": [
            "Using MLP-probe JRN which may not reflect true JRN",
            "rel-stack user-entity joins have only 2 user tasks",
            "Missing complete task coverage for some joins",
        ],
        "suggested_next_steps": [
            "Repeat with GBM-probe or GT JRN",
            "Add more diverse tasks per dataset",
            "Test cross-dataset join stability",
        ],
    })

    # 5. Training-free proxies can substitute for probes
    entropy_rho = abs(results_D["rel_f1_proxy_correlations"].get("proxy_entropy_reduction", {}).get("spearman_rho", 0))

    scorecard.append({
        "prediction": "Training-free proxies can substitute for probes",
        "rating": "Inconclusive",
        "evidence_summary": (
            f"Conditional entropy achieves ρ={entropy_rho:.3f} on rel-f1 (strong), "
            f"but only tested on one dataset. Fan-out shows mixed results across datasets "
            f"(rel-f1: ρ={fanout_rho:.3f}, rel-hm user-churn: ρ=0.94, rel-hm item-sales: ρ=-0.43). "
            f"Cannot confirm generalization without more cross-dataset testing."
        ),
        "key_numbers": {
            "entropy_reduction_rho_rel_f1": float(entropy_rho),
            "fanout_rho_rel_f1": float(fanout_rho),
        },
        "caveats": [
            "Conditional entropy only tested on rel-f1",
            "Fan-out correlation varies wildly across datasets",
            "No cross-dataset validation of best proxy (entropy)",
        ],
        "suggested_next_steps": [
            "Compute conditional entropy on rel-stack and rel-hm",
            "Test MI ratio as proxy across all datasets",
            "Compare proxy rankings across datasets",
        ],
    })

    # 6. JRN-guided architecture
    scorecard.append({
        "prediction": "JRN-guided architecture outperforms uniform",
        "rating": "Not Yet Testable",
        "evidence_summary": (
            "Phase 4 (JRN-guided architecture selection) has not been implemented. "
            "No experimental evidence available to test this prediction."
        ),
        "key_numbers": {},
        "caveats": [
            "Requires Phase 4 implementation",
            "Would need comparison with uniform architecture baselines",
        ],
        "suggested_next_steps": [
            "Implement JRN-guided aggregation strategy selection",
            "Compare JRN-guided vs uniform on all 3 datasets",
            "Test on downstream task performance",
        ],
    })

    return scorecard


# ═══════════════════════════════════════════════════════════════════════
# Overall Assessment
# ═══════════════════════════════════════════════════════════════════════

def overall_assessment(scorecard: list, results: dict) -> dict:
    """Generate overall assessment of JRN hypothesis."""
    ratings = [s["rating"] for s in scorecard if s["rating"] != "Not Yet Testable"]
    support_count = sum(1 for r in ratings if "Support" in r)
    disconfirm_count = sum(1 for r in ratings if "Disconfirmation" in r)
    inconclusive_count = sum(1 for r in ratings if r == "Inconclusive")

    if support_count >= 3 and disconfirm_count == 0:
        viability = "Promising — multiple predictions supported, no disconfirmation"
    elif support_count >= 2:
        viability = "Cautiously promising — some support but gaps remain"
    elif disconfirm_count >= 2:
        viability = "Concerning — multiple predictions disconfirmed"
    else:
        viability = "Uncertain — mostly inconclusive evidence"

    return {
        "hypothesis_viability": viability,
        "ratings_summary": {
            "strong_support": sum(1 for r in ratings if r == "Strong Support"),
            "moderate_support": sum(1 for r in ratings if r == "Moderate Support"),
            "inconclusive": inconclusive_count,
            "moderate_disconfirmation": sum(1 for r in ratings if r == "Moderate Disconfirmation"),
            "strong_disconfirmation": sum(1 for r in ratings if r == "Strong Disconfirmation"),
            "not_yet_testable": sum(1 for s in scorecard if s["rating"] == "Not Yet Testable"),
        },
        "strongest_evidence": (
            "GBM probe-GT correlation (ρ=0.960) demonstrates that JRN is a valid, measurable quantity. "
            "Multiplicative compounding R²=0.83 on rel-stack shows composability."
        ),
        "weakest_link": (
            "MLP probe failure (ρ=-0.067) undermines the 'lightweight probe' claim. "
            "The inverted-U relationship is only marginally significant and based on unreliable MLP probes. "
            "Metric direction bug in exp3 item-sales affects cross-dataset comparisons."
        ),
        "critical_next_experiments": [
            "1. Validate GBM probes on rel-stack and rel-hm (currently only tested on rel-f1)",
            "2. Diagnose MLP probe failure — test convergence, architecture, hyperparameters",
            "3. Test conditional entropy proxy across all 3 datasets",
            "4. Fix metric direction in rel-hm and re-run fan-out bucket analysis",
            "5. Implement Phase 4: JRN-guided architecture selection vs uniform baseline",
            "6. Increase sample sizes — 14 chains and 19 join-task pairs limit statistical power",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# Format output for schema compliance
# ═══════════════════════════════════════════════════════════════════════

def format_schema_output(all_results: dict) -> dict:
    """Format output to comply with exp_eval_sol_out.json schema."""

    # Compute aggregate metrics
    metrics_agg = {}

    # From Analysis A
    a = all_results["analysis_A"]
    metrics_agg["total_jrn_measurements"] = a["total_jrn_measurements"]
    metrics_agg["pooled_mlp_jrn_mean"] = a["pooled_stats"]["mean"]
    metrics_agg["pooled_mlp_jrn_std"] = a["pooled_stats"]["std"]
    metrics_agg["dip_test_pooled_p_value"] = a["dip_test_pooled_mlp"]["p_value"]
    metrics_agg["fraction_near_threshold_mlp"] = a["fraction_near_threshold_mlp"]

    # From Analysis B
    b = all_results["analysis_B"]
    metrics_agg["kendalls_W_rel_f1"] = b["kendalls_W_rel_f1"]["W"]
    metrics_agg["kendalls_W_rel_stack"] = b["kendalls_W_rel_stack"]["W"]

    # From Analysis C
    c = all_results["analysis_C"]
    metrics_agg["gbm_probe_spearman_rho"] = c["gbm_vs_gt_spearman"]["rho"]
    metrics_agg["mlp_probe_spearman_rho"] = c["mlp_vs_gt_spearman"]["rho"]
    metrics_agg["gbm_probe_p_value"] = c["gbm_vs_gt_spearman"]["p_value"]
    metrics_agg["mlp_probe_p_value"] = c["mlp_vs_gt_spearman"]["p_value"]

    # From Analysis D
    d = all_results["analysis_D"]
    entropy_rho = d["rel_f1_proxy_correlations"].get("proxy_entropy_reduction", {}).get("spearman_rho", 0)
    metrics_agg["entropy_proxy_spearman_rho"] = entropy_rho
    fanout_rho = d["rel_f1_proxy_correlations"].get("proxy_fanout", {}).get("spearman_rho", 0)
    metrics_agg["fanout_proxy_spearman_rho"] = fanout_rho

    # From Analysis E
    e = all_results["analysis_E"]
    metrics_agg["compounding_r_squared"] = e.get("bootstrap_r2", {}).get("original", 0)
    metrics_agg["compounding_r2_ci_lower"] = e.get("bootstrap_r2", {}).get("ci_95", [0, 0])[0]
    metrics_agg["compounding_r2_ci_upper"] = e.get("bootstrap_r2", {}).get("ci_95", [0, 0])[1]
    metrics_agg["loo_r_squared"] = e.get("loo_r2", 0)

    # From Analysis F
    f = all_results["analysis_F"]
    metrics_agg["n_metric_corrections_needed"] = f["n_values_affected"]

    # Scorecard summary
    g = all_results["analysis_G"]
    n_support = sum(1 for s in g if "Support" in s["rating"])
    n_inconclusive = sum(1 for s in g if s["rating"] == "Inconclusive")
    n_disconfirm = sum(1 for s in g if "Disconfirmation" in s["rating"])
    metrics_agg["scorecard_n_supported"] = n_support
    metrics_agg["scorecard_n_inconclusive"] = n_inconclusive
    metrics_agg["scorecard_n_disconfirmed"] = n_disconfirm

    # Build examples for each analysis
    examples = []

    # One example per analysis
    analysis_names = [
        ("A", "JRN Distribution & Threshold Testing", "analysis_A"),
        ("B", "Task Stability (Kendall's W)", "analysis_B"),
        ("C", "Probe Type Comparison (MLP vs GBM)", "analysis_C"),
        ("D", "Training-Free Proxy Generalization", "analysis_D"),
        ("E", "Compounding Robustness", "analysis_E"),
        ("F", "Metric Consistency Audit", "analysis_F"),
        ("G", "Hypothesis Scorecard", "analysis_G"),
    ]

    # Map analysis code to a key numeric result for eval_ field
    eval_key_metrics = {
        "A": ("eval_dip_test_p_value", a.get("dip_test_pooled_mlp", {}).get("p_value", 0)),
        "B": ("eval_kendalls_W_rel_f1", b.get("kendalls_W_rel_f1", {}).get("W", 0)),
        "C": ("eval_gbm_spearman_rho", c.get("gbm_vs_gt_spearman", {}).get("rho", 0)),
        "D": ("eval_entropy_proxy_rho", d.get("rel_f1_proxy_correlations", {}).get("proxy_entropy_reduction", {}).get("spearman_rho", 0)),
        "E": ("eval_compounding_r2", e.get("bootstrap_r2", {}).get("original", 0)),
        "F": ("eval_n_corrections_needed", f.get("n_values_affected", 0)),
        "G": ("eval_n_predictions_supported", n_support),
    }

    for code, name, key in analysis_names:
        data = all_results[key]
        data_str = json.dumps(data, default=str)
        # Truncate for schema compliance if needed
        if len(data_str) > 50000:
            data_str = data_str[:49990] + "...(truncated)"
        eval_field_name, eval_field_val = eval_key_metrics.get(code, ("eval_score", 0))
        examples.append({
            "input": f"Analysis {code}: {name}",
            "output": data_str,
            "metadata_analysis_code": code,
            "metadata_analysis_name": name,
            eval_field_name: float(eval_field_val),
        })

    # Add overall assessment as an example
    oa = all_results["overall_assessment"]
    examples.append({
        "input": "Overall Assessment: JRN Hypothesis Viability",
        "output": json.dumps(oa, default=str),
        "metadata_analysis_code": "overall",
        "metadata_analysis_name": "Overall Assessment",
        "eval_n_predictions_supported": float(n_support),
    })

    return {
        "metadata": all_results.get("metadata", {}),
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "cross_dataset_jrn_consolidation",
                "examples": examples,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("Starting Cross-Dataset JRN Consolidation Evaluation")

    # Step 0: Load data
    experiments = load_experiments()
    df_exp1 = parse_exp1(experiments["exp1"])
    df_exp2_p1, df_exp2_p3 = parse_exp2(experiments["exp2"])
    exp3_data = parse_exp3(experiments["exp3"])
    df_exp4 = parse_exp4(experiments["exp4"])

    # Count total JRN measurements
    total_jrn = len(df_exp1) + len(df_exp2_p1) + 2 + len(exp3_data.get("bucket_jrn", {})) + len(df_exp4) * 3
    logger.info(f"Total JRN measurements across all experiments: ~{total_jrn}")

    # Run analyses
    results_A = analysis_A(df_exp1, df_exp2_p1, exp3_data, df_exp4)
    results_B = analysis_B(df_exp1, df_exp2_p1)
    results_C = analysis_C(df_exp4, df_exp1)
    results_D = analysis_D(df_exp4, df_exp2_p1, exp3_data)
    results_E = analysis_E(df_exp2_p3)
    results_F = analysis_F(exp3_data)
    results_G = analysis_G(results_A, results_B, results_C, results_D, results_E, results_F)
    assessment = overall_assessment(results_G, {
        "A": results_A, "B": results_B, "C": results_C,
        "D": results_D, "E": results_E, "F": results_F,
    })

    # Assemble full results
    all_results = {
        "metadata": {
            "title": "Cross-Dataset JRN Consolidation",
            "n_experiments_consolidated": 4,
            "n_datasets": 3,
            "total_jrn_measurements": results_A["total_jrn_measurements"],
            "analysis_date": "2026-03-09",
        },
        "analysis_A": results_A,
        "analysis_B": results_B,
        "analysis_C": results_C,
        "analysis_D": results_D,
        "analysis_E": results_E,
        "analysis_F": results_F,
        "analysis_G": results_G,
        "overall_assessment": assessment,
    }

    # Format for schema compliance
    schema_output = format_schema_output(all_results)

    # Save outputs
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(schema_output, indent=2, default=str))
    logger.info(f"Saved schema-compliant output to {out_path}")

    # Also save full results for reference
    full_path = WORKSPACE / "eval_full_results.json"
    full_path.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"Saved full results to {full_path}")

    logger.info("Evaluation complete!")
    logger.info(f"Key metrics: GBM ρ={schema_output['metrics_agg']['gbm_probe_spearman_rho']:.3f}, "
                f"MLP ρ={schema_output['metrics_agg']['mlp_probe_spearman_rho']:.3f}, "
                f"Compounding R²={schema_output['metrics_agg']['compounding_r_squared']:.3f}, "
                f"Kendall's W (f1)={schema_output['metrics_agg']['kendalls_W_rel_f1']:.3f}")


if __name__ == "__main__":
    main()
