#!/usr/bin/env python3
"""Negative-Result Diagnostic: Why Threshold and Multiplicative Compounding Predictions Failed.

Systematic post-hoc analysis of two key hypothesis failures:
1. Inverted-U threshold prediction for aggregation sensitivity
2. Multiplicative compounding model for chain JRN

Uses data from three prior experiments (exp_id1_it3, exp_id1_it2, exp_id2_it6).
"""

import json
import math
import os
import resource
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
from scipy import stats
from scipy.optimize import minimize

from loguru import logger

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection & memory limits
# ---------------------------------------------------------------------------

def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

# Set memory limit — this evaluation is lightweight (~200MB)
RAM_BUDGET = int(4 * 1024**3)  # 4 GB
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
DEP_IT3 = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_3/gen_art/exp_id1_it3__opus")
DEP_IT2 = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus")
DEP_IT6 = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_6/gen_art/exp_id2_it6__opus")

MAX_EXAMPLES = None  # Set to int for testing, None for full run


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval. Handles 1D and 2D arrays by resampling rows."""
    rng = np.random.RandomState(42)
    data = np.asarray(data)
    n = data.shape[0]
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = data[idx]
        try:
            boot_stats.append(stat_fn(sample))
        except Exception:
            pass
    boot_stats = np.array(boot_stats)
    if len(boot_stats) == 0:
        return 0.0, 0.0
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, alpha * 100)
    hi = np.percentile(boot_stats, (1 - alpha) * 100)
    return float(lo), float(hi)


def safe_spearman(x, y):
    """Spearman correlation with safety checks."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return 0.0, 1.0
    # Check for constant arrays (would produce NaN)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    rho, p = stats.spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(p)


def entropy(counts_dict: dict) -> float:
    """Shannon entropy of a distribution from a count dict."""
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in counts_dict.values() if v > 0]
    return float(-sum(p * np.log2(p) for p in probs))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    logger.info(f"Loading {path.name} from {path.parent.name}")
    data = json.loads(path.read_text())
    return data


@logger.catch
def main():
    # Load all three dependency datasets
    data_it3 = load_json(DEP_IT3 / "full_method_out.json")
    data_it2 = load_json(DEP_IT2 / "full_method_out.json")
    data_it6 = load_json(DEP_IT6 / "full_method_out.json")

    # Extract examples
    examples_it3 = data_it3["datasets"][0]["examples"]
    examples_it2 = data_it2["datasets"][0]["examples"]
    examples_it6 = data_it6["datasets"][0]["examples"]

    if MAX_EXAMPLES is not None:
        examples_it3 = examples_it3[:MAX_EXAMPLES]
        examples_it2 = examples_it2[:MAX_EXAMPLES]
        examples_it6 = examples_it6[:MAX_EXAMPLES]

    logger.info(f"exp_id1_it3: {len(examples_it3)} examples (GBM probes)")
    logger.info(f"exp_id1_it2: {len(examples_it2)} examples (MLP probes)")
    logger.info(f"exp_id2_it6: {len(examples_it6)} examples (compounding)")

    # Metadata from it3
    meta_it3 = data_it3["metadata"]
    per_join_summary = meta_it3["per_join_summary"]
    winning_agg_by_bin = meta_it3["winning_agg_by_jrn_bin"]

    # -------------------------------------------------------------------
    # Parse exp_id1_it3 (GBM, 65 pairs, 5 agg strategies)
    # -------------------------------------------------------------------
    gbm_pairs = []
    for ex in examples_it3:
        out = json.loads(ex["output"])
        inp = json.loads(ex["input"])
        agg_perfs = out.get("agg_perfs", {})
        if not agg_perfs:
            continue
        perf_values = [v for v in agg_perfs.values() if isinstance(v, (int, float))]
        if len(perf_values) < 2:
            cov = 0.0
        else:
            mean_perf = np.mean(perf_values)
            std_perf = np.std(perf_values, ddof=0)
            cov = float(std_perf / mean_perf) if mean_perf > 0 else 0.0
        perf_range = max(perf_values) - min(perf_values) if perf_values else 0.0
        winning_agg = max(agg_perfs, key=agg_perfs.get)
        gbm_pairs.append({
            "join_idx": int(inp.get("join_idx", ex.get("metadata_join_idx", 0))),
            "join_name": inp.get("join_name", ""),
            "task_name": inp.get("task_name", ex.get("metadata_task_name", "")),
            "jrn": float(out.get("jrn", 0)),
            "sensitivity_cov": cov,
            "perf_range": float(perf_range),
            "best_agg_perf": float(out.get("best_agg_perf", 0)),
            "winning_agg": winning_agg,
            "agg_perfs": agg_perfs,
            "gt_jrn": float(out.get("gt_jrn", 0)),
        })
    logger.info(f"Parsed {len(gbm_pairs)} GBM join-task pairs from exp_id1_it3")

    # Build join_idx -> fanout_mean mapping from per_join_summary
    join_fanout = {js["join_idx"]: js["fanout_mean"] for js in per_join_summary}
    join_name_map = {js["join_idx"]: js["join_name"] for js in per_join_summary}

    # -------------------------------------------------------------------
    # Parse exp_id1_it2 (MLP, 50 pairs, 4 agg strategies)
    # -------------------------------------------------------------------
    mlp_pairs = []
    for ex in examples_it2:
        inp_raw = ex.get("input", "")
        try:
            inp = json.loads(inp_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if inp.get("analysis_type") == "summary":
            continue
        out_raw = ex.get("output", "")
        try:
            out = json.loads(out_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        agg_perfs_raw = out.get("agg_performances", {})
        # agg_performances has nested structure: {"mean": {"mean": val, "std": val}, ...}
        agg_perfs = {}
        for agg_name, perf_dict in agg_perfs_raw.items():
            if isinstance(perf_dict, dict):
                agg_perfs[agg_name] = perf_dict.get("mean", 0.0)
            else:
                agg_perfs[agg_name] = float(perf_dict)
        perf_values = [v for v in agg_perfs.values() if isinstance(v, (int, float))]
        if len(perf_values) < 2:
            cov = 0.0
        else:
            mean_perf = np.mean(perf_values)
            std_perf = np.std(perf_values, ddof=0)
            cov = float(std_perf / mean_perf) if mean_perf > 0 else 0.0
        mlp_pairs.append({
            "join_idx": int(inp.get("join_idx", ex.get("metadata_join_idx", 0))),
            "task_name": inp.get("task_name", ex.get("metadata_task_name", "")),
            "jrn": float(out.get("jrn", 0)),
            "sensitivity_cov": cov,
            "agg_sensitivity": float(out.get("agg_sensitivity", 0)),
            "fanout_mean": float(inp.get("fanout_mean", 0)),
            "num_edges": int(inp.get("num_edges", 0)),
        })
    logger.info(f"Parsed {len(mlp_pairs)} MLP join-task pairs from exp_id1_it2")

    # -------------------------------------------------------------------
    # Parse exp_id2_it6 (chains, compounding)
    # -------------------------------------------------------------------
    chain_examples = []
    individual_jrn_map = {}  # (join_idx, probe) -> jrn_mean
    compounding_summaries = []

    for ex in examples_it6:
        mt = ex.get("metadata_measurement_type", "")
        if mt == "individual_jrn":
            key = (ex["metadata_join_idx"], ex["metadata_probe"])
            individual_jrn_map[key] = ex["metadata_jrn_mean"]
        elif mt == "chain_jrn":
            chain_examples.append({
                "chain_id": ex["metadata_chain_id"],
                "chain_desc": ex.get("metadata_chain_desc", ""),
                "task": ex["metadata_task"],
                "probe": ex["metadata_probe"],
                "end_table": ex.get("metadata_end_table", ""),
                "measured_jrn": float(ex["metadata_measured_jrn"]),
                "onehop_jrn": float(ex["metadata_onehop_jrn"]),
                "predict_multiplicative": float(ex.get("predict_multiplicative", 0)),
                "predict_additive": float(ex.get("predict_additive", 0)),
                "predict_bottleneck": float(ex.get("predict_bottleneck", 0)),
            })
        elif mt == "compounding_summary":
            compounding_summaries.append({
                "model": ex["metadata_model"],
                "probe": ex["metadata_probe"],
                "r2": float(ex["metadata_r2"]),
                "spearman_r": float(ex["metadata_spearman_r"]),
                "mae": float(ex["metadata_mae"]),
                "rmse": float(ex["metadata_rmse"]),
            })

    logger.info(f"Parsed {len(chain_examples)} chain examples, {len(individual_jrn_map)} individual JRNs, {len(compounding_summaries)} compounding summaries")

    # ===================================================================
    # PART 1: Threshold Failure Analysis
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PART 1: Threshold Failure Analysis (GBM data)")
    logger.info("=" * 60)

    metrics_agg = {}
    eval_examples = []

    # ----- M1: Aggregation Sensitivity Distribution Statistics -----
    logger.info("Computing M1: Aggregation Sensitivity Distribution Statistics")
    covs = np.array([p["sensitivity_cov"] for p in gbm_pairs])
    ranges_arr = np.array([p["perf_range"] for p in gbm_pairs])

    m1_mean_cov = float(np.mean(covs))
    m1_median_cov = float(np.median(covs))
    m1_std_cov = float(np.std(covs, ddof=1)) if len(covs) > 1 else 0.0
    m1_pct_insensitive = float(np.mean(covs < 0.05) * 100)
    m1_pct_sensitive = float(np.mean(covs > 0.15) * 100)
    m1_mean_range = float(np.mean(ranges_arr))
    m1_max_range = float(np.max(ranges_arr)) if len(ranges_arr) > 0 else 0.0

    metrics_agg["M1_mean_cov"] = round(m1_mean_cov, 6)
    metrics_agg["M1_median_cov"] = round(m1_median_cov, 6)
    metrics_agg["M1_std_cov"] = round(m1_std_cov, 6)
    metrics_agg["M1_pct_insensitive_lt005"] = round(m1_pct_insensitive, 2)
    metrics_agg["M1_pct_sensitive_gt015"] = round(m1_pct_sensitive, 2)
    metrics_agg["M1_mean_perf_range"] = round(m1_mean_range, 6)
    metrics_agg["M1_max_perf_range"] = round(m1_max_range, 6)
    metrics_agg["M1_n_pairs"] = len(gbm_pairs)

    logger.info(f"  Mean CoV: {m1_mean_cov:.4f}, Median: {m1_median_cov:.4f}, Std: {m1_std_cov:.4f}")
    logger.info(f"  Insensitive (<0.05): {m1_pct_insensitive:.1f}%, Sensitive (>0.15): {m1_pct_sensitive:.1f}%")

    # ----- M2: Uniformity Test -----
    logger.info("Computing M2: KS Uniformity Test on CoV distribution")
    # Test if CoV is concentrated at zero vs uniform
    # KS test against uniform(0, max(covs))
    if np.max(covs) > 0:
        scaled_covs = covs / np.max(covs)
        ks_stat, ks_pvalue = stats.kstest(scaled_covs, 'uniform', args=(0, 1))
    else:
        ks_stat, ks_pvalue = 0.0, 1.0

    # Also test if concentrated at zero — one-sample test vs exponential
    if np.mean(covs) > 0:
        ks_exp_stat, ks_exp_pvalue = stats.kstest(covs, 'expon', args=(0, np.mean(covs)))
    else:
        ks_exp_stat, ks_exp_pvalue = 0.0, 1.0

    metrics_agg["M2_ks_uniform_stat"] = round(float(ks_stat), 6)
    metrics_agg["M2_ks_uniform_pvalue"] = round(float(ks_pvalue), 6)
    metrics_agg["M2_ks_expon_stat"] = round(float(ks_exp_stat), 6)
    metrics_agg["M2_ks_expon_pvalue"] = round(float(ks_exp_pvalue), 6)
    metrics_agg["M2_all_insensitive"] = 1 if m1_mean_cov < 0.05 else 0

    logger.info(f"  KS uniform: stat={ks_stat:.4f}, p={ks_pvalue:.4f}")
    logger.info(f"  KS expon: stat={ks_exp_stat:.4f}, p={ks_exp_pvalue:.4f}")
    logger.info(f"  All insensitive (mean CoV < 0.05): {m1_mean_cov < 0.05}")

    # ----- M3: Sensitivity vs Schema Properties Correlations -----
    logger.info("Computing M3: Sensitivity vs Schema Properties Correlations")

    jrns = np.array([p["jrn"] for p in gbm_pairs])
    fanouts = np.array([join_fanout.get(p["join_idx"], 0) for p in gbm_pairs])
    log_fanouts = np.log1p(fanouts)

    rho_cov_fanout, p_cov_fanout = safe_spearman(covs, fanouts)
    rho_cov_logfanout, p_cov_logfanout = safe_spearman(covs, log_fanouts)
    rho_cov_jrn, p_cov_jrn = safe_spearman(covs, jrns)

    metrics_agg["M3_rho_cov_vs_fanout"] = round(rho_cov_fanout, 6)
    metrics_agg["M3_p_cov_vs_fanout"] = round(p_cov_fanout, 6)
    metrics_agg["M3_rho_cov_vs_log_fanout"] = round(rho_cov_logfanout, 6)
    metrics_agg["M3_p_cov_vs_log_fanout"] = round(p_cov_logfanout, 6)
    metrics_agg["M3_rho_cov_vs_jrn"] = round(rho_cov_jrn, 6)
    metrics_agg["M3_p_cov_vs_jrn"] = round(p_cov_jrn, 6)

    logger.info(f"  CoV vs fanout: rho={rho_cov_fanout:.4f}, p={p_cov_fanout:.4f}")
    logger.info(f"  CoV vs log(fanout): rho={rho_cov_logfanout:.4f}, p={p_cov_logfanout:.4f}")
    logger.info(f"  CoV vs JRN: rho={rho_cov_jrn:.4f}, p={p_cov_jrn:.4f}")

    # Multiple regression: CoV ~ fanout + log(fanout) + JRN + JRN^2
    try:
        X = np.column_stack([fanouts, log_fanouts, jrns, jrns**2])
        X = np.column_stack([np.ones(len(X)), X])
        y = covs
        # OLS via pseudo-inverse
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        metrics_agg["M3_multiple_regression_r2"] = round(float(r2_multi), 6)
        metrics_agg["M3_beta_fanout"] = round(float(beta[1]), 6)
        metrics_agg["M3_beta_log_fanout"] = round(float(beta[2]), 6)
        metrics_agg["M3_beta_jrn"] = round(float(beta[3]), 6)
        metrics_agg["M3_beta_jrn_sq"] = round(float(beta[4]), 6)
        logger.info(f"  Multiple regression R²={r2_multi:.4f}")
        logger.info(f"  Betas: fanout={beta[1]:.6f}, log_fanout={beta[2]:.6f}, JRN={beta[3]:.6f}, JRN²={beta[4]:.6f}")
    except Exception:
        logger.exception("Multiple regression failed")
        metrics_agg["M3_multiple_regression_r2"] = 0.0

    # ----- M4: Fan-out Heterogeneity Test -----
    logger.info("Computing M4: Fan-out Heterogeneity Test")

    # For each join, compute max_fanout/mean_fanout ratio from per_join_summary
    # We don't have per-row fanout data, but we can use mean_sensitivity and fanout_mean
    # As proxy: use fanout_mean itself (higher fanout → more heterogeneity possible)
    # Correlate with performance gap between best non-mean strategy and mean
    m4_rel_gaps = []
    m4_fanouts = []
    for p in gbm_pairs:
        agg_perfs = p["agg_perfs"]
        all_vals = list(agg_perfs.values())
        avg_perf = np.mean(all_vals) if all_vals else 1.0
        mean_perf = agg_perfs.get("mean", 0)
        non_mean_perfs = {k: v for k, v in agg_perfs.items() if k != "mean"}
        if non_mean_perfs and abs(avg_perf) > 1e-12:
            best_non_mean = max(non_mean_perfs.values())
            rel_gap = (best_non_mean - mean_perf) / abs(avg_perf)
        else:
            rel_gap = 0.0
        m4_rel_gaps.append(rel_gap)
        m4_fanouts.append(join_fanout.get(p["join_idx"], 0))

    m4_rel_gaps = np.array(m4_rel_gaps)
    m4_fanouts = np.array(m4_fanouts)
    rho_gap_fanout, p_gap_fanout = safe_spearman(m4_rel_gaps, m4_fanouts)

    # Also use all_combined vs mean relative gap
    m4_allcomb_rel_gaps = []
    for p in gbm_pairs:
        agg_perfs = p["agg_perfs"]
        all_vals = list(agg_perfs.values())
        avg_perf = np.mean(all_vals) if all_vals else 1.0
        mean_perf = agg_perfs.get("mean", 0)
        allcomb_perf = agg_perfs.get("all_combined", mean_perf)
        if abs(avg_perf) > 1e-12:
            m4_allcomb_rel_gaps.append((allcomb_perf - mean_perf) / abs(avg_perf))
        else:
            m4_allcomb_rel_gaps.append(0.0)
    m4_allcomb_rel_gaps = np.array(m4_allcomb_rel_gaps)
    rho_allcomb_fanout, p_allcomb_fanout = safe_spearman(m4_allcomb_rel_gaps, m4_fanouts)

    metrics_agg["M4_rho_rel_gap_vs_fanout"] = round(rho_gap_fanout, 6)
    metrics_agg["M4_p_rel_gap_vs_fanout"] = round(p_gap_fanout, 6)
    metrics_agg["M4_rho_allcomb_rel_gap_vs_fanout"] = round(rho_allcomb_fanout, 6)
    metrics_agg["M4_p_allcomb_rel_gap_vs_fanout"] = round(p_allcomb_fanout, 6)
    metrics_agg["M4_mean_rel_gap_best_vs_mean"] = round(float(np.mean(m4_rel_gaps)), 6)
    metrics_agg["M4_pct_mean_is_best"] = round(float(np.mean(m4_rel_gaps <= 0) * 100), 2)

    logger.info(f"  RelGap(best_non_mean - mean)/avg vs fanout: rho={rho_gap_fanout:.4f}, p={p_gap_fanout:.4f}")
    logger.info(f"  RelGap(all_combined - mean)/avg vs fanout: rho={rho_allcomb_fanout:.4f}, p={p_allcomb_fanout:.4f}")
    logger.info(f"  Mean relative gap: {np.mean(m4_rel_gaps):.4f}, % where mean is best: {np.mean(m4_rel_gaps <= 0)*100:.1f}%")

    # ----- M5: MLP vs GBM Sensitivity Comparison -----
    logger.info("Computing M5: MLP vs GBM Sensitivity Comparison (Cross-Experiment)")

    # Build lookup for GBM pairs by (join_idx, task_name_normalized)
    def normalize_task(tn):
        """Normalize task names across experiments."""
        tn = tn.replace("rel-f1/", "").replace("-", "_").lower()
        return tn

    gbm_cov_by_key = {}
    for p in gbm_pairs:
        key = (p["join_idx"], normalize_task(p["task_name"]))
        gbm_cov_by_key[key] = p["sensitivity_cov"]

    mlp_cov_by_key = {}
    for p in mlp_pairs:
        key = (p["join_idx"], normalize_task(p["task_name"]))
        mlp_cov_by_key[key] = p["sensitivity_cov"]

    # Find overlapping keys
    common_keys = set(gbm_cov_by_key.keys()) & set(mlp_cov_by_key.keys())
    logger.info(f"  GBM keys: {len(gbm_cov_by_key)}, MLP keys: {len(mlp_cov_by_key)}, Common: {len(common_keys)}")

    if len(common_keys) >= 3:
        paired_mlp = np.array([mlp_cov_by_key[k] for k in sorted(common_keys)])
        paired_gbm = np.array([gbm_cov_by_key[k] for k in sorted(common_keys)])
        diff = paired_mlp - paired_gbm

        m5_mean_mlp_cov = float(np.mean(paired_mlp))
        m5_mean_gbm_cov = float(np.mean(paired_gbm))
        m5_mean_diff = float(np.mean(diff))

        # Wilcoxon signed-rank test
        try:
            # Filter out zero differences for Wilcoxon
            nonzero_diff = diff[diff != 0]
            if len(nonzero_diff) >= 3:
                wilcox_stat, wilcox_p = stats.wilcoxon(nonzero_diff)
                # Rank-biserial correlation as effect size
                n_nonzero = len(nonzero_diff)
                rank_biserial = 1 - (2 * wilcox_stat) / (n_nonzero * (n_nonzero + 1) / 2)
            else:
                wilcox_stat, wilcox_p = 0.0, 1.0
                rank_biserial = 0.0
        except Exception:
            logger.exception("Wilcoxon test failed")
            wilcox_stat, wilcox_p, rank_biserial = 0.0, 1.0, 0.0

        metrics_agg["M5_n_common_pairs"] = len(common_keys)
        metrics_agg["M5_mean_mlp_cov"] = round(m5_mean_mlp_cov, 6)
        metrics_agg["M5_mean_gbm_cov"] = round(m5_mean_gbm_cov, 6)
        metrics_agg["M5_mean_diff_mlp_minus_gbm"] = round(m5_mean_diff, 6)
        metrics_agg["M5_wilcoxon_stat"] = round(float(wilcox_stat), 6)
        metrics_agg["M5_wilcoxon_pvalue"] = round(float(wilcox_p), 6)
        metrics_agg["M5_rank_biserial"] = round(float(rank_biserial), 6)
        metrics_agg["M5_mlp_higher_sensitivity"] = 1 if m5_mean_diff > 0 and wilcox_p < 0.05 else 0

        logger.info(f"  Mean MLP CoV: {m5_mean_mlp_cov:.4f}, Mean GBM CoV: {m5_mean_gbm_cov:.4f}")
        logger.info(f"  Paired diff (MLP-GBM): {m5_mean_diff:.4f}")
        logger.info(f"  Wilcoxon: stat={wilcox_stat:.4f}, p={wilcox_p:.4f}, rank-biserial={rank_biserial:.4f}")
    else:
        logger.warning(f"  Only {len(common_keys)} common pairs, skipping paired test")
        metrics_agg["M5_n_common_pairs"] = len(common_keys)
        metrics_agg["M5_mean_mlp_cov"] = 0.0
        metrics_agg["M5_mean_gbm_cov"] = 0.0
        metrics_agg["M5_mean_diff_mlp_minus_gbm"] = 0.0
        metrics_agg["M5_wilcoxon_pvalue"] = 1.0
        metrics_agg["M5_mlp_higher_sensitivity"] = 0

    # ----- M6: Winning Aggregation Strategy Analysis -----
    logger.info("Computing M6: Winning Aggregation Strategy Analysis by JRN bin")

    # Entropy per JRN bin
    for bin_name, counts in winning_agg_by_bin.items():
        ent = entropy(counts)
        n_strategies = len(counts)
        max_ent = np.log2(n_strategies) if n_strategies > 0 else 1.0
        norm_ent = ent / max_ent if max_ent > 0 else 0.0
        metrics_agg[f"M6_entropy_{bin_name}"] = round(ent, 6)
        metrics_agg[f"M6_norm_entropy_{bin_name}"] = round(norm_ent, 6)
        logger.info(f"  {bin_name}: entropy={ent:.4f}, normalized={norm_ent:.4f}, counts={counts}")

    # Chi-squared test: winning strategy independent of JRN bin?
    try:
        all_strategies = set()
        for counts in winning_agg_by_bin.values():
            all_strategies.update(counts.keys())
        all_strategies = sorted(all_strategies)
        bin_names = sorted(winning_agg_by_bin.keys())
        contingency = []
        for bn in bin_names:
            row = [winning_agg_by_bin[bn].get(s, 0) for s in all_strategies]
            contingency.append(row)
        contingency = np.array(contingency)
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
        metrics_agg["M6_chi2_stat"] = round(float(chi2), 6)
        metrics_agg["M6_chi2_pvalue"] = round(float(chi2_p), 6)
        metrics_agg["M6_chi2_dof"] = int(dof)
        metrics_agg["M6_strategy_independent_of_jrn"] = 1 if chi2_p > 0.05 else 0
        logger.info(f"  Chi-squared: stat={chi2:.4f}, p={chi2_p:.4f}, dof={dof}")
    except Exception:
        logger.exception("Chi-squared test failed")
        metrics_agg["M6_chi2_pvalue"] = 1.0
        metrics_agg["M6_strategy_independent_of_jrn"] = 1

    # ===================================================================
    # PART 2: Compounding Failure Analysis
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PART 2: Compounding Failure Analysis (Chain data)")
    logger.info("=" * 60)

    # ----- M7: Residual Analysis for Each Compounding Model -----
    logger.info("Computing M7: Residual Analysis for Each Compounding Model")

    model_names = ["multiplicative", "additive", "bottleneck"]
    for probe_type in ["MLP", "GBM"]:
        probe_chains = [c for c in chain_examples if c["probe"] == probe_type]
        if not probe_chains:
            continue
        measured = np.array([c["measured_jrn"] for c in probe_chains])

        for model_name in model_names:
            predicted = np.array([c[f"predict_{model_name}"] for c in probe_chains])
            residuals = measured - predicted
            abs_residuals = np.abs(residuals)

            rmse = float(np.sqrt(np.mean(residuals**2)))
            mae = float(np.mean(abs_residuals))
            mean_res = float(np.mean(residuals))
            std_res = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((measured - np.mean(measured))**2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            rho_res, p_rho_res = safe_spearman(measured, predicted)

            # One-sample t-test on residuals (is bias significant?)
            if len(residuals) > 2:
                t_stat, t_p = stats.ttest_1samp(residuals, 0)
            else:
                t_stat, t_p = 0.0, 1.0

            pfx = f"M7_{model_name}_{probe_type}"
            metrics_agg[f"{pfx}_mean_residual"] = round(mean_res, 6)
            metrics_agg[f"{pfx}_std_residual"] = round(std_res, 6)
            metrics_agg[f"{pfx}_rmse"] = round(rmse, 6)
            metrics_agg[f"{pfx}_r2"] = round(r2, 6)
            metrics_agg[f"{pfx}_spearman_rho"] = round(rho_res, 6)
            metrics_agg[f"{pfx}_bias_ttest_p"] = round(float(t_p), 6)
            metrics_agg[f"{pfx}_n_chains"] = len(probe_chains)

            logger.info(f"  {pfx}: mean_res={mean_res:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, ρ={rho_res:.4f}, bias_p={t_p:.4f}")

    # ----- M8: Residual by Chain Length -----
    logger.info("Computing M8: Residual by Chain Length")

    # Parse chain descriptions to count hops
    def count_hops(desc: str) -> int:
        """Count hops from chain description like 'results -> races -> circuits'."""
        parts = [p.strip() for p in desc.replace("enrich", "->").split("->") if p.strip()]
        return max(len(parts) - 1, 1)

    for probe_type in ["MLP", "GBM"]:
        probe_chains = [c for c in chain_examples if c["probe"] == probe_type]
        if not probe_chains:
            continue

        measured = np.array([c["measured_jrn"] for c in probe_chains])
        predicted_mult = np.array([c["predict_multiplicative"] for c in probe_chains])
        residuals_mult = measured - predicted_mult
        abs_residuals = np.abs(residuals_mult)

        hops = np.array([count_hops(c["chain_desc"]) for c in probe_chains])
        unique_hops = sorted(set(hops))

        for h in unique_hops:
            mask = hops == h
            if np.sum(mask) > 0:
                metrics_agg[f"M8_{probe_type}_{h}hop_mean_abs_residual"] = round(float(np.mean(abs_residuals[mask])), 6)
                metrics_agg[f"M8_{probe_type}_{h}hop_n"] = int(np.sum(mask))

        # Mann-Whitney U between hop groups (if we have 2+ groups)
        if len(unique_hops) >= 2:
            # Compare 2-hop vs 3-hop (or lowest vs highest)
            group_a = abs_residuals[hops == unique_hops[0]]
            group_b = abs_residuals[hops == unique_hops[-1]]
            if len(group_a) >= 2 and len(group_b) >= 2:
                u_stat, u_p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
                metrics_agg[f"M8_{probe_type}_mannwhitney_u"] = round(float(u_stat), 6)
                metrics_agg[f"M8_{probe_type}_mannwhitney_p"] = round(float(u_p), 6)
                logger.info(f"  {probe_type}: {unique_hops[0]}-hop mean |res|={np.mean(group_a):.4f}, {unique_hops[-1]}-hop mean |res|={np.mean(group_b):.4f}, U p={u_p:.4f}")
            else:
                metrics_agg[f"M8_{probe_type}_mannwhitney_p"] = 1.0
        else:
            logger.info(f"  {probe_type}: only {unique_hops} hop lengths found")

    # ----- M9: Residual vs Fan-out -----
    logger.info("Computing M9: Residual vs Fan-out along chain")

    # Build join_label -> fanout mapping from per_join_summary
    join_label_fanout = {}
    for js in per_join_summary:
        join_label_fanout[js["join_name"]] = js["fanout_mean"]

    for probe_type in ["MLP", "GBM"]:
        probe_chains = [c for c in chain_examples if c["probe"] == probe_type]
        if not probe_chains:
            continue
        measured = np.array([c["measured_jrn"] for c in probe_chains])
        predicted_mult = np.array([c["predict_multiplicative"] for c in probe_chains])
        abs_residuals = np.abs(measured - predicted_mult)

        # For each chain, find max fanout among joins in the chain path
        max_fanouts_in_chain = []
        for c in probe_chains:
            desc = c["chain_desc"]
            # Parse chain: "results -> races -> circuits"
            tables = [t.strip() for t in desc.replace("enrich", "->").split("->") if t.strip()]
            chain_fanouts = []
            for js in per_join_summary:
                jn = js["join_name"]
                parts = jn.split("->")
                if len(parts) == 2:
                    src = parts[0].strip()
                    tgt = parts[1].strip()
                    if src in tables and tgt in tables:
                        chain_fanouts.append(js["fanout_mean"])
            max_fanouts_in_chain.append(max(chain_fanouts) if chain_fanouts else 0)

        max_fanouts_arr = np.array(max_fanouts_in_chain)
        rho_res_fanout, p_res_fanout = safe_spearman(abs_residuals, max_fanouts_arr)

        metrics_agg[f"M9_{probe_type}_rho_absres_vs_maxfanout"] = round(rho_res_fanout, 6)
        metrics_agg[f"M9_{probe_type}_p_absres_vs_maxfanout"] = round(p_res_fanout, 6)
        logger.info(f"  {probe_type}: |res| vs max_fanout: rho={rho_res_fanout:.4f}, p={p_res_fanout:.4f}")

    # ----- M10: Information Redundancy / Feature Overlap -----
    logger.info("Computing M10: Information Redundancy / Feature Overlap")

    for probe_type in ["MLP", "GBM"]:
        probe_chains = [c for c in chain_examples if c["probe"] == probe_type]
        if not probe_chains:
            continue
        measured = np.array([c["measured_jrn"] for c in probe_chains])
        predicted_mult = np.array([c["predict_multiplicative"] for c in probe_chains])
        abs_residuals = np.abs(measured - predicted_mult)

        # Count unique parent tables in each chain as redundancy proxy
        redundancy_scores = []
        for c in probe_chains:
            desc = c["chain_desc"]
            tables = [t.strip() for t in desc.replace("enrich", "->").split("->") if t.strip()]
            n_unique = len(set(tables))
            n_total = len(tables)
            # Higher ratio of unique/total means less redundancy
            redundancy = 1.0 - (n_unique / n_total) if n_total > 0 else 0.0
            redundancy_scores.append(redundancy)

        redundancy_arr = np.array(redundancy_scores)
        rho_red, p_red = safe_spearman(abs_residuals, redundancy_arr)

        metrics_agg[f"M10_{probe_type}_rho_absres_vs_redundancy"] = round(rho_red, 6)
        metrics_agg[f"M10_{probe_type}_p_absres_vs_redundancy"] = round(p_red, 6)
        metrics_agg[f"M10_{probe_type}_mean_redundancy"] = round(float(np.mean(redundancy_arr)), 6)
        logger.info(f"  {probe_type}: |res| vs redundancy: rho={rho_red:.4f}, p={p_red:.4f}")

    # ----- M11: Why Log-Linear Works Better — Diagnostic -----
    logger.info("Computing M11: Log-Linear Model Coefficient Analysis")

    for probe_type in ["MLP", "GBM"]:
        probe_chains = [c for c in chain_examples if c["probe"] == probe_type]
        if not probe_chains:
            continue

        # Group chains by chain_id to get unique chains
        chains_by_id = defaultdict(list)
        for c in probe_chains:
            chains_by_id[c["chain_id"]].append(c)

        # For each chain, we need: log(measured_chain_jrn) and log(individual JRNs along the path)
        # We have onehop_jrn for each chain
        # For 2-hop chains: log(chain_jrn) = alpha + beta1*log(j1) + beta2*log(j2)
        # But we only have the product prediction; individual hop JRNs need to be extracted

        # Simpler approach: fit log(measured) = alpha + beta * log(predicted_multiplicative)
        # If beta < 1 → sub-multiplicative compounding
        measured = np.array([c["measured_jrn"] for c in probe_chains])
        predicted_mult = np.array([c["predict_multiplicative"] for c in probe_chains])

        # Filter positive values for log
        mask = (measured > 0) & (predicted_mult > 0)
        if np.sum(mask) < 3:
            logger.warning(f"  {probe_type}: insufficient positive values for log-linear fit")
            continue

        log_measured = np.log(measured[mask])
        log_predicted = np.log(predicted_mult[mask])

        # Fit: log(measured) = alpha + beta * log(predicted_mult)
        X_ll = np.column_stack([np.ones(np.sum(mask)), log_predicted])
        try:
            beta_ll = np.linalg.lstsq(X_ll, log_measured, rcond=None)[0]
            alpha_ll = beta_ll[0]
            beta_coeff = beta_ll[1]

            y_pred_ll = X_ll @ beta_ll
            ss_res_ll = np.sum((log_measured - y_pred_ll) ** 2)
            ss_tot_ll = np.sum((log_measured - np.mean(log_measured)) ** 2)
            r2_ll = 1 - ss_res_ll / ss_tot_ll if ss_tot_ll > 0 else 0.0

            # Bootstrap CI for beta coefficient
            n_ll = len(log_measured)
            rng = np.random.RandomState(42)
            boot_betas = []
            for _ in range(2000):
                idx = rng.choice(n_ll, size=n_ll, replace=True)
                X_b = X_ll[idx]
                y_b = log_measured[idx]
                try:
                    b_b = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
                    boot_betas.append(b_b[1])
                except Exception:
                    pass
            boot_betas = np.array(boot_betas)
            beta_ci_lo = float(np.percentile(boot_betas, 2.5))
            beta_ci_hi = float(np.percentile(boot_betas, 97.5))
            beta_sig_diff_from_1 = not (beta_ci_lo <= 1.0 <= beta_ci_hi)

            metrics_agg[f"M11_{probe_type}_alpha"] = round(float(alpha_ll), 6)
            metrics_agg[f"M11_{probe_type}_beta"] = round(float(beta_coeff), 6)
            metrics_agg[f"M11_{probe_type}_beta_ci_lo"] = round(beta_ci_lo, 6)
            metrics_agg[f"M11_{probe_type}_beta_ci_hi"] = round(beta_ci_hi, 6)
            metrics_agg[f"M11_{probe_type}_beta_sig_diff_from_1"] = 1 if beta_sig_diff_from_1 else 0
            metrics_agg[f"M11_{probe_type}_r2_loglinear"] = round(float(r2_ll), 6)
            metrics_agg[f"M11_{probe_type}_sub_multiplicative"] = 1 if beta_coeff < 1 else 0

            logger.info(f"  {probe_type}: alpha={alpha_ll:.4f}, beta={beta_coeff:.4f} [{beta_ci_lo:.4f}, {beta_ci_hi:.4f}]")
            logger.info(f"  {probe_type}: R²={r2_ll:.4f}, sub-multiplicative={beta_coeff < 1}, sig diff from 1={beta_sig_diff_from_1}")
        except Exception:
            logger.exception(f"Log-linear fit failed for {probe_type}")

    # Also fit using onehop JRN as predictor
    logger.info("  Fitting log-linear using onehop JRN as predictor...")
    for probe_type in ["MLP", "GBM"]:
        probe_chains = [c for c in chain_examples if c["probe"] == probe_type]
        if not probe_chains:
            continue
        measured = np.array([c["measured_jrn"] for c in probe_chains])
        onehop = np.array([c["onehop_jrn"] for c in probe_chains])
        mask = (measured > 0) & (onehop > 0)
        if np.sum(mask) < 3:
            continue
        log_m = np.log(measured[mask])
        log_oh = np.log(onehop[mask])
        X_oh = np.column_stack([np.ones(np.sum(mask)), log_oh])
        try:
            beta_oh = np.linalg.lstsq(X_oh, log_m, rcond=None)[0]
            metrics_agg[f"M11_{probe_type}_onehop_beta"] = round(float(beta_oh[1]), 6)
            logger.info(f"  {probe_type} onehop: beta={beta_oh[1]:.4f}")
        except Exception:
            logger.exception(f"Onehop log-linear fit failed for {probe_type}")

    # ----- M12: MLP vs GBM Compounding Comparison -----
    logger.info("Computing M12: MLP vs GBM Compounding Comparison")

    for cs in compounding_summaries:
        pfx = f"M12_{cs['model']}_{cs['probe']}"
        metrics_agg[f"{pfx}_r2"] = round(cs["r2"], 6)
        metrics_agg[f"{pfx}_spearman"] = round(cs["spearman_r"], 6)
        metrics_agg[f"{pfx}_mae"] = round(cs["mae"], 6)
        metrics_agg[f"{pfx}_rmse"] = round(cs["rmse"], 6)

    # Compare best model R² between probes
    for model_name in ["multiplicative", "additive", "bottleneck", "log_linear"]:
        mlp_cs = [cs for cs in compounding_summaries if cs["model"] == model_name and cs["probe"] == "MLP"]
        gbm_cs = [cs for cs in compounding_summaries if cs["model"] == model_name and cs["probe"] == "GBM"]
        if mlp_cs and gbm_cs:
            delta_r2 = gbm_cs[0]["r2"] - mlp_cs[0]["r2"]
            metrics_agg[f"M12_{model_name}_delta_r2_gbm_minus_mlp"] = round(delta_r2, 6)
            logger.info(f"  {model_name}: MLP R²={mlp_cs[0]['r2']:.4f}, GBM R²={gbm_cs[0]['r2']:.4f}, ΔR²={delta_r2:.4f}")

    # ===================================================================
    # PART 3: Synthesis Metrics
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PART 3: Synthesis Metrics")
    logger.info("=" * 60)

    # ----- M13: Effect Size Summary Table -----
    logger.info("Computing M13: Effect Size Summary Table")

    # (a) JRN-sensitivity Spearman ρ with CI
    rho_jrn_sens = metrics_agg.get("M3_rho_cov_vs_jrn", 0.0)
    rho_ci_lo, rho_ci_hi = bootstrap_ci(
        np.column_stack([covs, jrns]),
        stat_fn=lambda x: float(stats.spearmanr(x[:, 0], x[:, 1])[0]) if len(x) > 2 else 0.0,
        n_boot=2000
    )
    metrics_agg["M13_jrn_sensitivity_rho"] = round(rho_jrn_sens, 6)
    metrics_agg["M13_jrn_sensitivity_rho_ci_lo"] = round(rho_ci_lo, 6)
    metrics_agg["M13_jrn_sensitivity_rho_ci_hi"] = round(rho_ci_hi, 6)

    # (b) Quadratic β₂ with CI from metadata
    quad_beta2 = meta_it3["quadratic_fit"]["beta2"]
    quad_beta2_p = meta_it3["quadratic_fit"]["beta2_pvalue"]
    metrics_agg["M13_quadratic_beta2"] = round(quad_beta2, 6)
    metrics_agg["M13_quadratic_beta2_pvalue"] = round(quad_beta2_p, 6)

    # (c) MLP vs GBM sensitivity difference (already computed in M5)
    metrics_agg["M13_mlp_gbm_sensitivity_diff"] = metrics_agg.get("M5_mean_diff_mlp_minus_gbm", 0.0)
    metrics_agg["M13_mlp_gbm_rank_biserial"] = metrics_agg.get("M5_rank_biserial", 0.0)

    # (d) Best compounding model R² for each probe
    for probe_type in ["MLP", "GBM"]:
        best_r2 = -999
        best_model = "none"
        for cs in compounding_summaries:
            if cs["probe"] == probe_type and cs["r2"] > best_r2:
                best_r2 = cs["r2"]
                best_model = cs["model"]
        metrics_agg[f"M13_best_compounding_r2_{probe_type}"] = round(best_r2, 6)
        logger.info(f"  Best compounding for {probe_type}: {best_model} (R²={best_r2:.4f})")

    # (e) Log-linear β coefficients (already in M11)

    # ----- M14: Key Takeaway Statistics -----
    logger.info("Computing M14: Key Takeaway Statistics")

    # Takeaway 1: "Aggregation robustness"
    # Probability that changing agg strategy changes performance by >5%
    pct_changes_gt_5 = []
    for p in gbm_pairs:
        agg_perfs = p["agg_perfs"]
        perf_values = list(agg_perfs.values())
        if len(perf_values) < 2:
            continue
        max_perf = max(perf_values)
        min_perf = min(perf_values)
        if max_perf > 0:
            pct_change = (max_perf - min_perf) / max_perf * 100
        else:
            pct_change = 0.0
        pct_changes_gt_5.append(pct_change > 5.0)

    prob_change_gt5 = float(np.mean(pct_changes_gt_5)) * 100 if pct_changes_gt_5 else 0.0
    prob_ci = bootstrap_ci(
        np.array(pct_changes_gt_5, dtype=float),
        stat_fn=lambda x: float(np.mean(x) * 100),
        n_boot=2000
    )
    metrics_agg["M14_prob_agg_change_gt5pct"] = round(prob_change_gt5, 2)
    metrics_agg["M14_prob_agg_change_gt5pct_ci_lo"] = round(prob_ci[0], 2)
    metrics_agg["M14_prob_agg_change_gt5pct_ci_hi"] = round(prob_ci[1], 2)

    logger.info(f"  Takeaway 1: P(agg change > 5%) = {prob_change_gt5:.1f}% [{prob_ci[0]:.1f}, {prob_ci[1]:.1f}]")

    # Takeaway 2: "Conditional non-independence"
    # Multiplicative overestimates chain JRN by Z% on average
    all_chain_residuals = []
    for c in chain_examples:
        measured = c["measured_jrn"]
        predicted = c["predict_multiplicative"]
        if measured > 0:
            pct_overest = (predicted - measured) / measured * 100
            all_chain_residuals.append(pct_overest)

    if all_chain_residuals:
        mean_overest = float(np.mean(all_chain_residuals))
        overest_ci = bootstrap_ci(np.array(all_chain_residuals), n_boot=2000)
        metrics_agg["M14_mult_overestimate_pct"] = round(mean_overest, 2)
        metrics_agg["M14_mult_overestimate_ci_lo"] = round(overest_ci[0], 2)
        metrics_agg["M14_mult_overestimate_ci_hi"] = round(overest_ci[1], 2)
        logger.info(f"  Takeaway 2: Mult overestimate = {mean_overest:.1f}% [{overest_ci[0]:.1f}, {overest_ci[1]:.1f}]")

    # Takeaway 3: "Log-linear correction"
    # ΔR² between log-linear and multiplicative with bootstrap CI
    for probe_type in ["MLP", "GBM"]:
        probe_chains = [c for c in chain_examples if c["probe"] == probe_type]
        if not probe_chains:
            continue
        measured = np.array([c["measured_jrn"] for c in probe_chains])
        predicted_mult = np.array([c["predict_multiplicative"] for c in probe_chains])

        # Get compounding summary R²
        ll_cs = [cs for cs in compounding_summaries if cs["model"] == "log_linear" and cs["probe"] == probe_type]
        mult_cs = [cs for cs in compounding_summaries if cs["model"] == "multiplicative" and cs["probe"] == probe_type]
        if ll_cs and mult_cs:
            delta_r2 = ll_cs[0]["r2"] - mult_cs[0]["r2"]
            metrics_agg[f"M14_delta_r2_loglinear_vs_mult_{probe_type}"] = round(delta_r2, 6)
            logger.info(f"  Takeaway 3 ({probe_type}): ΔR² (log-linear - mult) = {delta_r2:.4f}")

    # ===================================================================
    # Build output examples for all three datasets
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Building output examples")
    logger.info("=" * 60)

    # Dataset 1: exp_id1_it3 (GBM aggregation sensitivity)
    ds1_examples = []
    for p in gbm_pairs:
        ex = {
            "input": json.dumps({"join_idx": p["join_idx"], "join_name": p["join_name"], "task_name": p["task_name"], "experiment": "exp_id1_it3"}),
            "output": json.dumps({
                "jrn": p["jrn"],
                "sensitivity_cov": p["sensitivity_cov"],
                "perf_range": p["perf_range"],
                "winning_agg": p["winning_agg"],
                "diagnosis": "threshold_failure_analysis"
            }),
            "predict_jrn": str(round(p["jrn"], 6)),
            "predict_sensitivity_cov": str(round(p["sensitivity_cov"], 6)),
            "metadata_join_idx": str(p["join_idx"]),
            "metadata_task_name": p["task_name"],
            "metadata_fanout": str(round(join_fanout.get(p["join_idx"], 0), 4)),
            "eval_sensitivity_cov": round(p["sensitivity_cov"], 6),
            "eval_perf_range": round(p["perf_range"], 6),
        }
        ds1_examples.append(ex)

    # Dataset 2: exp_id1_it2 (MLP aggregation sensitivity)
    ds2_examples = []
    for p in mlp_pairs:
        ex = {
            "input": json.dumps({"join_idx": p["join_idx"], "task_name": p["task_name"], "experiment": "exp_id1_it2"}),
            "output": json.dumps({
                "jrn": p["jrn"],
                "sensitivity_cov": p["sensitivity_cov"],
                "agg_sensitivity": p["agg_sensitivity"],
                "diagnosis": "mlp_sensitivity_comparison"
            }),
            "predict_jrn": str(round(p["jrn"], 6)),
            "predict_sensitivity_cov": str(round(p["sensitivity_cov"], 6)),
            "metadata_join_idx": str(p["join_idx"]),
            "metadata_task_name": p["task_name"],
            "eval_sensitivity_cov": round(p["sensitivity_cov"], 6),
        }
        ds2_examples.append(ex)

    # Dataset 3: exp_id2_it6 (chain compounding)
    ds3_examples = []
    for c in chain_examples:
        residual_mult = c["measured_jrn"] - c["predict_multiplicative"]
        residual_add = c["measured_jrn"] - c["predict_additive"]
        residual_bott = c["measured_jrn"] - c["predict_bottleneck"]
        ex = {
            "input": json.dumps({
                "chain_id": c["chain_id"],
                "chain_desc": c["chain_desc"],
                "task": c["task"],
                "probe": c["probe"],
                "experiment": "exp_id2_it6"
            }),
            "output": json.dumps({
                "measured_jrn": c["measured_jrn"],
                "residual_multiplicative": residual_mult,
                "residual_additive": residual_add,
                "residual_bottleneck": residual_bott,
                "diagnosis": "compounding_failure_analysis"
            }),
            "predict_measured_jrn": str(round(c["measured_jrn"], 6)),
            "predict_multiplicative": str(round(c["predict_multiplicative"], 6)),
            "metadata_chain_id": str(c["chain_id"]),
            "metadata_probe": c["probe"],
            "metadata_task": c["task"],
            "eval_residual_mult": round(residual_mult, 6),
            "eval_abs_residual_mult": round(abs(residual_mult), 6),
            "eval_residual_add": round(residual_add, 6),
            "eval_residual_bott": round(residual_bott, 6),
        }
        ds3_examples.append(ex)

    # ===================================================================
    # Assemble final output
    # ===================================================================

    output = {
        "metadata": {
            "evaluation_name": "Negative-Result Diagnostic: Threshold and Compounding Failures",
            "description": "Systematic post-hoc analysis of inverted-U threshold prediction failure and multiplicative compounding model failure using data from exp_id1_it3 (GBM), exp_id1_it2 (MLP), and exp_id2_it6 (chains).",
            "dependencies": ["exp_id1_it3__opus", "exp_id1_it2__opus", "exp_id2_it6__opus"],
            "n_gbm_pairs": len(gbm_pairs),
            "n_mlp_pairs": len(mlp_pairs),
            "n_chain_examples": len(chain_examples),
            "n_compounding_summaries": len(compounding_summaries),
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "exp_id1_it3_gbm_sensitivity",
                "examples": ds1_examples,
            },
            {
                "dataset": "exp_id1_it2_mlp_sensitivity",
                "examples": ds2_examples,
            },
            {
                "dataset": "exp_id2_it6_chain_compounding",
                "examples": ds3_examples,
            },
        ],
    }

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved evaluation output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Print key metrics summary
    logger.info("=" * 60)
    logger.info("KEY METRICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"M1: Mean CoV = {metrics_agg.get('M1_mean_cov', 'N/A')}")
    logger.info(f"M2: All insensitive = {metrics_agg.get('M2_all_insensitive', 'N/A')}")
    logger.info(f"M3: CoV vs JRN rho = {metrics_agg.get('M3_rho_cov_vs_jrn', 'N/A')}")
    logger.info(f"M5: MLP higher sensitivity = {metrics_agg.get('M5_mlp_higher_sensitivity', 'N/A')}")
    logger.info(f"M6: Strategy independent of JRN = {metrics_agg.get('M6_strategy_independent_of_jrn', 'N/A')}")
    logger.info(f"M12: Best GBM compounding R² = {metrics_agg.get('M13_best_compounding_r2_GBM', 'N/A')}")
    logger.info(f"M14: P(agg change >5%) = {metrics_agg.get('M14_prob_agg_change_gt5pct', 'N/A')}%")

    return output


if __name__ == "__main__":
    main()
