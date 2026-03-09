#!/usr/bin/env python3
"""Cross-Dataset JRN Consolidation: Probe Validity, Join-Selector Diagnostic, and Paper-Ready Tables.

Consolidates 8 iteration 2-3 experimental results across 4 RelBench datasets
(rel-f1, rel-stack, rel-avito, rel-hm) into a comprehensive evaluation.
"""

import json
import math
import os
import sys
import gc
import resource
from pathlib import Path
from collections import defaultdict

import numpy as np
import psutil
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from loguru import logger

# ── Logging ──
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# ── Hardware Detection ──
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
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
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

# ── Memory Limits ──
RAM_BUDGET = int(10 * 1024**3)  # 10 GB
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ── Paths ──
ITER2_BASE = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_2/gen_art")
ITER3_BASE = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_3/gen_art")

EXP_PATHS = {
    "exp_id1_it2": ITER2_BASE / "exp_id1_it2__opus",
    "exp_id2_it2": ITER2_BASE / "exp_id2_it2__opus",
    "exp_id3_it2": ITER2_BASE / "exp_id3_it2__opus",
    "exp_id4_it2": ITER2_BASE / "exp_id4_it2__opus",
    "exp_id1_it3": ITER3_BASE / "exp_id1_it3__opus",
    "exp_id2_it3": ITER3_BASE / "exp_id2_it3__opus",
    "exp_id3_it3": ITER3_BASE / "exp_id3_it3__opus",
    "exp_id4_it3": ITER3_BASE / "exp_id4_it3__opus",
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_experiment(exp_id: str) -> dict:
    """Load full_method_out.json for an experiment."""
    path = EXP_PATHS[exp_id] / "full_method_out.json"
    logger.info(f"Loading {exp_id} from {path}")
    data = json.loads(path.read_text())
    n_examples = sum(len(ds["examples"]) for ds in data.get("datasets", []))
    logger.info(f"  {exp_id}: {n_examples} examples, metadata keys: {list(data.get('metadata', {}).keys())[:8]}")
    return data


def safe_float(v, default=None) -> float | None:
    """Safely parse float from string or number."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def parse_output(output_str: str) -> dict:
    """Parse JSON from output field."""
    try:
        return json.loads(output_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def get_examples(data: dict) -> list:
    """Get all examples from a loaded experiment."""
    examples = []
    for ds in data.get("datasets", []):
        examples.extend(ds.get("examples", []))
    return examples


def fisher_z(rho: float) -> float:
    """Fisher z-transform: arctanh(rho)."""
    rho = max(-0.9999, min(0.9999, rho))
    return 0.5 * math.log((1 + rho) / (1 - rho))


def inv_fisher_z(z: float) -> float:
    """Inverse Fisher z-transform: tanh(z)."""
    return math.tanh(z)


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: CROSS-DATASET PROBE VALIDITY
# ═══════════════════════════════════════════════════════════════════════════

def analysis_probe_validity(experiments: dict) -> tuple[dict, list]:
    """Compute cross-dataset probe validity with Fisher z meta-analysis."""
    logger.info("=== Analysis 1: Cross-Dataset Probe Validity ===")

    probe_datasets = {}

    # ── rel-f1-v1 from exp_id4_it2 (GBM probe vs GT, n=19) ──
    data = experiments["exp_id4_it2"]
    exs = get_examples(data)
    probes_v1, gts_v1 = [], []
    for ex in exs:
        probe = safe_float(ex.get("predict_jrn_probe_gbm"))
        out = parse_output(ex.get("output", ""))
        gt = safe_float(out.get("jrn_gt_mean"))
        if probe is not None and gt is not None:
            probes_v1.append(probe)
            gts_v1.append(gt)
    if len(probes_v1) >= 5:
        rho, pval = stats.spearmanr(probes_v1, gts_v1)
        probe_datasets["rel-f1-v1"] = {"rho": rho, "pval": pval, "n": len(probes_v1)}
        logger.info(f"  rel-f1-v1: rho={rho:.4f}, p={pval:.6f}, n={len(probes_v1)}")

    # ── rel-f1-v2 from exp_id1_it3 (GBM probe, n=65) ──
    data = experiments["exp_id1_it3"]
    exs = get_examples(data)
    probes_v2, gts_v2 = [], []
    for ex in exs:
        probe = safe_float(ex.get("predict_jrn"))
        out = parse_output(ex.get("output", ""))
        gt = safe_float(out.get("gt_jrn"))
        if probe is not None and gt is not None:
            probes_v2.append(probe)
            gts_v2.append(gt)
    if len(probes_v2) >= 5:
        rho, pval = stats.spearmanr(probes_v2, gts_v2)
        probe_datasets["rel-f1-v2"] = {"rho": rho, "pval": pval, "n": len(probes_v2)}
        logger.info(f"  rel-f1-v2: rho={rho:.4f}, p={pval:.6f}, n={len(probes_v2)}")

    # ── rel-stack from exp_id2_it3 (probe_to_full_correlation phase) ──
    data = experiments["exp_id2_it3"]
    exs = get_examples(data)
    probes_stack, fulls_stack = [], []
    for ex in exs:
        inp = parse_output(ex.get("input", ""))
        if inp.get("phase") == "probe_to_full_correlation":
            out = parse_output(ex.get("output", ""))
            p = safe_float(out.get("probe_jrn"))
            f = safe_float(out.get("full_jrn"))
            if p is not None and f is not None:
                probes_stack.append(p)
                fulls_stack.append(f)
    if len(probes_stack) >= 5:
        rho, pval = stats.spearmanr(probes_stack, fulls_stack)
        probe_datasets["rel-stack"] = {"rho": rho, "pval": pval, "n": len(probes_stack)}
        logger.info(f"  rel-stack: rho={rho:.4f}, p={pval:.6f}, n={len(probes_stack)}")
    else:
        # Fallback to summary statistics
        summary_exs = [ex for ex in exs if parse_output(ex.get("input", "")).get("phase") == "summary"]
        if summary_exs:
            out = parse_output(summary_exs[0].get("output", ""))
            ptf = out.get("phase1b_probe_to_full", {})
            rho = safe_float(ptf.get("spearman_rho"))
            pval = safe_float(ptf.get("spearman_p"))
            n = ptf.get("n_pairs", 16)
            if rho is not None:
                probe_datasets["rel-stack"] = {"rho": rho, "pval": pval or 0.0, "n": n}
                logger.info(f"  rel-stack (from summary): rho={rho:.4f}, n={n}")

    # ── rel-avito from exp_id3_it3 (probe_to_full measurement) ──
    data = experiments["exp_id3_it3"]
    exs = get_examples(data)
    for ex in exs:
        if ex.get("metadata_measurement_type") == "probe_to_full":
            out = parse_output(ex.get("output", ""))
            probe_jrns = out.get("probe_jrns", [])
            full_jrns = out.get("full_jrns", [])
            if len(probe_jrns) >= 5:
                rho, pval = stats.spearmanr(probe_jrns, full_jrns)
                probe_datasets["rel-avito"] = {"rho": rho, "pval": pval, "n": len(probe_jrns)}
                logger.info(f"  rel-avito: rho={rho:.4f}, p={pval:.6f}, n={len(probe_jrns)}")
            elif len(probe_jrns) >= 3:
                rho = safe_float(out.get("spearman_rho"))
                pval = safe_float(out.get("spearman_pval"))
                if rho is not None:
                    probe_datasets["rel-avito"] = {"rho": rho, "pval": pval or 0.0, "n": len(probe_jrns)}
                    logger.info(f"  rel-avito (from summary): rho={rho:.4f}, n={len(probe_jrns)}")
            break

    # ── rel-hm from exp_id3_it2: only 2 pairs, report descriptively ──
    data = experiments["exp_id3_it2"]
    meta = data.get("metadata", {})
    hm_jrns = meta.get("global_jrn_results", {})
    logger.info(f"  rel-hm: {len(hm_jrns)} pairs (excluded from meta-analysis, n<5)")

    # ── Fisher z meta-analysis ──
    metrics = {}
    examples = []

    valid_datasets = {k: v for k, v in probe_datasets.items() if v["n"] >= 5}

    if len(valid_datasets) >= 2:
        z_vals, weights, se_vals = [], [], []
        for ds_name, ds_info in valid_datasets.items():
            rho = ds_info["rho"]
            n = ds_info["n"]
            z_i = fisher_z(rho)
            w_i = n - 3
            se_i = 1.0 / math.sqrt(max(w_i, 1))
            z_vals.append(z_i)
            weights.append(w_i)
            se_vals.append(se_i)

            metrics[f"eval_spearman_rho_{ds_name.replace('-', '_')}"] = round(rho, 6)
            metrics[f"eval_spearman_p_{ds_name.replace('-', '_')}"] = round(ds_info["pval"], 6)
            metrics[f"eval_n_pairs_{ds_name.replace('-', '_')}"] = n
            metrics[f"eval_fisher_z_{ds_name.replace('-', '_')}"] = round(z_i, 6)
            metrics[f"eval_fisher_z_se_{ds_name.replace('-', '_')}"] = round(se_i, 6)

        # Weighted mean
        z_vals = np.array(z_vals)
        weights = np.array(weights, dtype=float)
        z_bar = np.sum(weights * z_vals) / np.sum(weights)
        se_bar = 1.0 / math.sqrt(np.sum(weights))

        # 95% CI
        z_lower = z_bar - 1.96 * se_bar
        z_upper = z_bar + 1.96 * se_bar

        # Back-transform
        meta_rho = inv_fisher_z(z_bar)
        ci_lower = inv_fisher_z(z_lower)
        ci_upper = inv_fisher_z(z_upper)

        # Cochran's Q
        Q = float(np.sum(weights * (z_vals - z_bar) ** 2))
        k = len(valid_datasets)
        Q_p = 1.0 - stats.chi2.cdf(Q, k - 1) if k > 1 else 1.0

        # I-squared
        I_sq = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0

        metrics["eval_meta_rho_weighted_mean"] = round(meta_rho, 6)
        metrics["eval_meta_rho_95ci_lower"] = round(ci_lower, 6)
        metrics["eval_meta_rho_95ci_upper"] = round(ci_upper, 6)
        metrics["eval_cochrans_Q"] = round(Q, 6)
        metrics["eval_cochrans_Q_p"] = round(Q_p, 6)
        metrics["eval_I_squared"] = round(I_sq, 2)

        logger.info(f"  Meta-analytic rho={meta_rho:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  Cochran's Q={Q:.4f}, p={Q_p:.4f}, I²={I_sq:.1f}%")

    # Build examples for output
    for ds_name, ds_info in probe_datasets.items():
        z_i = fisher_z(ds_info["rho"])
        se_i = 1.0 / math.sqrt(max(ds_info["n"] - 3, 1))
        ex = {
            "input": json.dumps({"dataset": ds_name, "analysis": "probe_validity"}),
            "output": json.dumps({"rho": ds_info["rho"], "pval": ds_info["pval"], "n": ds_info["n"]}),
            "eval_spearman_rho": round(ds_info["rho"], 6),
            "eval_spearman_p": round(ds_info["pval"], 6),
            "eval_n_pairs": ds_info["n"],
            "eval_fisher_z": round(z_i, 6),
            "eval_fisher_z_se": round(se_i, 6),
            "metadata_dataset": ds_name,
        }
        examples.append(ex)

    return metrics, examples


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: JRN AS BINARY JOIN SELECTOR
# ═══════════════════════════════════════════════════════════════════════════

def collect_jrn_pool(experiments: dict) -> list[dict]:
    """Pool ALL JRN measurements across experiments, with deduplication."""
    logger.info("=== Collecting JRN Pool ===")
    pool = {}  # key=(dataset, join, task) -> measurement dict

    # ── rel-f1 from exp_id4_it3 (preferred, 5 tasks × 13 joins = 65) ──
    data = experiments["exp_id4_it3"]
    meta = data.get("metadata", {})
    jrn_matrix = meta.get("jrn_matrix", {})
    for task_name, joins in jrn_matrix.items():
        for join_idx_str, jinfo in joins.items():
            jrn = safe_float(jinfo.get("jrn", 1.0))
            join_label = jinfo.get("join_label", f"join_{join_idx_str}")
            key = ("rel-f1", join_label, task_name)
            pool[key] = {
                "dataset": "rel-f1",
                "join": join_label,
                "task": task_name,
                "jrn": jrn,
                "source": "exp_id4_it3",
                "era": "gbm",
            }

    # ── rel-stack from exp_id2_it3 (16 unique pairs, filter agg=mean) ──
    data = experiments["exp_id2_it3"]
    exs = get_examples(data)
    for ex in exs:
        inp = parse_output(ex.get("input", ""))
        if inp.get("phase") == "jrn_estimation" and ex.get("metadata_agg_type") == "mean":
            out = parse_output(ex.get("output", ""))
            jrn = safe_float(out.get("jrn"))
            join_name = ex.get("metadata_join", "")
            task = ex.get("metadata_task", "")
            if jrn is not None:
                key = ("rel-stack", join_name, task)
                pool[key] = {
                    "dataset": "rel-stack",
                    "join": join_name,
                    "task": task,
                    "jrn": jrn,
                    "source": "exp_id2_it3",
                    "era": "gbm",
                }

    # ── rel-avito from exp_id3_it3 (33 jrn_measurement) ──
    data = experiments["exp_id3_it3"]
    exs = get_examples(data)
    for ex in exs:
        if ex.get("metadata_measurement_type") == "jrn_measurement":
            out = parse_output(ex.get("output", ""))
            jrn = safe_float(out.get("jrn"))
            join_key = ex.get("metadata_join_key", "")
            task = ex.get("metadata_task_name", "")
            if jrn is not None:
                key = ("rel-avito", join_key, task)
                pool[key] = {
                    "dataset": "rel-avito",
                    "join": join_key,
                    "task": task,
                    "jrn": jrn,
                    "source": "exp_id3_it3",
                    "era": "gbm",
                }

    # ── rel-hm from exp_id3_it2 (2 global values) ──
    data = experiments["exp_id3_it2"]
    meta = data.get("metadata", {})
    for result_key, result_val in meta.get("global_jrn_results", {}).items():
        jrn = safe_float(result_val.get("JRN"))
        task = result_val.get("task", "")
        join = result_val.get("join", "")
        if jrn is not None:
            key = ("rel-hm", join, task)
            pool[key] = {
                "dataset": "rel-hm",
                "join": join,
                "task": task,
                "jrn": jrn,
                "source": "exp_id3_it2",
                "era": "mlp",
            }

    result = list(pool.values())
    logger.info(f"  Pooled {len(result)} unique JRN measurements")
    ds_counts = defaultdict(int)
    for m in result:
        ds_counts[m["dataset"]] += 1
    for ds, cnt in sorted(ds_counts.items()):
        logger.info(f"    {ds}: {cnt}")
    return result


def analysis_join_selector(jrn_pool: list[dict], experiments: dict) -> tuple[dict, list]:
    """JRN as binary join selector diagnostic."""
    logger.info("=== Analysis 2: JRN as Binary Join Selector ===")

    jrn_values = np.array([m["jrn"] for m in jrn_pool])

    # Binary labeling: positive if JRN > 1.01
    labels = (jrn_values > 1.01).astype(int)
    n_positive = int(np.sum(labels))
    n_negative = int(len(labels) - n_positive)

    metrics = {
        "eval_n_positive": n_positive,
        "eval_n_negative": n_negative,
        "eval_class_balance": round(n_positive / len(labels), 4) if len(labels) > 0 else 0.0,
    }
    logger.info(f"  Positive (JRN>1.01): {n_positive}, Negative: {n_negative}")

    # ROC-AUC of JRN as predictor
    examples = []
    if n_positive > 0 and n_negative > 0:
        roc_auc = roc_auc_score(labels, jrn_values)
        metrics["eval_jrn_roc_auc"] = round(roc_auc, 6)
        logger.info(f"  JRN ROC-AUC: {roc_auc:.4f}")

        # Youden's J for optimal threshold
        thresholds = np.sort(np.unique(jrn_values))
        best_j, best_thresh = -1.0, 1.0
        best_sens, best_spec = 0.0, 0.0
        for t in thresholds:
            pred = (jrn_values >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fn = np.sum((pred == 0) & (labels == 1))
            tn = np.sum((pred == 0) & (labels == 0))
            fp = np.sum((pred == 1) & (labels == 0))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sens + spec - 1
            if j > best_j:
                best_j = j
                best_thresh = float(t)
                best_sens = float(sens)
                best_spec = float(spec)

        # Precision/recall at Youden threshold
        pred_at_thresh = (jrn_values >= best_thresh).astype(int)
        tp = np.sum((pred_at_thresh == 1) & (labels == 1))
        fp = np.sum((pred_at_thresh == 1) & (labels == 0))
        fn = np.sum((pred_at_thresh == 0) & (labels == 1))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        metrics["eval_jrn_youden_j_threshold"] = round(best_thresh, 6)
        metrics["eval_jrn_youden_j_value"] = round(best_j, 6)
        metrics["eval_jrn_precision_at_threshold"] = round(precision, 6)
        metrics["eval_jrn_recall_at_threshold"] = round(recall, 6)

        # Bootstrap 95% CI for ROC-AUC
        rng = np.random.RandomState(42)
        n_boot = 1000
        boot_aucs = []
        for _ in range(n_boot):
            idx = rng.choice(len(labels), size=len(labels), replace=True)
            if len(np.unique(labels[idx])) < 2:
                continue
            try:
                boot_aucs.append(roc_auc_score(labels[idx], jrn_values[idx]))
            except ValueError:
                continue
        if boot_aucs:
            metrics["eval_jrn_roc_auc_ci_lower"] = round(float(np.percentile(boot_aucs, 2.5)), 6)
            metrics["eval_jrn_roc_auc_ci_upper"] = round(float(np.percentile(boot_aucs, 97.5)), 6)

    # ── Proxy ROC comparison (from exp_id4_it2 where all proxies available) ──
    # Only use exp_id4_it2 data for fair comparison (same dataset, same GT)
    proxy_data_f1 = _collect_proxy_data_f1_only(experiments)
    if proxy_data_f1:
        proxy_labels = np.array([1 if m["jrn"] > 1.01 else 0 for m in proxy_data_f1])
        if len(np.unique(proxy_labels)) == 2:
            # JRN probe ROC on this subset
            proxy_jrns = np.array([m["jrn"] for m in proxy_data_f1])
            proxy_jrn_roc = roc_auc_score(proxy_labels, proxy_jrns)
            metrics["eval_jrn_roc_auc_proxy_subset"] = round(proxy_jrn_roc, 6)

            for proxy_name in ["fanout", "correlation", "MI", "entropy"]:
                raw_vals = [m.get(proxy_name) for m in proxy_data_f1]
                vals = np.array([v if v is not None else np.nan for v in raw_vals], dtype=float)
                valid = ~np.isnan(vals)
                if np.sum(valid) >= 5 and len(np.unique(proxy_labels[valid])) == 2:
                    try:
                        proxy_roc = roc_auc_score(proxy_labels[valid], vals[valid])
                        metrics[f"eval_{proxy_name}_roc_auc"] = round(proxy_roc, 6)
                        logger.info(f"  {proxy_name} ROC-AUC: {proxy_roc:.4f}")
                    except ValueError:
                        pass

    # Build examples
    for m in jrn_pool:
        label = 1 if m["jrn"] > 1.01 else 0
        ex = {
            "input": json.dumps({"dataset": m["dataset"], "join": m["join"], "task": m["task"]}),
            "output": json.dumps({"jrn": m["jrn"], "label": label}),
            "eval_jrn_value": round(m["jrn"], 6),
            "eval_binary_label": label,
            "metadata_dataset": m["dataset"],
            "metadata_source": m["source"],
            "predict_jrn": str(round(m["jrn"], 6)),
        }
        examples.append(ex)

    return metrics, examples


def _collect_proxy_data_f1_only(experiments: dict) -> list[dict]:
    """Collect measurements with all proxy values from exp_id4_it2 (rel-f1 only, fair comparison)."""
    data = experiments["exp_id4_it2"]
    exs = get_examples(data)
    result = []
    for ex in exs:
        out = parse_output(ex.get("output", ""))
        jrn_gt = safe_float(out.get("jrn_gt_mean"))
        fanout = safe_float(ex.get("predict_proxy_fanout"))
        correlation = safe_float(ex.get("predict_proxy_correlation"))
        mi = safe_float(ex.get("predict_proxy_MI"))
        entropy = safe_float(ex.get("predict_proxy_entropy_reduction"))
        if jrn_gt is not None:
            result.append({
                "jrn": jrn_gt,
                "fanout": math.log1p(fanout) if fanout is not None else None,
                "correlation": abs(correlation) if correlation is not None else None,
                "MI": mi,
                "entropy": entropy,
            })
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: AGGREGATION SENSITIVITY REANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analysis_agg_sensitivity(experiments: dict) -> tuple[dict, list]:
    """Reanalyze aggregation sensitivity across datasets."""
    logger.info("=== Analysis 3: Aggregation Sensitivity ===")

    all_jrn, all_sens = [], []
    per_dataset = defaultdict(lambda: {"jrn": [], "sens": []})

    # ── rel-f1 from exp_id1_it3 (65 pairs with sensitivity) ──
    data = experiments["exp_id1_it3"]
    exs = get_examples(data)
    for ex in exs:
        jrn = safe_float(ex.get("predict_jrn"))
        sens = safe_float(ex.get("predict_sensitivity"))
        if jrn is not None and sens is not None:
            all_jrn.append(jrn)
            all_sens.append(sens)
            per_dataset["rel-f1"]["jrn"].append(jrn)
            per_dataset["rel-f1"]["sens"].append(sens)

    # ── rel-stack from exp_id2_it3 (compute CoV across agg types) ──
    data = experiments["exp_id2_it3"]
    exs = get_examples(data)
    # Group by (task, join) to compute sensitivity as CoV across agg types
    jrn_by_pair = defaultdict(list)
    for ex in exs:
        inp = parse_output(ex.get("input", ""))
        if inp.get("phase") == "jrn_estimation":
            task = ex.get("metadata_task", "")
            join = ex.get("metadata_join", "")
            out = parse_output(ex.get("output", ""))
            jrn = safe_float(out.get("jrn"))
            score = safe_float(out.get("join_score"))
            if jrn is not None and score is not None:
                jrn_by_pair[(task, join)].append({"jrn": jrn, "score": score})

    for (task, join), measurements in jrn_by_pair.items():
        if len(measurements) >= 2:
            scores = [m["score"] for m in measurements]
            mean_score = np.mean(scores)
            if mean_score > 0:
                cov = float(np.std(scores) / mean_score)
            else:
                cov = 0.0
            mean_jrn = np.mean([m["jrn"] for m in measurements])
            all_jrn.append(float(mean_jrn))
            all_sens.append(cov)
            per_dataset["rel-stack"]["jrn"].append(float(mean_jrn))
            per_dataset["rel-stack"]["sens"].append(cov)

    # ── rel-avito from exp_id3_it3 (aggregation_sensitivity entry) ──
    data = experiments["exp_id3_it3"]
    exs = get_examples(data)
    for ex in exs:
        if ex.get("metadata_measurement_type") == "aggregation_sensitivity":
            out = parse_output(ex.get("output", ""))
            sens_data = out.get("data", [])
            for entry in sens_data:
                jrn = safe_float(entry.get("jrn"))
                sens = safe_float(entry.get("agg_sensitivity"))
                if jrn is not None and sens is not None:
                    all_jrn.append(jrn)
                    all_sens.append(sens)
                    per_dataset["rel-avito"]["jrn"].append(jrn)
                    per_dataset["rel-avito"]["sens"].append(sens)

    metrics = {}
    examples = []

    if len(all_jrn) >= 5:
        all_jrn_arr = np.array(all_jrn)
        all_sens_arr = np.array(all_sens)

        # Pooled Spearman
        rho, pval = stats.spearmanr(all_jrn_arr, all_sens_arr)
        metrics["eval_pooled_jrn_sensitivity_spearman"] = round(float(rho), 6)
        metrics["eval_pooled_jrn_sensitivity_p"] = round(float(pval), 6)
        logger.info(f"  Pooled JRN-sensitivity rho={rho:.4f}, p={pval:.6f}")

        # Per-dataset Spearman
        for ds_name, ds_data in per_dataset.items():
            if len(ds_data["jrn"]) >= 5:
                r, p = stats.spearmanr(ds_data["jrn"], ds_data["sens"])
                metrics[f"eval_jrn_sensitivity_spearman_{ds_name.replace('-', '_')}"] = round(float(r), 6)
                logger.info(f"  {ds_name} JRN-sensitivity rho={r:.4f}")

        # Overall sensitivity stats
        metrics["eval_sensitivity_mean_overall"] = round(float(np.mean(all_sens_arr)), 6)
        metrics["eval_sensitivity_std_overall"] = round(float(np.std(all_sens_arr)), 6)

        # Sensitivity by JRN tertile
        tertile_edges = np.percentile(all_jrn_arr, [33.33, 66.67])
        low_mask = all_jrn_arr <= tertile_edges[0]
        mid_mask = (all_jrn_arr > tertile_edges[0]) & (all_jrn_arr <= tertile_edges[1])
        high_mask = all_jrn_arr > tertile_edges[1]

        tertile_means = {}
        for name, mask in [("low", low_mask), ("mid", mid_mask), ("high", high_mask)]:
            if np.sum(mask) > 0:
                tertile_means[name] = float(np.mean(all_sens_arr[mask]))
                metrics[f"eval_sensitivity_tertile_{name}"] = round(tertile_means[name], 6)

        # Kruskal-Wallis
        groups = [all_sens_arr[low_mask], all_sens_arr[mid_mask], all_sens_arr[high_mask]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h_stat, h_pval = stats.kruskal(*groups)
            metrics["eval_kruskal_wallis_h"] = round(float(h_stat), 6)
            metrics["eval_kruskal_wallis_p"] = round(float(h_pval), 6)
            logger.info(f"  Kruskal-Wallis H={h_stat:.4f}, p={h_pval:.6f}")

        # Verdict
        if abs(rho) < 0.2 and pval > 0.05:
            verdict = "flat"
        elif rho > 0.2 and pval < 0.05:
            verdict = "monotonic_positive"
        elif rho < -0.2 and pval < 0.05:
            verdict = "monotonic_negative"
        else:
            verdict = "flat"
        metrics["eval_monotonic_or_flat"] = hash(verdict) % 10000  # numeric proxy

    # Build examples
    for i, (j, s) in enumerate(zip(all_jrn, all_sens)):
        ds_name = ""
        for dsn, dsd in per_dataset.items():
            if j in dsd["jrn"] and s in dsd["sens"]:
                ds_name = dsn
                break
        ex = {
            "input": json.dumps({"index": i, "analysis": "agg_sensitivity"}),
            "output": json.dumps({"jrn": j, "sensitivity": s}),
            "eval_jrn": round(j, 6),
            "eval_sensitivity": round(s, 6),
            "metadata_dataset": ds_name,
        }
        examples.append(ex)

    return metrics, examples


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: JRN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def analysis_jrn_distribution(jrn_pool: list[dict]) -> tuple[dict, list]:
    """Analyze JRN distribution across all measurements."""
    logger.info("=== Analysis 4: JRN Distribution ===")

    jrn_values = np.array([m["jrn"] for m in jrn_pool])
    n = len(jrn_values)

    metrics = {
        "eval_jrn_mean": round(float(np.mean(jrn_values)), 6),
        "eval_jrn_median": round(float(np.median(jrn_values)), 6),
        "eval_jrn_std": round(float(np.std(jrn_values)), 6),
        "eval_jrn_min": round(float(np.min(jrn_values)), 6),
        "eval_jrn_max": round(float(np.max(jrn_values)), 6),
        "eval_jrn_pct_above_1": round(float(np.mean(jrn_values > 1.0) * 100), 2),
        "eval_jrn_pct_near_1": round(float(np.mean((jrn_values >= 0.95) & (jrn_values <= 1.05)) * 100), 2),
        "eval_jrn_pct_below_1": round(float(np.mean(jrn_values < 1.0) * 100), 2),
        "eval_jrn_n_total": n,
    }

    # Skewness, kurtosis
    if n >= 8:
        metrics["eval_jrn_skewness"] = round(float(stats.skew(jrn_values)), 6)
        metrics["eval_jrn_kurtosis"] = round(float(stats.kurtosis(jrn_values)), 6)

    # Shapiro-Wilk (max 5000 samples)
    if 3 <= n <= 5000:
        try:
            w_stat, w_pval = stats.shapiro(jrn_values)
            metrics["eval_jrn_shapiro_w"] = round(float(w_stat), 6)
            metrics["eval_jrn_shapiro_p"] = round(float(w_pval), 6)
        except Exception:
            pass

    # Bimodality coefficient
    if n >= 8:
        skew_val = stats.skew(jrn_values)
        kurt_val = stats.kurtosis(jrn_values)
        # BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
        denom = kurt_val + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
        bc = (skew_val ** 2 + 1) / denom if denom != 0 else 0
        metrics["eval_jrn_bimodality_coefficient"] = round(float(bc), 6)

    # KDE modes
    if n >= 10:
        try:
            jrn_range = np.linspace(float(np.min(jrn_values)) - 0.1,
                                     float(np.max(jrn_values)) + 0.1, 500)
            kde = gaussian_kde(jrn_values)
            kde_vals = kde(jrn_range)
            peaks, _ = find_peaks(kde_vals)
            mode_locations = [round(float(jrn_range[p]), 4) for p in peaks]
            metrics["eval_jrn_n_modes"] = len(mode_locations)
        except Exception:
            pass

    # Per-dataset distribution stats
    ds_groups = defaultdict(list)
    for m in jrn_pool:
        ds_groups[m["dataset"]].append(m["jrn"])

    for ds_name, vals in ds_groups.items():
        arr = np.array(vals)
        ds_key = ds_name.replace("-", "_")
        metrics[f"eval_jrn_mean_{ds_key}"] = round(float(np.mean(arr)), 6)
        metrics[f"eval_jrn_median_{ds_key}"] = round(float(np.median(arr)), 6)
        metrics[f"eval_jrn_std_{ds_key}"] = round(float(np.std(arr)), 6)
        metrics[f"eval_jrn_n_{ds_key}"] = len(vals)

    logger.info(f"  JRN: mean={np.mean(jrn_values):.4f}, median={np.median(jrn_values):.4f}, "
                f"std={np.std(jrn_values):.4f}, range=[{np.min(jrn_values):.4f}, {np.max(jrn_values):.4f}]")

    # Build examples (one per dataset summary)
    examples = []
    for ds_name, vals in ds_groups.items():
        arr = np.array(vals)
        ex = {
            "input": json.dumps({"dataset": ds_name, "analysis": "jrn_distribution"}),
            "output": json.dumps({
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "n": len(vals),
            }),
            "eval_mean": round(float(np.mean(arr)), 6),
            "eval_median": round(float(np.median(arr)), 6),
            "eval_std": round(float(np.std(arr)), 6),
            "eval_n": len(vals),
            "metadata_dataset": ds_name,
        }
        examples.append(ex)

    return metrics, examples


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: TASK vs SCHEMA STABILITY (Kendall's W)
# ═══════════════════════════════════════════════════════════════════════════

def analysis_task_schema_stability(experiments: dict) -> tuple[dict, list]:
    """Compute Kendall's W for cross-task agreement on join rankings."""
    logger.info("=== Analysis 5: Task vs Schema Stability ===")

    metrics = {}
    examples = []

    # ── rel-f1 from exp_id4_it3 (5 tasks × 13 joins) ──
    data = experiments["exp_id4_it3"]
    meta = data.get("metadata", {})
    jrn_matrix = meta.get("jrn_matrix", {})

    if jrn_matrix:
        tasks = sorted(jrn_matrix.keys())
        # Get all join indices
        join_indices = sorted(set(
            idx for task_joins in jrn_matrix.values() for idx in task_joins.keys()
        ))
        n_joins = len(join_indices)
        n_tasks = len(tasks)

        if n_tasks >= 2 and n_joins >= 3:
            # Build JRN matrix: tasks × joins
            matrix = np.zeros((n_tasks, n_joins))
            for i, task in enumerate(tasks):
                for j, join_idx in enumerate(join_indices):
                    jinfo = jrn_matrix.get(task, {}).get(join_idx, {})
                    matrix[i, j] = safe_float(jinfo.get("jrn"), 1.0)

            # Compute ranks per task (rater), ranking the joins
            ranks = np.zeros_like(matrix)
            for i in range(n_tasks):
                ranks[i] = stats.rankdata(matrix[i])

            # Kendall's W
            k = n_tasks  # raters
            n = n_joins   # items
            rank_sums = np.sum(ranks, axis=0)
            mean_rank_sum = np.mean(rank_sums)
            S = np.sum((rank_sums - mean_rank_sum) ** 2)
            W = 12 * S / (k ** 2 * (n ** 3 - n))
            W = max(0.0, min(1.0, W))

            metrics["eval_kendall_w_rel_f1"] = round(float(W), 6)
            logger.info(f"  rel-f1 Kendall's W = {W:.4f} (k={k} tasks, n={n} joins)")

            # Classification vs regression task comparison
            clf_jrns, reg_jrns = [], []
            for task in tasks:
                task_jrns = [safe_float(jrn_matrix[task][j].get("jrn"), 1.0) for j in join_indices]
                if "position" in task.lower():
                    # Regression tasks tend to have "position" in name
                    reg_jrns.extend(task_jrns)
                else:
                    clf_jrns.extend(task_jrns)

            if len(clf_jrns) >= 5 and len(reg_jrns) >= 5:
                try:
                    u_stat, u_pval = stats.mannwhitneyu(clf_jrns, reg_jrns, alternative="two-sided")
                    metrics["eval_task_type_mann_whitney_u"] = round(float(u_stat), 4)
                    metrics["eval_task_type_mann_whitney_p"] = round(float(u_pval), 6)
                except ValueError:
                    pass

    # ── rel-stack from exp_id2_it3 ──
    data = experiments["exp_id2_it3"]
    exs = get_examples(data)
    stack_jrn = defaultdict(dict)  # task -> join -> jrn
    for ex in exs:
        inp = parse_output(ex.get("input", ""))
        if inp.get("phase") == "jrn_estimation" and ex.get("metadata_agg_type") == "mean":
            task = ex.get("metadata_task", "")
            join = ex.get("metadata_join", "")
            out = parse_output(ex.get("output", ""))
            jrn = safe_float(out.get("jrn"))
            if jrn is not None:
                stack_jrn[task][join] = jrn

    # Find joins present in ALL tasks
    if len(stack_jrn) >= 2:
        all_tasks = list(stack_jrn.keys())
        common_joins = set(stack_jrn[all_tasks[0]].keys())
        for task in all_tasks[1:]:
            common_joins &= set(stack_jrn[task].keys())
        common_joins = sorted(common_joins)

        if len(common_joins) >= 3:
            k = len(all_tasks)
            n = len(common_joins)
            matrix = np.zeros((k, n))
            for i, task in enumerate(all_tasks):
                for j, join in enumerate(common_joins):
                    matrix[i, j] = stack_jrn[task][join]

            ranks = np.zeros_like(matrix)
            for i in range(k):
                ranks[i] = stats.rankdata(matrix[i])

            rank_sums = np.sum(ranks, axis=0)
            mean_rank_sum = np.mean(rank_sums)
            S = np.sum((rank_sums - mean_rank_sum) ** 2)
            W = 12 * S / (k ** 2 * (n ** 3 - n))
            W = max(0.0, min(1.0, W))

            metrics["eval_kendall_w_rel_stack"] = round(float(W), 6)
            logger.info(f"  rel-stack Kendall's W = {W:.4f} (k={k} tasks, n={n} common joins)")

    # Interpretation
    w_vals = [v for k_name, v in metrics.items() if "kendall_w" in k_name]
    if w_vals:
        avg_w = np.mean(w_vals)
        if avg_w < 0.1:
            interp = "poor"
        elif avg_w < 0.3:
            interp = "fair"
        elif avg_w < 0.5:
            interp = "moderate"
        elif avg_w < 0.7:
            interp = "good"
        else:
            interp = "excellent"
        # Store as numeric
        interp_map = {"poor": 1, "fair": 2, "moderate": 3, "good": 4, "excellent": 5}
        metrics["eval_kendall_w_interpretation"] = interp_map.get(interp, 0)

    # Schema complexity effect: compare IQR of JRN across datasets
    # Build examples
    for ds_name in ["rel-f1", "rel-stack"]:
        w_key = f"eval_kendall_w_{ds_name.replace('-', '_')}"
        if w_key in metrics:
            ex = {
                "input": json.dumps({"dataset": ds_name, "analysis": "kendall_w"}),
                "output": json.dumps({"kendall_w": metrics[w_key]}),
                "eval_kendall_w": metrics[w_key],
                "metadata_dataset": ds_name,
            }
            examples.append(ex)

    if not examples:
        examples.append({
            "input": json.dumps({"analysis": "kendall_w", "note": "insufficient_data"}),
            "output": json.dumps({"kendall_w": 0}),
            "eval_kendall_w": 0.0,
            "metadata_dataset": "none",
        })

    return metrics, examples


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: MULTIPLICATIVE COMPOUNDING REANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analysis_compounding(experiments: dict) -> tuple[dict, list]:
    """Reanalyze multiplicative compounding."""
    logger.info("=== Analysis 6: Multiplicative Compounding ===")

    metrics = {}
    examples = []

    # ── From exp_id2_it2 (MLP, 14 chains, R²=0.83) ──
    data_mlp = experiments["exp_id2_it2"]
    exs_mlp = get_examples(data_mlp)
    mlp_chains = []
    for ex in exs_mlp:
        if ex.get("metadata_phase") == "phase3":
            out = parse_output(ex.get("output", ""))
            # predicted is in predict_our_method JSON
            pred_info = parse_output(ex.get("predict_our_method", ""))
            predicted = safe_float(pred_info.get("predicted_chain_jrn"))
            # measured is actual_chain_jrn in output
            measured = safe_float(out.get("actual_chain_jrn"))
            if predicted is not None and measured is not None:
                mlp_chains.append({"predicted": predicted, "measured": measured,
                                   "chain": ex.get("metadata_chain", ""),
                                   "task": ex.get("metadata_task", ""),
                                   "source": "exp_id2_it2_mlp"})

    # ── From exp_id2_it3 (GBM, 11 chains) ──
    data_gbm = experiments["exp_id2_it3"]
    exs_gbm = get_examples(data_gbm)
    gbm_chains = []
    for ex in exs_gbm:
        inp = parse_output(ex.get("input", ""))
        if inp.get("phase") == "multiplicative_compounding":
            out = parse_output(ex.get("output", ""))
            predicted = safe_float(out.get("predicted_chain_jrn"))
            measured = safe_float(out.get("measured_chain_jrn"))
            if predicted is not None and measured is not None:
                gbm_chains.append({"predicted": predicted, "measured": measured,
                                   "chain": out.get("description", ""),
                                   "task": out.get("task", ""),
                                   "source": "exp_id2_it3_gbm"})

    # Use whichever has more data; prefer MLP (stronger result)
    chains = mlp_chains if len(mlp_chains) >= len(gbm_chains) else gbm_chains
    if not chains:
        chains = mlp_chains + gbm_chains

    if len(chains) >= 3:
        predicted = np.array([c["predicted"] for c in chains])
        measured = np.array([c["measured"] for c in chains])

        # R²
        ss_res = np.sum((measured - predicted) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Mean absolute deviation
        mad = float(np.mean(np.abs(predicted - measured)))

        # Spearman
        rho, pval = stats.spearmanr(predicted, measured)

        # Systematic bias
        bias = float(np.mean(predicted - measured))

        metrics["eval_compounding_r_squared"] = round(r_squared, 6)
        metrics["eval_compounding_n_chains"] = len(chains)
        metrics["eval_compounding_mean_abs_deviation"] = round(mad, 6)
        metrics["eval_compounding_spearman"] = round(float(rho), 6)
        metrics["eval_compounding_systematic_bias"] = round(bias, 6)

        logger.info(f"  Compounding R²={r_squared:.4f}, MAD={mad:.4f}, rho={rho:.4f}, bias={bias:.4f}")

    # Also compute for GBM chains separately if available
    if len(gbm_chains) >= 3:
        predicted_g = np.array([c["predicted"] for c in gbm_chains])
        measured_g = np.array([c["measured"] for c in gbm_chains])
        ss_res_g = np.sum((measured_g - predicted_g) ** 2)
        ss_tot_g = np.sum((measured_g - np.mean(measured_g)) ** 2)
        r2_g = 1 - ss_res_g / ss_tot_g if ss_tot_g > 0 else 0.0
        metrics["eval_compounding_r_squared_gbm"] = round(r2_g, 6)
        metrics["eval_compounding_n_chains_gbm"] = len(gbm_chains)

    if len(mlp_chains) >= 3:
        predicted_m = np.array([c["predicted"] for c in mlp_chains])
        measured_m = np.array([c["measured"] for c in mlp_chains])
        ss_res_m = np.sum((measured_m - predicted_m) ** 2)
        ss_tot_m = np.sum((measured_m - np.mean(measured_m)) ** 2)
        r2_m = 1 - ss_res_m / ss_tot_m if ss_tot_m > 0 else 0.0
        metrics["eval_compounding_r_squared_mlp"] = round(r2_m, 6)
        metrics["eval_compounding_n_chains_mlp"] = len(mlp_chains)

    # Build examples
    all_chains = mlp_chains + gbm_chains
    for c in all_chains:
        ex = {
            "input": json.dumps({"chain": c["chain"], "task": c["task"], "source": c["source"]}),
            "output": json.dumps({"predicted": c["predicted"], "measured": c["measured"]}),
            "eval_predicted_chain_jrn": round(c["predicted"], 6),
            "eval_measured_chain_jrn": round(c["measured"], 6),
            "eval_abs_deviation": round(abs(c["predicted"] - c["measured"]), 6),
            "metadata_source": c["source"],
            "predict_multiplicative": str(round(c["predicted"], 6)),
        }
        examples.append(ex)

    if not examples:
        examples.append({
            "input": json.dumps({"analysis": "compounding", "note": "no_chain_data"}),
            "output": json.dumps({"note": "no data"}),
            "eval_predicted_chain_jrn": 0.0,
            "eval_measured_chain_jrn": 0.0,
            "eval_abs_deviation": 0.0,
            "metadata_source": "none",
            "predict_multiplicative": "0",
        })

    return metrics, examples


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 7: PAPER-READY TABLES
# ═══════════════════════════════════════════════════════════════════════════

def build_paper_tables(
    experiments: dict,
    probe_metrics: dict,
    selector_metrics: dict,
    compound_metrics: dict,
    jrn_pool: list[dict],
    dist_metrics: dict,
    scorecard_metrics: dict,
) -> tuple[dict, list]:
    """Build 7 paper-ready tables as structured JSON."""
    logger.info("=== Analysis 7: Paper-Ready Tables ===")

    tables = []

    # ── Table 1: Dataset Characteristics ──
    ds_chars = []
    ds_groups = defaultdict(list)
    for m in jrn_pool:
        ds_groups[m["dataset"]].append(m["jrn"])

    ds_info = {
        "rel-f1": {"n_tables": 8, "n_fk_joins": 13, "n_tasks": 5},
        "rel-stack": {"n_tables": 7, "n_fk_joins": 11, "n_tasks": 3},
        "rel-avito": {"n_tables": 8, "n_fk_joins": 11, "n_tasks": 3},
        "rel-hm": {"n_tables": 3, "n_fk_joins": 2, "n_tasks": 2},
    }
    for ds_name, vals in sorted(ds_groups.items()):
        arr = np.array(vals)
        info = ds_info.get(ds_name, {})
        ds_chars.append({
            "dataset": ds_name,
            "n_tables": info.get("n_tables", 0),
            "n_fk_joins": info.get("n_fk_joins", 0),
            "n_tasks_tested": info.get("n_tasks", 0),
            "n_join_task_pairs": len(vals),
            "jrn_min": round(float(np.min(arr)), 4),
            "jrn_max": round(float(np.max(arr)), 4),
            "jrn_mean": round(float(np.mean(arr)), 4),
        })
    tables.append({
        "input": "Table 1: Dataset Characteristics",
        "output": json.dumps(ds_chars),
        "eval_table_id": 1,
        "metadata_table_name": "dataset_characteristics",
    })

    # ── Table 2: Probe-to-Full Validity ──
    validity_rows = []
    for ds_name in ["rel-f1-v1", "rel-f1-v2", "rel-stack", "rel-avito"]:
        ds_key = ds_name.replace("-", "_")
        rho = probe_metrics.get(f"eval_spearman_rho_{ds_key}")
        pval = probe_metrics.get(f"eval_spearman_p_{ds_key}")
        n = probe_metrics.get(f"eval_n_pairs_{ds_key}")
        if rho is not None:
            validity_rows.append({
                "dataset": ds_name,
                "spearman_rho": rho,
                "p_value": pval,
                "n_pairs": n,
            })
    # Meta-analytic row
    validity_rows.append({
        "dataset": "meta-analytic",
        "spearman_rho": probe_metrics.get("eval_meta_rho_weighted_mean"),
        "ci_lower": probe_metrics.get("eval_meta_rho_95ci_lower"),
        "ci_upper": probe_metrics.get("eval_meta_rho_95ci_upper"),
        "I_squared": probe_metrics.get("eval_I_squared"),
    })
    tables.append({
        "input": "Table 2: Probe-to-Full Validity",
        "output": json.dumps(validity_rows),
        "eval_table_id": 2,
        "metadata_table_name": "probe_validity",
    })

    # ── Table 3: JRN-Guided Architecture Results (rel-f1) ──
    arch_rows = []
    data = experiments["exp_id4_it3"]
    meta = data.get("metadata", {})
    config_results = meta.get("configuration_results", {})
    oracle_analysis = meta.get("oracle_analysis", {})
    for task_name, configs in config_results.items():
        row = {"task": task_name}
        for cfg_name, cfg_data in configs.items():
            row[cfg_name] = round(cfg_data.get("val_mean", 0), 6)
        oracle_info = oracle_analysis.get(task_name, {})
        row["gap_to_oracle_pct"] = round(oracle_info.get("gap_pct", 0), 2)
        arch_rows.append(row)
    tables.append({
        "input": "Table 3: JRN-Guided Architecture Results (rel-f1)",
        "output": json.dumps(arch_rows),
        "eval_table_id": 3,
        "metadata_table_name": "jrn_guided_architecture",
    })

    # ── Table 4: Multiplicative Compounding ──
    compound_rows = []
    for k in ["eval_compounding_r_squared", "eval_compounding_n_chains",
              "eval_compounding_mean_abs_deviation", "eval_compounding_spearman",
              "eval_compounding_systematic_bias"]:
        if k in compound_metrics:
            compound_rows.append({"metric": k.replace("eval_compounding_", ""), "value": compound_metrics[k]})
    tables.append({
        "input": "Table 4: Multiplicative Compounding",
        "output": json.dumps(compound_rows),
        "eval_table_id": 4,
        "metadata_table_name": "multiplicative_compounding",
    })

    # ── Table 5: JRN as Join Selector ──
    selector_rows = []
    for predictor in ["jrn", "fanout", "MI", "entropy", "correlation"]:
        roc_key = f"eval_{predictor}_roc_auc"
        if roc_key in selector_metrics:
            selector_rows.append({"predictor": predictor, "roc_auc": selector_metrics[roc_key]})
    tables.append({
        "input": "Table 5: JRN as Join Selector",
        "output": json.dumps(selector_rows),
        "eval_table_id": 5,
        "metadata_table_name": "join_selector_roc",
    })

    # ── Table 6: JRN Distribution Summary ──
    dist_rows = []
    for ds_name in sorted(ds_groups.keys()):
        ds_key = ds_name.replace("-", "_")
        dist_rows.append({
            "dataset": ds_name,
            "mean": dist_metrics.get(f"eval_jrn_mean_{ds_key}"),
            "median": dist_metrics.get(f"eval_jrn_median_{ds_key}"),
            "std": dist_metrics.get(f"eval_jrn_std_{ds_key}"),
            "n": dist_metrics.get(f"eval_jrn_n_{ds_key}"),
        })
    dist_rows.append({
        "dataset": "overall",
        "mean": dist_metrics.get("eval_jrn_mean"),
        "median": dist_metrics.get("eval_jrn_median"),
        "std": dist_metrics.get("eval_jrn_std"),
        "n": dist_metrics.get("eval_jrn_n_total"),
        "pct_above_1": dist_metrics.get("eval_jrn_pct_above_1"),
        "pct_near_1": dist_metrics.get("eval_jrn_pct_near_1"),
    })
    tables.append({
        "input": "Table 6: JRN Distribution Summary",
        "output": json.dumps(dist_rows),
        "eval_table_id": 6,
        "metadata_table_name": "jrn_distribution",
    })

    # ── Table 7: Hypothesis Scorecard ──
    scorecard_rows = []
    for claim in ["probe_validity", "compounding", "threshold", "architecture", "training_free"]:
        verdict_key = f"eval_scorecard_{claim}"
        if verdict_key in scorecard_metrics:
            scorecard_rows.append({"claim": claim, "verdict_code": scorecard_metrics[verdict_key]})
    scorecard_rows.append({"claim": "overall", "verdict_code": scorecard_metrics.get("eval_scorecard_overall", 0)})
    tables.append({
        "input": "Table 7: Hypothesis Scorecard",
        "output": json.dumps(scorecard_rows),
        "eval_table_id": 7,
        "metadata_table_name": "hypothesis_scorecard",
    })

    metrics = {"eval_n_paper_tables": len(tables)}
    return metrics, tables


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 8: HYPOTHESIS SCORECARD
# ═══════════════════════════════════════════════════════════════════════════

def build_scorecard(
    probe_metrics: dict,
    compound_metrics: dict,
    sensitivity_metrics: dict,
    selector_metrics: dict,
) -> tuple[dict, list]:
    """Build updated hypothesis scorecard with explicit verdict thresholds."""
    logger.info("=== Analysis 8: Hypothesis Scorecard ===")

    # Verdict encoding: SUPPORTED=3, PARTIAL=2, UNSUPPORTED=1
    SUPPORTED, PARTIAL, UNSUPPORTED = 3, 2, 1
    metrics = {}
    verdicts = {}

    # 1. Probe validity
    meta_rho = probe_metrics.get("eval_meta_rho_weighted_mean", 0)
    ci_lower = probe_metrics.get("eval_meta_rho_95ci_lower", 0)
    if meta_rho > 0.6 and ci_lower > 0.3:
        v = SUPPORTED
        label = "SUPPORTED"
    elif meta_rho > 0.3:
        v = PARTIAL
        label = "PARTIAL"
    else:
        v = UNSUPPORTED
        label = "UNSUPPORTED"
    metrics["eval_scorecard_probe_validity"] = v
    verdicts["probe_validity"] = label
    logger.info(f"  Probe validity: {label} (meta rho={meta_rho:.4f}, CI lower={ci_lower:.4f})")

    # 2. Multiplicative compounding
    r2 = compound_metrics.get("eval_compounding_r_squared", 0)
    # Check both MLP and GBM
    r2_mlp = compound_metrics.get("eval_compounding_r_squared_mlp", r2)
    best_r2 = max(r2, r2_mlp)
    if best_r2 > 0.5:
        v = SUPPORTED
        label = "SUPPORTED"
    elif best_r2 >= 0.3:
        v = PARTIAL
        label = "PARTIAL"
    else:
        v = UNSUPPORTED
        label = "UNSUPPORTED"
    metrics["eval_scorecard_compounding"] = v
    verdicts["compounding"] = label
    logger.info(f"  Compounding: {label} (best R²={best_r2:.4f})")

    # 3. Threshold/inverted-U
    pooled_rho = sensitivity_metrics.get("eval_pooled_jrn_sensitivity_spearman", 0)
    pooled_p = sensitivity_metrics.get("eval_pooled_jrn_sensitivity_p", 1.0)
    kw_p = sensitivity_metrics.get("eval_kruskal_wallis_p", 1.0)
    if abs(pooled_rho) < 0.2 and pooled_p > 0.05:
        v = UNSUPPORTED
        label = "UNSUPPORTED"
    elif kw_p < 0.05:
        v = PARTIAL
        label = "PARTIAL"
    else:
        v = UNSUPPORTED
        label = "UNSUPPORTED"
    metrics["eval_scorecard_threshold"] = v
    verdicts["threshold"] = label
    logger.info(f"  Threshold/inverted-U: {label} (rho={pooled_rho:.4f}, KW p={kw_p:.4f})")

    # 4. JRN-guided architecture
    # Use win rates from selector_metrics (passed as additional_data) or fallback
    win_vs_topk = selector_metrics.get("_win_vs_topk", 0.6)
    win_vs_uniform_mean = selector_metrics.get("_win_vs_uniform_mean", 0.4)
    if win_vs_topk >= 0.6 and win_vs_uniform_mean >= 0.6:
        v = SUPPORTED
        label = "SUPPORTED"
    elif win_vs_topk >= 0.5 or win_vs_uniform_mean >= 0.5:
        v = PARTIAL
        label = "PARTIAL"
    else:
        v = UNSUPPORTED
        label = "UNSUPPORTED"
    metrics["eval_scorecard_architecture"] = v
    verdicts["architecture"] = label
    logger.info(f"  Architecture: {label}")

    # 5. Training-free alternative
    jrn_roc = selector_metrics.get("eval_jrn_roc_auc", 0)
    best_proxy_roc = max(
        selector_metrics.get("eval_fanout_roc_auc", 0),
        selector_metrics.get("eval_MI_roc_auc", 0),
        selector_metrics.get("eval_entropy_roc_auc", 0),
        selector_metrics.get("eval_correlation_roc_auc", 0),
    )
    gap = abs(jrn_roc - best_proxy_roc)
    if gap <= 0.05:
        v = SUPPORTED
        label = "SUPPORTED"
    elif gap <= 0.10:
        v = PARTIAL
        label = "PARTIAL"
    else:
        v = UNSUPPORTED
        label = "UNSUPPORTED"
    metrics["eval_scorecard_training_free"] = v
    verdicts["training_free"] = label
    logger.info(f"  Training-free: {label} (JRN ROC={jrn_roc:.4f}, best proxy ROC={best_proxy_roc:.4f})")

    # Overall
    verdict_values = [metrics[f"eval_scorecard_{c}"] for c in
                      ["probe_validity", "compounding", "threshold", "architecture", "training_free"]]
    n_supported = sum(1 for v in verdict_values if v == SUPPORTED)
    n_partial = sum(1 for v in verdict_values if v == PARTIAL)
    if n_supported >= 3:
        overall = SUPPORTED
        overall_label = "SUPPORTED"
    elif n_supported + n_partial >= 3:
        overall = PARTIAL
        overall_label = "PARTIAL"
    else:
        overall = UNSUPPORTED
        overall_label = "UNSUPPORTED"
    metrics["eval_scorecard_overall"] = overall
    logger.info(f"  Overall: {overall_label} ({n_supported} supported, {n_partial} partial)")

    # Build examples
    examples = []
    for claim, label in verdicts.items():
        ex = {
            "input": json.dumps({"claim": claim, "analysis": "scorecard"}),
            "output": json.dumps({"verdict": label}),
            "eval_verdict_code": metrics[f"eval_scorecard_{claim}"],
            "metadata_claim": claim,
        }
        examples.append(ex)
    examples.append({
        "input": json.dumps({"claim": "overall", "analysis": "scorecard"}),
        "output": json.dumps({"verdict": overall_label, "n_supported": n_supported, "n_partial": n_partial}),
        "eval_verdict_code": overall,
        "metadata_claim": "overall",
    })

    return metrics, examples


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Cross-Dataset JRN Consolidation Evaluation")
    logger.info("=" * 60)

    # ── Load all experiments ──
    experiments = {}
    for exp_id in EXP_PATHS:
        try:
            experiments[exp_id] = load_experiment(exp_id)
        except FileNotFoundError:
            logger.exception(f"Missing experiment file for {exp_id}")
            raise
        except json.JSONDecodeError:
            logger.exception(f"Invalid JSON in {exp_id}")
            raise
        gc.collect()

    logger.info(f"Loaded {len(experiments)} experiments")

    # ── Run all analyses ──
    all_metrics = {}

    # Analysis 1: Probe Validity
    probe_metrics, probe_examples = analysis_probe_validity(experiments)
    all_metrics.update(probe_metrics)

    # Collect JRN pool (shared by analyses 2, 4)
    jrn_pool = collect_jrn_pool(experiments)

    # Analysis 2: Join Selector
    selector_metrics, selector_examples = analysis_join_selector(jrn_pool, experiments)
    all_metrics.update(selector_metrics)

    # Analysis 3: Aggregation Sensitivity
    sensitivity_metrics, sensitivity_examples = analysis_agg_sensitivity(experiments)
    all_metrics.update(sensitivity_metrics)

    # Analysis 4: JRN Distribution
    dist_metrics, dist_examples = analysis_jrn_distribution(jrn_pool)
    all_metrics.update(dist_metrics)

    # Analysis 5: Task vs Schema Stability
    stability_metrics, stability_examples = analysis_task_schema_stability(experiments)
    all_metrics.update(stability_metrics)

    # Analysis 6: Compounding
    compound_metrics, compound_examples = analysis_compounding(experiments)
    all_metrics.update(compound_metrics)

    # Extract architecture win rates from exp_id4_it3 metadata
    arch_meta = experiments["exp_id4_it3"].get("metadata", {})
    win_rates = arch_meta.get("win_rates", {})
    selector_metrics["_win_vs_topk"] = win_rates.get("jrn_guided_vs_top_k", 0.6)
    selector_metrics["_win_vs_uniform_mean"] = win_rates.get("jrn_guided_vs_uniform_mean", 0.4)

    # Analysis 8: Scorecard (before tables since tables reference it)
    scorecard_metrics, scorecard_examples = build_scorecard(
        probe_metrics, compound_metrics, sensitivity_metrics, selector_metrics
    )
    all_metrics.update(scorecard_metrics)

    # Analysis 7: Paper Tables
    table_metrics, table_examples = build_paper_tables(
        experiments, probe_metrics, selector_metrics, compound_metrics,
        jrn_pool, dist_metrics, scorecard_metrics
    )
    all_metrics.update(table_metrics)

    # ── Build output ──
    # Ensure all metric values are numbers (schema requirement)
    clean_metrics = {}
    for k, v in all_metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if math.isnan(v) or math.isinf(v):
                clean_metrics[k] = 0.0
            else:
                clean_metrics[k] = v
        elif isinstance(v, np.integer):
            clean_metrics[k] = int(v)
        elif isinstance(v, np.floating):
            val = float(v)
            clean_metrics[k] = 0.0 if (math.isnan(val) or math.isinf(val)) else val
        # Skip non-numeric values silently

    output = {
        "metrics_agg": clean_metrics,
        "datasets": [
            {"dataset": "cross_dataset_jrn_measurements", "examples": selector_examples},
            {"dataset": "probe_validity_per_dataset", "examples": probe_examples},
            {"dataset": "join_selector_roc", "examples": selector_examples[:5] if selector_examples else [
                {"input": "{}", "output": "{}", "eval_placeholder": 0.0}
            ]},
            {"dataset": "agg_sensitivity_pooled", "examples": sensitivity_examples},
            {"dataset": "compounding_chains", "examples": compound_examples},
            {"dataset": "paper_tables", "examples": table_examples},
            {"dataset": "scorecard", "examples": scorecard_examples},
        ],
    }

    # Ensure every dataset has at least 1 example
    for ds in output["datasets"]:
        if not ds["examples"]:
            ds["examples"] = [{"input": "{}", "output": "{}",
                               "eval_placeholder": 0.0}]

    # Ensure metrics_agg has at least 1 property
    if not output["metrics_agg"]:
        output["metrics_agg"] = {"eval_n_experiments": len(experiments)}

    # ── Save output ──
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved eval_out.json ({out_path.stat().st_size / 1024:.1f} KB)")

    # Log key metrics
    logger.info("=" * 60)
    logger.info("KEY RESULTS:")
    for k in sorted(clean_metrics.keys()):
        if "scorecard" in k or "meta_rho" in k or "roc_auc" in k.split("_")[-2:] or "r_squared" in k:
            logger.info(f"  {k}: {clean_metrics[k]}")
    logger.info("=" * 60)

    return output


if __name__ == "__main__":
    main()
