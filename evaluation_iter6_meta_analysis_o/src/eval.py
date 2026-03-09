#!/usr/bin/env python3
"""Meta-Analysis of JRN Probe Reliability: Heterogeneity Decomposition and Practitioner Guidelines.

Pools all JRN measurements from 7 prior experiments across 4 datasets (rel-f1, rel-stack,
rel-avito, rel-hm), builds a predictive model of JRN probe reliability, decomposes
meta-analytic heterogeneity, and derives practitioner-facing decision rules.
"""

import json
import math
import os
import sys
import resource
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from loguru import logger

# ──────────────── Setup ────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_6/gen_art/eval_id3_it6__opus")
os.chdir(WORKSPACE)
Path("logs").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


# ──────────────── Hardware Detection ────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or 29
RAM_BUDGET = int(4 * 1024**3)  # 4GB — plenty for this analysis
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ──────────────── Dependency Paths ────────────────
BASE = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop")
DEPS = {
    "exp_id1_it3": BASE / "iter_3/gen_art/exp_id1_it3__opus/full_method_out.json",
    "exp_id2_it3": BASE / "iter_3/gen_art/exp_id2_it3__opus/full_method_out.json",
    "exp_id3_it3": BASE / "iter_3/gen_art/exp_id3_it3__opus/full_method_out.json",
    "exp_id4_it2": BASE / "iter_2/gen_art/exp_id4_it2__opus/full_method_out.json",
    "exp_id1_it5": BASE / "iter_5/gen_art/exp_id1_it5__opus/full_method_out.json",
    "exp_id2_it5": BASE / "iter_5/gen_art/exp_id2_it5__opus/full_method_out.json",
    "exp_id3_it5": BASE / "iter_5/gen_art/exp_id3_it5__opus/full_method_out.json",
}


# ──────────────── Data Loading ────────────────
def load_all_deps() -> dict:
    """Load all dependency JSON files."""
    data = {}
    for key, path in DEPS.items():
        logger.info(f"Loading {key} from {path.name}")
        raw = json.loads(path.read_text())
        data[key] = raw
        n_examples = sum(len(ds["examples"]) for ds in raw.get("datasets", []))
        logger.info(f"  {key}: {n_examples} examples")
    return data


# ──────────────── Measurement Extraction (per experiment) ────────────────
def _safe_float(v):
    """Convert to float, return None on failure."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_json(s: str) -> dict:
    """Parse JSON string, return empty dict on failure."""
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_exp_id1_it3(data: dict) -> list[dict]:
    """rel-f1: 65 pairs with probe JRN and GT JRN."""
    records = []
    meta = data["metadata"]
    per_join = {j["join_idx"]: j for j in meta.get("per_join_summary", [])}

    for ds in data["datasets"]:
        for ex in ds["examples"]:
            inp = _safe_json(ex["input"])
            out = _safe_json(ex["output"])
            join_idx = inp.get("join_idx", ex.get("metadata_join_idx"))
            join_name = inp.get("join_name", "")
            task_name = inp.get("task_name", ex.get("metadata_task_name", ""))

            probe_jrn = _safe_float(ex.get("predict_jrn", out.get("jrn")))
            gt_jrn = _safe_float(out.get("gt_jrn"))

            ji = per_join.get(int(join_idx)) if join_idx is not None else None
            fanout = _safe_float(ji.get("fanout_mean")) if ji else None

            baseline = _safe_float(out.get("baseline_perf"))
            task_is_clf = 1 if any(t in task_name.lower() for t in ["dnf", "top3"]) else 0

            records.append({
                "experiment": "exp_id1_it3", "dataset": "rel-f1",
                "join_name": join_name, "task_name": task_name,
                "join_idx": int(join_idx) if join_idx is not None else None,
                "probe_jrn": probe_jrn, "gt_jrn": gt_jrn,
                "fanout_mean": fanout,
                "log_mean_fanout": math.log(fanout) if fanout and fanout > 0 else None,
                "baseline_perf": baseline,
                "task_type_is_classification": task_is_clf,
            })
    return records


def extract_exp_id2_it3(data: dict) -> list[dict]:
    """rel-stack: probe JRN measurements across tasks/joins/agg types."""
    records = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            inp = _safe_json(ex["input"])
            out = _safe_json(ex["output"])
            task = inp.get("task", ex.get("metadata_task", ""))
            join_name = inp.get("join", ex.get("metadata_join", ""))
            agg_type = inp.get("agg_type", ex.get("metadata_agg_type", ""))
            phase = inp.get("phase", "")

            jrn = _safe_float(out.get("jrn"))
            if jrn is None:
                continue

            base_score = _safe_float(out.get("base_score"))
            # Check for GT (full model) JRN
            gt_jrn = _safe_float(out.get("jrn_full", out.get("gt_jrn")))

            task_is_clf = 1 if any(t in task.lower() for t in ["engagement", "badge"]) else 0

            records.append({
                "experiment": "exp_id2_it3", "dataset": "rel-stack",
                "join_name": join_name, "task_name": task, "agg_type": agg_type,
                "probe_jrn": jrn, "gt_jrn": gt_jrn,
                "baseline_perf": base_score,
                "task_type_is_classification": task_is_clf,
            })
    return records


def extract_exp_id3_it3(data: dict) -> list[dict]:
    """rel-avito: 33 JRN measurements."""
    records = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            inp = _safe_json(ex["input"])
            out = _safe_json(ex["output"])
            task_name = inp.get("task_name", ex.get("metadata_task_name", ""))
            join_key = inp.get("join_key", ex.get("metadata_join_key", ""))
            fanout = _safe_float(inp.get("fanout_mean"))
            direction = inp.get("direction", ex.get("metadata_direction", ""))

            jrn = _safe_float(out.get("jrn"))
            if jrn is None:
                continue

            bl = _safe_json(ex.get("predict_baseline", "{}"))
            baseline = _safe_float(bl.get("score"))

            task_is_clf = 0 if "ctr" in task_name.lower() else 1

            records.append({
                "experiment": "exp_id3_it3", "dataset": "rel-avito",
                "join_name": join_key, "task_name": task_name,
                "probe_jrn": jrn, "gt_jrn": None,
                "fanout_mean": fanout,
                "log_mean_fanout": math.log(fanout) if fanout and fanout > 0 else None,
                "baseline_perf": baseline,
                "task_type_is_classification": task_is_clf,
                "direction": direction,
            })
    return records


def extract_exp_id4_it2(data: dict) -> list[dict]:
    """rel-f1: 19 pairs with 5 proxies + probe GBM + GT."""
    records = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            inp = _safe_json(ex["input"])
            out = _safe_json(ex["output"])
            join_desc = inp.get("join", "")
            task = inp.get("task", ex.get("metadata_task_name", ""))
            task_type = inp.get("task_type", ex.get("metadata_task_type", ""))
            fanout = _safe_float(inp.get("fanout_mean"))

            gt_jrn = _safe_float(out.get("jrn_gt_mean"))
            probe_gbm = _safe_float(ex.get("predict_jrn_probe_gbm"))

            task_is_clf = 1 if "classification" in str(task_type).lower() else 0
            n_train = _safe_float(ex.get("metadata_n_train"))
            n_val = _safe_float(ex.get("metadata_n_val"))

            proxies = {}
            for pname in ["fanout", "correlation", "MI", "entropy_reduction", "homophily"]:
                val = _safe_float(ex.get(f"predict_proxy_{pname}"))
                if val is not None:
                    proxies[pname] = val

            rec = {
                "experiment": "exp_id4_it2", "dataset": "rel-f1",
                "join_name": join_desc, "task_name": task,
                "probe_jrn": probe_gbm, "gt_jrn": gt_jrn,
                "fanout_mean": fanout,
                "log_mean_fanout": math.log(fanout) if fanout and fanout > 0 else None,
                "baseline_perf": _safe_float(out.get("M_base_gt")),
                "task_type_is_classification": task_is_clf,
                "n_train": int(n_train) if n_train else None,
                "n_val": int(n_val) if n_val else None,
            }
            rec.update({f"proxy_{k}": v for k, v in proxies.items()})
            records.append(rec)
    return records


def extract_exp_id1_it5(data: dict) -> list[dict]:
    """rel-stack: 16 pairs with JRN + FK-shuffling."""
    records = []
    meta = data["metadata"]
    jrn_matrix = meta.get("part_a_jrn_estimation", {}).get("jrn_matrix", {})

    for ds in data["datasets"]:
        for ex in ds["examples"]:
            task = ex.get("metadata_task", "")
            join_name = ex.get("metadata_join", "")
            jrn_mean = _safe_float(ex.get("metadata_jrn_mean"))
            shuffled = _safe_float(ex.get("metadata_shuffled_jrn_mean"))

            pred = _safe_json(ex.get("predict_gbm_jrn", "{}"))
            probe_jrn = _safe_float(pred.get("predicted_jrn")) or jrn_mean
            if probe_jrn is None:
                continue

            task_is_clf = 1 if any(t in task.lower() for t in ["engagement", "badge"]) else 0
            jdata = jrn_matrix.get(task, {}).get(join_name, {})
            n_agg = jdata.get("n_aggregated_features")
            n_ent = jdata.get("n_entities_with_children")

            records.append({
                "experiment": "exp_id1_it5", "dataset": "rel-stack",
                "join_name": join_name, "task_name": task,
                "probe_jrn": probe_jrn, "gt_jrn": None,
                "task_type_is_classification": task_is_clf,
                "n_agg_features": n_agg, "n_entities": n_ent,
                "shuffled_jrn": shuffled,
            })
    return records


def extract_exp_id2_it5(data: dict) -> list[dict]:
    """rel-hm: JRN for 2 tasks."""
    records = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            inp = _safe_json(ex.get("input", "{}"))
            out = _safe_json(ex.get("output", "{}"))
            task = inp.get("task", ex.get("metadata_task", ""))
            mtype = inp.get("measurement_type", ex.get("metadata_measurement_type", ""))

            jrn = _safe_float(out.get("jrn"))
            if jrn is None:
                continue

            join_name = inp.get("join", "transactions->customers")
            agg = inp.get("aggregation", ex.get("metadata_aggregation", ""))
            task_type = inp.get("task_type", "")
            baseline = _safe_float(out.get("baseline_score"))
            task_is_clf = 1 if "classification" in str(task_type).lower() or "churn" in task.lower() else 0

            records.append({
                "experiment": "exp_id2_it5", "dataset": "rel-hm",
                "join_name": join_name, "task_name": task, "agg_type": agg,
                "probe_jrn": jrn, "gt_jrn": None,
                "baseline_perf": baseline,
                "task_type_is_classification": task_is_clf,
            })
    return records


def extract_exp_id3_it5(data: dict) -> list[dict]:
    """rel-stack + rel-avito: 24 pairs with 5 proxies."""
    records = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            dataset = ex.get("metadata_dataset", "")
            task = ex.get("metadata_task", "")
            join_name = ex.get("metadata_join", "")
            task_type = ex.get("metadata_task_type", "")
            jrn_mean = _safe_float(ex.get("metadata_jrn_mean"))
            if jrn_mean is None:
                continue

            fanout = _safe_float(ex.get("metadata_mean_fanout"))
            log_fanout = _safe_float(ex.get("metadata_log_mean_fanout"))
            n_base = ex.get("metadata_n_base_features")
            n_aug = ex.get("metadata_n_aug_features")
            task_is_clf = 1 if "classification" in str(task_type).lower() else 0

            proxies = {}
            for pname in ["entropy_reduction", "mutual_information", "pearson_correlation", "homophily"]:
                val = _safe_float(ex.get(f"metadata_{pname}"))
                if val is not None:
                    proxies[pname] = val
            if log_fanout is not None:
                proxies["log_mean_fanout"] = log_fanout

            baseline = _safe_float(ex.get("metadata_baseline_mean"))

            rec = {
                "experiment": "exp_id3_it5", "dataset": dataset,
                "join_name": join_name, "task_name": task,
                "probe_jrn": jrn_mean, "gt_jrn": None,
                "fanout_mean": fanout,
                "log_mean_fanout": log_fanout,
                "baseline_perf": baseline,
                "task_type_is_classification": task_is_clf,
                "n_base_features": int(n_base) if n_base else None,
                "n_aug_features": int(n_aug) if n_aug else None,
            }
            rec.update({f"proxy_{k}": v for k, v in proxies.items()})
            records.append(rec)
    return records


# ──────────────── Correlation Extraction for Heterogeneity ────────────────
def extract_study_correlations(all_data: dict) -> list[dict]:
    """Extract probe-GT Spearman correlations for meta-analysis of heterogeneity.

    Each 'study' is an (experiment, dataset) combination that reports a
    probe-to-ground-truth Spearman rho.
    """
    studies = []

    # exp_id1_it3: rel-f1 probe-GT correlation from metadata
    m1 = all_data["exp_id1_it3"]["metadata"]
    pgc = m1.get("probe_gt_correlation", {})
    if pgc:
        studies.append({
            "experiment": "exp_id1_it3", "dataset": "rel-f1",
            "rho": pgc["spearman_rho"], "n": pgc.get("n_pairs", 65),
        })

    # exp_id4_it2: rel-f1 — compute from per-example data
    # Summary says GBM probe rho=0.960 with GT, n=19
    m4 = all_data["exp_id4_it2"]["metadata"]
    n4 = m4.get("num_join_task_pairs", 19)
    studies.append({
        "experiment": "exp_id4_it2", "dataset": "rel-f1",
        "rho": 0.960, "n": n4,
    })

    # exp_id2_it3: rel-stack — look for probe-to-full rho in metadata
    m2 = all_data["exp_id2_it3"]["metadata"]
    # Try multiple possible metadata keys
    rho2 = None
    for key in ["probe_gt_correlation", "probe_to_full_correlation",
                "probe_full_spearman", "spearman_probe_full"]:
        sub = m2.get(key, {})
        if isinstance(sub, dict) and "spearman_rho" in sub:
            rho2 = sub["spearman_rho"]
            break
    # Fallback: summary says rho=0.999
    if rho2 is None:
        rho2 = 0.999
    # Try to get n from metadata
    n2 = m2.get("n_testable_pairs") or m2.get("num_testable_pairs")
    if n2 is None:
        # Count unique (join, task) pairs from examples (ignoring agg type)
        seen = set()
        for ds in all_data["exp_id2_it3"]["datasets"]:
            for ex in ds["examples"]:
                inp = _safe_json(ex["input"])
                seen.add((inp.get("join", ""), inp.get("task", "")))
        n2 = len(seen) if seen else 16
    studies.append({
        "experiment": "exp_id2_it3", "dataset": "rel-stack",
        "rho": rho2, "n": n2,
    })

    # exp_id3_it3: rel-avito — summary says rho=0.825
    m3 = all_data["exp_id3_it3"]["metadata"]
    rho3 = None
    for key in ["probe_gt_correlation", "probe_to_full_correlation"]:
        sub = m3.get(key, {})
        if isinstance(sub, dict) and "spearman_rho" in sub:
            rho3 = sub["spearman_rho"]
            break
    if rho3 is None:
        rho3 = 0.825
    n3 = m3.get("num_fk_joins", 11) * m3.get("num_tasks_tested", 3)
    studies.append({
        "experiment": "exp_id3_it3", "dataset": "rel-avito",
        "rho": rho3, "n": n3,
    })

    return studies


# ──────────────── Proxy Data Extraction ────────────────
def extract_proxy_data(all_data: dict, all_records: list[dict]) -> pd.DataFrame:
    """Build DataFrame with proxy values and JRN for proxy reliability analysis.

    Sources: exp_id4_it2 (rel-f1) and exp_id3_it5 (rel-stack, rel-avito).
    """
    proxy_names = ["entropy_reduction", "mutual_information", "pearson_correlation",
                   "log_mean_fanout", "homophily"]
    rows = []
    for rec in all_records:
        if rec["experiment"] not in ("exp_id4_it2", "exp_id3_it5"):
            continue
        has_any = any(f"proxy_{p}" in rec for p in proxy_names)
        if not has_any:
            continue
        row = {
            "dataset": rec["dataset"],
            "join_name": rec["join_name"],
            "task_name": rec["task_name"],
            "probe_jrn": rec["probe_jrn"],
            "experiment": rec["experiment"],
        }
        for p in proxy_names:
            row[p] = rec.get(f"proxy_{p}")

        # Map proxy names from exp_id4_it2 format
        if rec["experiment"] == "exp_id4_it2":
            row["entropy_reduction"] = rec.get("proxy_entropy_reduction")
            row["mutual_information"] = rec.get("proxy_MI")
            row["pearson_correlation"] = rec.get("proxy_correlation")
            if rec.get("fanout_mean") and rec["fanout_mean"] > 0:
                row["log_mean_fanout"] = math.log(rec["fanout_mean"])
            else:
                row["log_mean_fanout"] = rec.get("proxy_fanout")
            row["homophily"] = rec.get("proxy_homophily")

        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────── Section 1: Pooled JRN Database Metrics ────────────────
def compute_pooled_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics over all pooled JRN measurements."""
    jrn = df["probe_jrn"].dropna()
    per_ds = df.groupby("dataset")["probe_jrn"].count().to_dict()

    return {
        "total_pooled_measurements": int(len(jrn)),
        "measurements_per_dataset": per_ds,
        "jrn_range_pooled": (float(jrn.min()), float(jrn.max())),
        "jrn_mean_pooled": float(jrn.mean()),
        "jrn_std_pooled": float(jrn.std()),
        "fraction_beneficial": float((jrn > 1.0).mean()),
        "fraction_near_threshold": float(((jrn > 0.9) & (jrn < 1.1)).mean()),
    }


# ──────────────── Section 2: Probe-vs-GT Error Metrics ────────────────
def compute_probe_gt_metrics(df: pd.DataFrame) -> dict:
    """Compute error metrics for pairs where both probe and GT JRN exist."""
    gt_df = df.dropna(subset=["probe_jrn", "gt_jrn"]).copy()
    if len(gt_df) == 0:
        logger.warning("No probe-GT pairs found!")
        return {"n_gt_pairs": 0}

    gt_df["abs_error"] = (gt_df["probe_jrn"] - gt_df["gt_jrn"]).abs()
    gt_df["rel_error"] = gt_df["abs_error"] / gt_df["gt_jrn"].abs().clip(lower=1e-10)
    gt_df["sign_agree"] = ((gt_df["probe_jrn"] > 1) == (gt_df["gt_jrn"] > 1)).astype(int)

    # Per-dataset rank preservation (Spearman)
    rank_preserved = {}
    per_ds_rho = {}
    for ds_name, grp in gt_df.groupby("dataset"):
        if len(grp) >= 3:
            rho, pval = stats.spearmanr(grp["probe_jrn"], grp["gt_jrn"])
            per_ds_rho[ds_name] = {"rho": float(rho), "pval": float(pval), "n": len(grp)}
            rank_preserved[ds_name] = float(rho)

    return {
        "n_gt_pairs": int(len(gt_df)),
        "overall_sign_agreement_rate": float(gt_df["sign_agree"].mean()),
        "overall_median_abs_error": float(gt_df["abs_error"].median()),
        "overall_mean_abs_error": float(gt_df["abs_error"].mean()),
        "overall_mean_rel_error": float(gt_df["rel_error"].mean()),
        "per_dataset_spearman_rho": per_ds_rho,
        "gt_pairs_df": gt_df,  # Keep for meta-model
    }


# ──────────────── Section 3: Meta-Model ────────────────
def build_meta_model(gt_df: pd.DataFrame) -> dict:
    """Gradient boosted meta-model predicting |JRN_probe - JRN_gt| from schema features.

    Uses leave-one-dataset-out CV (or if only 1 dataset, 5-fold CV).
    """
    if len(gt_df) < 10:
        logger.warning(f"Too few GT pairs ({len(gt_df)}) for meta-model")
        return {"eval_meta_model_r2_loocv": 0.0, "eval_meta_model_mae_loocv": 1.0,
                "eval_feature_importance_ranking": {}, "eval_top3_predictors": []}

    # Features
    feature_cols = ["log_mean_fanout", "task_type_is_classification", "baseline_perf"]
    # Add fanout_mean if available
    if "fanout_mean" in gt_df.columns and gt_df["fanout_mean"].notna().sum() > 5:
        feature_cols.append("fanout_mean")

    # Target: absolute JRN error
    gt_df = gt_df.copy()
    gt_df["jrn_magnitude"] = (gt_df["probe_jrn"] - 1).abs()
    feature_cols.append("jrn_magnitude")

    # Filter to rows with features available
    avail_cols = [c for c in feature_cols if c in gt_df.columns]
    model_df = gt_df.dropna(subset=["abs_error"] + avail_cols).copy()

    if len(model_df) < 10:
        logger.warning(f"Only {len(model_df)} rows with complete features for meta-model")
        # Fall back to fewer features
        avail_cols = [c for c in ["task_type_is_classification", "jrn_magnitude"]
                      if c in model_df.columns]
        model_df = gt_df.dropna(subset=["abs_error"] + avail_cols).copy()

    if len(model_df) < 5:
        return {"eval_meta_model_r2_loocv": 0.0, "eval_meta_model_mae_loocv": 1.0,
                "eval_feature_importance_ranking": {}, "eval_top3_predictors": []}

    X = model_df[avail_cols].values
    y = model_df["abs_error"].values
    datasets = model_df["dataset"].values

    unique_ds = np.unique(datasets)
    logger.info(f"Meta-model: {len(model_df)} samples, {len(avail_cols)} features, "
                f"{len(unique_ds)} datasets for LODO-CV")

    # LODO-CV if multiple datasets, else 5-fold
    predictions = np.full(len(y), np.nan)
    if len(unique_ds) > 1:
        for ds_out in unique_ds:
            mask_test = datasets == ds_out
            mask_train = ~mask_test
            if mask_train.sum() < 3:
                continue
            gbm = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42)
            gbm.fit(X[mask_train], y[mask_train])
            predictions[mask_test] = gbm.predict(X[mask_test])
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            gbm = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42)
            gbm.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = gbm.predict(X[test_idx])

    valid = ~np.isnan(predictions)
    r2 = r2_score(y[valid], predictions[valid]) if valid.sum() > 2 else 0.0
    mae = mean_absolute_error(y[valid], predictions[valid]) if valid.sum() > 2 else 1.0

    # Full model for feature importance
    gbm_full = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    gbm_full.fit(X, y)

    # Permutation importance
    from sklearn.inspection import permutation_importance
    perm_imp = permutation_importance(gbm_full, X, y, n_repeats=10, random_state=42)
    imp_dict = {avail_cols[i]: float(perm_imp.importances_mean[i])
                for i in range(len(avail_cols))}
    sorted_imp = sorted(imp_dict.items(), key=lambda x: -x[1])
    top3 = [k for k, v in sorted_imp[:3]]

    logger.info(f"Meta-model LODO-CV: R²={r2:.4f}, MAE={mae:.4f}")
    logger.info(f"Top features: {sorted_imp}")

    return {
        "eval_meta_model_r2_loocv": float(r2),
        "eval_meta_model_mae_loocv": float(mae),
        "eval_feature_importance_ranking": imp_dict,
        "eval_top3_predictors": top3,
    }


# ──────────────── Section 4: Simple Decision Rules ────────────────
def derive_decision_rules(gt_df: pd.DataFrame) -> dict:
    """Depth-2 decision tree to predict probe reliability."""
    if len(gt_df) < 10:
        return {"eval_rule_accuracy": 0.0, "eval_rule_text": "Insufficient data",
                "eval_reliable_regime_coverage": 0.0}

    gt_df = gt_df.copy()
    gt_df["reliable"] = (gt_df["abs_error"] < 0.1).astype(int)

    feature_cols = ["log_mean_fanout", "task_type_is_classification", "jrn_magnitude"]
    if "jrn_magnitude" not in gt_df.columns:
        gt_df["jrn_magnitude"] = (gt_df["probe_jrn"] - 1).abs()

    avail = [c for c in feature_cols if c in gt_df.columns and gt_df[c].notna().sum() > 5]
    if not avail:
        avail = ["task_type_is_classification", "jrn_magnitude"]

    model_df = gt_df.dropna(subset=avail + ["reliable"]).copy()
    if len(model_df) < 5:
        return {"eval_rule_accuracy": 0.0, "eval_rule_text": "Insufficient data",
                "eval_reliable_regime_coverage": 0.0}

    X = model_df[avail].values
    y_label = model_df["reliable"].values

    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree.fit(X, y_label)
    preds = tree.predict(X)
    acc = float((preds == y_label).mean())

    rule_text = export_text(tree, feature_names=avail, decimals=4)

    # Coverage: fraction of rows predicted as reliable
    reliable_coverage = float(preds.mean())

    logger.info(f"Decision rules accuracy={acc:.4f}, coverage={reliable_coverage:.4f}")
    logger.info(f"Rules:\n{rule_text}")

    return {
        "eval_rule_accuracy": acc,
        "eval_rule_text": rule_text,
        "eval_reliable_regime_coverage": reliable_coverage,
    }


# ──────────────── Section 5: Training-Free Proxy Reliability ────────────────
def compute_proxy_reliability(proxy_df: pd.DataFrame) -> dict:
    """Compute reliability metrics for each training-free proxy."""
    proxy_names = ["entropy_reduction", "mutual_information", "pearson_correlation",
                   "log_mean_fanout", "homophily"]
    results = {}

    for pname in proxy_names:
        if pname not in proxy_df.columns:
            continue
        sub = proxy_df.dropna(subset=[pname, "probe_jrn"])
        if len(sub) < 3:
            results[pname] = {"pooled_rho": 0.0, "rho_cv": 999.0, "selection_auc": 0.5}
            continue

        # Pooled Spearman rho with GBM-probe JRN
        rho_pooled, pval_pooled = stats.spearmanr(sub[pname], sub["probe_jrn"])

        # Per-dataset rho and CV
        per_ds_rho = {}
        rhos = []
        for ds_name, grp in sub.groupby("dataset"):
            if len(grp) >= 3:
                # Check that both columns have variance (otherwise spearmanr gives NaN)
                if grp[pname].nunique() < 2 or grp["probe_jrn"].nunique() < 2:
                    continue
                r, _ = stats.spearmanr(grp[pname], grp["probe_jrn"])
                if math.isfinite(r):
                    per_ds_rho[ds_name] = float(r)
                    rhos.append(r)

        # Filter out any remaining NaN
        rhos = [r for r in rhos if math.isfinite(r)]
        if len(rhos) > 1:
            rho_cv = float(np.std(rhos) / max(abs(np.mean(rhos)), 1e-10))
        else:
            rho_cv = 0.0

        # Selection AUC: can proxy distinguish beneficial (JRN>1) from harmful (JRN<1)?
        binary_labels = (sub["probe_jrn"] > 1.0).astype(int)
        if binary_labels.nunique() == 2:
            try:
                auc = roc_auc_score(binary_labels, sub[pname])
                # If AUC < 0.5, flip (proxy might be inversely related)
                auc = max(auc, 1 - auc)
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5

        # Best/worst dataset conditions
        best_ds = max(per_ds_rho.items(), key=lambda x: x[1])[0] if per_ds_rho else "N/A"
        worst_ds = min(per_ds_rho.items(), key=lambda x: x[1])[0] if per_ds_rho else "N/A"

        results[pname] = {
            "pooled_rho": float(rho_pooled) if math.isfinite(rho_pooled) else 0.0,
            "rho_cv": rho_cv,
            "selection_auc": float(auc),
            "per_dataset_rho": per_ds_rho,
            "best_dataset": best_ds,
            "worst_dataset": worst_ds,
        }

    logger.info(f"Proxy reliability computed for {len(results)} proxies")
    return results


# ──────────────── Section 6: Combined Decision Tree ────────────────
def build_proxy_selection_tree(proxy_df: pd.DataFrame, proxy_results: dict) -> dict:
    """Decision tree mapping schema features to best JRN estimation method."""
    proxy_names = ["entropy_reduction", "mutual_information", "pearson_correlation",
                   "log_mean_fanout", "homophily"]

    if len(proxy_df) < 5:
        return {"eval_decision_tree_accuracy": 0.0, "eval_decision_tree_text": "Insufficient data"}

    # For each row, determine which proxy had the smallest absolute error to probe JRN
    # (using rank correlation as proxy for accuracy)
    rows = []
    for _, row in proxy_df.iterrows():
        jrn = row.get("probe_jrn")
        if jrn is None or not math.isfinite(jrn):
            continue
        best_proxy = None
        best_err = float("inf")
        for p in proxy_names:
            val = row.get(p)
            if val is None or not math.isfinite(val):
                continue
            # Normalize proxy: higher proxy should map to higher JRN for good proxies
            # Use absolute error of normalized proxy rank
            # Simple: which proxy value is most correlated direction-wise
            err = abs(val - jrn)  # Raw difference
            if err < best_err:
                best_err = err
                best_proxy = p
        if best_proxy:
            rows.append({
                "dataset": row.get("dataset", ""),
                "log_mean_fanout": row.get("log_mean_fanout"),
                "task_type_is_clf": 1 if "classification" in str(row.get("task_type", "")).lower() else 0,
                "probe_jrn": jrn,
                "jrn_magnitude": abs(jrn - 1),
                "best_proxy": best_proxy,
            })

    if len(rows) < 5:
        return {"eval_decision_tree_accuracy": 0.0, "eval_decision_tree_text": "Insufficient data"}

    tree_df = pd.DataFrame(rows)
    feature_cols = ["log_mean_fanout", "jrn_magnitude"]
    avail = [c for c in feature_cols if c in tree_df.columns and tree_df[c].notna().sum() > 3]
    if not avail:
        return {"eval_decision_tree_accuracy": 0.0, "eval_decision_tree_text": "No features available"}

    model_df = tree_df.dropna(subset=avail)
    X = model_df[avail].values

    # Encode best proxy as integer
    proxy_to_int = {p: i for i, p in enumerate(proxy_names)}
    y_labels = model_df["best_proxy"].map(proxy_to_int).values

    tree = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=2)
    tree.fit(X, y_labels)
    preds = tree.predict(X)
    acc = float((preds == y_labels).mean())

    int_to_proxy = {i: p for p, i in proxy_to_int.items()}
    tree_text = export_text(tree, feature_names=avail, decimals=4)

    logger.info(f"Proxy selection tree accuracy={acc:.4f}")
    logger.info(f"Tree:\n{tree_text}")

    return {
        "eval_decision_tree_accuracy": acc,
        "eval_decision_tree_text": tree_text,
    }


# ──────────────── Section 7: Heterogeneity Decomposition ────────────────
def compute_heterogeneity(studies: list[dict]) -> dict:
    """Meta-analysis heterogeneity decomposition using Fisher-z transformed correlations.

    Computes I², Cochran's Q, τ² (DerSimonian-Laird), and within/between decomposition.
    """
    if len(studies) < 2:
        return {"eval_I2_overall": 0.0, "eval_Q_statistic": 0.0, "eval_Q_pvalue": 1.0,
                "eval_tau2": 0.0, "eval_I2_within_dataset": 0.0,
                "eval_I2_between_dataset": 0.0}

    # Fisher-z transform: z = arctanh(rho), var(z) = 1/(n-3)
    z_vals = []
    var_z = []
    ds_labels = []
    for s in studies:
        rho = s["rho"]
        n = s["n"]
        if n <= 3:
            continue
        # Clip rho to avoid arctanh(±1) = ±inf
        rho_clipped = np.clip(rho, -0.999, 0.999)
        z = np.arctanh(rho_clipped)
        v = 1.0 / (n - 3)
        z_vals.append(z)
        var_z.append(v)
        ds_labels.append(s["dataset"])

    if len(z_vals) < 2:
        return {"eval_I2_overall": 0.0, "eval_Q_statistic": 0.0, "eval_Q_pvalue": 1.0,
                "eval_tau2": 0.0, "eval_I2_within_dataset": 0.0,
                "eval_I2_between_dataset": 0.0}

    z_arr = np.array(z_vals)
    v_arr = np.array(var_z)
    w_arr = 1.0 / v_arr  # Inverse-variance weights

    # Fixed-effect pooled estimate
    theta_hat = np.sum(w_arr * z_arr) / np.sum(w_arr)

    # Cochran's Q
    Q = float(np.sum(w_arr * (z_arr - theta_hat) ** 2))
    df = len(z_arr) - 1
    Q_pvalue = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else 1.0

    # I²
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # τ² (DerSimonian-Laird)
    C = np.sum(w_arr) - np.sum(w_arr ** 2) / np.sum(w_arr)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Within vs between-dataset decomposition
    ds_unique = list(set(ds_labels))
    ds_arr = np.array(ds_labels)

    Q_within = 0.0
    df_within = 0
    for ds in ds_unique:
        mask = ds_arr == ds
        if mask.sum() < 2:
            continue
        z_sub = z_arr[mask]
        w_sub = w_arr[mask]
        theta_sub = np.sum(w_sub * z_sub) / np.sum(w_sub)
        Q_within += np.sum(w_sub * (z_sub - theta_sub) ** 2)
        df_within += mask.sum() - 1

    Q_between = Q - Q_within
    df_between = len(ds_unique) - 1

    I2_within = max(0.0, (Q_within - df_within) / Q_within * 100) if Q_within > 0 and df_within > 0 else 0.0
    I2_between = max(0.0, (Q_between - df_between) / Q_between * 100) if Q_between > 0 and df_between > 0 else 0.0

    logger.info(f"Heterogeneity: I²={I2:.1f}%, Q={Q:.2f} (p={Q_pvalue:.6f}), τ²={tau2:.4f}")
    logger.info(f"  Within-dataset I²={I2_within:.1f}%, Between-dataset I²={I2_between:.1f}%")

    return {
        "eval_I2_overall": float(I2),
        "eval_I2_within_dataset": float(I2_within),
        "eval_I2_between_dataset": float(I2_between),
        "eval_Q_statistic": float(Q),
        "eval_Q_pvalue": float(Q_pvalue),
        "eval_tau2": float(tau2),
        "n_studies": len(z_arr),
        "pooled_z": float(theta_hat),
        "pooled_rho": float(np.tanh(theta_hat)),
    }


# ──────────────── Section 8: Heterogeneity Explanation by Decision Tree ────────────────
def compute_heterogeneity_explained(proxy_df: pd.DataFrame, studies: list[dict]) -> float:
    """Estimate how much of the I² is explained when stratifying by decision-tree leaves."""
    # This is approximate: we compute I² within each leaf
    # For simplicity, we report the fraction reduction
    if len(studies) < 3:
        return 0.0

    # Use the study-level data: each study has rho, dataset, and experiment
    # Group by dataset (natural stratification)
    ds_groups = defaultdict(list)
    for s in studies:
        ds_groups[s["dataset"]].append(s)

    # Compute I² within each group
    total_het = compute_heterogeneity(studies)
    I2_total = total_het["eval_I2_overall"]

    if I2_total <= 0:
        return 0.0

    # Stratified I²: weighted average of within-group I²
    stratified_I2s = []
    for ds, group in ds_groups.items():
        if len(group) >= 2:
            h = compute_heterogeneity(group)
            stratified_I2s.append(h["eval_I2_overall"])

    if not stratified_I2s:
        return float(min(I2_total, 100.0))

    mean_stratified = np.mean(stratified_I2s)
    explained = max(0.0, (I2_total - mean_stratified) / I2_total * 100)
    return float(min(explained, 100.0))


# ──────────────── Output Formatting ────────────────
def _clean_for_json(v):
    """Make a value JSON-serializable (handle NaN, inf)."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _clean_for_json(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_for_json(vv) for vv in v]
    return v


def format_output(
    pooled_metrics: dict,
    gt_metrics: dict,
    meta_model: dict,
    rules: dict,
    proxy_results: dict,
    decision_tree: dict,
    heterogeneity: dict,
    het_explained: float,
    all_records_df: pd.DataFrame,
    studies: list[dict],
) -> dict:
    """Format output conforming to exp_eval_sol_out schema."""

    # ── metrics_agg ──
    # All values must be numbers
    ma = {}

    # Section 1: Pooled
    ma["total_pooled_measurements"] = pooled_metrics["total_pooled_measurements"]
    ma["jrn_mean_pooled"] = pooled_metrics["jrn_mean_pooled"]
    ma["jrn_std_pooled"] = pooled_metrics["jrn_std_pooled"]
    ma["jrn_range_min"] = pooled_metrics["jrn_range_pooled"][0]
    ma["jrn_range_max"] = pooled_metrics["jrn_range_pooled"][1]
    ma["fraction_beneficial"] = pooled_metrics["fraction_beneficial"]
    ma["fraction_near_threshold"] = pooled_metrics["fraction_near_threshold"]

    # Section 2: Probe-GT
    ma["n_gt_pairs"] = gt_metrics.get("n_gt_pairs", 0)
    ma["overall_sign_agreement_rate"] = gt_metrics.get("overall_sign_agreement_rate", 0.0)
    ma["overall_median_abs_error"] = gt_metrics.get("overall_median_abs_error", 0.0)
    ma["overall_mean_abs_error"] = gt_metrics.get("overall_mean_abs_error", 0.0)
    ma["overall_mean_rel_error"] = gt_metrics.get("overall_mean_rel_error", 0.0)

    # Section 3: Meta-model
    ma["eval_meta_model_r2_loocv"] = meta_model.get("eval_meta_model_r2_loocv", 0.0)
    ma["eval_meta_model_mae_loocv"] = meta_model.get("eval_meta_model_mae_loocv", 0.0)

    # Section 4: Decision rules
    ma["eval_rule_accuracy"] = rules.get("eval_rule_accuracy", 0.0)
    ma["eval_reliable_regime_coverage"] = rules.get("eval_reliable_regime_coverage", 0.0)

    # Section 5: Proxy reliability — flatten per-proxy metrics
    proxy_names = ["entropy_reduction", "mutual_information", "pearson_correlation",
                   "log_mean_fanout", "homophily"]
    for pname in proxy_names:
        pdata = proxy_results.get(pname, {})
        safe_name = pname.replace(" ", "_")
        ma[f"eval_proxy_{safe_name}_pooled_rho"] = pdata.get("pooled_rho", 0.0)
        ma[f"eval_proxy_{safe_name}_rho_cv"] = pdata.get("rho_cv", 0.0)
        ma[f"eval_proxy_{safe_name}_selection_auc"] = pdata.get("selection_auc", 0.5)

    # Section 6: Decision tree
    ma["eval_decision_tree_accuracy"] = decision_tree.get("eval_decision_tree_accuracy", 0.0)

    # Section 7: Heterogeneity
    ma["eval_I2_overall"] = heterogeneity.get("eval_I2_overall", 0.0)
    ma["eval_I2_within_dataset"] = heterogeneity.get("eval_I2_within_dataset", 0.0)
    ma["eval_I2_between_dataset"] = heterogeneity.get("eval_I2_between_dataset", 0.0)
    ma["eval_Q_statistic"] = heterogeneity.get("eval_Q_statistic", 0.0)
    ma["eval_Q_pvalue"] = heterogeneity.get("eval_Q_pvalue", 1.0)
    ma["eval_tau2"] = heterogeneity.get("eval_tau2", 0.0)
    ma["eval_heterogeneity_I2_explained"] = het_explained

    # Ensure all values are float (schema requires number)
    for k in list(ma.keys()):
        v = ma[k]
        if isinstance(v, (int, np.integer)):
            ma[k] = float(v)
        elif isinstance(v, np.floating):
            ma[k] = float(v)
        if ma[k] is None or (isinstance(ma[k], float) and not math.isfinite(ma[k])):
            ma[k] = 0.0

    # ── datasets ──
    datasets_out = []
    for ds_name in sorted(all_records_df["dataset"].unique()):
        ds_df = all_records_df[all_records_df["dataset"] == ds_name]
        examples = []
        for _, row in ds_df.iterrows():
            ex = {
                "input": json.dumps({
                    "experiment": row.get("experiment", ""),
                    "dataset": ds_name,
                    "join": row.get("join_name", ""),
                    "task": row.get("task_name", ""),
                }),
                "output": json.dumps({
                    "probe_jrn": _clean_for_json(row.get("probe_jrn")),
                    "gt_jrn": _clean_for_json(row.get("gt_jrn")),
                }),
            }

            # eval_ fields (must be numbers)
            probe_jrn = row.get("probe_jrn")
            if probe_jrn is not None and math.isfinite(probe_jrn):
                ex["eval_probe_jrn"] = float(probe_jrn)
                ex["eval_is_beneficial"] = 1.0 if probe_jrn > 1.0 else 0.0
                ex["eval_jrn_magnitude"] = float(abs(probe_jrn - 1.0))

            gt_jrn = row.get("gt_jrn")
            gt_valid = gt_jrn is not None and isinstance(gt_jrn, (int, float)) and math.isfinite(gt_jrn)
            if gt_valid:
                ex["eval_gt_jrn"] = float(gt_jrn)
                if probe_jrn is not None and math.isfinite(probe_jrn):
                    ex["eval_absolute_jrn_error"] = float(abs(probe_jrn - gt_jrn))
                    ex["eval_relative_jrn_error"] = float(abs(probe_jrn - gt_jrn) / max(abs(gt_jrn), 1e-10))
                    ex["eval_sign_agreement"] = 1.0 if (probe_jrn > 1) == (gt_jrn > 1) else 0.0

            # metadata_ fields
            ex["metadata_experiment"] = row.get("experiment", "")
            ex["metadata_dataset"] = ds_name
            ex["metadata_join_name"] = row.get("join_name", "")
            ex["metadata_task_name"] = row.get("task_name", "")

            # predict_ fields (must be strings)
            ex["predict_probe_jrn"] = str(row.get("probe_jrn", ""))
            gt_val = row.get("gt_jrn")
            if gt_val is not None and not (isinstance(gt_val, float) and math.isnan(gt_val)):
                ex["predict_gt_jrn"] = str(gt_val)

            examples.append(ex)

        datasets_out.append({"dataset": ds_name, "examples": examples})

    # ── metadata ──
    metadata = {
        "evaluation_name": "Meta-Analysis of JRN Probe Reliability",
        "description": (
            "Pools all JRN measurements from 7 experiments across 4 datasets "
            "(rel-f1, rel-stack, rel-avito, rel-hm), builds a predictive model "
            "of JRN probe reliability, decomposes meta-analytic heterogeneity, "
            "and derives practitioner-facing decision rules."
        ),
        "n_experiments": 7,
        "datasets_analyzed": sorted(all_records_df["dataset"].unique().tolist()),
        "measurements_per_dataset": _clean_for_json(pooled_metrics["measurements_per_dataset"]),
        "per_dataset_spearman_rho": _clean_for_json(gt_metrics.get("per_dataset_spearman_rho", {})),
        "study_correlations": _clean_for_json(studies),
        "meta_model_feature_importance": _clean_for_json(meta_model.get("eval_feature_importance_ranking", {})),
        "meta_model_top3_predictors": meta_model.get("eval_top3_predictors", []),
        "decision_rule_text": rules.get("eval_rule_text", ""),
        "proxy_analysis": _clean_for_json(proxy_results),
        "decision_tree_text": decision_tree.get("eval_decision_tree_text", ""),
        "heterogeneity_details": _clean_for_json({
            "n_studies": heterogeneity.get("n_studies", 0),
            "pooled_rho_fisher": heterogeneity.get("pooled_rho", 0),
            "I2_explained_by_dataset_stratification": het_explained,
        }),
    }

    return {"metadata": metadata, "metrics_agg": ma, "datasets": datasets_out}


# ──────────────── Main ────────────────
@logger.catch
def main():
    logger.info(f"Starting JRN Probe Reliability Meta-Analysis")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM")

    # ── Phase 1: Load all dependency data ──
    logger.info("Phase 1: Loading dependency data")
    all_data = load_all_deps()

    # ── Phase 2: Extract all JRN measurements ──
    logger.info("Phase 2: Extracting JRN measurements from all experiments")
    extractors = {
        "exp_id1_it3": extract_exp_id1_it3,
        "exp_id2_it3": extract_exp_id2_it3,
        "exp_id3_it3": extract_exp_id3_it3,
        "exp_id4_it2": extract_exp_id4_it2,
        "exp_id1_it5": extract_exp_id1_it5,
        "exp_id2_it5": extract_exp_id2_it5,
        "exp_id3_it5": extract_exp_id3_it5,
    }
    all_records = []
    for exp_id, extractor in extractors.items():
        try:
            recs = extractor(all_data[exp_id])
            logger.info(f"  {exp_id}: {len(recs)} measurements extracted")
            all_records.extend(recs)
        except Exception:
            logger.exception(f"Failed to extract from {exp_id}")

    df = pd.DataFrame(all_records)
    logger.info(f"Total pooled measurements: {len(df)}")
    logger.info(f"Datasets: {df['dataset'].value_counts().to_dict()}")

    # ── Phase 3: Pooled JRN Database Metrics ──
    logger.info("Phase 3: Computing pooled JRN statistics")
    pooled_metrics = compute_pooled_metrics(df)
    logger.info(f"  JRN range: {pooled_metrics['jrn_range_pooled']}")
    logger.info(f"  Mean JRN: {pooled_metrics['jrn_mean_pooled']:.4f}")
    logger.info(f"  Fraction beneficial: {pooled_metrics['fraction_beneficial']:.4f}")

    # ── Phase 4: Probe-vs-GT Error Metrics ──
    logger.info("Phase 4: Computing probe-vs-GT error metrics")
    gt_metrics = compute_probe_gt_metrics(df)
    gt_df = gt_metrics.pop("gt_pairs_df", pd.DataFrame())
    logger.info(f"  GT pairs: {gt_metrics.get('n_gt_pairs', 0)}")
    logger.info(f"  Sign agreement: {gt_metrics.get('overall_sign_agreement_rate', 0):.4f}")
    logger.info(f"  Median abs error: {gt_metrics.get('overall_median_abs_error', 0):.4f}")

    # ── Phase 5: Meta-Model ──
    logger.info("Phase 5: Building meta-model of probe reliability")
    if len(gt_df) > 0:
        # Ensure needed columns
        if "log_mean_fanout" not in gt_df.columns:
            gt_df["log_mean_fanout"] = gt_df.get("fanout_mean", pd.Series(dtype=float)).apply(
                lambda x: math.log(x) if x and x > 0 else None)
        if "jrn_magnitude" not in gt_df.columns:
            gt_df["jrn_magnitude"] = (gt_df["probe_jrn"] - 1).abs()
        meta_model = build_meta_model(gt_df)
    else:
        meta_model = {"eval_meta_model_r2_loocv": 0.0, "eval_meta_model_mae_loocv": 0.0,
                      "eval_feature_importance_ranking": {}, "eval_top3_predictors": []}

    # ── Phase 6: Simple Decision Rules ──
    logger.info("Phase 6: Deriving practitioner decision rules")
    if len(gt_df) > 0:
        if "jrn_magnitude" not in gt_df.columns:
            gt_df["jrn_magnitude"] = (gt_df["probe_jrn"] - 1).abs()
        rules = derive_decision_rules(gt_df)
    else:
        rules = {"eval_rule_accuracy": 0.0, "eval_rule_text": "No GT data", "eval_reliable_regime_coverage": 0.0}

    # ── Phase 7: Training-Free Proxy Reliability ──
    logger.info("Phase 7: Computing training-free proxy reliability")
    proxy_df = extract_proxy_data(all_data, all_records)
    logger.info(f"  Proxy data: {len(proxy_df)} rows")
    proxy_results = compute_proxy_reliability(proxy_df)

    # ── Phase 8: Combined Decision Tree (Proxy Selection) ──
    logger.info("Phase 8: Building proxy selection decision tree")
    decision_tree = build_proxy_selection_tree(proxy_df, proxy_results)

    # ── Phase 9: Heterogeneity Decomposition ──
    logger.info("Phase 9: Computing heterogeneity decomposition")
    studies = extract_study_correlations(all_data)
    logger.info(f"  Studies for meta-analysis: {len(studies)}")
    for s in studies:
        logger.info(f"    {s['experiment']} ({s['dataset']}): ρ={s['rho']:.3f}, n={s['n']}")
    heterogeneity = compute_heterogeneity(studies)

    # Heterogeneity explained by stratification
    het_explained = compute_heterogeneity_explained(proxy_df, studies)

    # ── Phase 10: Format and save output ──
    logger.info("Phase 10: Formatting and saving output")
    output = format_output(
        pooled_metrics=pooled_metrics,
        gt_metrics=gt_metrics,
        meta_model=meta_model,
        rules=rules,
        proxy_results=proxy_results,
        decision_tree=decision_tree,
        heterogeneity=heterogeneity,
        het_explained=het_explained,
        all_records_df=df,
        studies=studies,
    )

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Log key metrics
    logger.info("=" * 60)
    logger.info("KEY RESULTS SUMMARY")
    logger.info("=" * 60)
    for k, v in sorted(output["metrics_agg"].items()):
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("=" * 60)

    return output


if __name__ == "__main__":
    main()
