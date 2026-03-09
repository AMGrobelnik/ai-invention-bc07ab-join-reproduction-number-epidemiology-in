#!/usr/bin/env python3
"""
GBM-Probe JRN Estimation on rel-hm (H&M) with Corrected Metric Direction,
Fan-Out Stratification, FK-Shuffling Confound Decomposition, and JRN-Guided
Architecture Comparison.

Self-contained: method.py -> method_out.json
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

# ============================================================
# LOGGING
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION (cgroup-aware)
# ============================================================

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
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM (container), {AVAILABLE_RAM_GB:.1f} GB available")

# Set RAM budget to 70% of container limit
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.70 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB, CPU time limit: 3600s")

# ============================================================
# PATHS & CONFIG
# ============================================================
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id5_it1__opus/data_out")

SEEDS = [42, 123, 456]
PROBE_CONFIG = {
    "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1, "n_jobs": NUM_CPUS,
}

FULL_CONFIG = {
    "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "num_leaves": 127,
    "verbose": -1, "n_jobs": NUM_CPUS,
}

TASK_CONFIG = {
    "user-churn": {
        "entity_table": "customers", "entity_col": "customer_id",
        "task_type": "classification", "metric": "AUROC", "higher_is_better": True,
        "join": {"child_table": "transactions", "fk_col": "customer_id"},
    },
    "item-sales": {
        "entity_table": "articles", "entity_col": "article_id",
        "task_type": "regression", "metric": "MAE", "higher_is_better": False,
        "join": {"child_table": "transactions", "fk_col": "article_id"},
    },
}

AGG_TYPES = ["mean", "sum", "max", "std", "all_combined"]

FAN_OUT_BUCKETS = [
    (1, 5, "1-5"), (6, 20, "6-20"), (21, 50, "21-50"),
    (51, 200, "51-200"), (201, float("inf"), "200+"),
]

# ============================================================
# PHASE 0: DATA LOADING & FEATURE ENGINEERING
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load examples from 3 split JSON files, return per-table DataFrames."""
    logger.info("Loading data from split JSON files...")
    customers_records: list[dict] = []
    articles_records: list[dict] = []
    transactions_records: list[dict] = []

    for part_file in sorted(DATA_DIR.glob("full_data_out_*.json")):
        logger.info(f"  Loading {part_file.name} ...")
        t0 = time.time()
        raw = json.loads(part_file.read_text())
        examples = raw["datasets"][0]["examples"]
        logger.info(f"    {len(examples)} examples, parsed in {time.time()-t0:.1f}s")

        for ex in examples:
            table = ex["metadata_table"]
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])
            fold = ex["metadata_fold"]
            fan_out = ex.get("metadata_fan_out", 0)
            fan_out_bucket = ex.get("metadata_fan_out_bucket", "")

            feats = inp["features"]
            row_id = inp["row_id"]

            if table == "customers":
                customers_records.append({
                    "_row_id": row_id, "_fold": fold,
                    "_fan_out": fan_out, "_fan_out_bucket": fan_out_bucket,
                    "_label_churn": out.get("user-churn"),
                    "FN": feats.get("FN"), "Active": feats.get("Active"),
                    "club_member_status": feats.get("club_member_status"),
                    "fashion_news_frequency": feats.get("fashion_news_frequency"),
                    "age": feats.get("age"),
                    # DROP postal_code (too-high-cardinality hash)
                })
            elif table == "articles":
                rec = {
                    "_row_id": row_id, "_fold": fold,
                    "_fan_out": fan_out, "_fan_out_bucket": fan_out_bucket,
                    "_label_sales": out.get("item-sales"),
                }
                # Keep numeric cols; label-encode low-cardinality strings; drop text
                for k, v in feats.items():
                    if k in ("prod_name", "detail_desc", "article_id"):
                        continue  # drop text / primary key
                    rec[k] = v
                articles_records.append(rec)
            elif table == "transactions":
                transactions_records.append({
                    "_row_id": row_id, "_fold": fold,
                    "_fan_out": fan_out, "_fan_out_bucket": fan_out_bucket,
                    "customer_id": feats.get("customer_id"),
                    "article_id": feats.get("article_id"),
                    "price": feats.get("price"),
                    "sales_channel_id": feats.get("sales_channel_id"),
                    # DROP t_dat (time)
                })

        del raw, examples
        gc.collect()

    customers_df = pd.DataFrame(customers_records)
    articles_df = pd.DataFrame(articles_records)
    transactions_df = pd.DataFrame(transactions_records)

    del customers_records, articles_records, transactions_records
    gc.collect()

    logger.info(f"Data loaded: customers={len(customers_df)}, articles={len(articles_df)}, transactions={len(transactions_df)}")
    return customers_df, articles_df, transactions_df


def encode_features(customers_df: pd.DataFrame, articles_df: pd.DataFrame,
                    transactions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encode categorical features for LightGBM."""
    logger.info("Encoding features...")

    # --- Customers ---
    # FN, Active: already numeric (float with NaN)
    for col in ["FN", "Active", "age"]:
        customers_df[col] = pd.to_numeric(customers_df[col], errors="coerce")

    # Encode categoricals
    for col in ["club_member_status", "fashion_news_frequency"]:
        customers_df[col] = customers_df[col].astype("category").cat.codes.replace(-1, np.nan)

    # --- Articles ---
    # Identify numeric vs string columns (excluding meta cols)
    meta_cols = [c for c in articles_df.columns if c.startswith("_")]
    feat_cols = [c for c in articles_df.columns if not c.startswith("_")]

    for col in feat_cols:
        # Try numeric conversion first
        converted = pd.to_numeric(articles_df[col], errors="coerce")
        non_null_orig = articles_df[col].notna().sum()
        non_null_conv = converted.notna().sum()
        if non_null_conv >= non_null_orig * 0.8:  # mostly numeric
            articles_df[col] = converted
        else:
            # Label encode string columns with reasonable cardinality
            nunique = articles_df[col].nunique()
            if nunique < 1000:
                articles_df[col] = articles_df[col].astype("category").cat.codes.replace(-1, np.nan)
            else:
                articles_df.drop(columns=[col], inplace=True)

    # --- Transactions ---
    for col in ["price", "sales_channel_id"]:
        transactions_df[col] = pd.to_numeric(transactions_df[col], errors="coerce")

    logger.info(f"Customer features: {[c for c in customers_df.columns if not c.startswith('_')]}")
    logger.info(f"Article features: {[c for c in articles_df.columns if not c.startswith('_')]}")
    logger.info(f"Transaction features: {[c for c in transactions_df.columns if not c.startswith('_')]}")

    return customers_df, articles_df, transactions_df


# ============================================================
# HELPER: Feature columns
# ============================================================

def get_feat_cols(df: pd.DataFrame) -> list[str]:
    """Return non-meta, non-label columns."""
    return [c for c in df.columns if not c.startswith("_") and c not in ("customer_id", "article_id")]


def get_entity_feat_cols(df: pd.DataFrame) -> list[str]:
    """Return entity feature columns (non-meta)."""
    return [c for c in df.columns if not c.startswith("_")]


# ============================================================
# PHASE 1: BASELINE + JRN ESTIMATION
# ============================================================

def aggregate_transactions(transactions_df: pd.DataFrame, fk_col: str,
                           agg_type: str) -> pd.DataFrame:
    """Aggregate transaction features to entity level."""
    txn_feat_cols = ["price", "sales_channel_id"]
    available_cols = [c for c in txn_feat_cols if c in transactions_df.columns]

    if agg_type == "all_combined":
        agg_funcs = ["mean", "sum", "max", "std", "min"]
        agg_dict = {col: agg_funcs for col in available_cols}
        grouped = transactions_df.groupby(fk_col)[available_cols].agg(agg_dict)
        # Flatten column names
        grouped.columns = [f"txn_{col}_{func}" for col, func in grouped.columns]
    else:
        grouped = transactions_df.groupby(fk_col)[available_cols].agg(agg_type)
        grouped.columns = [f"txn_{col}_{agg_type}" for col in grouped.columns]

    # Add count feature
    counts = transactions_df.groupby(fk_col).size().rename("txn_count")
    grouped = grouped.join(counts)

    grouped = grouped.reset_index()
    return grouped


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       task_type: str, config: dict, seed: int) -> float:
    """Train LightGBM and evaluate."""
    import lightgbm as lgb
    import warnings

    params = {**config, "random_state": seed}
    if task_type == "classification":
        # Check for single-class validation set
        if y_val.nunique() < 2:
            logger.warning("  Single class in val set — AUROC undefined, returning NaN")
            return float("nan")
        model = lgb.LGBMClassifier(**params, class_weight="balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
    else:  # regression
        model = lgb.LGBMRegressor(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = mean_absolute_error(y_val, preds)
    del model
    return score


def compute_jrn(baseline_score: float, augmented_score: float,
                higher_is_better: bool) -> float:
    """Compute JRN with CORRECTED direction for lower-is-better metrics."""
    if higher_is_better:
        # AUROC: higher=better, so JRN = augmented/baseline > 1 means improvement
        return augmented_score / baseline_score if baseline_score > 0 else 1.0
    else:
        # MAE: lower=better, so JRN = baseline/augmented > 1 means improvement
        return baseline_score / augmented_score if augmented_score > 0 else 1.0


def run_phase1(customers_df: pd.DataFrame, articles_df: pd.DataFrame,
               transactions_df: pd.DataFrame) -> dict:
    """Phase 1: Baseline + JRN estimation for 2 tasks x 5 agg types x 3 seeds."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Baseline + JRN Estimation")
    logger.info("=" * 60)

    results = {}

    for task_name, task_cfg in TASK_CONFIG.items():
        logger.info(f"\n--- Task: {task_name} ({task_cfg['task_type']}) ---")
        entity_table = task_cfg["entity_table"]
        entity_col = task_cfg["entity_col"]
        fk_col = task_cfg["join"]["fk_col"]
        task_type = task_cfg["task_type"]
        higher_is_better = task_cfg["higher_is_better"]

        # Select entity DataFrame
        if entity_table == "customers":
            entity_df = customers_df.copy()
            label_col = "_label_churn"
        else:
            entity_df = articles_df.copy()
            label_col = "_label_sales"

        # Entity feature columns
        entity_feat_cols = get_entity_feat_cols(entity_df)
        entity_feat_cols = [c for c in entity_feat_cols if c != label_col]

        # Split by fold
        train_mask = entity_df["_fold"] == 0

        # Drop rows without labels
        has_label = entity_df[label_col].notna()
        train_idx = entity_df[train_mask & has_label].index

        # For val: combine fold 1+2 when either is too small (< 100)
        val_fold1 = entity_df[(entity_df["_fold"] == 1) & has_label]
        val_fold2 = entity_df[(entity_df["_fold"] == 2) & has_label]
        if len(val_fold1) < 100 or len(val_fold2) < 100:
            # Combine both folds for validation
            val_mask = (entity_df["_fold"] == 1) | (entity_df["_fold"] == 2)
            val_idx = entity_df[val_mask & has_label].index
            logger.info(f"  Combined fold 1+2 for val: {len(val_fold1)}+{len(val_fold2)}={len(val_idx)}")
        else:
            val_mask = entity_df["_fold"] == 1
            val_idx = entity_df[val_mask & has_label].index

        logger.info(f"  Entity: {entity_table}, train={len(train_idx)}, val={len(val_idx)}")

        X_train_base = entity_df.loc[train_idx, entity_feat_cols].copy()
        y_train = entity_df.loc[train_idx, label_col].copy()
        X_val_base = entity_df.loc[val_idx, entity_feat_cols].copy()
        y_val = entity_df.loc[val_idx, label_col].copy()

        # Ensure numeric
        for col in X_train_base.columns:
            X_train_base[col] = pd.to_numeric(X_train_base[col], errors="coerce")
            X_val_base[col] = pd.to_numeric(X_val_base[col], errors="coerce")

        # --- Baseline ---
        logger.info(f"  Training baseline ({len(entity_feat_cols)} features)...")
        baseline_scores = []
        for seed in SEEDS:
            try:
                score = train_and_evaluate(X_train_base, y_train, X_val_base, y_val,
                                           task_type, PROBE_CONFIG, seed)
                if not np.isnan(score):
                    baseline_scores.append(score)
            except Exception:
                logger.exception(f"  Baseline failed seed={seed}")
        baseline_mean = np.mean(baseline_scores) if baseline_scores else 0.0
        baseline_std = np.std(baseline_scores) if len(baseline_scores) > 1 else 0.0
        logger.info(f"  Baseline {task_cfg['metric']}: {baseline_mean:.4f} +/- {baseline_std:.4f}")

        # --- With-join for each agg type ---
        task_results = {
            "baseline": {"mean": baseline_mean, "std": baseline_std, "scores": baseline_scores},
            "agg_results": {},
        }

        for agg_type in AGG_TYPES:
            logger.info(f"  Aggregation: {agg_type}")
            try:
                agg_df = aggregate_transactions(transactions_df, fk_col, agg_type)
                agg_feat_cols = [c for c in agg_df.columns if c != fk_col]

                # Merge entity features with aggregated transaction features
                # Need to map entity_df._row_id to fk_col values
                # For customers: _row_id is customer_id; for articles: _row_id is article_id
                merged_train = entity_df.loc[train_idx].merge(
                    agg_df, left_on="_row_id", right_on=fk_col, how="left"
                )
                merged_val = entity_df.loc[val_idx].merge(
                    agg_df, left_on="_row_id", right_on=fk_col, how="left"
                )

                all_feat_cols = entity_feat_cols + agg_feat_cols
                # Remove duplicate fk_col if it appeared
                all_feat_cols = [c for c in all_feat_cols if c in merged_train.columns]
                all_feat_cols = list(dict.fromkeys(all_feat_cols))  # dedupe preserving order

                X_train_aug = merged_train[all_feat_cols].copy()
                X_val_aug = merged_val[all_feat_cols].copy()

                # Ensure numeric
                for col in X_train_aug.columns:
                    X_train_aug[col] = pd.to_numeric(X_train_aug[col], errors="coerce")
                    X_val_aug[col] = pd.to_numeric(X_val_aug[col], errors="coerce")

                augmented_scores = []
                for seed in SEEDS:
                    try:
                        score = train_and_evaluate(X_train_aug, y_train, X_val_aug, y_val,
                                                   task_type, PROBE_CONFIG, seed)
                        if not np.isnan(score):
                            augmented_scores.append(score)
                    except Exception:
                        logger.exception(f"  Augmented failed agg={agg_type}, seed={seed}")

                aug_mean = np.mean(augmented_scores) if augmented_scores else baseline_mean
                aug_std = np.std(augmented_scores) if len(augmented_scores) > 1 else 0.0
                jrn = compute_jrn(baseline_mean, aug_mean, higher_is_better)

                logger.info(f"    Augmented {task_cfg['metric']}: {aug_mean:.4f} +/- {aug_std:.4f}")
                logger.info(f"    JRN = {jrn:.4f} (baseline={baseline_mean:.4f}, augmented={aug_mean:.4f})")

                # Verify JRN direction for regression
                if not higher_is_better:
                    if aug_mean < baseline_mean:
                        assert jrn > 1.0, f"BUG: aug_MAE < baseline_MAE but JRN={jrn} <= 1"
                    elif aug_mean > baseline_mean:
                        assert jrn < 1.0, f"BUG: aug_MAE > baseline_MAE but JRN={jrn} >= 1"

                task_results["agg_results"][agg_type] = {
                    "augmented_mean": aug_mean, "augmented_std": aug_std,
                    "augmented_scores": augmented_scores,
                    "jrn": jrn, "n_agg_features": len(agg_feat_cols),
                }

                del agg_df, merged_train, merged_val, X_train_aug, X_val_aug
                gc.collect()

            except Exception:
                logger.exception(f"  Failed agg_type={agg_type}")
                task_results["agg_results"][agg_type] = {
                    "augmented_mean": baseline_mean, "augmented_std": 0.0,
                    "jrn": 1.0, "error": "failed",
                }

        results[task_name] = task_results
        del entity_df
        gc.collect()

    return results


# ============================================================
# PHASE 2: FAN-OUT STRATIFICATION
# ============================================================

def run_phase2(customers_df: pd.DataFrame, articles_df: pd.DataFrame,
               transactions_df: pd.DataFrame) -> dict:
    """Phase 2: Fan-out stratified JRN estimation."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Fan-Out Stratification")
    logger.info("=" * 60)

    results = {}
    agg_type = "mean"  # Use mean aggregation for stratification

    for task_name, task_cfg in TASK_CONFIG.items():
        logger.info(f"\n--- Task: {task_name} ---")
        entity_table = task_cfg["entity_table"]
        fk_col = task_cfg["join"]["fk_col"]
        task_type = task_cfg["task_type"]
        higher_is_better = task_cfg["higher_is_better"]

        if entity_table == "customers":
            entity_df = customers_df
            label_col = "_label_churn"
        else:
            entity_df = articles_df
            label_col = "_label_sales"

        entity_feat_cols = get_entity_feat_cols(entity_df)
        entity_feat_cols = [c for c in entity_feat_cols if c != label_col]

        has_label = entity_df[label_col].notna()

        # Pre-compute aggregated features
        agg_df = aggregate_transactions(transactions_df, fk_col, agg_type)
        agg_feat_cols = [c for c in agg_df.columns if c != fk_col]

        bucket_results = []

        for lo, hi, bucket_label in FAN_OUT_BUCKETS:
            bucket_mask = (entity_df["_fan_out"] >= lo) & (entity_df["_fan_out"] <= hi)
            train_mask = (entity_df["_fold"] == 0) & bucket_mask & has_label
            # Combine fold 1+2 for val in stratification (small buckets)
            val_mask = ((entity_df["_fold"] == 1) | (entity_df["_fold"] == 2)) & bucket_mask & has_label

            n_train = train_mask.sum()
            n_val = val_mask.sum()

            logger.info(f"  Bucket {bucket_label}: train={n_train}, val={n_val}")

            if n_train < 30 or n_val < 10:
                logger.info(f"    Skipping (too few samples)")
                bucket_results.append({
                    "bucket": bucket_label, "n_train": int(n_train), "n_val": int(n_val),
                    "skipped": True,
                })
                continue

            train_idx = entity_df[train_mask].index
            val_idx = entity_df[val_mask].index

            X_train_base = entity_df.loc[train_idx, entity_feat_cols].copy()
            y_train = entity_df.loc[train_idx, label_col].copy()
            X_val_base = entity_df.loc[val_idx, entity_feat_cols].copy()
            y_val = entity_df.loc[val_idx, label_col].copy()

            for col in X_train_base.columns:
                X_train_base[col] = pd.to_numeric(X_train_base[col], errors="coerce")
                X_val_base[col] = pd.to_numeric(X_val_base[col], errors="coerce")

            # Baseline within bucket
            baseline_scores = []
            for seed in SEEDS:
                try:
                    score = train_and_evaluate(X_train_base, y_train, X_val_base, y_val,
                                               task_type, PROBE_CONFIG, seed)
                    if not np.isnan(score):
                        baseline_scores.append(score)
                except Exception:
                    logger.exception(f"    Baseline failed bucket={bucket_label}, seed={seed}")

            if not baseline_scores:
                bucket_results.append({
                    "bucket": bucket_label, "n_train": int(n_train), "n_val": int(n_val),
                    "skipped": True, "reason": "baseline_failed_or_single_class",
                })
                continue

            baseline_mean = np.mean(baseline_scores)

            # With-join within bucket
            merged_train = entity_df.loc[train_idx].merge(
                agg_df, left_on="_row_id", right_on=fk_col, how="left"
            )
            merged_val = entity_df.loc[val_idx].merge(
                agg_df, left_on="_row_id", right_on=fk_col, how="left"
            )

            all_feat_cols = entity_feat_cols + agg_feat_cols
            all_feat_cols = [c for c in all_feat_cols if c in merged_train.columns]
            all_feat_cols = list(dict.fromkeys(all_feat_cols))

            X_train_aug = merged_train[all_feat_cols].copy()
            X_val_aug = merged_val[all_feat_cols].copy()
            for col in X_train_aug.columns:
                X_train_aug[col] = pd.to_numeric(X_train_aug[col], errors="coerce")
                X_val_aug[col] = pd.to_numeric(X_val_aug[col], errors="coerce")

            aug_scores = []
            for seed in SEEDS:
                try:
                    score = train_and_evaluate(X_train_aug, y_train, X_val_aug, y_val,
                                               task_type, PROBE_CONFIG, seed)
                    if not np.isnan(score):
                        aug_scores.append(score)
                except Exception:
                    logger.exception(f"    Augmented failed bucket={bucket_label}, seed={seed}")

            aug_mean = np.mean(aug_scores) if aug_scores else baseline_mean
            jrn = compute_jrn(baseline_mean, aug_mean, higher_is_better)

            # Midpoint of bucket for correlation
            midpoint = (lo + min(hi, 500)) / 2

            logger.info(f"    Baseline={baseline_mean:.4f}, Augmented={aug_mean:.4f}, JRN={jrn:.4f}")

            bucket_results.append({
                "bucket": bucket_label, "lo": lo, "hi": hi if hi != float("inf") else 99999,
                "midpoint": midpoint,
                "n_train": int(n_train), "n_val": int(n_val),
                "baseline": baseline_mean, "augmented": aug_mean, "jrn": jrn,
                "skipped": False,
            })

            del merged_train, merged_val, X_train_aug, X_val_aug
            gc.collect()

        # Compute Spearman correlation between midpoint and JRN
        valid_buckets = [b for b in bucket_results if not b.get("skipped", False)]
        if len(valid_buckets) >= 3:
            midpoints = [b["midpoint"] for b in valid_buckets]
            jrns = [b["jrn"] for b in valid_buckets]
            rho, pval = spearmanr(midpoints, jrns)
        else:
            rho, pval = float("nan"), float("nan")

        logger.info(f"  Spearman rho(fan-out midpoint, JRN) = {rho:.4f}, p={pval:.4f}")

        results[task_name] = {
            "buckets": bucket_results,
            "spearman_rho": rho if not np.isnan(rho) else None,
            "spearman_pval": pval if not np.isnan(pval) else None,
        }

        del agg_df
        gc.collect()

    return results


# ============================================================
# PHASE 3: FK-SHUFFLING CONFOUND DECOMPOSITION
# ============================================================

def run_phase3(customers_df: pd.DataFrame, articles_df: pd.DataFrame,
               transactions_df: pd.DataFrame, phase1_results: dict) -> dict:
    """Phase 3: FK-shuffling to decompose structural vs feature JRN."""
    logger.info("=" * 60)
    logger.info("PHASE 3: FK-Shuffling Confound Decomposition")
    logger.info("=" * 60)

    N_PERMUTATIONS = 5
    results = {}
    agg_type = "mean"

    for task_name, task_cfg in TASK_CONFIG.items():
        logger.info(f"\n--- Task: {task_name} ---")
        entity_table = task_cfg["entity_table"]
        fk_col = task_cfg["join"]["fk_col"]
        task_type = task_cfg["task_type"]
        higher_is_better = task_cfg["higher_is_better"]

        if entity_table == "customers":
            entity_df = customers_df
            label_col = "_label_churn"
        else:
            entity_df = articles_df
            label_col = "_label_sales"

        entity_feat_cols = get_entity_feat_cols(entity_df)
        entity_feat_cols = [c for c in entity_feat_cols if c != label_col]

        has_label = entity_df[label_col].notna()
        train_idx = entity_df[(entity_df["_fold"] == 0) & has_label].index
        # Combine fold 1+2 for val when small
        val_fold1 = entity_df[(entity_df["_fold"] == 1) & has_label]
        val_fold2 = entity_df[(entity_df["_fold"] == 2) & has_label]
        if len(val_fold1) < 100 or len(val_fold2) < 100:
            val_idx = entity_df[((entity_df["_fold"] == 1) | (entity_df["_fold"] == 2)) & has_label].index
        else:
            val_idx = val_fold1.index

        y_train = entity_df.loc[train_idx, label_col]
        y_val = entity_df.loc[val_idx, label_col]

        baseline_mean = phase1_results[task_name]["baseline"]["mean"]
        actual_jrn = phase1_results[task_name]["agg_results"].get(agg_type, {}).get("jrn", 1.0)

        shuffled_jrns = []
        shuffled_scores_all = []

        for perm_i in range(N_PERMUTATIONS):
            logger.info(f"  Permutation {perm_i+1}/{N_PERMUTATIONS}")
            # Shuffle FK column
            rng = np.random.RandomState(perm_i * 1000 + 7)
            txn_shuffled = transactions_df.copy()
            txn_shuffled[fk_col] = rng.permutation(txn_shuffled[fk_col].values)

            # Re-aggregate with shuffled FK
            agg_df = aggregate_transactions(txn_shuffled, fk_col, agg_type)
            agg_feat_cols = [c for c in agg_df.columns if c != fk_col]

            merged_train = entity_df.loc[train_idx].merge(
                agg_df, left_on="_row_id", right_on=fk_col, how="left"
            )
            merged_val = entity_df.loc[val_idx].merge(
                agg_df, left_on="_row_id", right_on=fk_col, how="left"
            )

            all_feat_cols = entity_feat_cols + agg_feat_cols
            all_feat_cols = [c for c in all_feat_cols if c in merged_train.columns]
            all_feat_cols = list(dict.fromkeys(all_feat_cols))

            X_train_shuf = merged_train[all_feat_cols].copy()
            X_val_shuf = merged_val[all_feat_cols].copy()
            for col in X_train_shuf.columns:
                X_train_shuf[col] = pd.to_numeric(X_train_shuf[col], errors="coerce")
                X_val_shuf[col] = pd.to_numeric(X_val_shuf[col], errors="coerce")

            # Use single seed for shuffled to save time
            seed = SEEDS[0]
            try:
                score = train_and_evaluate(X_train_shuf, y_train, X_val_shuf, y_val,
                                           task_type, PROBE_CONFIG, seed)
                shuffled_jrn = compute_jrn(baseline_mean, score, higher_is_better)
                shuffled_jrns.append(shuffled_jrn)
                shuffled_scores_all.append(score)
                logger.info(f"    Shuffled score={score:.4f}, shuffled_JRN={shuffled_jrn:.4f}")
            except Exception:
                logger.exception(f"    Shuffled evaluation failed perm={perm_i}")

            del txn_shuffled, agg_df, merged_train, merged_val, X_train_shuf, X_val_shuf
            gc.collect()

        structural_jrn = np.mean(shuffled_jrns) if shuffled_jrns else 1.0
        feature_jrn = actual_jrn - structural_jrn + 1.0

        logger.info(f"  Actual JRN={actual_jrn:.4f}, Structural JRN={structural_jrn:.4f}, Feature JRN={feature_jrn:.4f}")

        results[task_name] = {
            "actual_jrn": actual_jrn,
            "structural_jrn": structural_jrn,
            "feature_jrn": feature_jrn,
            "shuffled_jrns": shuffled_jrns,
            "shuffled_scores": shuffled_scores_all,
            "n_permutations": len(shuffled_jrns),
        }

    return results


# ============================================================
# PHASE 4: AGGREGATION SENSITIVITY ANALYSIS
# ============================================================

def run_phase4(phase1_results: dict) -> dict:
    """Phase 4: Aggregation sensitivity (CoV, spread) from Phase 1 results."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Aggregation Sensitivity Analysis")
    logger.info("=" * 60)

    results = {}
    for task_name, task_data in phase1_results.items():
        agg_jrns = {}
        agg_scores = {}
        for agg_type, agg_data in task_data["agg_results"].items():
            agg_jrns[agg_type] = agg_data["jrn"]
            agg_scores[agg_type] = agg_data["augmented_mean"]

        jrn_values = list(agg_jrns.values())
        score_values = list(agg_scores.values())

        jrn_mean = np.mean(jrn_values)
        jrn_std = np.std(jrn_values)
        cov = jrn_std / jrn_mean if jrn_mean > 0 else 0.0
        jrn_spread = max(jrn_values) - min(jrn_values)

        best_agg = max(agg_jrns, key=agg_jrns.get)
        worst_agg = min(agg_jrns, key=agg_jrns.get)

        logger.info(f"  {task_name}: JRN mean={jrn_mean:.4f}, std={jrn_std:.4f}, CoV={cov:.4f}")
        logger.info(f"    Spread={jrn_spread:.4f}, Best={best_agg} ({agg_jrns[best_agg]:.4f}), Worst={worst_agg} ({agg_jrns[worst_agg]:.4f})")

        results[task_name] = {
            "jrn_per_agg": agg_jrns,
            "score_per_agg": agg_scores,
            "jrn_mean": jrn_mean,
            "jrn_std": jrn_std,
            "cov": cov,
            "jrn_spread": jrn_spread,
            "best_agg": best_agg,
            "worst_agg": worst_agg,
        }

    return results


# ============================================================
# PHASE 5: JRN-GUIDED VS UNIFORM ARCHITECTURE COMPARISON
# ============================================================

def run_phase5(customers_df: pd.DataFrame, articles_df: pd.DataFrame,
               transactions_df: pd.DataFrame, phase1_results: dict) -> dict:
    """Phase 5: Compare uniform-mean, uniform-rich, JRN-guided strategies."""
    logger.info("=" * 60)
    logger.info("PHASE 5: JRN-Guided vs Uniform Architecture Comparison")
    logger.info("=" * 60)

    results = {}

    for task_name, task_cfg in TASK_CONFIG.items():
        logger.info(f"\n--- Task: {task_name} ---")
        entity_table = task_cfg["entity_table"]
        fk_col = task_cfg["join"]["fk_col"]
        task_type = task_cfg["task_type"]
        higher_is_better = task_cfg["higher_is_better"]

        if entity_table == "customers":
            entity_df = customers_df
            label_col = "_label_churn"
        else:
            entity_df = articles_df
            label_col = "_label_sales"

        entity_feat_cols = get_entity_feat_cols(entity_df)
        entity_feat_cols = [c for c in entity_feat_cols if c != label_col]

        has_label = entity_df[label_col].notna()
        train_idx = entity_df[(entity_df["_fold"] == 0) & has_label].index
        # Combine fold 1+2 for val when small
        val_f1 = entity_df[(entity_df["_fold"] == 1) & has_label]
        val_f2 = entity_df[(entity_df["_fold"] == 2) & has_label]
        if len(val_f1) < 100 or len(val_f2) < 100:
            val_idx = entity_df[((entity_df["_fold"] == 1) | (entity_df["_fold"] == 2)) & has_label].index
        else:
            val_idx = val_f1.index

        y_train = entity_df.loc[train_idx, label_col]
        y_val = entity_df.loc[val_idx, label_col]

        # Get JRN from Phase 1 (mean agg)
        mean_jrn = phase1_results[task_name]["agg_results"].get("mean", {}).get("jrn", 1.0)

        strategies = {}

        # Strategy 1: Uniform-mean
        logger.info("  Strategy: uniform-mean")
        agg_df = aggregate_transactions(transactions_df, fk_col, "mean")
        agg_feat_cols = [c for c in agg_df.columns if c != fk_col]
        merged_train = entity_df.loc[train_idx].merge(agg_df, left_on="_row_id", right_on=fk_col, how="left")
        merged_val = entity_df.loc[val_idx].merge(agg_df, left_on="_row_id", right_on=fk_col, how="left")
        all_cols = entity_feat_cols + agg_feat_cols
        all_cols = [c for c in all_cols if c in merged_train.columns]
        all_cols = list(dict.fromkeys(all_cols))
        X_train = merged_train[all_cols].copy()
        X_val = merged_val[all_cols].copy()
        for col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
        scores = []
        for seed in SEEDS:
            try:
                s = train_and_evaluate(X_train, y_train, X_val, y_val, task_type, FULL_CONFIG, seed)
                scores.append(s)
            except Exception:
                logger.exception(f"    uniform-mean failed seed={seed}")
        strategies["uniform_mean"] = {"mean": np.mean(scores) if scores else 0.0, "std": np.std(scores) if len(scores) > 1 else 0.0, "scores": scores}
        del agg_df, merged_train, merged_val, X_train, X_val
        gc.collect()

        # Strategy 2: Uniform-rich (all_combined)
        logger.info("  Strategy: uniform-rich")
        agg_df = aggregate_transactions(transactions_df, fk_col, "all_combined")
        agg_feat_cols = [c for c in agg_df.columns if c != fk_col]
        merged_train = entity_df.loc[train_idx].merge(agg_df, left_on="_row_id", right_on=fk_col, how="left")
        merged_val = entity_df.loc[val_idx].merge(agg_df, left_on="_row_id", right_on=fk_col, how="left")
        all_cols = entity_feat_cols + agg_feat_cols
        all_cols = [c for c in all_cols if c in merged_train.columns]
        all_cols = list(dict.fromkeys(all_cols))
        X_train = merged_train[all_cols].copy()
        X_val = merged_val[all_cols].copy()
        for col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
        scores = []
        for seed in SEEDS:
            try:
                s = train_and_evaluate(X_train, y_train, X_val, y_val, task_type, FULL_CONFIG, seed)
                scores.append(s)
            except Exception:
                logger.exception(f"    uniform-rich failed seed={seed}")
        strategies["uniform_rich"] = {"mean": np.mean(scores) if scores else 0.0, "std": np.std(scores) if len(scores) > 1 else 0.0, "scores": scores}
        del agg_df, merged_train, merged_val, X_train, X_val
        gc.collect()

        # Strategy 3: JRN-guided
        logger.info(f"  Strategy: jrn-guided (JRN={mean_jrn:.4f})")
        if mean_jrn > 1.2:
            # JRN >> 1: use mean (cheap, signal is strong)
            chosen_agg = "mean"
            guidance = "strong_signal_use_mean"
        elif mean_jrn >= 0.8:
            # JRN ~ 1: use all_combined (invest at threshold)
            chosen_agg = "all_combined"
            guidance = "threshold_use_rich"
        else:
            # JRN << 1: prune the join
            chosen_agg = None
            guidance = "weak_signal_prune"

        if chosen_agg is not None:
            agg_df = aggregate_transactions(transactions_df, fk_col, chosen_agg)
            agg_feat_cols = [c for c in agg_df.columns if c != fk_col]
            merged_train = entity_df.loc[train_idx].merge(agg_df, left_on="_row_id", right_on=fk_col, how="left")
            merged_val = entity_df.loc[val_idx].merge(agg_df, left_on="_row_id", right_on=fk_col, how="left")
            all_cols = entity_feat_cols + agg_feat_cols
            all_cols = [c for c in all_cols if c in merged_train.columns]
            all_cols = list(dict.fromkeys(all_cols))
            X_train = merged_train[all_cols].copy()
            X_val = merged_val[all_cols].copy()
            for col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
            scores = []
            for seed in SEEDS:
                try:
                    s = train_and_evaluate(X_train, y_train, X_val, y_val, task_type, FULL_CONFIG, seed)
                    scores.append(s)
                except Exception:
                    logger.exception(f"    jrn-guided failed seed={seed}")
            strategies["jrn_guided"] = {
                "mean": np.mean(scores) if scores else 0.0,
                "std": np.std(scores) if len(scores) > 1 else 0.0,
                "scores": scores,
                "chosen_agg": chosen_agg, "guidance": guidance,
                "jrn_value": mean_jrn,
            }
            del agg_df, merged_train, merged_val, X_train, X_val
            gc.collect()
        else:
            # Pruned: use baseline only
            X_train_base = entity_df.loc[train_idx, entity_feat_cols].copy()
            X_val_base = entity_df.loc[val_idx, entity_feat_cols].copy()
            for col in X_train_base.columns:
                X_train_base[col] = pd.to_numeric(X_train_base[col], errors="coerce")
                X_val_base[col] = pd.to_numeric(X_val_base[col], errors="coerce")
            scores = []
            for seed in SEEDS:
                try:
                    s = train_and_evaluate(X_train_base, y_train, X_val_base, y_val, task_type, FULL_CONFIG, seed)
                    scores.append(s)
                except Exception:
                    logger.exception(f"    jrn-guided (pruned) failed seed={seed}")
            strategies["jrn_guided"] = {
                "mean": np.mean(scores) if scores else 0.0,
                "std": np.std(scores) if len(scores) > 1 else 0.0,
                "scores": scores,
                "chosen_agg": None, "guidance": guidance,
                "jrn_value": mean_jrn,
            }

        logger.info(f"  Results: uniform_mean={strategies['uniform_mean']['mean']:.4f}, "
                     f"uniform_rich={strategies['uniform_rich']['mean']:.4f}, "
                     f"jrn_guided={strategies['jrn_guided']['mean']:.4f}")

        results[task_name] = strategies

    return results


# ============================================================
# PHASE 6: OUTPUT ASSEMBLY
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _jdumps(obj, **kwargs) -> str:
    """json.dumps with numpy support."""
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)


def _json_safe(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def assemble_output(phase1_results: dict, phase2_results: dict,
                    phase3_results: dict, phase4_results: dict,
                    phase5_results: dict, total_time: float) -> dict:
    """Assemble method_out.json in exp_gen_sol_out format."""
    logger.info("=" * 60)
    logger.info("PHASE 6: Output Assembly")
    logger.info("=" * 60)

    examples = []

    # --- 1. JRN measurements: one per (task x agg_type) ---
    for task_name, task_data in phase1_results.items():
        task_cfg = TASK_CONFIG[task_name]
        baseline = task_data["baseline"]
        for agg_type, agg_data in task_data["agg_results"].items():
            inp = _jdumps({
                "measurement_type": "jrn_measurement",
                "task": task_name, "task_type": task_cfg["task_type"],
                "metric": task_cfg["metric"],
                "join": f"transactions.{task_cfg['join']['fk_col']} -> {task_cfg['entity_table']}",
                "aggregation": agg_type,
                "n_agg_features": agg_data.get("n_agg_features", 0),
                "seeds": SEEDS,
            })
            out = _jdumps({
                "jrn": round(agg_data["jrn"], 6),
                "baseline_score": round(baseline["mean"], 6),
                "augmented_score": round(agg_data["augmented_mean"], 6),
                "baseline_std": round(baseline["std"], 6),
                "augmented_std": round(agg_data.get("augmented_std", 0.0), 6),
                "higher_is_better": task_cfg["higher_is_better"],
                "jrn_direction_note": "JRN > 1 means join improves prediction (corrected for MAE)",
            })
            examples.append({
                "input": inp, "output": out,
                "predict_baseline": _jdumps({"score": round(baseline["mean"], 6)}),
                "predict_our_method": _jdumps({"jrn": round(agg_data["jrn"], 6), "score": round(agg_data["augmented_mean"], 6)}),
                "metadata_measurement_type": "jrn_measurement",
                "metadata_task": task_name,
                "metadata_aggregation": agg_type,
            })

    # --- 2. JRN matrix summary ---
    matrix = {}
    for task_name, task_data in phase1_results.items():
        matrix[task_name] = {
            "baseline": round(task_data["baseline"]["mean"], 6),
            "jrn_by_agg": {agg: round(d["jrn"], 6) for agg, d in task_data["agg_results"].items()},
            "best_agg": max(task_data["agg_results"], key=lambda a: task_data["agg_results"][a]["jrn"]),
        }
    inp = _jdumps({"measurement_type": "jrn_matrix", "description": "JRN values across 2 tasks x 5 aggregation types"})
    out = _jdumps(matrix)
    examples.append({
        "input": inp, "output": out,
        "predict_baseline": _jdumps({"description": "No JRN estimation (baseline only)"}),
        "predict_our_method": _jdumps(matrix),
        "metadata_measurement_type": "jrn_matrix",
    })

    # --- 3. Fan-out stratification ---
    for task_name, strat_data in phase2_results.items():
        inp = _jdumps({
            "measurement_type": "fanout_stratification",
            "task": task_name,
            "n_buckets": len(FAN_OUT_BUCKETS),
            "bucket_ranges": [f"{lo}-{hi}" for lo, hi, _ in FAN_OUT_BUCKETS],
        })
        out = _jdumps({
            "buckets": strat_data["buckets"],
            "spearman_rho": strat_data["spearman_rho"],
            "spearman_pval": strat_data["spearman_pval"],
        })
        examples.append({
            "input": inp, "output": out,
            "predict_baseline": _jdumps({"description": "No fan-out stratification analysis"}),
            "predict_our_method": _jdumps({
                "spearman_rho": strat_data["spearman_rho"],
                "n_valid_buckets": len([b for b in strat_data["buckets"] if not b.get("skipped", False)]),
            }),
            "metadata_measurement_type": "fanout_stratification",
            "metadata_task": task_name,
        })

    # --- 4. FK-shuffling decomposition ---
    for task_name, shuf_data in phase3_results.items():
        inp = _jdumps({
            "measurement_type": "fk_shuffling",
            "task": task_name,
            "n_permutations": shuf_data["n_permutations"],
        })
        out = _jdumps({
            "actual_jrn": round(shuf_data["actual_jrn"], 6),
            "structural_jrn": round(shuf_data["structural_jrn"], 6),
            "feature_jrn": round(shuf_data["feature_jrn"], 6),
            "shuffled_jrns": [round(j, 6) for j in shuf_data["shuffled_jrns"]],
            "interpretation": (
                "structural_jrn captures improvement from adding ANY aggregated features "
                "(regularization/more features effect). feature_jrn captures the actual "
                "FK-link predictive signal beyond random features."
            ),
        })
        examples.append({
            "input": inp, "output": out,
            "predict_baseline": _jdumps({"description": "No confound decomposition"}),
            "predict_our_method": _jdumps({
                "actual_jrn": round(shuf_data["actual_jrn"], 6),
                "structural_jrn": round(shuf_data["structural_jrn"], 6),
                "feature_jrn": round(shuf_data["feature_jrn"], 6),
            }),
            "metadata_measurement_type": "fk_shuffling",
            "metadata_task": task_name,
        })

    # --- 5. Aggregation sensitivity ---
    for task_name, sens_data in phase4_results.items():
        inp = _jdumps({
            "measurement_type": "aggregation_sensitivity",
            "task": task_name,
            "agg_types": AGG_TYPES,
        })
        out = _jdumps({
            "jrn_per_agg": {k: round(v, 6) for k, v in sens_data["jrn_per_agg"].items()},
            "score_per_agg": {k: round(v, 6) for k, v in sens_data["score_per_agg"].items()},
            "cov": round(sens_data["cov"], 6),
            "jrn_spread": round(sens_data["jrn_spread"], 6),
            "best_agg": sens_data["best_agg"],
            "worst_agg": sens_data["worst_agg"],
        })
        examples.append({
            "input": inp, "output": out,
            "predict_baseline": _jdumps({"description": "No aggregation sensitivity analysis"}),
            "predict_our_method": _jdumps({
                "cov": round(sens_data["cov"], 6),
                "best_agg": sens_data["best_agg"],
            }),
            "metadata_measurement_type": "aggregation_sensitivity",
            "metadata_task": task_name,
        })

    # --- 6. Architecture comparison ---
    for task_name, arch_data in phase5_results.items():
        task_cfg = TASK_CONFIG[task_name]
        inp = _jdumps({
            "measurement_type": "architecture_comparison",
            "task": task_name,
            "strategies": ["uniform_mean", "uniform_rich", "jrn_guided"],
        })
        strategy_results = {}
        for strat_name, strat_data in arch_data.items():
            strategy_results[strat_name] = {
                "score_mean": round(strat_data["mean"], 6),
                "score_std": round(strat_data["std"], 6),
            }
            if "chosen_agg" in strat_data:
                strategy_results[strat_name]["chosen_agg"] = strat_data["chosen_agg"]
                strategy_results[strat_name]["guidance"] = strat_data["guidance"]

        # Determine best strategy
        if task_cfg["higher_is_better"]:
            best_strat = max(strategy_results, key=lambda s: strategy_results[s]["score_mean"])
        else:
            best_strat = min(strategy_results, key=lambda s: strategy_results[s]["score_mean"])

        out = _jdumps({
            "strategy_results": strategy_results,
            "best_strategy": best_strat,
            "metric": task_cfg["metric"],
            "higher_is_better": task_cfg["higher_is_better"],
        })
        examples.append({
            "input": inp, "output": out,
            "predict_baseline": _jdumps({"strategy": "uniform_mean", "score": round(arch_data["uniform_mean"]["mean"], 6)}),
            "predict_our_method": _jdumps({"strategy": "jrn_guided", "score": round(arch_data["jrn_guided"]["mean"], 6)}),
            "metadata_measurement_type": "architecture_comparison",
            "metadata_task": task_name,
        })

    # --- 7. Overall summary ---
    summary = {
        "dataset": "rel-hm",
        "n_tables": 3,
        "n_joins": 2,
        "tasks": {},
        "total_runtime_seconds": round(total_time, 1),
        "n_seeds": len(SEEDS),
        "n_agg_types": len(AGG_TYPES),
        "jrn_direction_fix": "For MAE (lower=better), JRN = baseline_MAE / augmented_MAE so JRN > 1 always means improvement",
    }
    for task_name, task_data in phase1_results.items():
        best_agg = max(task_data["agg_results"], key=lambda a: task_data["agg_results"][a]["jrn"])
        summary["tasks"][task_name] = {
            "baseline": round(task_data["baseline"]["mean"], 6),
            "best_jrn": round(task_data["agg_results"][best_agg]["jrn"], 6),
            "best_agg": best_agg,
            "join_helpful": task_data["agg_results"][best_agg]["jrn"] > 1.0,
        }
        if task_name in phase3_results:
            summary["tasks"][task_name]["structural_jrn"] = round(phase3_results[task_name]["structural_jrn"], 6)
            summary["tasks"][task_name]["feature_jrn"] = round(phase3_results[task_name]["feature_jrn"], 6)

    inp = _jdumps({"measurement_type": "summary", "dataset": "rel-hm"})
    out = _jdumps(summary)
    examples.append({
        "input": inp, "output": out,
        "predict_baseline": _jdumps({"description": "No JRN analysis performed"}),
        "predict_our_method": _jdumps(summary),
        "metadata_measurement_type": "summary",
    })

    # Assemble final output
    output = {
        "metadata": {
            "method_name": "GBM-Probe JRN Estimation",
            "dataset": "rel-hm",
            "description": (
                "JRN estimation on rel-hm (H&M) dataset using LightGBM probes. "
                "Covers 2 tasks (user-churn classification, item-sales regression), "
                "5 aggregation types, fan-out stratification, FK-shuffling decomposition, "
                "and JRN-guided architecture comparison. "
                "CRITICAL FIX: uses JRN = baseline_MAE / augmented_MAE for regression "
                "so JRN > 1 always means improvement."
            ),
            "seeds": SEEDS,
            "probe_config": PROBE_CONFIG,
            "full_config": FULL_CONFIG,
            "agg_types": AGG_TYPES,
            "tasks": list(TASK_CONFIG.keys()),
            "total_runtime_seconds": round(total_time, 1),
        },
        "datasets": [{
            "dataset": "rel-hm",
            "examples": examples,
        }],
    }

    return output


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main():
    start_time = time.time()
    logger.info("Starting GBM-Probe JRN Estimation on rel-hm")
    logger.info(f"Working directory: {WORKSPACE}")

    # --- Load & encode data ---
    customers_df, articles_df, transactions_df = load_data()
    customers_df, articles_df, transactions_df = encode_features(
        customers_df, articles_df, transactions_df
    )

    # Log data statistics
    for name, df in [("customers", customers_df), ("articles", articles_df), ("transactions", transactions_df)]:
        fold_counts = df["_fold"].value_counts().to_dict()
        logger.info(f"  {name}: {len(df)} rows, folds={fold_counts}")
        nan_pct = df.isna().mean()
        high_nan = nan_pct[nan_pct > 0.1]
        if len(high_nan) > 0:
            logger.info(f"    High NaN columns: {dict(high_nan.round(3))}")

    mem_used = psutil.virtual_memory().used / 1e9
    logger.info(f"Memory used after data load: {mem_used:.1f} GB")

    # --- Phase 1: Baseline + JRN ---
    phase1_results = run_phase1(customers_df, articles_df, transactions_df)

    t1 = time.time()
    logger.info(f"Phase 1 completed in {t1 - start_time:.0f}s")

    # --- Phase 2: Fan-out stratification ---
    phase2_results = run_phase2(customers_df, articles_df, transactions_df)

    t2 = time.time()
    logger.info(f"Phase 2 completed in {t2 - t1:.0f}s")

    # --- Phase 3: FK-shuffling ---
    phase3_results = run_phase3(customers_df, articles_df, transactions_df, phase1_results)

    t3 = time.time()
    logger.info(f"Phase 3 completed in {t3 - t2:.0f}s")

    # --- Phase 4: Aggregation sensitivity ---
    phase4_results = run_phase4(phase1_results)

    t4 = time.time()
    logger.info(f"Phase 4 completed in {t4 - t3:.0f}s")

    # --- Phase 5: Architecture comparison ---
    phase5_results = run_phase5(customers_df, articles_df, transactions_df, phase1_results)

    t5 = time.time()
    logger.info(f"Phase 5 completed in {t5 - t4:.0f}s")

    total_time = time.time() - start_time
    logger.info(f"All phases completed in {total_time:.0f}s")

    # --- Phase 6: Assemble output ---
    output = assemble_output(phase1_results, phase2_results, phase3_results,
                             phase4_results, phase5_results, total_time)

    # Ensure all numpy types are converted to native Python for JSON
    output = _json_safe(output)

    # Write output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(_jdumps(output, indent=2))
    logger.info(f"Output written to {out_path}")
    logger.info(f"Total examples: {len(output['datasets'][0]['examples'])}")

    # Verify output parses
    verify = json.loads(out_path.read_text())
    assert "datasets" in verify
    assert len(verify["datasets"][0]["examples"]) > 0
    logger.info("Output verification passed!")

    # Clean up
    del customers_df, articles_df, transactions_df
    gc.collect()

    logger.info(f"DONE. Total runtime: {total_time:.0f}s")


if __name__ == "__main__":
    main()
