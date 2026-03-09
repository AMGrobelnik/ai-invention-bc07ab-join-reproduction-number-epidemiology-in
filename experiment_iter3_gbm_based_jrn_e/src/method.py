#!/usr/bin/env python3
"""GBM-Based JRN Estimation for 11 FK Joins x 3 Tasks on rel-avito.

Estimates Join Reproduction Number (JRN) for all 11 foreign key joins in
rel-avito across 3 tractable tasks (ad-ctr, user-clicks, user-visits) using
LightGBM probes. Tests JRN estimation at extreme cardinality scales
(fan-out 1.18 to 114,625). Computes JRN matrix, aggregation sensitivity,
training-free proxy correlations, and probe-to-full Spearman rho.

user-ad-visit (link_prediction/MAP) is excluded as it requires specialized
handling incompatible with tabular GBM probes.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_absolute_error, roc_auc_score

# ============================================================
# LOGGING SETUP
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION AND RESOURCE LIMITS
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


def _container_ram_gb() -> Optional[float]:
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

# Set RAM budget: 70% of available (~20GB for 29GB container)
RAM_BUDGET_BYTES = int(AVAILABLE_RAM_GB * 0.70 * 1e9)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET_BYTES < _avail, f"Budget {RAM_BUDGET_BYTES/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM (container), "
            f"budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ============================================================
# CONFIGURATION
# ============================================================

WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/"
                "3_invention_loop/iter_2/gen_art/data_id5_it2__opus/data_out")

SEEDS = [42, 123, 456]
PROBE_CONFIG = {
    "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
    "n_jobs": NUM_CPUS,
}
FULL_CONFIG = {
    "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
    "num_leaves": 127, "n_jobs": NUM_CPUS,
}
MAX_TRAIN_ROWS = 200_000

# Columns to drop (text and time)
TEXT_COLS = {"Title", "SearchQuery"}
TIME_COLS = {"SearchDate", "timestamp"}
DROP_COLS = TEXT_COLS | TIME_COLS

# Task configuration (excluding user-ad-visit link_prediction)
TASK_CONFIG = {
    "rel-avito/ad-ctr": {
        "entity_table": "AdsInfo", "entity_col": "AdID",
        "task_type": "regression", "metric": "MAE",
        "higher_is_better": False,
    },
    "rel-avito/user-clicks": {
        "entity_table": "UserInfo", "entity_col": "UserID",
        "task_type": "binary_classification", "metric": "AUROC",
        "higher_is_better": True,
    },
    "rel-avito/user-visits": {
        "entity_table": "UserInfo", "entity_col": "UserID",
        "task_type": "binary_classification", "metric": "AUROC",
        "higher_is_better": True,
    },
}

AGG_TYPES = ["mean", "sum", "max", "std", "all_combined"]

# ============================================================
# DATA LOADING
# ============================================================

def load_all_data() -> Tuple[Dict[str, List[dict]], List[dict], Dict[str, List[dict]]]:
    """Load all data parts sequentially, extracting table rows, FK metadata, task samples."""
    table_records: Dict[str, List[dict]] = defaultdict(list)
    fk_metadata: List[dict] = []
    task_records: Dict[str, List[dict]] = defaultdict(list)

    part_files = sorted(DATA_DIR.glob("full_data_out_*.json"))
    logger.info(f"Found {len(part_files)} data parts in {DATA_DIR}")

    for part_file in part_files:
        t0 = time.time()
        logger.info(f"Loading {part_file.name} ...")
        raw = json.loads(part_file.read_text())
        examples = raw["datasets"][0]["examples"]
        logger.info(f"  {len(examples)} examples in {time.time()-t0:.1f}s")

        for ex in examples:
            row_type = ex.get("metadata_row_type")

            if row_type == "table_row":
                table_name = ex["metadata_table_name"]
                pk_col = ex.get("metadata_primary_key_col")
                pk_val = ex.get("metadata_primary_key_value")
                features = json.loads(ex["input"])
                if pk_col and pk_col not in ("None", "null"):
                    # Convert PK value to appropriate type
                    try:
                        features[pk_col] = int(pk_val)
                    except (ValueError, TypeError):
                        try:
                            features[pk_col] = float(pk_val)
                        except (ValueError, TypeError):
                            features[pk_col] = pk_val
                table_records[table_name].append(features)

            elif row_type == "fk_join_metadata":
                inp = json.loads(ex["input"])
                out = json.loads(ex["output"])
                fk_metadata.append({
                    "idx": ex["metadata_row_index"],
                    **inp, **out,
                })

            elif row_type == "task_sample":
                task_name = ex.get("metadata_task_name", "")
                if task_name not in TASK_CONFIG:
                    continue  # Skip user-ad-visit
                fold = ex.get("metadata_fold", -1)
                if fold not in (0, 1):
                    continue  # Skip test fold
                output_val = ex.get("output", "masked")
                if output_val == "masked":
                    continue
                inp = json.loads(ex["input"])
                inp["_label"] = output_val
                inp["_fold"] = fold
                task_records[task_name].append(inp)

        # Free memory
        del raw, examples
        gc.collect()

    logger.info(f"Tables loaded: {dict((k, len(v)) for k, v in table_records.items())}")
    logger.info(f"FK joins: {len(fk_metadata)}")
    logger.info(f"Task samples: {dict((k, len(v)) for k, v in task_records.items())}")

    return dict(table_records), fk_metadata, dict(task_records)


# ============================================================
# TABLE RECONSTRUCTION
# ============================================================

def build_tables(table_records: Dict[str, List[dict]]) -> Dict[str, pd.DataFrame]:
    """Convert table records to DataFrames with proper types."""
    tables = {}
    for table_name, records in table_records.items():
        if not records:
            logger.warning(f"  Table '{table_name}' has 0 rows, skipping")
            continue

        df = pd.DataFrame(records)

        # Drop text and time columns
        cols_to_drop = [c for c in df.columns if c in DROP_COLS]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Convert to numeric where possible
        for col in df.columns:
            if df[col].dtype == object:
                # Try numeric conversion
                converted = pd.to_numeric(df[col], errors="coerce")
                non_null_orig = df[col].notna().sum()
                non_null_conv = converted.notna().sum()
                # If most values convert successfully, use numeric
                if non_null_conv >= non_null_orig * 0.5:
                    df[col] = converted
                else:
                    # Label encode low-cardinality string cols
                    nunique = df[col].nunique()
                    if nunique < 1000:
                        df[col] = df[col].astype("category").cat.codes.astype(float)
                        df.loc[df[col] < 0, col] = np.nan
                    else:
                        df = df.drop(columns=[col])

        tables[table_name] = df
        logger.info(f"  Table '{table_name}': {len(df)} rows, {len(df.columns)} cols: "
                     f"{list(df.columns)}")

    return tables


def build_task_dfs(task_records: Dict[str, List[dict]]) -> Dict[str, pd.DataFrame]:
    """Build task DataFrames with proper label encoding."""
    task_dfs = {}
    for task_name, records in task_records.items():
        if not records:
            logger.warning(f"  Task '{task_name}' has 0 samples, skipping")
            continue

        config = TASK_CONFIG[task_name]
        df = pd.DataFrame(records)

        # Drop timestamp column
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        # Convert label
        if config["task_type"] == "regression":
            df["_label"] = pd.to_numeric(df["_label"], errors="coerce")
            df = df.dropna(subset=["_label"])
        elif config["task_type"] == "binary_classification":
            df["_label"] = df["_label"].map({"True": 1, "False": 0, True: 1, False: 0})
            df = df.dropna(subset=["_label"])
            df["_label"] = df["_label"].astype(int)

        # Convert entity col to numeric
        entity_col = config["entity_col"]
        if entity_col in df.columns:
            df[entity_col] = pd.to_numeric(df[entity_col], errors="coerce")

        task_dfs[task_name] = df
        logger.info(f"  Task '{task_name}': {len(df)} samples "
                     f"(train={int((df['_fold']==0).sum())}, val={int((df['_fold']==1).sum())})")

    return task_dfs


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def get_entity_features(
    tables: Dict[str, pd.DataFrame],
    task_df: pd.DataFrame,
    entity_table: str,
    entity_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Join entity table features to task samples."""
    if entity_table not in tables:
        logger.warning(f"Entity table '{entity_table}' not found")
        return task_df.copy(), []

    entity_df = tables[entity_table].copy()
    if entity_col not in entity_df.columns:
        logger.warning(f"Entity col '{entity_col}' not in '{entity_table}'")
        return task_df.copy(), []

    # Select numeric feature columns (exclude PK)
    feature_cols = [c for c in entity_df.columns
                    if c != entity_col and entity_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float]]
    if not feature_cols:
        logger.warning(f"No numeric features in '{entity_table}'")
        return task_df.copy(), []

    entity_subset = entity_df[[entity_col] + feature_cols].copy()
    merged = task_df.merge(entity_subset, on=entity_col, how="left")

    # Fill NaN with 0 for entity features
    merged[feature_cols] = merged[feature_cols].fillna(0)
    match_rate = merged[feature_cols[0]].notna().mean()
    logger.debug(f"  Entity merge match rate: {match_rate:.3f}")

    return merged, feature_cols


def add_forward_join_features(
    features_df: pd.DataFrame,
    fk: dict,
    tables: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    """Many-to-one lookup: entity has FK column pointing to parent table."""
    parent_table = fk["target_table"]
    fk_col = fk["source_fk_col"]
    pk_col = fk["target_pk_col"]

    if parent_table not in tables:
        logger.debug(f"  Parent table '{parent_table}' not in tables")
        return features_df, []
    if fk_col not in features_df.columns:
        logger.debug(f"  FK col '{fk_col}' not in features_df")
        return features_df, []

    parent_df = tables[parent_table].copy()
    if pk_col not in parent_df.columns:
        logger.debug(f"  PK col '{pk_col}' not in parent table")
        return features_df, []

    # Get numeric columns from parent (exclude PK)
    parent_numeric = [c for c in parent_df.columns
                      if c != pk_col and parent_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float]]
    if not parent_numeric:
        return features_df, []

    # Prefix parent features to avoid collisions
    rename_map = {c: f"fwd_{parent_table}_{c}" for c in parent_numeric}
    parent_subset = parent_df[[pk_col] + parent_numeric].rename(columns=rename_map)

    merged = features_df.merge(parent_subset, left_on=fk_col, right_on=pk_col, how="left")
    # Drop duplicate PK column if different name
    if pk_col != fk_col and pk_col in merged.columns:
        merged = merged.drop(columns=[pk_col])

    new_cols = list(rename_map.values())
    merged[new_cols] = merged[new_cols].fillna(0)
    return merged, new_cols


def compute_reverse_join_features(
    fk: dict,
    entity_col: str,
    tables: Dict[str, pd.DataFrame],
    agg_type: str = "mean",
) -> Tuple[pd.DataFrame, List[str]]:
    """One-to-many aggregation: child table FK -> entity table PK."""
    child_table = fk["source_table"]
    fk_col = fk["source_fk_col"]

    if child_table not in tables:
        logger.debug(f"  Child table '{child_table}' not in tables")
        return pd.DataFrame(), []
    child_df = tables[child_table]
    if fk_col not in child_df.columns:
        logger.debug(f"  FK col '{fk_col}' not in child table '{child_table}'")
        return pd.DataFrame(), []

    # Get numeric columns (exclude FK cols and other non-feature cols)
    fk_cols_in_child = set()
    # Identify all FK columns in this child table by checking all fk_joins
    # For now just exclude the groupby col
    child_numeric = [c for c in child_df.columns
                     if c != fk_col
                     and child_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float]]
    if not child_numeric:
        logger.debug(f"  No numeric features in child table '{child_table}'")
        return pd.DataFrame(), []

    # Perform aggregation
    if agg_type == "all_combined":
        agg_funcs = ["mean", "sum", "max", "std", "min"]
    elif agg_type in ("mean", "sum", "max", "std"):
        agg_funcs = [agg_type]
    elif agg_type == "median":
        agg_funcs = ["median"]
    elif agg_type == "trimmed_mean":
        # Custom: clip to 5th-95th percentile then mean
        clipped = child_df[child_numeric].copy()
        for col in child_numeric:
            q05 = clipped[col].quantile(0.05)
            q95 = clipped[col].quantile(0.95)
            clipped[col] = clipped[col].clip(q05, q95)
        clipped[fk_col] = child_df[fk_col]
        agg_df = clipped.groupby(fk_col)[child_numeric].mean()
        new_cols = [f"rev_{child_table}_{col}_trimmed_mean" for col in child_numeric]
        agg_df.columns = new_cols
        agg_df = agg_df.reset_index().rename(columns={fk_col: entity_col})
        return agg_df, new_cols
    else:
        agg_funcs = ["mean"]

    try:
        agg_df = child_df.groupby(fk_col)[child_numeric].agg(agg_funcs)
    except Exception:
        logger.exception(f"  Aggregation failed for {child_table}.{fk_col}")
        return pd.DataFrame(), []

    # Flatten multi-level columns
    if isinstance(agg_df.columns, pd.MultiIndex):
        agg_df.columns = [f"rev_{child_table}_{col}_{agg}" for col, agg in agg_df.columns]
    else:
        agg_df.columns = [f"rev_{child_table}_{col}_{agg_funcs[0]}" for col in agg_df.columns]

    new_cols = list(agg_df.columns)
    agg_df = agg_df.reset_index().rename(columns={fk_col: entity_col})

    # Handle NaN/Inf
    for col in new_cols:
        agg_df[col] = agg_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return agg_df, new_cols


# ============================================================
# GBM TRAINING
# ============================================================

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    task_type: str,
    config: dict,
    seed: int,
) -> float:
    """Train LightGBM and return metric on validation set."""
    params = {**config, "random_state": seed}

    if task_type == "regression":
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = mean_absolute_error(y_val, preds)
    elif task_type == "binary_classification":
        params["class_weight"] = "balanced"
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        try:
            score = roc_auc_score(y_val, preds)
        except ValueError:
            score = 0.5  # Single class in val
    else:
        score = 0.5

    return score


# ============================================================
# JRN COMPUTATION
# ============================================================

def compute_jrn_matrix(
    tables: Dict[str, pd.DataFrame],
    fk_joins: List[dict],
    task_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, dict]:
    """Compute JRN for all task x join combinations."""
    results = {}

    for task_name, config in TASK_CONFIG.items():
        if task_name not in task_dfs:
            logger.warning(f"  Task '{task_name}' has no data, skipping")
            continue

        entity_table = config["entity_table"]
        entity_col = config["entity_col"]
        task_type = config["task_type"]
        higher_is_better = config["higher_is_better"]

        tdf = task_dfs[task_name]
        train_mask = tdf["_fold"] == 0
        val_mask = tdf["_fold"] == 1

        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        logger.info(f"Task '{task_name}': {n_train} train, {n_val} val samples")

        if n_val < 10:
            logger.warning(f"  Too few val samples for '{task_name}', skipping")
            continue

        # Get entity base features
        base_df, base_feature_cols = get_entity_features(tables, tdf, entity_table, entity_col)
        if not base_feature_cols:
            logger.warning(f"  No entity features for '{task_name}', skipping")
            continue

        # Subsample if needed
        if n_train > MAX_TRAIN_ROWS:
            logger.info(f"  Subsampling train from {n_train} to {MAX_TRAIN_ROWS}")
            train_idx = base_df[train_mask].sample(MAX_TRAIN_ROWS, random_state=42).index
            train_mask = base_df.index.isin(train_idx)

        # Train baseline (entity features only)
        logger.info(f"  Training baseline ({len(base_feature_cols)} features)...")
        baseline_scores = []
        for seed in SEEDS:
            try:
                score = train_and_evaluate(
                    base_df[train_mask][base_feature_cols],
                    base_df[train_mask]["_label"].values,
                    base_df[val_mask][base_feature_cols],
                    base_df[val_mask]["_label"].values,
                    task_type, PROBE_CONFIG, seed,
                )
                baseline_scores.append(score)
            except Exception:
                logger.exception(f"  Baseline training failed (seed={seed})")
                baseline_scores.append(0.5 if higher_is_better else 999.0)

        baseline_mean = float(np.mean(baseline_scores))
        baseline_std = float(np.std(baseline_scores))
        logger.info(f"  Baseline: {baseline_mean:.6f} +/- {baseline_std:.6f}")

        # For EACH of the 11 FK joins:
        for fk in fk_joins:
            join_idx = fk["idx"]
            join_key = f"{fk['source_table']}.{fk['source_fk_col']}->{fk['target_table']}"
            t0 = time.time()

            # Determine direction relative to entity table
            if fk["source_table"] == entity_table:
                direction = "forward"
            elif fk["target_table"] == entity_table:
                direction = "reverse"
            else:
                direction = "none"

            fanout_mean = fk.get("fanout_mean", 0)

            # Determine which agg types to test
            if direction == "none":
                # Non-connected join -> JRN = 1.0
                results[(task_name, join_idx)] = {
                    "task": task_name, "join_idx": join_idx, "join_key": join_key,
                    "direction": direction, "fanout_mean": fanout_mean,
                    "baseline": baseline_mean, "baseline_std": baseline_std,
                    "best_agg_score": baseline_mean, "best_agg_type": "none",
                    "jrn": 1.0, "agg_sensitivity": 0.0,
                    "agg_scores": {}, "subsampled_fanout": 0.0,
                }
                logger.info(f"  Join {join_idx} ({join_key}): direction=none -> JRN=1.000")
                continue

            # Determine agg types
            test_agg_types = list(AGG_TYPES)
            if fanout_mean > 10000:
                test_agg_types.extend(["trimmed_mean", "median"])

            # For forward joins, aggregation type doesn't matter (many-to-one)
            if direction == "forward":
                test_agg_types = ["mean"]  # Only one type needed for lookup joins

            agg_scores_dict = {}
            for agg_type in test_agg_types:
                try:
                    if direction == "forward":
                        enriched, new_cols = add_forward_join_features(base_df, fk, tables)
                        if not new_cols:
                            agg_scores_dict[agg_type] = baseline_mean
                            continue
                        all_cols = base_feature_cols + new_cols
                    else:  # reverse
                        agg_feats, new_cols = compute_reverse_join_features(
                            fk, entity_col, tables, agg_type
                        )
                        if agg_feats.empty or not new_cols:
                            agg_scores_dict[agg_type] = baseline_mean
                            continue
                        enriched = base_df.merge(agg_feats, on=entity_col, how="left")
                        enriched[new_cols] = enriched[new_cols].fillna(0)
                        all_cols = base_feature_cols + new_cols

                    # Train with-join model, 3 seeds
                    join_scores = []
                    for seed in SEEDS:
                        s = train_and_evaluate(
                            enriched[train_mask][all_cols],
                            enriched[train_mask]["_label"].values,
                            enriched[val_mask][all_cols],
                            enriched[val_mask]["_label"].values,
                            task_type, PROBE_CONFIG, seed,
                        )
                        join_scores.append(s)
                    agg_scores_dict[agg_type] = float(np.mean(join_scores))

                except Exception:
                    logger.exception(f"  Failed agg_type={agg_type} for join {join_idx}")
                    agg_scores_dict[agg_type] = baseline_mean

            # Compute JRN for best aggregation
            valid_agg_scores = {k: v for k, v in agg_scores_dict.items() if v is not None}
            if not valid_agg_scores:
                jrn = 1.0
                best_agg_type = "none"
                best_agg_score = baseline_mean
            elif higher_is_better:
                best_agg_type = max(valid_agg_scores, key=valid_agg_scores.get)
                best_agg_score = valid_agg_scores[best_agg_type]
                jrn = best_agg_score / baseline_mean if baseline_mean > 0 else 1.0
            else:  # lower is better (MAE)
                best_agg_type = min(valid_agg_scores, key=valid_agg_scores.get)
                best_agg_score = valid_agg_scores[best_agg_type]
                jrn = baseline_mean / best_agg_score if best_agg_score > 0 else 1.0

            # Compute aggregation sensitivity = CoV across types
            agg_values = list(valid_agg_scores.values())
            if len(agg_values) > 1 and np.mean(agg_values) > 0:
                agg_sensitivity = float(np.std(agg_values) / np.mean(agg_values))
            else:
                agg_sensitivity = 0.0

            # Compute subsampled fan-out
            subsampled_fanout = 0.0
            if direction == "reverse":
                child_table = fk["source_table"]
                fk_col_name = fk["source_fk_col"]
                if child_table in tables and fk_col_name in tables[child_table].columns:
                    child_fk_vals = tables[child_table][fk_col_name]
                    entity_ids = tables[entity_table][entity_col].unique() if entity_col in tables[entity_table].columns else []
                    if len(entity_ids) > 0:
                        matching = child_fk_vals.isin(entity_ids).sum()
                        subsampled_fanout = float(matching / len(entity_ids))

            elapsed = time.time() - t0
            results[(task_name, join_idx)] = {
                "task": task_name, "join_idx": join_idx, "join_key": join_key,
                "direction": direction, "fanout_mean": fanout_mean,
                "baseline": baseline_mean, "baseline_std": baseline_std,
                "best_agg_score": best_agg_score, "best_agg_type": best_agg_type,
                "jrn": jrn, "agg_sensitivity": agg_sensitivity,
                "agg_scores": valid_agg_scores, "subsampled_fanout": subsampled_fanout,
            }
            logger.info(f"  Join {join_idx} ({join_key}): dir={direction}, "
                         f"JRN={jrn:.4f}, best_agg={best_agg_type}, "
                         f"baseline={baseline_mean:.6f}, best={best_agg_score:.6f} "
                         f"({elapsed:.1f}s)")

    return results


# ============================================================
# TRAINING-FREE PROXIES
# ============================================================

def compute_training_free_proxies(
    tables: Dict[str, pd.DataFrame],
    fk_joins: List[dict],
    task_dfs: Dict[str, pd.DataFrame],
    jrn_results: Dict[str, dict],
) -> dict:
    """Compute training-free proxy metrics and correlate with JRN."""
    proxy_results = {}

    for task_name, config in TASK_CONFIG.items():
        if task_name not in task_dfs:
            continue

        entity_table = config["entity_table"]
        entity_col = config["entity_col"]
        task_type = config["task_type"]
        tdf = task_dfs[task_name]

        # Get labels
        train_mask = tdf["_fold"] == 0
        labels = tdf[train_mask]["_label"].values

        for fk in fk_joins:
            join_idx = fk["idx"]
            key = (task_name, join_idx)

            if key not in jrn_results:
                continue
            if jrn_results[key]["direction"] == "none":
                proxy_results[key] = {
                    "fanout_mean": fk.get("fanout_mean", 0),
                    "fanout_std": fk.get("fanout_std", 0),
                    "pearson_r_max": 0.0,
                    "mutual_info": 0.0,
                    "cond_entropy_reduction": 0.0,
                }
                continue

            direction = jrn_results[key]["direction"]

            try:
                if direction == "reverse":
                    agg_feats, new_cols = compute_reverse_join_features(
                        fk, entity_col, tables, "mean"
                    )
                    if agg_feats.empty or not new_cols:
                        proxy_results[key] = {
                            "fanout_mean": fk.get("fanout_mean", 0),
                            "fanout_std": fk.get("fanout_std", 0),
                            "pearson_r_max": 0.0, "mutual_info": 0.0,
                            "cond_entropy_reduction": 0.0,
                        }
                        continue
                    # Merge with task df
                    merged = tdf[train_mask].merge(agg_feats, on=entity_col, how="left")
                    merged[new_cols] = merged[new_cols].fillna(0)
                    feat_matrix = merged[new_cols].values
                elif direction == "forward":
                    parent_table = fk["target_table"]
                    fk_col_name = fk["source_fk_col"]
                    pk_col = fk["target_pk_col"]
                    if parent_table not in tables:
                        continue
                    parent_df = tables[parent_table]
                    parent_numeric = [c for c in parent_df.columns
                                      if c != pk_col and parent_df[c].dtype in
                                      [np.float64, np.float32, np.int64, np.int32, float]]
                    if not parent_numeric:
                        proxy_results[key] = {
                            "fanout_mean": fk.get("fanout_mean", 0),
                            "fanout_std": fk.get("fanout_std", 0),
                            "pearson_r_max": 0.0, "mutual_info": 0.0,
                            "cond_entropy_reduction": 0.0,
                        }
                        continue
                    # Need entity table to link task samples to parent
                    if entity_table not in tables:
                        continue
                    entity_df = tables[entity_table]
                    if entity_col not in entity_df.columns or fk_col_name not in entity_df.columns:
                        continue
                    # Join: task samples -> entity table -> parent table
                    task_entity = tdf[train_mask].merge(
                        entity_df[[entity_col, fk_col_name]], on=entity_col, how="left"
                    )
                    task_parent = task_entity.merge(
                        parent_df[[pk_col] + parent_numeric],
                        left_on=fk_col_name, right_on=pk_col, how="left"
                    )
                    new_cols = parent_numeric
                    feat_matrix = task_parent[new_cols].fillna(0).values
                else:
                    continue

                # (b) Pearson correlation
                if len(labels) > 10 and feat_matrix.shape[0] == len(labels):
                    correlations = []
                    for j in range(feat_matrix.shape[1]):
                        col_data = feat_matrix[:, j]
                        if np.std(col_data) > 0:
                            r = np.corrcoef(col_data, labels.astype(float))[0, 1]
                            if not np.isnan(r):
                                correlations.append(abs(r))
                    pearson_max = float(max(correlations)) if correlations else 0.0
                else:
                    pearson_max = 0.0

                # (c) Mutual information
                mi_score = 0.0
                if feat_matrix.shape[0] > 10:
                    try:
                        # Subsample for MI computation (expensive)
                        n_mi = min(5000, feat_matrix.shape[0])
                        rng = np.random.RandomState(42)
                        mi_idx = rng.choice(feat_matrix.shape[0], n_mi, replace=False)
                        X_mi = feat_matrix[mi_idx]
                        y_mi = labels[mi_idx]

                        if task_type == "regression":
                            mi_vals = mutual_info_regression(X_mi, y_mi, random_state=42)
                        else:
                            mi_vals = mutual_info_classif(X_mi, y_mi, random_state=42)
                        mi_score = float(np.mean(mi_vals))
                    except Exception:
                        logger.exception("  MI computation failed")

                # (d) Conditional entropy reduction via discretized distributions
                cond_entropy_red = 0.0
                if feat_matrix.shape[0] > 50:
                    try:
                        # Discretize labels and features into bins
                        n_bins = 10
                        y_binned = np.digitize(
                            labels.astype(float),
                            np.linspace(labels.astype(float).min(),
                                        labels.astype(float).max() + 1e-10, n_bins + 1)[1:-1]
                        )
                        # H(Y)
                        _, y_counts = np.unique(y_binned, return_counts=True)
                        p_y = y_counts / y_counts.sum()
                        h_y = float(-np.sum(p_y * np.log2(p_y + 1e-10)))

                        # H(Y | X) using best feature (lowest conditional entropy)
                        best_cond_h = h_y
                        for j in range(min(feat_matrix.shape[1], 10)):
                            x_col = feat_matrix[:, j]
                            x_binned = np.digitize(
                                x_col,
                                np.linspace(x_col.min(), x_col.max() + 1e-10, n_bins + 1)[1:-1]
                            )
                            x_vals = np.unique(x_binned)
                            cond_h = 0.0
                            for xv in x_vals:
                                mask_xv = x_binned == xv
                                if mask_xv.sum() == 0:
                                    continue
                                _, yxv_counts = np.unique(y_binned[mask_xv], return_counts=True)
                                p_yxv = yxv_counts / yxv_counts.sum()
                                h_yxv = float(-np.sum(p_yxv * np.log2(p_yxv + 1e-10)))
                                cond_h += (mask_xv.sum() / len(y_binned)) * h_yxv
                            best_cond_h = min(best_cond_h, cond_h)

                        cond_entropy_red = h_y - best_cond_h
                    except Exception:
                        logger.exception("  Conditional entropy computation failed")

                proxy_results[key] = {
                    "fanout_mean": fk.get("fanout_mean", 0),
                    "fanout_std": fk.get("fanout_std", 0),
                    "pearson_r_max": pearson_max,
                    "mutual_info": mi_score,
                    "cond_entropy_reduction": cond_entropy_red,
                }

            except Exception:
                logger.exception(f"  Proxy computation failed for {task_name}, join {join_idx}")
                proxy_results[key] = {
                    "fanout_mean": fk.get("fanout_mean", 0),
                    "fanout_std": fk.get("fanout_std", 0),
                    "pearson_r_max": 0.0, "mutual_info": 0.0,
                    "cond_entropy_reduction": 0.0,
                }

    # Compute Spearman correlations between proxies and JRN
    spearman_results = {}
    jrn_vals = []
    fanout_vals = []
    pearson_vals = []
    mi_vals = []
    entropy_vals = []

    for key in proxy_results:
        if key in jrn_results and jrn_results[key]["direction"] != "none":
            jrn_vals.append(jrn_results[key]["jrn"])
            fanout_vals.append(proxy_results[key]["fanout_mean"])
            pearson_vals.append(proxy_results[key]["pearson_r_max"])
            mi_vals.append(proxy_results[key]["mutual_info"])
            entropy_vals.append(proxy_results[key]["cond_entropy_reduction"])

    if len(jrn_vals) >= 3:
        for name, proxy_vals in [
            ("fanout_vs_jrn", fanout_vals),
            ("pearson_vs_jrn", pearson_vals),
            ("mutual_info_vs_jrn", mi_vals),
            ("cond_entropy_vs_jrn", entropy_vals),
        ]:
            try:
                rho, pval = spearmanr(proxy_vals, jrn_vals)
                spearman_results[name] = {"rho": float(rho) if not np.isnan(rho) else 0.0,
                                          "pval": float(pval) if not np.isnan(pval) else 1.0}
            except Exception:
                spearman_results[name] = {"rho": 0.0, "pval": 1.0}
    else:
        for name in ["fanout_vs_jrn", "pearson_vs_jrn", "mutual_info_vs_jrn", "cond_entropy_vs_jrn"]:
            spearman_results[name] = {"rho": 0.0, "pval": 1.0}

    return {
        "proxy_values": {f"{k[0]}__join{k[1]}": v for k, v in proxy_results.items()},
        "spearman_correlations": spearman_results,
    }


# ============================================================
# PROBE-TO-FULL VALIDATION
# ============================================================

def probe_to_full_validation(
    tables: Dict[str, pd.DataFrame],
    fk_joins: List[dict],
    task_dfs: Dict[str, pd.DataFrame],
    jrn_results: Dict[str, dict],
) -> dict:
    """Validate probe JRN rankings against full-model JRN rankings."""
    # Select pairs: top 4 highest JRN and bottom 4 lowest JRN (among connected joins)
    connected = [(k, v) for k, v in jrn_results.items() if v["direction"] != "none"]
    if len(connected) < 4:
        logger.warning("  Too few connected joins for probe-to-full validation")
        return {"spearman_rho": 0.0, "probe_jrns": [], "full_jrns": [], "n_pairs": 0}

    sorted_by_jrn = sorted(connected, key=lambda x: x[1]["jrn"])
    # Pick bottom 4 and top 4 (or as many as available)
    n_select = min(4, len(sorted_by_jrn) // 2)
    if n_select < 2:
        n_select = min(2, len(sorted_by_jrn))
    selected = sorted_by_jrn[:n_select] + sorted_by_jrn[-n_select:]
    # Deduplicate
    seen = set()
    selected_dedup = []
    for item in selected:
        if item[0] not in seen:
            seen.add(item[0])
            selected_dedup.append(item)
    selected = selected_dedup

    logger.info(f"  Probe-to-full: testing {len(selected)} (task, join) pairs")

    probe_jrns = []
    full_jrns = []

    for (task_name, join_idx), result in selected:
        config = TASK_CONFIG.get(task_name)
        if not config or task_name not in task_dfs:
            continue

        entity_table = config["entity_table"]
        entity_col = config["entity_col"]
        task_type = config["task_type"]
        higher_is_better = config["higher_is_better"]

        tdf = task_dfs[task_name]
        train_mask = tdf["_fold"] == 0
        val_mask = tdf["_fold"] == 1

        # Get entity features
        base_df, base_feature_cols = get_entity_features(tables, tdf, entity_table, entity_col)
        if not base_feature_cols:
            continue

        # Train full baseline
        full_baseline_scores = []
        for seed in SEEDS:
            try:
                s = train_and_evaluate(
                    base_df[train_mask][base_feature_cols],
                    base_df[train_mask]["_label"].values,
                    base_df[val_mask][base_feature_cols],
                    base_df[val_mask]["_label"].values,
                    task_type, FULL_CONFIG, seed,
                )
                full_baseline_scores.append(s)
            except Exception:
                logger.exception("  Full baseline training failed")
                full_baseline_scores.append(0.5 if higher_is_better else 999.0)

        full_baseline = float(np.mean(full_baseline_scores))

        # Find the FK join
        fk = next((f for f in fk_joins if f["idx"] == join_idx), None)
        if fk is None:
            continue

        direction = result["direction"]
        best_agg = result["best_agg_type"]
        if best_agg == "none" or direction == "none":
            continue

        # Get enriched features
        try:
            if direction == "forward":
                enriched, new_cols = add_forward_join_features(base_df, fk, tables)
            else:
                agg_feats, new_cols = compute_reverse_join_features(
                    fk, entity_col, tables, best_agg
                )
                if agg_feats.empty:
                    continue
                enriched = base_df.merge(agg_feats, on=entity_col, how="left")
                enriched[new_cols] = enriched[new_cols].fillna(0)

            all_cols = base_feature_cols + new_cols

            # Train full model with join features
            full_join_scores = []
            for seed in SEEDS:
                s = train_and_evaluate(
                    enriched[train_mask][all_cols],
                    enriched[train_mask]["_label"].values,
                    enriched[val_mask][all_cols],
                    enriched[val_mask]["_label"].values,
                    task_type, FULL_CONFIG, seed,
                )
                full_join_scores.append(s)

            full_join_mean = float(np.mean(full_join_scores))

            # Compute full JRN
            if higher_is_better:
                full_jrn = full_join_mean / full_baseline if full_baseline > 0 else 1.0
            else:
                full_jrn = full_baseline / full_join_mean if full_join_mean > 0 else 1.0

            probe_jrns.append(result["jrn"])
            full_jrns.append(full_jrn)
            logger.info(f"    {task_name}/join{join_idx}: probe_JRN={result['jrn']:.4f}, "
                         f"full_JRN={full_jrn:.4f}")

        except Exception:
            logger.exception(f"  Probe-to-full failed for {task_name}/join{join_idx}")

    # Compute Spearman correlation
    if len(probe_jrns) >= 3:
        rho, pval = spearmanr(probe_jrns, full_jrns)
        rho = float(rho) if not np.isnan(rho) else 0.0
        pval = float(pval) if not np.isnan(pval) else 1.0
    else:
        rho, pval = 0.0, 1.0

    logger.info(f"  Probe-to-full Spearman rho = {rho:.4f} (p={pval:.4f})")

    return {
        "spearman_rho": rho, "spearman_pval": pval,
        "probe_jrns": probe_jrns, "full_jrns": full_jrns,
        "n_pairs": len(probe_jrns),
    }


# ============================================================
# OUTPUT ASSEMBLY
# ============================================================

def assemble_output(
    jrn_results: Dict[str, dict],
    proxy_data: dict,
    probe_full_data: dict,
    fk_joins: List[dict],
) -> dict:
    """Assemble output in exp_gen_sol_out.json format."""
    examples = []

    # 1) JRN measurement examples
    for (task_name, join_idx), result in sorted(jrn_results.items()):
        input_data = {
            "task_name": result["task"],
            "join_key": result["join_key"],
            "join_idx": result["join_idx"],
            "direction": result["direction"],
            "fanout_mean": result["fanout_mean"],
            "subsampled_fanout": result.get("subsampled_fanout", 0),
        }
        output_data = {
            "jrn": round(result["jrn"], 6),
            "best_agg_type": result["best_agg_type"],
            "agg_sensitivity": round(result["agg_sensitivity"], 6),
            "agg_scores": {k: round(v, 6) for k, v in result["agg_scores"].items()},
        }
        baseline_data = {
            "score": round(result["baseline"], 6),
            "std": round(result["baseline_std"], 6),
            "metric": TASK_CONFIG.get(task_name, {}).get("metric", "unknown"),
        }
        method_data = {
            "score": round(result["best_agg_score"], 6),
            "best_agg_type": result["best_agg_type"],
            "metric": TASK_CONFIG.get(task_name, {}).get("metric", "unknown"),
        }

        examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "predict_baseline": json.dumps(baseline_data),
            "predict_our_method": json.dumps(method_data),
            "metadata_measurement_type": "jrn_measurement",
            "metadata_task_name": task_name,
            "metadata_join_idx": join_idx,
            "metadata_join_key": result["join_key"],
            "metadata_direction": result["direction"],
        })

    # 2) Training-free proxy correlations
    examples.append({
        "input": json.dumps({
            "analysis": "training_free_proxy_correlations",
            "proxies": ["fanout", "pearson_r", "mutual_info", "cond_entropy_reduction"],
            "n_connected_joins": sum(1 for v in jrn_results.values() if v["direction"] != "none"),
        }),
        "output": json.dumps(proxy_data["spearman_correlations"]),
        "predict_baseline": json.dumps({"note": "no proxy baseline"}),
        "predict_our_method": json.dumps(proxy_data["spearman_correlations"]),
        "metadata_measurement_type": "proxy_correlations",
    })

    # 3) Probe-to-full validation
    examples.append({
        "input": json.dumps({
            "analysis": "probe_to_full_validation",
            "probe_config": PROBE_CONFIG,
            "full_config": FULL_CONFIG,
            "n_pairs": probe_full_data["n_pairs"],
        }),
        "output": json.dumps({
            "spearman_rho": round(probe_full_data["spearman_rho"], 4),
            "spearman_pval": round(probe_full_data.get("spearman_pval", 1.0), 4),
            "probe_jrns": [round(v, 4) for v in probe_full_data["probe_jrns"]],
            "full_jrns": [round(v, 4) for v in probe_full_data["full_jrns"]],
        }),
        "predict_baseline": json.dumps({"probe_config": PROBE_CONFIG}),
        "predict_our_method": json.dumps({"full_config": FULL_CONFIG}),
        "metadata_measurement_type": "probe_to_full",
    })

    # 4) JRN matrix summary
    tasks = sorted(TASK_CONFIG.keys())
    join_keys = [f"{fk['source_table']}.{fk['source_fk_col']}->{fk['target_table']}" for fk in fk_joins]
    jrn_matrix = []
    for fk in fk_joins:
        row = []
        for task in tasks:
            key = (task, fk["idx"])
            if key in jrn_results:
                row.append(round(jrn_results[key]["jrn"], 4))
            else:
                row.append(1.0)
        jrn_matrix.append(row)

    examples.append({
        "input": json.dumps({
            "analysis": "jrn_matrix_summary",
            "joins": join_keys,
            "tasks": tasks,
        }),
        "output": json.dumps({
            "jrn_matrix": jrn_matrix,
            "joins": join_keys,
            "tasks": tasks,
            "connected_joins_per_task": {
                t: sum(1 for fk in fk_joins
                       if (t, fk["idx"]) in jrn_results
                       and jrn_results[(t, fk["idx"])]["direction"] != "none")
                for t in tasks
            },
        }),
        "predict_baseline": json.dumps({"note": "all JRN=1.0 (no joins)"}),
        "predict_our_method": json.dumps({"note": "JRN matrix with probe model"}),
        "metadata_measurement_type": "jrn_matrix",
    })

    # 5) Extreme cardinality analysis
    sorted_by_fanout = sorted(
        [(k, v) for k, v in jrn_results.items() if v["direction"] != "none"],
        key=lambda x: x[1]["fanout_mean"],
    )
    extreme_data = {
        "joins_by_fanout": [
            {"join_key": v["join_key"], "fanout_mean": v["fanout_mean"],
             "jrn": round(v["jrn"], 4), "task": v["task"]}
            for _, v in sorted_by_fanout
        ],
    }
    # Correlation between fan-out and JRN (connected joins only)
    connected_fanouts = [v["fanout_mean"] for _, v in sorted_by_fanout]
    connected_jrns = [v["jrn"] for _, v in sorted_by_fanout]
    if len(connected_fanouts) >= 3:
        rho, _ = spearmanr(connected_fanouts, connected_jrns)
        extreme_data["jrn_vs_fanout_spearman"] = round(float(rho) if not np.isnan(rho) else 0.0, 4)
    else:
        extreme_data["jrn_vs_fanout_spearman"] = 0.0

    examples.append({
        "input": json.dumps({"analysis": "extreme_cardinality_analysis"}),
        "output": json.dumps(extreme_data),
        "predict_baseline": json.dumps({"note": "no cardinality baseline"}),
        "predict_our_method": json.dumps(extreme_data),
        "metadata_measurement_type": "extreme_cardinality",
    })

    # 6) Aggregation sensitivity analysis
    agg_sensitivity_data = []
    for (task_name, join_idx), result in sorted(jrn_results.items()):
        if result["direction"] != "none" and len(result["agg_scores"]) > 1:
            agg_sensitivity_data.append({
                "task": task_name, "join_key": result["join_key"],
                "jrn": round(result["jrn"], 4),
                "agg_sensitivity": round(result["agg_sensitivity"], 6),
                "agg_scores": {k: round(v, 6) for k, v in result["agg_scores"].items()},
            })

    examples.append({
        "input": json.dumps({"analysis": "aggregation_sensitivity"}),
        "output": json.dumps({
            "data": agg_sensitivity_data,
            "summary": "Aggregation sensitivity across joins and tasks",
        }),
        "predict_baseline": json.dumps({"note": "single aggregation baseline"}),
        "predict_our_method": json.dumps({"note": "multi-aggregation JRN analysis"}),
        "metadata_measurement_type": "aggregation_sensitivity",
    })

    # 7) Cross-dataset comparison note
    examples.append({
        "input": json.dumps({
            "analysis": "cross_dataset_comparison",
            "note": "Compare with rel-f1 rho=0.960 from prior experiments",
        }),
        "output": json.dumps({
            "rel_avito_proxy_rho": proxy_data["spearman_correlations"],
            "rel_avito_probe_to_full_rho": round(probe_full_data["spearman_rho"], 4),
            "note": "Cross-dataset validation: rel-f1 showed proxy-JRN rho=0.960. "
                    "rel-avito results here provide independent validation at extreme cardinality scales.",
        }),
        "predict_baseline": json.dumps({"note": "single-dataset analysis"}),
        "predict_our_method": json.dumps({"note": "cross-dataset JRN validation"}),
        "metadata_measurement_type": "cross_dataset",
    })

    # Build full output
    output = {
        "metadata": {
            "dataset": "rel-avito",
            "num_tables": 8,
            "num_fk_joins": 11,
            "num_tasks_tested": 3,
            "tasks_excluded": ["rel-avito/user-ad-visit (link_prediction - requires specialized approach)"],
            "probe_config": PROBE_CONFIG,
            "full_config": FULL_CONFIG,
            "seeds": SEEDS,
            "max_train_rows": MAX_TRAIN_ROWS,
            "agg_types_tested": AGG_TYPES,
        },
        "datasets": [
            {
                "dataset": "rel-avito",
                "examples": examples,
            }
        ],
    }

    return output


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main():
    t_start = time.time()

    # Phase 0: Data loading
    logger.info("=" * 60)
    logger.info("PHASE 0: DATA LOADING")
    logger.info("=" * 60)

    table_records, fk_joins, task_records = load_all_data()

    # Sort FK joins by index
    fk_joins = sorted(fk_joins, key=lambda x: x["idx"])

    # Build tables
    logger.info("Building tables...")
    tables = build_tables(table_records)
    del table_records
    gc.collect()

    # Build task DataFrames
    logger.info("Building task DataFrames...")
    task_dfs = build_task_dfs(task_records)
    del task_records
    gc.collect()

    mem_used = psutil.virtual_memory().used / 1e9
    logger.info(f"Memory after loading: {mem_used:.1f} GB")

    # Phase 1-3: JRN Computation
    logger.info("=" * 60)
    logger.info("PHASES 1-3: JRN COMPUTATION (3 tasks x 11 joins)")
    logger.info("=" * 60)

    jrn_results = compute_jrn_matrix(tables, fk_joins, task_dfs)
    logger.info(f"Computed {len(jrn_results)} JRN values")

    # Summary of JRN results
    for (task, jidx), res in sorted(jrn_results.items()):
        logger.info(f"  {task} / join{jidx} ({res['join_key']}): "
                     f"JRN={res['jrn']:.4f} dir={res['direction']}")

    elapsed = time.time() - t_start
    logger.info(f"JRN computation done in {elapsed:.1f}s")

    # Phase 4: Training-free proxies
    logger.info("=" * 60)
    logger.info("PHASE 4: TRAINING-FREE PROXIES")
    logger.info("=" * 60)

    proxy_data = compute_training_free_proxies(tables, fk_joins, task_dfs, jrn_results)
    logger.info(f"Proxy Spearman correlations: {proxy_data['spearman_correlations']}")

    # Phase 5: Probe-to-full validation
    logger.info("=" * 60)
    logger.info("PHASE 5: PROBE-TO-FULL VALIDATION")
    logger.info("=" * 60)

    probe_full_data = probe_to_full_validation(tables, fk_joins, task_dfs, jrn_results)

    # Phase 7: Output assembly
    logger.info("=" * 60)
    logger.info("PHASE 7: OUTPUT ASSEMBLY")
    logger.info("=" * 60)

    output = assemble_output(jrn_results, proxy_data, probe_full_data, fk_joins)

    # Write output
    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {output_path} ({output_size_mb:.1f} MB)")

    elapsed_total = time.time() - t_start
    logger.info(f"Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
