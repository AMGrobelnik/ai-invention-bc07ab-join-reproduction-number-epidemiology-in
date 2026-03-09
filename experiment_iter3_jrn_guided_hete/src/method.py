#!/usr/bin/env python3
"""JRN-Guided Heterogeneous Architecture: GBM Probe Estimation & 5-Config Comparison on rel-f1.

Estimates Join Reproduction Number (JRN) for all 13 FK joins across tasks using
LightGBM probes, classifies joins as HIGH/CRITICAL/LOW, constructs a JRN-guided
model with per-join aggregation strategy, and compares against 4 baselines
including an exhaustive oracle search over all 2^13 join subsets.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ─── Hardware detection ────────────────────────────────────────────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

# Set memory limit to 80% of container RAM
RAM_BUDGET = int(TOTAL_RAM_GB * 0.80 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU time

# ─── Constants ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id3_it1__opus/full_data_out.json")

VAL_TIMESTAMP = "2005-01-01"
TEST_TIMESTAMP = "2010-01-01"

TASKS = [
    "rel-f1/driver-dnf",
    "rel-f1/driver-top3",
    "rel-f1/driver-position",
    "rel-f1/qualifying-position",
    "rel-f1/results-position",
]

TASK_META = {
    "rel-f1/driver-dnf":          {"type": "clf", "entity_table": "drivers",     "entity_col": "driverId",  "target": "did_not_finish"},
    "rel-f1/driver-top3":         {"type": "clf", "entity_table": "drivers",     "entity_col": "driverId",  "target": "qualifying"},
    "rel-f1/driver-position":     {"type": "reg", "entity_table": "drivers",     "entity_col": "driverId",  "target": "position"},
    "rel-f1/qualifying-position": {"type": "reg", "entity_table": "qualifying",  "entity_col": "qualifyId", "target": "position"},
    "rel-f1/results-position":    {"type": "reg", "entity_table": "results",     "entity_col": "resultId",  "target": "position"},
}

FK_JOINS = [
    {"idx": 0,  "child": "races",                 "fk_col": "circuitId",     "parent": "circuits"},
    {"idx": 1,  "child": "constructor_standings",  "fk_col": "raceId",       "parent": "races"},
    {"idx": 2,  "child": "constructor_standings",  "fk_col": "constructorId","parent": "constructors"},
    {"idx": 3,  "child": "standings",              "fk_col": "raceId",       "parent": "races"},
    {"idx": 4,  "child": "standings",              "fk_col": "driverId",     "parent": "drivers"},
    {"idx": 5,  "child": "constructor_results",    "fk_col": "raceId",       "parent": "races"},
    {"idx": 6,  "child": "constructor_results",    "fk_col": "constructorId","parent": "constructors"},
    {"idx": 7,  "child": "qualifying",             "fk_col": "raceId",       "parent": "races"},
    {"idx": 8,  "child": "qualifying",             "fk_col": "driverId",     "parent": "drivers"},
    {"idx": 9,  "child": "qualifying",             "fk_col": "constructorId","parent": "constructors"},
    {"idx": 10, "child": "results",                "fk_col": "raceId",       "parent": "races"},
    {"idx": 11, "child": "results",                "fk_col": "driverId",     "parent": "drivers"},
    {"idx": 12, "child": "results",                "fk_col": "constructorId","parent": "constructors"},
]

JRN_HIGH = 1.15
JRN_LOW  = 0.85

SEEDS = [42, 123, 7]
PROBE_PARAMS  = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "verbose": -1}
FINAL_PARAMS  = {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "verbose": -1}
ORACLE_PARAMS = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "verbose": -1}

# ─── Join relevance mapping ───────────────────────────────────────────────────
# For driver-* tasks: which joins are DIRECT (child→drivers via driverId),
# which are 2-HOP (child→parent, then aggregate by driverId via another FK), N/A
DRIVER_DIRECT_JOINS = {4, 8, 11}  # standings→drivers, qualifying→drivers, results→drivers
DRIVER_2HOP_JOINS = {
    # join_idx: (child_table, intermediate_fk, entity_fk_in_child)
    3:  ("standings",  "raceId",        "driverId"),  # standings→races, agg by driverId
    7:  ("qualifying", "raceId",        "driverId"),  # qualifying→races, agg by driverId
    9:  ("qualifying", "constructorId", "driverId"),  # qualifying→constructors, agg by driverId
    10: ("results",    "raceId",        "driverId"),  # results→races, agg by driverId
    12: ("results",    "constructorId", "driverId"),  # results→constructors, agg by driverId
}
DRIVER_NA_JOINS = {0, 1, 2, 5, 6}  # 3+ hops from driver entity

# For qualifying/results tasks: ALL 13 joins are reachable within 2 hops
# Direct lookup: the entity table's own FK joins
# 2-hop: through the entity table's parent tables
QUAL_DIRECT_JOINS = {7, 8, 9}   # qualifying→races, qualifying→drivers, qualifying→constructors
RESULTS_DIRECT_JOINS = {10, 11, 12}  # results→races, results→drivers, results→constructors

# ─── Helpers ───────────────────────────────────────────────────────────────────
def try_numeric(val):
    """Try to convert a string to int or float."""
    if val is None:
        return np.nan
    try:
        f = float(val)
        if f == int(f) and "." not in str(val):
            return int(f)
        return f
    except (ValueError, TypeError):
        return val


def get_numeric_cols(df: pd.DataFrame, exclude: set | None = None) -> list[str]:
    """Return list of numeric columns, excluding specified ones."""
    exclude = exclude or set()
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def compute_agg_features(df: pd.DataFrame, group_col: str, numeric_cols: list[str],
                         strategies: list[str] = None, prefix: str = "") -> pd.DataFrame:
    """Aggregate numeric columns by group_col using specified strategies."""
    if strategies is None:
        strategies = ["mean"]
    if not numeric_cols or df.empty:
        return pd.DataFrame()

    agg_dict = {}
    for col in numeric_cols:
        for strat in strategies:
            agg_dict[f"{prefix}{col}_{strat}"] = pd.NamedAgg(column=col, aggfunc=strat)

    try:
        result = df.groupby(group_col).agg(**agg_dict)
    except Exception:
        # Fallback: manual aggregation
        result = pd.DataFrame(index=df[group_col].unique())
        result.index.name = group_col
        for col in numeric_cols:
            for strat in strategies:
                try:
                    result[f"{prefix}{col}_{strat}"] = df.groupby(group_col)[col].agg(strat)
                except Exception:
                    pass

    result[f"{prefix}__count__"] = df.groupby(group_col).size()
    return result


def get_temporal_table(table_dfs: dict, table_name: str, fold: int) -> pd.DataFrame:
    """Return temporally filtered table rows for the given fold."""
    df = table_dfs[table_name].copy()
    if table_name == "races":
        date_col = pd.to_datetime(df["date"], errors="coerce")
    elif "date" in df.columns:
        date_col = pd.to_datetime(df["date"], errors="coerce")
    elif "raceId" in df.columns and "races" in table_dfs:
        race_dates = table_dfs["races"][["raceId", "date"]].copy()
        race_dates["__race_date__"] = pd.to_datetime(race_dates["date"], errors="coerce")
        df = df.merge(race_dates[["raceId", "__race_date__"]], on="raceId", how="left")
        date_col = df["__race_date__"]
    else:
        return df  # no temporal filtering

    cutoff_map = {0: VAL_TIMESTAMP, 1: TEST_TIMESTAMP}
    cutoff = cutoff_map.get(fold)
    if cutoff:
        mask = date_col < pd.Timestamp(cutoff)
        df = df[mask].copy()

    # Drop helper columns
    for c in ["__race_date__"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df


# ─── GBM training/evaluation ──────────────────────────────────────────────────
def train_eval_gbm(X_train, y_train, X_val, y_val, task_type: str, seed: int,
                   params: dict | None = None) -> float:
    """Train LightGBM and return validation metric (higher = better)."""
    import lightgbm as lgb

    if params is None:
        params = PROBE_PARAMS
    model_params = {**params, "random_state": seed, "n_jobs": NUM_CPUS}

    # Clean data
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Deduplicate columns
    if X_train.columns.duplicated().any():
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    if X_val.columns.duplicated().any():
        X_val = X_val.loc[:, ~X_val.columns.duplicated()]

    # Align columns
    common_cols = list(X_train.columns.intersection(X_val.columns))
    if common_cols:
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]

    # Replace inf with nan, then fill
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)

    # Ensure all columns are numeric
    for c in list(X_train.columns):
        try:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0)
        except TypeError:
            X_train = X_train.drop(columns=[c])
            X_val = X_val.drop(columns=[c], errors="ignore")
    for c in list(X_val.columns):
        try:
            X_val[c] = pd.to_numeric(X_val[c], errors="coerce").fillna(0)
        except TypeError:
            X_val = X_val.drop(columns=[c], errors="ignore")
            X_train = X_train.drop(columns=[c], errors="ignore")

    y_train_clean = pd.to_numeric(y_train, errors="coerce").fillna(0)
    y_val_clean = pd.to_numeric(y_val, errors="coerce").fillna(0)

    if len(X_train) == 0 or len(X_val) == 0:
        return 0.5 if task_type == "clf" else 0.0

    try:
        if task_type == "clf":
            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train, y_train_clean)
            y_pred = model.predict_proba(X_val)
            if y_pred.shape[1] > 1:
                y_pred = y_pred[:, 1]
            else:
                y_pred = y_pred[:, 0]
            # Check for constant y_val
            if len(y_val_clean.unique()) < 2:
                return 0.5
            return roc_auc_score(y_val_clean, y_pred)
        else:
            model = lgb.LGBMRegressor(**model_params)
            model.fit(X_train, y_train_clean)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val_clean, y_pred)
            return 1.0 / max(mae, 1e-8)
    except Exception as e:
        logger.warning(f"GBM training failed: {e}")
        return 0.5 if task_type == "clf" else 0.0


def train_predict_gbm(X_train, y_train, X_pred, task_type: str, seed: int,
                      params: dict | None = None):
    """Train LightGBM and return predictions on X_pred."""
    import lightgbm as lgb

    if params is None:
        params = FINAL_PARAMS
    model_params = {**params, "random_state": seed, "n_jobs": NUM_CPUS}

    X_train = X_train.copy()
    X_pred = X_pred.copy()

    # Deduplicate columns
    if X_train.columns.duplicated().any():
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    if X_pred.columns.duplicated().any():
        X_pred = X_pred.loc[:, ~X_pred.columns.duplicated()]

    common_cols = list(X_train.columns.intersection(X_pred.columns))
    if common_cols:
        X_train = X_train[common_cols]
        X_pred = X_pred[common_cols]

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_pred.fillna(0, inplace=True)

    for c in list(X_train.columns):
        try:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0)
        except TypeError:
            X_train = X_train.drop(columns=[c])
            X_pred = X_pred.drop(columns=[c], errors="ignore")
    for c in list(X_pred.columns):
        try:
            X_pred[c] = pd.to_numeric(X_pred[c], errors="coerce").fillna(0)
        except TypeError:
            X_pred = X_pred.drop(columns=[c], errors="ignore")
            X_train = X_train.drop(columns=[c], errors="ignore")

    y_train_clean = pd.to_numeric(y_train, errors="coerce").fillna(0)

    try:
        if task_type == "clf":
            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train, y_train_clean)
            y_pred = model.predict_proba(X_pred)
            return y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0]
        else:
            model = lgb.LGBMRegressor(**model_params)
            model.fit(X_train, y_train_clean)
            return model.predict(X_pred)
    except Exception as e:
        logger.warning(f"GBM predict failed: {e}")
        if task_type == "clf":
            return np.full(len(X_pred), 0.5)
        else:
            return np.full(len(X_pred), y_train_clean.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@logger.catch
def main():
    t0 = time.time()
    logger.info("Phase 1: Loading data...")

    with open(DATA_PATH) as f:
        raw = json.load(f)
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples in {time.time()-t0:.1f}s")

    # ── Separate by row_type ──
    table_rows: dict[str, list] = {}
    fk_metadata: list = []
    task_samples: dict[str, list] = {}

    for ex in examples:
        row_type = ex["metadata_row_type"]
        if row_type == "table_row":
            tname = ex["metadata_table_name"]
            row_data = json.loads(ex["input"])
            pk_col = ex["metadata_primary_key_col"]
            pk_val = ex["metadata_primary_key_value"]
            row_data[pk_col] = try_numeric(pk_val)
            table_rows.setdefault(tname, []).append(row_data)
        elif row_type == "fk_join_metadata":
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])
            fk_metadata.append({**inp, **out})
        elif row_type == "task_sample":
            tname = ex.get("metadata_task_name", "")
            if tname not in TASKS:
                continue
            inp = json.loads(ex["input"])
            label = ex["output"]
            fold = ex["metadata_fold"]
            task_samples.setdefault(tname, []).append({
                "features": inp,
                "label": try_numeric(label) if label != "masked" else np.nan,
                "fold": fold,
                "input_str": ex["input"],
                "output_str": ex["output"],
            })

    del examples, raw
    gc.collect()

    # ── Build pandas DataFrames for tables ──
    logger.info("Building table DataFrames...")
    table_dfs: dict[str, pd.DataFrame] = {}
    for tname, rows in table_rows.items():
        df = pd.DataFrame(rows)
        for col in df.columns:
            numeric_attempt = pd.to_numeric(df[col], errors="coerce")
            if numeric_attempt.notna().mean() > 0.5:
                df[col] = numeric_attempt
        table_dfs[tname] = df
        logger.debug(f"  Table '{tname}': {len(df)} rows, {len(df.columns)} cols")

    del table_rows
    gc.collect()

    for tname, df in table_dfs.items():
        logger.info(f"  {tname}: {df.shape} cols={list(df.columns)}")

    # ── Build task DataFrames ──
    logger.info("Building task DataFrames...")
    task_dfs: dict[str, pd.DataFrame] = {}
    task_raw_samples: dict[str, list] = {}  # keep raw for output

    for tname, samples in task_samples.items():
        rows = []
        for s in samples:
            row = {**s["features"], "__label__": s["label"], "__fold__": s["fold"]}
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col not in ("__label__", "__fold__"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["__label__"] = pd.to_numeric(df["__label__"], errors="coerce")
        task_dfs[tname] = df
        task_raw_samples[tname] = samples
        fold_counts = df["__fold__"].value_counts().to_dict()
        logger.info(f"  {tname}: {len(df)} samples, folds={fold_counts}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: ENTITY + JOIN FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Phase 2: Feature engineering...")

    # Determine which tasks are trainable
    trainable_tasks = []
    for tname in TASKS:
        df = task_dfs[tname]
        n_train = (df["__fold__"] == 0).sum()
        n_val = (df["__fold__"] == 1).sum()
        if n_train >= 10 and n_val >= 10:
            trainable_tasks.append(tname)
        elif n_val >= 50:
            # Split val into pseudo-train/val (70/30)
            logger.info(f"  {tname}: no train data, splitting val 70/30")
            val_idx = df[df["__fold__"] == 1].index
            np.random.seed(42)
            shuffled = np.random.permutation(val_idx)
            split_pt = int(len(shuffled) * 0.7)
            df.loc[shuffled[:split_pt], "__fold__"] = 0
            df.loc[shuffled[split_pt:], "__fold__"] = 1
            task_dfs[tname] = df
            trainable_tasks.append(tname)
        else:
            logger.warning(f"  {tname}: insufficient data, skipping")

    logger.info(f"Trainable tasks: {trainable_tasks}")

    # ── Entity feature builder ──
    def build_entity_features(task_name: str) -> pd.DataFrame:
        """Build entity-level features for each task sample."""
        meta = TASK_META[task_name]
        entity_table = meta["entity_table"]
        entity_col = meta["entity_col"]
        tdf = task_dfs[task_name].copy()

        if entity_table == "drivers":
            # Look up driver features
            drivers = table_dfs["drivers"].copy()
            # Encode nationality with frequency encoding
            nat_freq = drivers["nationality"].value_counts(normalize=True).to_dict()
            drivers["nationality_freq"] = drivers["nationality"].map(nat_freq).fillna(0)
            # Birth year
            drivers["dob_dt"] = pd.to_datetime(drivers["dob"], errors="coerce")
            drivers["birth_year"] = drivers["dob_dt"].dt.year
            driver_feats = drivers[["driverId", "nationality_freq", "birth_year"]].copy()
            driver_feats["birth_year"] = pd.to_numeric(driver_feats["birth_year"], errors="coerce").fillna(1970)

            # Merge with task samples
            tdf = tdf.merge(driver_feats, on="driverId", how="left")
            # Add date features
            tdf["date_dt"] = pd.to_datetime(tdf["date"], errors="coerce")
            tdf["sample_year"] = tdf["date_dt"].dt.year.fillna(2000)
            tdf["driver_age"] = tdf["sample_year"] - tdf["birth_year"]
            tdf.drop(columns=["date", "date_dt", "birth_year"], inplace=True, errors="ignore")

        elif entity_table in ("qualifying", "results"):
            # Entity IS the row — use the entity table's features
            entity_df = table_dfs[entity_table].copy()
            # Exclude target column and PK
            exclude_cols = {"position", entity_col, "date"}
            feat_cols = [c for c in get_numeric_cols(entity_df) if c not in exclude_cols]

            # Merge entity features with task samples
            entity_feats = entity_df[[entity_col] + feat_cols].copy()
            tdf = tdf.merge(entity_feats, on=entity_col, how="left")
            # Add date features
            tdf["date_dt"] = pd.to_datetime(tdf["date"], errors="coerce")
            tdf["sample_year"] = tdf["date_dt"].dt.year.fillna(2005)
            tdf.drop(columns=["date", "date_dt"], inplace=True, errors="ignore")

        # Drop non-numeric columns
        drop_cols = [c for c in tdf.columns if tdf[c].dtype == object and c not in ("__label__", "__fold__")]
        tdf.drop(columns=drop_cols, inplace=True, errors="ignore")

        return tdf

    # ── Join feature builder ──
    def build_join_features_for_task(task_name: str, join_idx: int, fold: int,
                                     strategies: list[str] = None) -> pd.DataFrame | None:
        """Build features from a specific FK join for a task's entities in a given fold."""
        if strategies is None:
            strategies = ["mean"]

        meta = TASK_META[task_name]
        entity_table = meta["entity_table"]
        entity_col = meta["entity_col"]
        join = FK_JOINS[join_idx]
        child_table = join["child"]
        fk_col = join["fk_col"]
        parent_table = join["parent"]
        prefix = f"j{join_idx}_"

        # ─ DRIVER TASKS ─
        if entity_table == "drivers":
            if join_idx in DRIVER_NA_JOINS:
                return None

            if join_idx in DRIVER_DIRECT_JOINS:
                # Direct: child table has driverId, aggregate by driverId
                child_df = get_temporal_table(table_dfs, child_table, fold)
                if "driverId" not in child_df.columns or child_df.empty:
                    return None
                num_cols = get_numeric_cols(child_df, exclude={"driverId", "raceId", "constructorId",
                                                                child_table + "Id", "driverStandingsId",
                                                                "qualifyId", "resultId", "statusId"})
                if not num_cols:
                    return None
                return compute_agg_features(child_df, "driverId", num_cols, strategies, prefix)

            if join_idx in DRIVER_2HOP_JOINS:
                # 2-hop: enrich child with parent features, then aggregate by driverId
                child_tbl, inter_fk, entity_fk = DRIVER_2HOP_JOINS[join_idx]
                child_df = get_temporal_table(table_dfs, child_tbl, fold)
                if entity_fk not in child_df.columns or child_df.empty:
                    return None
                parent_df = table_dfs[parent_table].copy()
                # Get parent numeric features
                parent_pk = fk_col  # FK in child = PK in parent
                parent_num_cols = get_numeric_cols(parent_df, exclude={parent_pk, "date"})
                if not parent_num_cols:
                    return None
                # Rename parent cols with prefix
                rename_map = {c: f"{prefix}p_{c}" for c in parent_num_cols}
                parent_subset = parent_df[[parent_pk] + parent_num_cols].rename(columns=rename_map)
                # Merge child with parent
                enriched = child_df.merge(parent_subset, left_on=inter_fk, right_on=parent_pk, how="left")
                enriched_num_cols = [f"{prefix}p_{c}" for c in parent_num_cols]
                return compute_agg_features(enriched, entity_fk, enriched_num_cols, strategies, prefix)

            return None

        # ─ QUALIFYING / RESULTS TASKS ─
        elif entity_table in ("qualifying", "results"):
            # For these tasks, each row IS an entity with its own FK columns
            tdf = task_dfs[task_name]
            entity_df = table_dfs[entity_table]

            # Determine which FK columns the entity table has
            entity_fk_map = {}
            if entity_table == "qualifying":
                entity_fk_map = {"raceId": "races", "driverId": "drivers", "constructorId": "constructors"}
            else:  # results
                entity_fk_map = {"raceId": "races", "driverId": "drivers", "constructorId": "constructors"}

            if join_idx in (QUAL_DIRECT_JOINS if entity_table == "qualifying" else RESULTS_DIRECT_JOINS):
                # Direct lookup: entity's FK → parent table
                if fk_col not in entity_df.columns:
                    return None
                parent_df = table_dfs[parent_table].copy()
                parent_pk = fk_col
                parent_num_cols = get_numeric_cols(parent_df, exclude={parent_pk, "date", parent_table + "Id"})
                if not parent_num_cols:
                    return None
                rename_map = {c: f"{prefix}{c}" for c in parent_num_cols}
                parent_feats = parent_df[[parent_pk] + parent_num_cols].rename(columns=rename_map)
                # Create per-entity lookup via entity_col → FK col mapping
                entity_with_fk = entity_df[[entity_col, fk_col]].copy()
                merged = entity_with_fk.merge(parent_feats, on=fk_col, how="left")
                result = merged.drop(columns=[fk_col]).set_index(entity_col)
                return result

            else:
                # 2-hop: through entity's FK to a parent, then aggregate another child of that parent
                # Find the intermediate path
                # The join is child_table(fk_col) → parent_table
                # We need to connect via a shared FK between entity_table and child_table
                shared_fks = set(entity_fk_map.keys()) & {fk_col}
                if shared_fks:
                    shared_fk = fk_col
                else:
                    # Try indirect: child's parent is one of entity's parents
                    # e.g., join 4 (standings→drivers): entity=qualifying, qualifying has driverId
                    if parent_table in entity_fk_map.values():
                        # Entity has FK to the parent table
                        for efk, etgt in entity_fk_map.items():
                            if etgt == parent_table:
                                shared_fk = efk
                                break
                        else:
                            return None
                    elif fk_col in entity_df.columns:
                        shared_fk = fk_col
                    else:
                        return None

                child_df = get_temporal_table(table_dfs, child_table, fold)
                if child_df.empty or shared_fk not in entity_df.columns:
                    return None

                # Aggregate child table by the shared FK
                num_cols = get_numeric_cols(child_df, exclude={fk_col, "raceId", "driverId",
                                                                "constructorId", "date",
                                                                "driverStandingsId", "constructorStandingsId",
                                                                "constructorResultsId", "qualifyId",
                                                                "resultId", "statusId", "circuitId"})
                if not num_cols:
                    return None
                agg_feats = compute_agg_features(child_df, fk_col, num_cols, strategies, prefix)

                # Map entity rows to the FK value, then look up aggregated features
                entity_fk_vals = entity_df[[entity_col, shared_fk]].copy()
                merged = entity_fk_vals.merge(agg_feats, left_on=shared_fk, right_index=True, how="left")
                result = merged.drop(columns=[shared_fk]).set_index(entity_col)
                return result

        return None

    # ── Build all features ──
    logger.info("Building entity + join features for all tasks...")
    all_features: dict[str, dict] = {}

    for task_name in trainable_tasks:
        logger.info(f"  Processing {task_name}...")
        entity_df = build_entity_features(task_name)
        meta = TASK_META[task_name]
        entity_col = meta["entity_col"]

        # Build join features per fold
        join_mean_feats: dict[int, pd.DataFrame] = {}
        join_rich_feats: dict[int, pd.DataFrame] = {}

        folds_present = entity_df["__fold__"].unique()

        for join_idx in range(13):
            mean_parts = []
            rich_parts = []
            for fold in sorted(folds_present):
                fold = int(fold)
                mean_f = build_join_features_for_task(task_name, join_idx, fold, ["mean"])
                rich_f = build_join_features_for_task(task_name, join_idx, fold,
                                                      ["mean", "sum", "max", "min", "std"])
                if mean_f is not None and not mean_f.empty:
                    # Get entity IDs for this fold
                    fold_mask = entity_df["__fold__"] == fold
                    fold_entity_ids = entity_df.loc[fold_mask, entity_col]
                    # Align join features with task samples
                    mean_aligned = mean_f.reindex(fold_entity_ids.values)
                    mean_aligned.index = fold_entity_ids.index
                    mean_parts.append(mean_aligned)

                if rich_f is not None and not rich_f.empty:
                    fold_mask = entity_df["__fold__"] == fold
                    fold_entity_ids = entity_df.loc[fold_mask, entity_col]
                    rich_aligned = rich_f.reindex(fold_entity_ids.values)
                    rich_aligned.index = fold_entity_ids.index
                    rich_parts.append(rich_aligned)

            if mean_parts:
                join_mean_feats[join_idx] = pd.concat(mean_parts).sort_index()
            if rich_parts:
                join_rich_feats[join_idx] = pd.concat(rich_parts).sort_index()

        all_features[task_name] = {
            "entity": entity_df,
            "join_mean": join_mean_feats,
            "join_rich": join_rich_feats,
        }

        n_mean = sum(1 for v in join_mean_feats.values() if not v.empty)
        n_rich = sum(1 for v in join_rich_feats.values() if not v.empty)
        logger.info(f"    Entity shape: {entity_df.shape}, "
                     f"mean_joins: {n_mean}/13, rich_joins: {n_rich}/13")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: JRN ESTIMATION
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Phase 3: JRN estimation...")

    jrn_results: dict[tuple, dict] = {}

    for task_name in trainable_tasks:
        logger.info(f"  JRN estimation for {task_name}...")
        meta = TASK_META[task_name]
        entity_feats = all_features[task_name]["entity"]
        entity_col = meta["entity_col"]

        train_mask = entity_feats["__fold__"] == 0
        val_mask = entity_feats["__fold__"] == 1
        y_train = entity_feats.loc[train_mask, "__label__"]
        y_val = entity_feats.loc[val_mask, "__label__"]

        drop_cols = ["__fold__", "__label__", entity_col]
        drop_cols = [c for c in drop_cols if c in entity_feats.columns]
        X_base_train = entity_feats.loc[train_mask].drop(columns=drop_cols, errors="ignore")
        X_base_val = entity_feats.loc[val_mask].drop(columns=drop_cols, errors="ignore")

        # Remove non-numeric columns
        X_base_train = X_base_train.select_dtypes(include=[np.number])
        X_base_val = X_base_val.select_dtypes(include=[np.number])

        logger.info(f"    Base features: train={X_base_train.shape}, val={X_base_val.shape}")

        # Baseline: entity features only
        base_metrics = []
        for seed in SEEDS:
            m = train_eval_gbm(X_base_train, y_train, X_base_val, y_val, meta["type"], seed)
            base_metrics.append(m)
        base_mean = float(np.mean(base_metrics))
        logger.info(f"    Baseline metric: {base_mean:.4f} (seeds: {[f'{m:.4f}' for m in base_metrics]})")

        # Per-join probes
        for join_idx in range(13):
            join_mean_feats = all_features[task_name]["join_mean"].get(join_idx)
            if join_mean_feats is None or join_mean_feats.empty:
                jrn_results[(task_name, join_idx)] = {
                    "jrn": 1.0, "category": "NA",
                    "base_metrics": base_metrics, "join_metrics": base_metrics,
                    "base_mean": base_mean, "join_mean": base_mean,
                }
                continue

            # Align join features with base features
            jf_train = join_mean_feats.reindex(X_base_train.index).fillna(0)
            jf_val = join_mean_feats.reindex(X_base_val.index).fillna(0)

            # Remove non-numeric
            jf_train = jf_train.select_dtypes(include=[np.number])
            jf_val = jf_val.select_dtypes(include=[np.number])

            X_with_train = pd.concat([X_base_train, jf_train], axis=1)
            X_with_val = pd.concat([X_base_val, jf_val], axis=1)

            join_metrics = []
            for seed in SEEDS:
                m = train_eval_gbm(X_with_train, y_train, X_with_val, y_val, meta["type"], seed)
                join_metrics.append(m)
            join_mean_val = float(np.mean(join_metrics))

            jrn = join_mean_val / max(base_mean, 1e-8)
            if jrn > JRN_HIGH:
                category = "HIGH"
            elif jrn < JRN_LOW:
                category = "LOW"
            else:
                category = "CRITICAL"

            jrn_results[(task_name, join_idx)] = {
                "jrn": jrn, "category": category,
                "base_metrics": base_metrics, "join_metrics": join_metrics,
                "base_mean": base_mean, "join_mean": join_mean_val,
            }
            join_label = f"{FK_JOINS[join_idx]['child']}({FK_JOINS[join_idx]['fk_col']})→{FK_JOINS[join_idx]['parent']}"
            logger.info(f"    Join {join_idx} [{join_label}]: "
                         f"JRN={jrn:.4f} [{category}] "
                         f"(base={base_mean:.4f}, with_join={join_mean_val:.4f})")

    # Print JRN matrix
    logger.info("\n=== JRN MATRIX ===")
    header = f"{'Join':<50} " + " ".join(f"{t.split('/')[-1]:>12}" for t in trainable_tasks)
    logger.info(header)
    for j in range(13):
        jlabel = f"{FK_JOINS[j]['child']}({FK_JOINS[j]['fk_col']})→{FK_JOINS[j]['parent']}"
        vals = []
        for t in trainable_tasks:
            r = jrn_results.get((t, j), {"jrn": 1.0, "category": "NA"})
            vals.append(f"{r['jrn']:>8.3f}[{r['category'][:1]}]")
        logger.info(f"  J{j:2d} {jlabel:<47} " + " ".join(vals))

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4: BUILD 5 CONFIGURATIONS AND EVALUATE
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Phase 4: Building 5 configurations and evaluating...")

    results_all: dict[str, dict] = {}
    oracle_info: dict[str, dict] = {}
    task_predictions: dict[str, dict] = {}  # task -> config -> predictions_on_val

    for task_name in trainable_tasks:
        logger.info(f"  Evaluating configs for {task_name}...")
        t_start = time.time()
        meta = TASK_META[task_name]
        entity_feats = all_features[task_name]["entity"]
        entity_col = meta["entity_col"]

        train_mask = entity_feats["__fold__"] == 0
        val_mask = entity_feats["__fold__"] == 1

        y_train = entity_feats.loc[train_mask, "__label__"]
        y_val = entity_feats.loc[val_mask, "__label__"]

        drop_cols = ["__fold__", "__label__", entity_col]
        drop_cols = [c for c in drop_cols if c in entity_feats.columns]
        X_base_train = entity_feats.loc[train_mask].drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        X_base_val = entity_feats.loc[val_mask].drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])

        # Precompute join feature columns per join for train and val
        join_mean_train: dict[int, pd.DataFrame] = {}
        join_mean_val: dict[int, pd.DataFrame] = {}
        join_rich_train: dict[int, pd.DataFrame] = {}
        join_rich_val: dict[int, pd.DataFrame] = {}
        join_mean_col_names: dict[int, list] = {}
        join_rich_col_names: dict[int, list] = {}

        for j in range(13):
            jm = all_features[task_name]["join_mean"].get(j)
            if jm is not None and not jm.empty:
                jm_n = jm.select_dtypes(include=[np.number])
                join_mean_train[j] = jm_n.reindex(X_base_train.index).fillna(0)
                join_mean_val[j] = jm_n.reindex(X_base_val.index).fillna(0)
                join_mean_col_names[j] = list(jm_n.columns)
            else:
                join_mean_train[j] = pd.DataFrame(index=X_base_train.index)
                join_mean_val[j] = pd.DataFrame(index=X_base_val.index)
                join_mean_col_names[j] = []

            jr = all_features[task_name]["join_rich"].get(j)
            if jr is not None and not jr.empty:
                jr_n = jr.select_dtypes(include=[np.number])
                join_rich_train[j] = jr_n.reindex(X_base_train.index).fillna(0)
                join_rich_val[j] = jr_n.reindex(X_base_val.index).fillna(0)
                join_rich_col_names[j] = list(jr_n.columns)
            else:
                join_rich_train[j] = pd.DataFrame(index=X_base_train.index)
                join_rich_val[j] = pd.DataFrame(index=X_base_val.index)
                join_rich_col_names[j] = []

        # ── Config (a): JRN-guided heterogeneous ──
        jrn_guided_joins = []
        for j in range(13):
            cat = jrn_results.get((task_name, j), {"category": "NA"})["category"]
            if cat == "HIGH":
                jrn_guided_joins.append((j, "mean"))
            elif cat == "CRITICAL":
                jrn_guided_joins.append((j, "rich"))

        X_jrn_train = X_base_train.copy()
        X_jrn_val = X_base_val.copy()
        for j, agg_type in jrn_guided_joins:
            if agg_type == "mean" and join_mean_col_names[j]:
                X_jrn_train = pd.concat([X_jrn_train, join_mean_train[j]], axis=1)
                X_jrn_val = pd.concat([X_jrn_val, join_mean_val[j]], axis=1)
            elif agg_type == "rich" and join_rich_col_names[j]:
                X_jrn_train = pd.concat([X_jrn_train, join_rich_train[j]], axis=1)
                X_jrn_val = pd.concat([X_jrn_val, join_rich_val[j]], axis=1)

        # ── Config (b): Uniform-all-mean ──
        X_uni_mean_train = X_base_train.copy()
        X_uni_mean_val = X_base_val.copy()
        for j in range(13):
            if join_mean_col_names[j]:
                X_uni_mean_train = pd.concat([X_uni_mean_train, join_mean_train[j]], axis=1)
                X_uni_mean_val = pd.concat([X_uni_mean_val, join_mean_val[j]], axis=1)

        # ── Config (c): Uniform-all-rich ──
        X_uni_rich_train = X_base_train.copy()
        X_uni_rich_val = X_base_val.copy()
        for j in range(13):
            if join_rich_col_names[j]:
                X_uni_rich_train = pd.concat([X_uni_rich_train, join_rich_train[j]], axis=1)
                X_uni_rich_val = pd.concat([X_uni_rich_val, join_rich_val[j]], axis=1)

        # ── Config (d): Top-K by JRN ──
        K = max(len(jrn_guided_joins), 1)
        sorted_joins = sorted(range(13),
                              key=lambda jj: jrn_results.get((task_name, jj), {"jrn": 1.0})["jrn"],
                              reverse=True)
        top_k_joins = sorted_joins[:K]
        X_topk_train = X_base_train.copy()
        X_topk_val = X_base_val.copy()
        for j in top_k_joins:
            if join_mean_col_names[j]:
                X_topk_train = pd.concat([X_topk_train, join_mean_train[j]], axis=1)
                X_topk_val = pd.concat([X_topk_val, join_mean_val[j]], axis=1)

        # ── Config (e): Oracle — exhaustive search over 2^13 subsets ──
        # Only search over joins that have features (non-NA)
        active_joins = [j for j in range(13) if join_mean_col_names[j]]
        n_active = len(active_joins)
        total_subsets = (2 ** n_active) - 1
        logger.info(f"    Running oracle search ({n_active} active joins, {total_subsets} subsets)...")
        t_oracle = time.time()

        # Pre-stack all mean join features for fast column selection
        all_mean_cols = list(X_base_train.columns)
        col_ranges: dict[int, tuple[int, int]] = {}
        stacked_train_parts = [X_base_train]
        stacked_val_parts = [X_base_val]
        cur = len(all_mean_cols)
        for j in active_joins:
            stacked_train_parts.append(join_mean_train[j])
            stacked_val_parts.append(join_mean_val[j])
            col_ranges[j] = (cur, cur + len(join_mean_col_names[j]))
            cur += len(join_mean_col_names[j])
            all_mean_cols.extend(join_mean_col_names[j])

        stacked_train = pd.concat(stacked_train_parts, axis=1).fillna(0)
        stacked_val = pd.concat(stacked_val_parts, axis=1).fillna(0)
        # Deduplicate columns before renaming
        if stacked_train.columns.duplicated().any():
            stacked_train = stacked_train.loc[:, ~stacked_train.columns.duplicated()]
            stacked_val = stacked_val.loc[:, ~stacked_val.columns.duplicated()]
        stacked_train.columns = range(len(stacked_train.columns))
        stacked_val.columns = range(len(stacked_val.columns))
        base_col_count = len(X_base_train.columns)
        base_cols = list(range(base_col_count))

        # Convert to numpy for speed
        stacked_train_np = stacked_train.values.astype(np.float64)
        stacked_val_np = stacked_val.values.astype(np.float64)
        np.nan_to_num(stacked_train_np, copy=False)
        np.nan_to_num(stacked_val_np, copy=False)
        y_train_np = y_train.values.astype(float)
        y_val_np = y_val.values.astype(float)

        best_val_score = -1
        best_subset = []

        import lightgbm as lgb
        eval_count = 0

        for mask_int in range(1, 2 ** n_active):
            subset = [active_joins[i] for i in range(n_active) if (mask_int >> i) & 1]
            # Get column indices
            cols = list(base_cols)
            for j in subset:
                if j in col_ranges:
                    start, end = col_ranges[j]
                    cols.extend(range(start, end))

            X_sub_train = stacked_train_np[:, cols]
            X_sub_val = stacked_val_np[:, cols]

            try:
                if meta["type"] == "clf":
                    model = lgb.LGBMClassifier(**ORACLE_PARAMS, random_state=42, n_jobs=NUM_CPUS)
                    model.fit(X_sub_train, y_train_np)
                    y_pred = model.predict_proba(X_sub_val)
                    if y_pred.shape[1] > 1:
                        y_pred = y_pred[:, 1]
                    else:
                        y_pred = y_pred[:, 0]
                    if len(np.unique(y_val_np)) < 2:
                        score = 0.5
                    else:
                        score = roc_auc_score(y_val_np, y_pred)
                else:
                    model = lgb.LGBMRegressor(**ORACLE_PARAMS, random_state=42, n_jobs=NUM_CPUS)
                    model.fit(X_sub_train, y_train_np)
                    y_pred = model.predict(X_sub_val)
                    mae = mean_absolute_error(y_val_np, y_pred)
                    score = 1.0 / max(mae, 1e-8)
            except Exception:
                score = -1

            if score > best_val_score:
                best_val_score = score
                best_subset = subset

            eval_count += 1
            if eval_count % 500 == 0:
                elapsed = time.time() - t_oracle
                logger.info(f"      Oracle: {eval_count}/{total_subsets} subsets evaluated "
                             f"({elapsed:.0f}s), best_score={best_val_score:.4f}")

        oracle_elapsed = time.time() - t_oracle
        logger.info(f"    Oracle search done in {oracle_elapsed:.0f}s. Best subset: {best_subset}")

        # Build oracle features
        X_oracle_train = X_base_train.copy()
        X_oracle_val = X_base_val.copy()
        for j in best_subset:
            if join_mean_col_names[j]:
                X_oracle_train = pd.concat([X_oracle_train, join_mean_train[j]], axis=1)
                X_oracle_val = pd.concat([X_oracle_val, join_mean_val[j]], axis=1)

        # ── Evaluate all 5 configs ──
        configs = [
            ("jrn_guided",   X_jrn_train,      X_jrn_val),
            ("uniform_mean", X_uni_mean_train,  X_uni_mean_val),
            ("uniform_rich", X_uni_rich_train,  X_uni_rich_val),
            ("top_k",        X_topk_train,      X_topk_val),
            ("oracle",       X_oracle_train,    X_oracle_val),
        ]

        task_results = {}
        task_preds = {}

        for config_name, X_tr, X_va in configs:
            val_scores = []
            preds_accum = []
            for seed in SEEDS:
                m = train_eval_gbm(X_tr, y_train, X_va, y_val, meta["type"], seed, FINAL_PARAMS)
                val_scores.append(m)
                p = train_predict_gbm(X_tr, y_train, X_va, meta["type"], seed, FINAL_PARAMS)
                preds_accum.append(p)

            # Average predictions across seeds
            avg_preds = np.mean(preds_accum, axis=0)

            task_results[config_name] = {
                "val_mean": float(np.mean(val_scores)),
                "val_std": float(np.std(val_scores)),
            }
            task_preds[config_name] = avg_preds
            logger.info(f"    {config_name}: val={np.mean(val_scores):.4f}±{np.std(val_scores):.4f}")

        results_all[task_name] = task_results
        oracle_info[task_name] = {
            "best_subset": best_subset,
            "best_subset_labels": [
                f"{FK_JOINS[j]['child']}→{FK_JOINS[j]['parent']}" for j in best_subset
            ],
            "oracle_val": task_results["oracle"]["val_mean"],
            "jrn_guided_val": task_results["jrn_guided"]["val_mean"],
        }
        task_predictions[task_name] = task_preds

        elapsed = time.time() - t_start
        logger.info(f"    Task done in {elapsed:.0f}s")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 5: ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Phase 5: Analysis...")

    # Win rates
    win_vs_uniform_mean = 0
    win_vs_uniform_rich = 0
    win_vs_topk = 0
    n_tasks = len(trainable_tasks)

    for t in trainable_tasks:
        jrn_val = results_all[t]["jrn_guided"]["val_mean"]
        if jrn_val > results_all[t]["uniform_mean"]["val_mean"]:
            win_vs_uniform_mean += 1
        if jrn_val > results_all[t]["uniform_rich"]["val_mean"]:
            win_vs_uniform_rich += 1
        if jrn_val > results_all[t]["top_k"]["val_mean"]:
            win_vs_topk += 1

    win_rates = {
        "jrn_guided_vs_uniform_mean": win_vs_uniform_mean / max(n_tasks, 1),
        "jrn_guided_vs_uniform_rich": win_vs_uniform_rich / max(n_tasks, 1),
        "jrn_guided_vs_top_k": win_vs_topk / max(n_tasks, 1),
    }
    logger.info(f"Win rates: {win_rates}")

    # Oracle gap analysis
    for t in trainable_tasks:
        oracle_val = results_all[t]["oracle"]["val_mean"]
        jrn_val = results_all[t]["jrn_guided"]["val_mean"]
        if oracle_val > 0:
            gap_pct = abs(oracle_val - jrn_val) / oracle_val * 100
        else:
            gap_pct = 0
        oracle_info[t]["gap_pct"] = gap_pct
        logger.info(f"  {t}: oracle={oracle_val:.4f}, jrn_guided={jrn_val:.4f}, gap={gap_pct:.1f}%")

    # JRN matrix for output
    jrn_matrix_out = {}
    join_classifications_out = {}
    for t in trainable_tasks:
        jrn_matrix_out[t] = {}
        join_classifications_out[t] = {}
        for j in range(13):
            r = jrn_results.get((t, j), {"jrn": 1.0, "category": "NA", "base_mean": 0, "join_mean": 0})
            join_label = f"{FK_JOINS[j]['child']}({FK_JOINS[j]['fk_col']})→{FK_JOINS[j]['parent']}"
            jrn_matrix_out[t][str(j)] = {
                "jrn": round(r["jrn"], 4),
                "category": r["category"],
                "base_metric": round(r.get("base_mean", 0), 4),
                "join_metric": round(r.get("join_mean", 0), 4),
                "join_label": join_label,
            }
            # Determine agg strategy used
            agg_strat = "excluded"
            if r["category"] == "HIGH":
                agg_strat = "mean"
            elif r["category"] == "CRITICAL":
                agg_strat = "rich"
            join_classifications_out[t][str(j)] = {
                "category": r["category"],
                "jrn": round(r["jrn"], 4),
                "agg_strategy": agg_strat,
                "join_label": join_label,
            }

    # Hypothesis support
    all_jrns = [jrn_results.get((t, j), {"jrn": 1.0})["jrn"]
                for t in trainable_tasks for j in range(13)
                if jrn_results.get((t, j), {"category": "NA"})["category"] != "NA"]
    jrn_range = max(all_jrns) - min(all_jrns) if all_jrns else 0
    all_cats = [jrn_results.get((t, j), {"category": "NA"})["category"]
                for t in trainable_tasks for j in range(13)]
    has_critical = "CRITICAL" in all_cats
    gaps = [oracle_info[t]["gap_pct"] for t in trainable_tasks]
    avg_gap = np.mean(gaps) if gaps else 100

    hypothesis_support = {
        "jrn_differentiates_joins": jrn_range > 0.3,
        "jrn_guided_beats_uniform": win_rates["jrn_guided_vs_uniform_mean"] > 0.5,
        "jrn_guided_close_to_oracle": avg_gap < 5.0,
        "critical_joins_exist": has_critical,
    }
    logger.info(f"Hypothesis support: {hypothesis_support}")

    # Domain analysis
    domain_lines = []
    for t in trainable_tasks:
        highs = [j for j in range(13) if jrn_results.get((t, j), {"category": "NA"})["category"] == "HIGH"]
        criticals = [j for j in range(13) if jrn_results.get((t, j), {"category": "NA"})["category"] == "CRITICAL"]
        lows = [j for j in range(13) if jrn_results.get((t, j), {"category": "NA"})["category"] == "LOW"]
        nas = [j for j in range(13) if jrn_results.get((t, j), {"category": "NA"})["category"] == "NA"]
        domain_lines.append(f"{t}: HIGH={highs}, CRITICAL={criticals}, LOW={lows}, NA={nas}")

    domain_lines.append("")
    domain_lines.append("Key observations:")
    # Check if standings→drivers (j4) and results→drivers (j11) help
    for jj, jlabel in [(4, "standings→drivers"), (8, "qualifying→drivers"), (11, "results→drivers")]:
        jrns = [jrn_results.get((t, jj), {"jrn": 1.0})["jrn"] for t in trainable_tasks if t.startswith("rel-f1/driver")]
        if jrns:
            domain_lines.append(f"  {jlabel}: avg JRN={np.mean(jrns):.3f} across driver tasks")
    domain_analysis = "\n".join(domain_lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 6: OUTPUT (exp_gen_sol_out.json schema)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Phase 6: Generating output...")

    # Build summary
    summary_parts = [
        f"JRN estimation across {len(trainable_tasks)} tasks and 13 FK joins on rel-f1.",
        f"JRN range: {jrn_range:.3f}, critical joins exist: {has_critical}.",
        f"Win rate vs uniform-mean: {win_rates['jrn_guided_vs_uniform_mean']:.0%}, "
        f"vs uniform-rich: {win_rates['jrn_guided_vs_uniform_rich']:.0%}, "
        f"vs top-k: {win_rates['jrn_guided_vs_top_k']:.0%}.",
        f"Average gap to oracle: {avg_gap:.1f}%.",
    ]
    summary = " ".join(summary_parts)

    # Build examples: one per val task sample across all trainable tasks
    output_examples = []
    for task_name in trainable_tasks:
        meta = TASK_META[task_name]
        entity_feats = all_features[task_name]["entity"]
        val_mask = entity_feats["__fold__"] == 1
        val_indices = entity_feats[val_mask].index.tolist()

        raw_samples = task_raw_samples[task_name]
        preds = task_predictions.get(task_name, {})

        # Map val indices to raw samples
        # Raw samples are in original order; val_indices are DataFrame indices
        # We need to align them properly
        val_raw = []
        for i, s in enumerate(raw_samples):
            if s["fold"] == 1 or (
                task_name in ("rel-f1/qualifying-position", "rel-f1/results-position")
                and i in val_indices
            ):
                val_raw.append((i, s))

        # For tasks where we split val into train/val, we need to use the actual val mask
        val_df_indices = list(val_indices)

        for enum_idx, (raw_idx, sample) in enumerate(val_raw):
            if enum_idx >= len(val_df_indices):
                break

            example = {
                "input": sample["input_str"],
                "output": sample["output_str"],
                "metadata_task_name": task_name,
                "metadata_task_type": meta["type"],
                "metadata_entity_col": meta["entity_col"],
                "metadata_fold": 1,
            }

            # Add predictions from each config
            for config_name in ["jrn_guided", "uniform_mean", "uniform_rich", "top_k", "oracle"]:
                pred_arr = preds.get(config_name)
                if pred_arr is not None and enum_idx < len(pred_arr):
                    example[f"predict_{config_name}"] = str(round(float(pred_arr[enum_idx]), 6))

            output_examples.append(example)

    # Full output structure
    output = {
        "metadata": {
            "title": "JRN-Guided Heterogeneous Architecture on rel-f1",
            "summary": summary,
            "method": "JRN-guided heterogeneous architecture with LightGBM probes",
            "jrn_matrix": jrn_matrix_out,
            "configuration_results": {t: results_all[t] for t in trainable_tasks},
            "win_rates": win_rates,
            "oracle_analysis": oracle_info,
            "join_classifications": join_classifications_out,
            "domain_analysis": domain_analysis,
            "hypothesis_support": hypothesis_support,
            "tasks_evaluated": trainable_tasks,
            "n_fk_joins": 13,
            "jrn_thresholds": {"HIGH": JRN_HIGH, "LOW": JRN_LOW},
            "seeds": SEEDS,
            "probe_params": PROBE_PARAMS,
            "final_params": FINAL_PARAMS,
            "oracle_params": ORACLE_PARAMS,
        },
        "datasets": [
            {
                "dataset": "rel-f1",
                "examples": output_examples,
            }
        ],
    }

    # Write output
    out_path = SCRIPT_DIR / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output written to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    total_time = time.time() - t0
    logger.info(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
