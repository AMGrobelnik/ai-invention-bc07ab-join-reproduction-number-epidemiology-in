#!/usr/bin/env python3
"""JRN Multiplicative Compounding on rel-f1 + Entropy vs GBM Probe Architecture Comparison.

Self-contained experiment on the rel-f1 (Formula 1) dataset testing two core JRN hypotheses:
  PART A: Multiplicative compounding of individual JRN values along multi-hop chains.
  PART B: Entropy-based JRN proxy vs GBM-probe, and architecture comparison (4 configs).

Output: method_out.json with all results in exp_gen_sol_out schema.
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
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0

# Set memory limit to 80% of available
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DATA_FILE = DATA_DIR / "full_data_out.json"

SEEDS = [42, 123, 456]
DRIVER_TASKS = ["rel-f1/driver-dnf", "rel-f1/driver-top3", "rel-f1/driver-position"]
ALL_TASKS = DRIVER_TASKS  # Focus on driver tasks as specified

# FK join definitions (from the data)
FK_DEFS = [
    {"id": "FK0",  "source": "races",                  "fk_col": "circuitId",      "target": "circuits",     "pk_col": "circuitId"},
    {"id": "FK1",  "source": "constructor_standings",   "fk_col": "raceId",         "target": "races",        "pk_col": "raceId"},
    {"id": "FK2",  "source": "constructor_standings",   "fk_col": "constructorId",  "target": "constructors", "pk_col": "constructorId"},
    {"id": "FK3",  "source": "standings",               "fk_col": "raceId",         "target": "races",        "pk_col": "raceId"},
    {"id": "FK4",  "source": "standings",               "fk_col": "driverId",       "target": "drivers",      "pk_col": "driverId"},
    {"id": "FK5",  "source": "constructor_results",     "fk_col": "raceId",         "target": "races",        "pk_col": "raceId"},
    {"id": "FK6",  "source": "constructor_results",     "fk_col": "constructorId",  "target": "constructors", "pk_col": "constructorId"},
    {"id": "FK7",  "source": "qualifying",              "fk_col": "raceId",         "target": "races",        "pk_col": "raceId"},
    {"id": "FK8",  "source": "qualifying",              "fk_col": "driverId",       "target": "drivers",      "pk_col": "driverId"},
    {"id": "FK9",  "source": "qualifying",              "fk_col": "constructorId",  "target": "constructors", "pk_col": "constructorId"},
    {"id": "FK10", "source": "results",                 "fk_col": "raceId",         "target": "races",        "pk_col": "raceId"},
    {"id": "FK11", "source": "results",                 "fk_col": "driverId",       "target": "drivers",      "pk_col": "driverId"},
    {"id": "FK12", "source": "results",                 "fk_col": "constructorId",  "target": "constructors", "pk_col": "constructorId"},
]

# Chain definitions
CHAINS = [
    # Linear 2-hop chains (through races to circuits)
    {"id": "chain_results_races_circuits",          "type": "linear", "joins": ["FK10", "FK0"], "desc": "results->races->circuits"},
    {"id": "chain_standings_races_circuits",         "type": "linear", "joins": ["FK3", "FK0"],  "desc": "standings->races->circuits"},
    {"id": "chain_qualifying_races_circuits",        "type": "linear", "joins": ["FK7", "FK0"],  "desc": "qualifying->races->circuits"},
    {"id": "chain_conresults_races_circuits",        "type": "linear", "joins": ["FK5", "FK0"],  "desc": "con_results->races->circuits"},
    {"id": "chain_constandings_races_circuits",      "type": "linear", "joins": ["FK1", "FK0"],  "desc": "con_standings->races->circuits"},
    # Convergent chains (two sources into same target)
    {"id": "chain_results_standings_to_drivers",     "type": "convergent", "joins": ["FK11", "FK4"],  "desc": "results+standings->drivers"},
    {"id": "chain_results_qualifying_to_drivers",    "type": "convergent", "joins": ["FK11", "FK8"],  "desc": "results+qualifying->drivers"},
    {"id": "chain_results_conresults_to_constructors","type": "convergent","joins": ["FK12", "FK6"],  "desc": "results+con_results->constructors"},
    {"id": "chain_constandings_conresults_to_constructors","type": "convergent","joins": ["FK2", "FK6"],"desc": "con_standings+con_results->constructors"},
    # 3-hop linear (driver-centric)
    {"id": "chain_drivers_results_races_circuits",   "type": "linear_3hop", "joins": ["FK11", "FK10", "FK0"], "desc": "drivers<-results->races->circuits"},
]


# ══════════════════════════════════════════════════════════════════════════
# STEP 0: LOAD AND RECONSTRUCT RELATIONAL DATABASE
# ══════════════════════════════════════════════════════════════════════════

def load_data() -> Tuple[Dict[str, pd.DataFrame], List[Dict], Dict[str, Dict]]:
    """Load full_data_out.json and reconstruct tables, FK metadata, and task samples."""
    logger.info(f"Loading data from {DATA_FILE}")
    t0 = time.time()
    raw = json.loads(DATA_FILE.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples in {time.time()-t0:.1f}s")

    # Separate by type
    table_rows: Dict[str, List[Dict]] = defaultdict(list)
    table_pk_cols: Dict[str, str] = {}
    fk_metadata: List[Dict] = []
    task_samples: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    task_meta: Dict[str, Dict] = {}

    for ex in examples:
        rt = ex["metadata_row_type"]
        if rt == "table_row":
            tname = ex["metadata_table_name"]
            pk_col = ex["metadata_primary_key_col"]
            pk_val = ex["metadata_primary_key_value"]
            table_pk_cols[tname] = pk_col
            features = json.loads(ex["input"])
            # Parse pk value
            try:
                pk_v = int(pk_val)
            except (ValueError, TypeError):
                pk_v = pk_val
            features[pk_col] = pk_v
            table_rows[tname].append(features)
        elif rt == "fk_join_metadata":
            inp = json.loads(ex["input"])
            outp = json.loads(ex["output"])
            fk_metadata.append({**inp, **outp})
        elif rt == "task_sample":
            tname = ex["metadata_task_name"]
            fold = ex.get("metadata_fold_name", "train")
            features = json.loads(ex["input"])
            label = ex["output"]
            features["__label__"] = label
            features["__fold__"] = fold
            task_samples[tname][fold].append(features)
            if tname not in task_meta:
                task_meta[tname] = {
                    "task_type": ex.get("metadata_task_type", "regression"),
                    "entity_col": ex.get("metadata_entity_col"),
                    "entity_table": ex.get("metadata_entity_table"),
                    "target_col": ex.get("metadata_target_col"),
                }

    # Convert table rows to DataFrames
    tables: Dict[str, pd.DataFrame] = {}
    for tname, rows in table_rows.items():
        df = pd.DataFrame(rows)
        pk_col = table_pk_cols[tname]
        # Ensure PK is numeric if possible
        if pk_col in df.columns:
            df[pk_col] = pd.to_numeric(df[pk_col], errors="coerce").fillna(0).astype(int)
        tables[tname] = df

    del raw, examples, table_rows
    gc.collect()

    logger.info(f"Reconstructed {len(tables)} tables:")
    for tname, df in tables.items():
        logger.info(f"  {tname}: {df.shape[0]} rows x {df.shape[1]} cols")

    logger.info(f"FK metadata: {len(fk_metadata)} joins")
    logger.info(f"Tasks: {list(task_meta.keys())}")

    return tables, fk_metadata, task_samples, task_meta, table_pk_cols


def encode_table(df: pd.DataFrame, pk_col: str) -> pd.DataFrame:
    """Encode a table's features: label-encode strings, parse dates, keep numerics."""
    df = df.copy()
    label_encoders = {}
    drop_cols = []

    for col in df.columns:
        if col == pk_col:
            continue
        # Try to parse as datetime
        if df[col].dtype == object:
            # Check if it looks like a date
            sample = df[col].dropna().head(5)
            is_date = False
            for v in sample:
                if isinstance(v, str) and ("T" in v or "-" in v) and len(v) > 8:
                    try:
                        pd.to_datetime(v)
                        is_date = True
                        break
                    except (ValueError, TypeError):
                        pass

            if is_date:
                try:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
                    df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
                    df[f"{col}_day"] = dt.dt.day.fillna(0).astype(int)
                    drop_cols.append(col)
                    continue
                except Exception:
                    pass

            # Label encode strings
            try:
                le = LabelEncoder()
                mask = df[col].notna()
                if mask.sum() > 0:
                    vals = df.loc[mask, col].astype(str)
                    df.loc[mask, col] = le.fit_transform(vals)
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                    label_encoders[col] = le
                else:
                    df[col] = 0
            except Exception:
                drop_cols.append(col)
        else:
            # Try numeric conversion
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Ensure all columns are numeric
    for col in df.columns:
        if col == pk_col:
            continue
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            except Exception:
                df = df.drop(columns=[col], errors="ignore")

    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: FEATURE AGGREGATION HELPER
# ══════════════════════════════════════════════════════════════════════════

def aggregate_child_features(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    parent_pk: str,
    child_fk: str,
    agg_funcs: List[str] = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Aggregate child numeric features to parent via FK relationship."""
    if agg_funcs is None:
        agg_funcs = ["mean"]

    # Get numeric columns from child (exclude FK/PK)
    numeric_cols = child_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != child_fk and c != parent_pk]

    if not numeric_cols:
        return parent_df.copy()

    # Group by FK, aggregate
    try:
        grouped = child_df.groupby(child_fk)[numeric_cols].agg(agg_funcs)
        # Flatten MultiIndex columns
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"agg_{prefix}{col}_{func}" for col, func in grouped.columns]
        else:
            grouped.columns = [f"agg_{prefix}{col}" for col in grouped.columns]

        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={child_fk: parent_pk})

        # Merge with parent
        merged = parent_df.merge(grouped, on=parent_pk, how="left")
        # Fill NaN
        agg_cols = [c for c in merged.columns if c.startswith("agg_")]
        merged[agg_cols] = merged[agg_cols].fillna(0)
        return merged
    except Exception as e:
        logger.warning(f"Aggregation failed: {e}")
        return parent_df.copy()


def get_agg_features_for_entity(
    entity_df: pd.DataFrame,
    entity_pk: str,
    source_df: pd.DataFrame,
    source_fk: str,
    agg_funcs: List[str] = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Get aggregated features from source table for entity table."""
    if agg_funcs is None:
        agg_funcs = ["mean"]

    source_encoded = encode_table(source_df, source_fk)
    return aggregate_child_features(entity_df, source_encoded, entity_pk, source_fk, agg_funcs, prefix)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: PREPARE TASK-LEVEL DATASETS
# ══════════════════════════════════════════════════════════════════════════

def build_task_dataset(
    task_name: str,
    task_samples: Dict,
    task_meta: Dict,
    tables: Dict[str, pd.DataFrame],
    table_pk_cols: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:
    """Build train/val feature matrices and labels for a task."""
    meta = task_meta[task_name]
    entity_col = meta["entity_col"]
    entity_table = meta["entity_table"]
    task_type = meta["task_type"]

    # Gather samples
    train_samples = task_samples[task_name].get("train", [])
    val_samples = task_samples[task_name].get("val", [])
    test_samples = task_samples[task_name].get("test", [])

    # Use val for evaluation, test as backup if val is small
    if len(val_samples) < 50 and len(test_samples) > 50:
        eval_samples = test_samples
    else:
        eval_samples = val_samples

    def samples_to_df(samples):
        rows = []
        labels = []
        for s in samples:
            label = s["__label__"]
            if label == "masked":
                continue
            try:
                if task_type == "binary_classification":
                    label = int(float(label))
                else:
                    label = float(label)
            except (ValueError, TypeError):
                continue
            row = {k: v for k, v in s.items() if k not in ("__label__", "__fold__")}
            rows.append(row)
            labels.append(label)
        if not rows:
            return pd.DataFrame(), pd.Series(dtype=float)
        df = pd.DataFrame(rows)
        # Parse entity_col to int
        if entity_col and entity_col in df.columns:
            df[entity_col] = pd.to_numeric(df[entity_col], errors="coerce").fillna(0).astype(int)
        # Parse date columns
        for col in df.columns:
            if col == entity_col:
                continue
            sample = df[col].dropna().head(3)
            is_date = False
            for v in sample:
                if isinstance(v, str) and "T" in v:
                    is_date = True
                    break
            if is_date:
                try:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
                    df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
                    df = df.drop(columns=[col])
                except Exception:
                    df = df.drop(columns=[col], errors="ignore")
        return df, pd.Series(labels)

    X_train, y_train = samples_to_df(train_samples)
    X_val, y_val = samples_to_df(eval_samples)

    if X_train.empty or X_val.empty:
        logger.warning(f"Task {task_name}: empty train or val set")
        return X_train, y_train, X_val, y_val, task_type

    # Merge entity features from entity table
    if entity_table and entity_table in tables and entity_col:
        entity_df = tables[entity_table].copy()
        pk_col = table_pk_cols.get(entity_table, entity_col)
        entity_encoded = encode_table(entity_df, pk_col)

        # Merge entity features into task samples
        entity_feats = [c for c in entity_encoded.columns if c != pk_col]
        if entity_feats and entity_col in X_train.columns:
            X_train = X_train.merge(
                entity_encoded, left_on=entity_col, right_on=pk_col, how="left"
            )
            X_val = X_val.merge(
                entity_encoded, left_on=entity_col, right_on=pk_col, how="left"
            )
            # Drop duplicate PK columns
            if pk_col != entity_col and pk_col in X_train.columns:
                X_train = X_train.drop(columns=[pk_col], errors="ignore")
                X_val = X_val.drop(columns=[pk_col], errors="ignore")

    # Drop non-numeric columns
    for df in [X_train, X_val]:
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                except Exception:
                    pass

    # Align columns
    common_cols = sorted(set(X_train.columns) & set(X_val.columns))
    numeric_cols = []
    for c in common_cols:
        if X_train[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            numeric_cols.append(c)
        else:
            try:
                X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0)
                X_val[c] = pd.to_numeric(X_val[c], errors="coerce").fillna(0)
                numeric_cols.append(c)
            except Exception:
                pass

    X_train = X_train[numeric_cols].fillna(0)
    X_val = X_val[numeric_cols].fillna(0)

    return X_train, y_train, X_val, y_val, task_type


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: COMPUTE JRN (GBM PROBE)
# ══════════════════════════════════════════════════════════════════════════

def train_eval_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: str,
    seed: int,
    n_estimators: int = 200,
    max_depth: int = 6,
) -> float:
    """Train LightGBM and return evaluation score. Higher is better."""
    if X_train.shape[0] < 5 or X_val.shape[0] < 5:
        return float("nan")
    if X_train.shape[1] == 0:
        return float("nan")

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "n_jobs": 1,
        "random_state": seed,
    }

    try:
        if task_type == "binary_classification":
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = model.predict_proba(X_val)
            if y_pred.ndim == 2:
                y_pred = y_pred[:, 1]
            # AUROC
            if len(np.unique(y_val)) < 2:
                return float("nan")
            score = roc_auc_score(y_val, y_pred)
        else:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = model.predict(X_val)
            # 1 / (1 + MAE) so higher is better
            mae = mean_absolute_error(y_val, y_pred)
            score = 1.0 / (1.0 + mae)
        return score
    except Exception as e:
        logger.warning(f"LightGBM failed: {e}")
        return float("nan")


def _make_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns in a DataFrame are numeric, dropping those that can't convert."""
    result = df.copy()
    to_drop = []
    for c in result.columns:
        if result[c].dtype not in [np.float64, np.float32, np.int64, np.int32, float, int]:
            try:
                result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0)
            except Exception:
                to_drop.append(c)
    if to_drop:
        result = result.drop(columns=to_drop)
    return result.fillna(0)


def _aggregate_to_entity(
    source_df: pd.DataFrame,
    entity_col: str,
    target_df: Optional[pd.DataFrame],
    fk_col: str,
    pk_col: str,
    agg_prefix: str,
    agg_funcs: List[str],
    tables: Dict[str, pd.DataFrame],
    table_pk_cols: Dict[str, str],
    source_table_name: str = "",
) -> Optional[pd.DataFrame]:
    """Compute aggregated features from a join, returning DataFrame keyed by entity_col.

    Three cases:
    1. DIRECT: source has entity_col AND target is entity table
       -> aggregate SOURCE features by entity_col
    2. INDIRECT: source has entity_col, target != entity table
       -> merge source with target via FK, aggregate TARGET features by entity_col
    3. BRIDGE: source does NOT have entity_col
       -> find a bridge table with entity_col and source's PK, route through it
    """
    NaN_RESULT = None

    # Case 1 & 2: source has entity_col
    if entity_col in source_df.columns:
        if target_df is not None and fk_col != entity_col:
            # CASE 2 (INDIRECT): source->target, aggregate TARGET features by entity_col
            # E.g. FK3: standings(has driverId).raceId -> races
            # -> merge standings with races via raceId, aggregate race features by driverId
            target_enc = encode_table(target_df, pk_col)
            target_numeric = target_enc.select_dtypes(include=[np.number]).columns.tolist()
            target_numeric = [c for c in target_numeric if c != pk_col and c != entity_col]

            if not target_numeric:
                return NaN_RESULT

            # Merge source with target to bring in target features
            merge_right = target_enc[[pk_col] + target_numeric]
            # source has fk_col, target has pk_col
            source_with_target = source_df[[entity_col, fk_col]].drop_duplicates().merge(
                merge_right, left_on=fk_col, right_on=pk_col, how="left"
            )
            if pk_col != fk_col and pk_col in source_with_target.columns:
                source_with_target = source_with_target.drop(columns=[pk_col], errors="ignore")

            # Aggregate target features by entity
            agg_cols = [c for c in target_numeric if c in source_with_target.columns]
            if not agg_cols:
                return NaN_RESULT

            grouped = source_with_target.groupby(entity_col)[agg_cols].agg(agg_funcs)
        else:
            # CASE 1 (DIRECT): fk_col == entity_col, target IS entity table
            # E.g. FK4: standings.driverId -> drivers, FK11: results.driverId -> drivers
            # -> aggregate SOURCE features by entity_col
            source_enc = encode_table(source_df, entity_col)
            numeric_cols = source_enc.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != entity_col]

            if not numeric_cols:
                return NaN_RESULT

            grouped = source_enc.groupby(entity_col)[numeric_cols].agg(agg_funcs)

        # Flatten MultiIndex columns
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"agg_{agg_prefix}{col}_{func}" for col, func in grouped.columns]
        else:
            grouped.columns = [f"agg_{agg_prefix}{col}" for col in grouped.columns]
        return grouped.reset_index()

    # Case 3: BRIDGE - source doesn't have entity_col
    # E.g. FK0: races.circuitId -> circuits (races doesn't have driverId)
    # Need a bridge table that has both entity_col and source's PK
    source_pk = table_pk_cols.get(source_table_name, None)
    if source_pk is None:
        for src_name, src_pk in table_pk_cols.items():
            if src_pk in source_df.columns:
                source_pk = src_pk
                break

    if source_pk is None:
        return NaN_RESULT

    for bridge_name, bridge_df in tables.items():
        if entity_col in bridge_df.columns and source_pk in bridge_df.columns:
            try:
                target_enc = encode_table(target_df, pk_col) if target_df is not None else None
                if target_enc is None:
                    continue
                target_numeric = target_enc.select_dtypes(include=[np.number]).columns.tolist()
                target_numeric = [c for c in target_numeric if c != pk_col and c != entity_col]
                if not target_numeric:
                    continue

                # source -> target via FK
                source_with_target = source_df.merge(
                    target_enc[[pk_col] + target_numeric],
                    left_on=fk_col, right_on=pk_col, how="left"
                )
                # bridge -> source via source_pk, bringing entity_col
                bridge_link = bridge_df[[entity_col, source_pk]].drop_duplicates()
                merged = source_with_target.merge(bridge_link, on=source_pk, how="inner")

                agg_cols = [c for c in target_numeric if c in merged.columns]
                if not agg_cols or merged.empty:
                    continue

                grouped = merged.groupby(entity_col)[agg_cols].agg(agg_funcs)
                if isinstance(grouped.columns, pd.MultiIndex):
                    grouped.columns = [f"agg_{agg_prefix}{col}_{func}" for col, func in grouped.columns]
                else:
                    grouped.columns = [f"agg_{agg_prefix}{col}" for col in grouped.columns]
                return grouped.reset_index()
            except Exception as e:
                logger.debug(f"Bridge {bridge_name} failed for {agg_prefix}: {e}")
                continue

    return NaN_RESULT


def compute_jrn_for_task_join(
    task_name: str,
    join_id: str,
    X_train_base: pd.DataFrame,
    y_train: pd.Series,
    X_val_base: pd.DataFrame,
    y_val: pd.Series,
    task_type: str,
    tables: Dict[str, pd.DataFrame],
    table_pk_cols: Dict[str, str],
    entity_col: str,
    agg_funcs: List[str] = None,
) -> Dict:
    """Compute entity-centric JRN for a single join on a single task.

    Properly differentiates:
    - Direct joins (FK4,8,11 for drivers): aggregate SOURCE features by entity
    - Indirect joins (FK3,7,10 etc.): aggregate TARGET features via source by entity
    - Bridge joins (FK0): route through intermediate tables
    """
    if agg_funcs is None:
        agg_funcs = ["mean"]
    NaN_RESULT = {"jrn": float("nan"), "baseline": float("nan"), "augmented": float("nan")}

    fk_def = next((f for f in FK_DEFS if f["id"] == join_id), None)
    if fk_def is None:
        return NaN_RESULT

    source_table = fk_def["source"]
    fk_col = fk_def["fk_col"]
    target_table = fk_def["target"]
    pk_col = fk_def["pk_col"]
    agg_prefix = f"{join_id}_{source_table}_{target_table}_"

    source_df = tables.get(source_table)
    target_df = tables.get(target_table)
    if source_df is None:
        return NaN_RESULT

    # Get aggregated features for this join
    grouped = _aggregate_to_entity(
        source_df=source_df,
        entity_col=entity_col,
        target_df=target_df,
        fk_col=fk_col,
        pk_col=pk_col,
        agg_prefix=agg_prefix,
        agg_funcs=agg_funcs,
        tables=tables,
        table_pk_cols=table_pk_cols,
        source_table_name=source_table,
    )

    if grouped is None or grouped.empty:
        return NaN_RESULT

    # Merge aggregated features into train/val sets
    if entity_col not in X_train_base.columns:
        return NaN_RESULT

    X_train_aug = X_train_base.merge(grouped, on=entity_col, how="left")
    X_val_aug = X_val_base.merge(grouped, on=entity_col, how="left")

    # Fill NaN and ensure numeric
    X_train_aug = _make_numeric_df(X_train_aug)
    X_val_aug = _make_numeric_df(X_val_aug)

    agg_cols = [c for c in X_train_aug.columns if c.startswith("agg_")]
    if not agg_cols:
        return NaN_RESULT

    # Align columns
    common_aug = sorted(set(X_train_aug.columns) & set(X_val_aug.columns))
    X_train_aug = X_train_aug[common_aug]
    X_val_aug = X_val_aug[common_aug]

    # Baseline columns
    base_cols = sorted(set(X_train_base.columns) & set(X_val_base.columns))
    X_train_b = _make_numeric_df(X_train_base[base_cols])
    X_val_b = _make_numeric_df(X_val_base[base_cols])

    # Compute scores across seeds
    baseline_scores = []
    augmented_scores = []
    for seed in SEEDS:
        sb = train_eval_lgbm(X_train_b, y_train, X_val_b, y_val, task_type, seed)
        sa = train_eval_lgbm(X_train_aug, y_train, X_val_aug, y_val, task_type, seed)
        if not math.isnan(sb) and not math.isnan(sa):
            baseline_scores.append(sb)
            augmented_scores.append(sa)

    if not baseline_scores:
        return NaN_RESULT

    mean_base = float(np.mean(baseline_scores))
    mean_aug = float(np.mean(augmented_scores))
    jrn = mean_aug / mean_base if mean_base > 1e-10 else float("nan")

    return {
        "jrn": jrn,
        "baseline": mean_base,
        "augmented": mean_aug,
        "baseline_scores": baseline_scores,
        "augmented_scores": augmented_scores,
    }


def compute_local_jrn(
    join_id: str,
    tables: Dict[str, pd.DataFrame],
    table_pk_cols: Dict[str, str],
) -> Dict:
    """Compute LOCAL JRN for any join: predict a proxy column of the target table
    with and without aggregated source features. Works even for joins unreachable
    from driver entity table."""
    NaN_RESULT = {"local_jrn": float("nan"), "proxy_col": None}

    fk_def = next((f for f in FK_DEFS if f["id"] == join_id), None)
    if fk_def is None:
        return NaN_RESULT

    source_table = fk_def["source"]
    fk_col = fk_def["fk_col"]
    target_table = fk_def["target"]
    pk_col = fk_def["pk_col"]

    target_df = tables.get(target_table)
    source_df = tables.get(source_table)
    if target_df is None or source_df is None:
        return NaN_RESULT

    target_enc = encode_table(target_df, pk_col)
    numeric_cols = target_enc.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != pk_col]

    if len(numeric_cols) < 2:
        return NaN_RESULT

    # Pick proxy target: column with highest variance
    variances = target_enc[numeric_cols].var()
    proxy_col = variances.idxmax()
    feature_cols = [c for c in numeric_cols if c != proxy_col]

    if not feature_cols:
        return NaN_RESULT

    y = target_enc[proxy_col]
    X_base = target_enc[feature_cols].fillna(0)

    # Augmented: add aggregated source features
    source_enc = encode_table(source_df, fk_col)
    src_numeric = source_enc.select_dtypes(include=[np.number]).columns.tolist()
    src_numeric = [c for c in src_numeric if c != fk_col]

    if not src_numeric:
        return NaN_RESULT

    try:
        grouped = source_enc.groupby(fk_col)[src_numeric].mean().reset_index()
        grouped.columns = [fk_col] + [f"local_agg_{c}" for c in src_numeric]
        augmented = target_enc.merge(grouped, left_on=pk_col, right_on=fk_col, how="left")
        aug_cols = [c for c in augmented.columns if c.startswith("local_agg_")]
        X_aug = augmented[feature_cols + aug_cols].fillna(0)
    except Exception:
        return NaN_RESULT

    # Train/test split
    scores_base = []
    scores_aug = []
    for seed in SEEDS:
        try:
            Xtr, Xte, ytr, yte = train_test_split(X_base, y, test_size=0.2, random_state=seed)
            Xtr_a, Xte_a = X_aug.loc[Xtr.index], X_aug.loc[Xte.index]

            sb = train_eval_lgbm(Xtr, ytr, Xte, yte, "regression", seed, n_estimators=100, max_depth=4)
            sa = train_eval_lgbm(Xtr_a, ytr, Xte_a, yte, "regression", seed, n_estimators=100, max_depth=4)
            if not math.isnan(sb) and not math.isnan(sa):
                scores_base.append(sb)
                scores_aug.append(sa)
        except Exception:
            continue

    if not scores_base:
        return NaN_RESULT

    mean_b = float(np.mean(scores_base))
    mean_a = float(np.mean(scores_aug))

    return {
        "local_jrn": mean_a / mean_b if mean_b > 1e-10 else float("nan"),
        "proxy_col": proxy_col,
        "baseline": mean_b,
        "augmented": mean_a,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE CHAIN JRN (MEASURED)
# ══════════════════════════════════════════════════════════════════════════

def compute_chain_features(
    chain_def: Dict,
    tables: Dict[str, pd.DataFrame],
    table_pk_cols: Dict[str, str],
    entity_col: str,
    entity_table: str,
    agg_funcs: List[str] = None,
) -> Optional[pd.DataFrame]:
    """Compute aggregated features for a chain of joins, returning a DataFrame
    indexed by entity_col with aggregated features from the chain's endpoint."""
    if agg_funcs is None:
        agg_funcs = ["mean"]

    chain_type = chain_def["type"]
    join_ids = chain_def["joins"]

    if chain_type == "linear":
        # Linear chain: source -> intermediate -> target
        # E.g., results->races->circuits: FK10(results.raceId->races.raceId), FK0(races.circuitId->circuits.circuitId)
        # We want: for each entity (driver), get the END of chain features
        # Path: entity <- first_source (via entity_col) -> first_target -> ... -> last_target

        # Get FK definitions
        fk_defs_chain = [next((f for f in FK_DEFS if f["id"] == jid), None) for jid in join_ids]
        if any(f is None for f in fk_defs_chain):
            return None

        # Start from the first source table
        first_fk = fk_defs_chain[0]
        current_df = tables.get(first_fk["source"])
        if current_df is None:
            return None

        # Check if first source has entity_col
        if entity_col not in current_df.columns:
            return None

        current_df = encode_table(current_df, entity_col)

        # Follow the chain
        for fk_def in fk_defs_chain:
            target_df = tables.get(fk_def["target"])
            if target_df is None:
                return None

            target_enc = encode_table(target_df, fk_def["pk_col"])
            target_numeric = target_enc.select_dtypes(include=[np.number]).columns.tolist()
            target_numeric = [c for c in target_numeric if c != fk_def["pk_col"]]

            if not target_numeric:
                continue

            # Merge current with target via FK
            current_df = current_df.merge(
                target_enc[target_numeric + [fk_def["pk_col"]]],
                left_on=fk_def["fk_col"], right_on=fk_def["pk_col"],
                how="left", suffixes=("", f"_{fk_def['id']}")
            )

        # Now aggregate by entity_col
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != entity_col]

        if not numeric_cols:
            return None

        grouped = current_df.groupby(entity_col)[numeric_cols].agg(agg_funcs)
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"chain_{chain_def['id']}_{col}_{func}" for col, func in grouped.columns]
        else:
            grouped.columns = [f"chain_{chain_def['id']}_{col}" for col in grouped.columns]
        grouped = grouped.reset_index()
        return grouped

    elif chain_type == "convergent":
        # Convergent: two sources both connect to the same target (entity)
        # E.g., results+standings->drivers: FK11(results->drivers) and FK4(standings->drivers)
        # We want: features from BOTH sources aggregated to entity
        fk_defs_chain = [next((f for f in FK_DEFS if f["id"] == jid), None) for jid in join_ids]
        if any(f is None for f in fk_defs_chain):
            return None

        all_grouped = None
        for fk_def in fk_defs_chain:
            source_df = tables.get(fk_def["source"])
            if source_df is None:
                continue

            source_enc = encode_table(source_df, fk_def["fk_col"])
            numeric_cols = source_enc.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != fk_def["fk_col"] and c != entity_col]

            if not numeric_cols:
                continue

            # Check if source has entity_col or if target is entity
            if entity_col in source_enc.columns:
                grp_col = entity_col
            elif fk_def["fk_col"] == entity_col or fk_def["pk_col"] == entity_col:
                grp_col = fk_def["fk_col"]
            else:
                continue

            grouped = source_enc.groupby(grp_col)[numeric_cols].agg(agg_funcs)
            if isinstance(grouped.columns, pd.MultiIndex):
                grouped.columns = [f"chain_{chain_def['id']}_{fk_def['source']}_{col}_{func}" for col, func in grouped.columns]
            else:
                grouped.columns = [f"chain_{chain_def['id']}_{fk_def['source']}_{col}" for col in grouped.columns]
            grouped = grouped.reset_index()

            if grp_col != entity_col:
                grouped = grouped.rename(columns={grp_col: entity_col})

            if all_grouped is None:
                all_grouped = grouped
            else:
                all_grouped = all_grouped.merge(grouped, on=entity_col, how="outer")

        return all_grouped

    elif chain_type == "linear_3hop":
        # 3-hop: drivers <- results -> races -> circuits
        # FK11 (results->drivers), FK10 (results->races), FK0 (races->circuits)
        # For each driver: get their results, for each result get its race,
        # for each race get its circuit features
        fk_defs_chain = [next((f for f in FK_DEFS if f["id"] == jid), None) for jid in join_ids]
        if any(f is None for f in fk_defs_chain):
            return None

        # Start from results (which has driverId)
        results_df = tables.get("results")
        if results_df is None or entity_col not in results_df.columns:
            return None

        current_df = results_df[[entity_col, "raceId"]].copy()

        # Merge races
        races_df = tables.get("races")
        if races_df is None:
            return None
        races_enc = encode_table(races_df, "raceId")

        current_df = current_df.merge(races_enc, on="raceId", how="left")

        # Merge circuits via races.circuitId
        circuits_df = tables.get("circuits")
        if circuits_df is None:
            return None
        circuits_enc = encode_table(circuits_df, "circuitId")
        circuit_numeric = circuits_enc.select_dtypes(include=[np.number]).columns.tolist()
        circuit_numeric = [c for c in circuit_numeric if c != "circuitId"]

        if "circuitId" in current_df.columns:
            current_df = current_df.merge(
                circuits_enc[circuit_numeric + ["circuitId"]],
                on="circuitId", how="left"
            )

        # Aggregate by driver
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != entity_col]

        if not numeric_cols:
            return None

        grouped = current_df.groupby(entity_col)[numeric_cols].agg(agg_funcs)
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"chain3hop_{col}_{func}" for col, func in grouped.columns]
        else:
            grouped.columns = [f"chain3hop_{col}" for col in grouped.columns]
        grouped = grouped.reset_index()
        return grouped

    return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: ENTROPY-BASED JRN PROXY
# ══════════════════════════════════════════════════════════════════════════

def compute_entropy(values: np.ndarray) -> float:
    """Compute Shannon entropy of a discrete array."""
    counts = np.bincount(values.astype(int))
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def compute_conditional_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Compute H(Y|X) using discretized X."""
    unique_x = np.unique(x)
    h_y_given_x = 0.0
    n = len(y)
    for xv in unique_x:
        mask = x == xv
        p_x = mask.sum() / n
        if mask.sum() > 0:
            h_y_given_x += p_x * compute_entropy(y[mask])
    return h_y_given_x


def compute_entropy_jrn(
    entity_ids: np.ndarray,
    labels: np.ndarray,
    source_df: pd.DataFrame,
    entity_col: str,
    source_fk: str,
    task_type: str,
    n_bins: int = 10,
) -> float:
    """Compute entropy-based JRN proxy: H(Y) - H(Y|agg_features)."""
    # Discretize labels if regression
    if task_type == "regression":
        try:
            y_disc = pd.cut(labels, bins=n_bins, labels=False)
            y_disc = np.nan_to_num(y_disc, nan=0).astype(int)
        except Exception:
            y_disc = np.zeros(len(labels), dtype=int)
    else:
        y_disc = labels.astype(int)

    # Compute H(Y)
    h_y = compute_entropy(y_disc)
    if h_y < 1e-10:
        return 0.0

    # Get aggregated features
    source_enc = encode_table(source_df, source_fk)
    numeric_cols = source_enc.select_dtypes(include=[np.number]).columns.tolist()
    link_col = entity_col if entity_col in source_enc.columns else source_fk
    numeric_cols = [c for c in numeric_cols if c != link_col and c != entity_col]

    if not numeric_cols:
        return 0.0

    # Aggregate
    grouped = source_enc.groupby(link_col)[numeric_cols].mean().reset_index()

    # Build entity->features mapping
    entity_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    feature_matrix = np.zeros((len(entity_ids), len(numeric_cols)))

    for _, row in grouped.iterrows():
        eid = row[link_col]
        if eid in entity_to_idx:
            idx = entity_to_idx[eid]
            feature_matrix[idx] = row[numeric_cols].values

    # For each feature, compute H(Y|X_binned)
    min_h_y_given_x = h_y  # worst case = no reduction
    for j in range(feature_matrix.shape[1]):
        col_vals = feature_matrix[:, j]
        if np.std(col_vals) < 1e-10:
            continue
        try:
            binned = pd.cut(col_vals, bins=min(n_bins, len(np.unique(col_vals))), labels=False)
            binned = np.nan_to_num(binned, nan=0).astype(int)
        except Exception:
            continue
        h_cond = compute_conditional_entropy(binned, y_disc)
        min_h_y_given_x = min(min_h_y_given_x, h_cond)

    entropy_reduction = h_y - min_h_y_given_x
    return max(0.0, entropy_reduction)


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: ARCHITECTURE COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def build_config_features(
    config_name: str,
    join_classifications: Dict[str, str],  # join_id -> HIGH/CRITICAL/LOW
    X_base: pd.DataFrame,
    entity_col: str,
    tables: Dict[str, pd.DataFrame],
    table_pk_cols: Dict[str, str],
) -> pd.DataFrame:
    """Build feature matrix for a given architecture configuration."""
    X = X_base.copy()

    for fk_def in FK_DEFS:
        join_id = fk_def["id"]
        source_table = fk_def["source"]
        target_table = fk_def["target"]
        fk_col = fk_def["fk_col"]
        pk_col = fk_def["pk_col"]
        source_df = tables.get(source_table)
        target_df = tables.get(target_table)
        if source_df is None:
            continue

        category = join_classifications.get(join_id, "LOW")

        if config_name == "probe_guided" or config_name == "entropy_guided":
            if category == "LOW":
                continue  # Skip low-value joins
            elif category == "HIGH":
                agg_funcs = ["mean"]
            else:  # CRITICAL
                agg_funcs = ["mean", "std", "max", "min"]
        elif config_name == "uniform_mean":
            agg_funcs = ["mean"]
        elif config_name == "uniform_rich":
            agg_funcs = ["mean", "std", "max", "min"]
        else:
            agg_funcs = ["mean"]

        agg_prefix = f"cfg_{join_id}_"

        try:
            grouped = _aggregate_to_entity(
                source_df=source_df,
                entity_col=entity_col,
                target_df=target_df,
                fk_col=fk_col,
                pk_col=pk_col,
                agg_prefix=agg_prefix,
                agg_funcs=agg_funcs,
                tables=tables,
                table_pk_cols=table_pk_cols,
                source_table_name=source_table,
            )

            if grouped is not None and not grouped.empty and entity_col in X.columns:
                X = X.merge(grouped, on=entity_col, how="left")
        except Exception as e:
            logger.debug(f"Config feature build failed for {join_id}: {e}")
            continue

    # Fill NaN and keep only numeric
    X = _make_numeric_df(X)
    return X


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("JRN Multiplicative Compounding + Entropy vs Probe Architecture")
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    tables, fk_metadata, task_samples, task_meta, table_pk_cols = load_data()

    # Encode all tables once
    encoded_tables: Dict[str, pd.DataFrame] = {}
    for tname, df in tables.items():
        pk = table_pk_cols.get(tname, df.columns[0])
        encoded_tables[tname] = encode_table(df, pk)

    # ── Build task datasets ──────────────────────────────────────────────
    logger.info("Building task datasets...")
    task_datasets = {}
    for task_name in ALL_TASKS:
        logger.info(f"  Building {task_name}...")
        X_train, y_train, X_val, y_val, task_type = build_task_dataset(
            task_name, task_samples, task_meta, tables, table_pk_cols
        )
        if X_train.empty or X_val.empty:
            logger.warning(f"  Skipping {task_name}: insufficient data")
            continue
        task_datasets[task_name] = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "task_type": task_type,
            "entity_col": task_meta[task_name]["entity_col"],
        }
        logger.info(f"  {task_name}: train={X_train.shape}, val={X_val.shape}, type={task_type}")

    # ══════════════════════════════════════════════════════════════════════
    # PART A: INDIVIDUAL JRN COMPUTATION
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PART A: Computing individual JRN for all joins x tasks")
    logger.info("=" * 70)

    individual_jrn: Dict[str, Dict[str, Dict]] = {}  # join_id -> task_name -> result

    for task_name, td in task_datasets.items():
        logger.info(f"\nTask: {task_name}")
        entity_col = td["entity_col"]

        for fk_def in FK_DEFS:
            join_id = fk_def["id"]
            t0 = time.time()

            result = compute_jrn_for_task_join(
                task_name=task_name,
                join_id=join_id,
                X_train_base=td["X_train"],
                y_train=td["y_train"],
                X_val_base=td["X_val"],
                y_val=td["y_val"],
                task_type=td["task_type"],
                tables=tables,
                table_pk_cols=table_pk_cols,
                entity_col=entity_col,
            )

            if join_id not in individual_jrn:
                individual_jrn[join_id] = {}
            individual_jrn[join_id][task_name] = result

            elapsed = time.time() - t0
            jrn_val = result.get("jrn", float("nan"))
            if not math.isnan(jrn_val):
                logger.info(f"  {join_id} ({fk_def['source']}->{fk_def['target']}): "
                           f"JRN={jrn_val:.4f} (base={result['baseline']:.4f}, aug={result['augmented']:.4f}) [{elapsed:.1f}s]")
            else:
                logger.info(f"  {join_id}: JRN=NaN (not reachable from {entity_col}) [{elapsed:.1f}s]")

    # Print JRN matrix
    logger.info("\n=== JRN MATRIX (task-specific) ===")
    for join_id in sorted(individual_jrn.keys()):
        vals = []
        for task_name in ALL_TASKS:
            if task_name in individual_jrn[join_id]:
                v = individual_jrn[join_id][task_name].get("jrn", float("nan"))
                vals.append(f"{v:.4f}" if not math.isnan(v) else "NaN")
            else:
                vals.append("N/A")
        logger.info(f"  {join_id}: {', '.join(vals)}")

    # ── Compute LOCAL JRN for ALL 13 joins ────────────────────────────
    logger.info("\n=== LOCAL JRN (proxy-target based) ===")
    local_jrn: Dict[str, Dict] = {}
    for fk_def in FK_DEFS:
        join_id = fk_def["id"]
        t0 = time.time()
        result = compute_local_jrn(join_id, tables, table_pk_cols)
        local_jrn[join_id] = result
        lj = result.get("local_jrn", float("nan"))
        proxy = result.get("proxy_col", "?")
        if not math.isnan(lj):
            logger.info(f"  {join_id} ({fk_def['source']}->{fk_def['target']}): "
                       f"local_JRN={lj:.4f} proxy={proxy} [{time.time()-t0:.1f}s]")
        else:
            logger.info(f"  {join_id}: local_JRN=NaN [{time.time()-t0:.1f}s]")

    # Build a unified JRN lookup: prefer task-specific, fallback to local
    def get_best_jrn(join_id: str, task_name: Optional[str] = None) -> float:
        """Get best available JRN for a join, preferring task-specific over local."""
        if task_name and join_id in individual_jrn and task_name in individual_jrn[join_id]:
            v = individual_jrn[join_id][task_name].get("jrn", float("nan"))
            if not math.isnan(v):
                return v
        # Fallback: average across tasks
        if join_id in individual_jrn:
            all_v = [individual_jrn[join_id][t].get("jrn", float("nan"))
                     for t in individual_jrn[join_id]
                     if not math.isnan(individual_jrn[join_id][t].get("jrn", float("nan")))]
            if all_v:
                return float(np.mean(all_v))
        # Fallback: local JRN
        if join_id in local_jrn:
            v = local_jrn[join_id].get("local_jrn", float("nan"))
            if not math.isnan(v):
                return v
        return float("nan")

    # ══════════════════════════════════════════════════════════════════════
    # PART A continued: CHAIN COMPOUNDING
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PART A: Computing chain JRN (multiplicative compounding test)")
    logger.info("=" * 70)

    chain_results = []

    for chain_def in CHAINS:
        logger.info(f"\nChain: {chain_def['id']} ({chain_def['desc']})")

        for task_name, td in task_datasets.items():
            entity_col = td["entity_col"]
            entity_table = task_meta[task_name].get("entity_table", "drivers")

            # Compute predicted JRN (product of individual JRNs)
            join_jrns = []
            for jid in chain_def["joins"]:
                v = get_best_jrn(jid, task_name)
                if not math.isnan(v):
                    join_jrns.append(v)

            if len(join_jrns) != len(chain_def["joins"]):
                logger.info(f"  {task_name}: Skipping, only {len(join_jrns)}/{len(chain_def['joins'])} JRNs available")
                continue

            predicted_jrn = np.prod(join_jrns)

            # Compute measured chain JRN
            chain_features = compute_chain_features(
                chain_def, tables, table_pk_cols, entity_col, entity_table
            )

            if chain_features is None or chain_features.empty:
                logger.info(f"  {task_name}: Could not compute chain features")
                continue

            # Merge chain features with task features
            X_train_aug = td["X_train"].copy()
            X_val_aug = td["X_val"].copy()

            if entity_col in X_train_aug.columns and entity_col in chain_features.columns:
                X_train_aug = X_train_aug.merge(chain_features, on=entity_col, how="left")
                X_val_aug = X_val_aug.merge(chain_features, on=entity_col, how="left")
            else:
                logger.info(f"  {task_name}: entity_col mismatch")
                continue

            # Fill NaN
            X_train_aug = X_train_aug.fillna(0)
            X_val_aug = X_val_aug.fillna(0)

            # Ensure all numeric
            numeric_cols = []
            for c in X_train_aug.columns:
                if X_train_aug[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    numeric_cols.append(c)
                else:
                    try:
                        X_train_aug[c] = pd.to_numeric(X_train_aug[c], errors="coerce").fillna(0)
                        X_val_aug[c] = pd.to_numeric(X_val_aug[c], errors="coerce").fillna(0)
                        numeric_cols.append(c)
                    except Exception:
                        pass

            common_cols = sorted(set(numeric_cols) & set(X_val_aug.columns))
            X_train_aug = X_train_aug[common_cols]
            X_val_aug = X_val_aug[common_cols]

            # Train and evaluate
            base_cols = sorted(set(td["X_train"].columns) & set(td["X_val"].columns))
            base_cols = [c for c in base_cols if td["X_train"][c].dtype in [np.float64, np.float32, np.int64, np.int32]]

            measured_scores_base = []
            measured_scores_aug = []
            for seed in SEEDS:
                sb = train_eval_lgbm(
                    td["X_train"][base_cols], td["y_train"],
                    td["X_val"][base_cols], td["y_val"],
                    td["task_type"], seed
                )
                sa = train_eval_lgbm(
                    X_train_aug, td["y_train"],
                    X_val_aug, td["y_val"],
                    td["task_type"], seed
                )
                if not math.isnan(sb) and not math.isnan(sa):
                    measured_scores_base.append(sb)
                    measured_scores_aug.append(sa)

            if not measured_scores_base:
                continue

            mean_base = np.mean(measured_scores_base)
            mean_aug = np.mean(measured_scores_aug)

            if mean_base < 1e-10:
                continue

            measured_jrn = mean_aug / mean_base

            chain_results.append({
                "chain_id": chain_def["id"],
                "chain_desc": chain_def["desc"],
                "chain_type": chain_def["type"],
                "task": task_name,
                "joins": chain_def["joins"],
                "individual_jrns": join_jrns,
                "predicted_jrn": float(predicted_jrn),
                "measured_jrn": float(measured_jrn),
                "measured_base": float(mean_base),
                "measured_aug": float(mean_aug),
            })

            logger.info(f"  {task_name}: predicted={predicted_jrn:.4f}, measured={measured_jrn:.4f}")

    # Compute R² between predicted and measured chain JRN
    if len(chain_results) >= 3:
        predicted = np.array([r["predicted_jrn"] for r in chain_results])
        measured = np.array([r["measured_jrn"] for r in chain_results])

        # Filter out any extreme outliers (> 3 std from mean)
        mask = np.ones(len(predicted), dtype=bool)
        for arr in [predicted, measured]:
            mu, sigma = np.mean(arr), np.std(arr)
            if sigma > 0:
                mask &= np.abs(arr - mu) < 3 * sigma

        predicted_clean = predicted[mask]
        measured_clean = measured[mask]

        if len(predicted_clean) >= 3:
            r2 = r2_score(measured_clean, predicted_clean)
            pearson_r, pearson_p = stats.pearsonr(predicted_clean, measured_clean)
            spearman_rho, spearman_p = stats.spearmanr(predicted_clean, measured_clean)
        else:
            r2 = float("nan")
            pearson_r = float("nan")
            spearman_rho = float("nan")

        logger.info(f"\n=== COMPOUNDING RESULTS ===")
        logger.info(f"  Chains evaluated: {len(chain_results)}")
        logger.info(f"  R² (predicted vs measured): {r2:.4f}")
        logger.info(f"  Pearson r: {pearson_r:.4f}")
        logger.info(f"  Spearman rho: {spearman_rho:.4f}")
        logger.info(f"  rel-stack reference R²: 0.83")
    else:
        r2 = float("nan")
        pearson_r = float("nan")
        spearman_rho = float("nan")
        logger.warning(f"Only {len(chain_results)} chain results, insufficient for R²")

    # ══════════════════════════════════════════════════════════════════════
    # PART B: ENTROPY-BASED JRN PROXY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PART B: Computing entropy-based JRN proxy")
    logger.info("=" * 70)

    entropy_values: Dict[str, Dict[str, float]] = {}  # join_id -> task -> entropy_reduction
    probe_jrn_values: Dict[str, Dict[str, float]] = {}  # join_id -> task -> JRN

    for task_name, td in task_datasets.items():
        entity_col = td["entity_col"]
        logger.info(f"\nTask: {task_name}")

        # Get entity IDs and labels from training data
        if entity_col not in td["X_train"].columns:
            logger.warning(f"  entity_col {entity_col} not in training data")
            continue

        entity_ids = td["X_train"][entity_col].values
        labels = td["y_train"].values

        for fk_def in FK_DEFS:
            join_id = fk_def["id"]
            source_table = fk_def["source"]
            source_df = tables.get(source_table)
            if source_df is None:
                continue

            # Check if source has entity_col
            if entity_col not in source_df.columns:
                continue

            # Compute entropy-based proxy
            try:
                entropy_red = compute_entropy_jrn(
                    entity_ids, labels, source_df, entity_col,
                    fk_def["fk_col"], td["task_type"]
                )
            except Exception as e:
                logger.warning(f"  {join_id} entropy failed: {e}")
                entropy_red = 0.0

            if join_id not in entropy_values:
                entropy_values[join_id] = {}
            entropy_values[join_id][task_name] = float(entropy_red)

            # Also store probe JRN for comparison
            if join_id not in probe_jrn_values:
                probe_jrn_values[join_id] = {}
            if join_id in individual_jrn and task_name in individual_jrn[join_id]:
                probe_jrn_values[join_id][task_name] = individual_jrn[join_id][task_name].get("jrn", float("nan"))

            logger.info(f"  {join_id}: entropy={entropy_red:.4f}, probe_jrn={probe_jrn_values.get(join_id, {}).get(task_name, 'N/A')}")

    # Compute Spearman correlation between entropy and probe rankings
    per_task_spearman = {}
    all_entropy = []
    all_probe = []

    for task_name in ALL_TASKS:
        ent_list = []
        probe_list = []
        for join_id in sorted(entropy_values.keys()):
            if task_name in entropy_values.get(join_id, {}):
                ev = entropy_values[join_id][task_name]
                pv = probe_jrn_values.get(join_id, {}).get(task_name, float("nan"))
                if not math.isnan(ev) and not math.isnan(pv):
                    ent_list.append(ev)
                    probe_list.append(pv)

        if len(ent_list) >= 3:
            rho, p = stats.spearmanr(ent_list, probe_list)
            per_task_spearman[task_name] = {"rho": float(rho), "p_value": float(p), "n": len(ent_list)}
            all_entropy.extend(ent_list)
            all_probe.extend(probe_list)
            logger.info(f"  {task_name}: Spearman rho={rho:.4f} (n={len(ent_list)})")
        else:
            per_task_spearman[task_name] = {"rho": float("nan"), "n": len(ent_list)}

    if len(all_entropy) >= 3:
        overall_rho, overall_p = stats.spearmanr(all_entropy, all_probe)
        logger.info(f"  Overall Spearman rho: {overall_rho:.4f} (n={len(all_entropy)})")
    else:
        overall_rho = float("nan")
        logger.warning("  Insufficient data for overall Spearman correlation")

    # ══════════════════════════════════════════════════════════════════════
    # PART B: ARCHITECTURE COMPARISON (4 configs)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PART B: Architecture comparison (4 configs)")
    logger.info("=" * 70)

    # Classify joins by probe JRN
    def classify_joins(jrn_values: Dict[str, Dict[str, float]], tasks: List[str]) -> Dict[str, str]:
        """Classify joins into HIGH/CRITICAL/LOW based on average JRN across tasks."""
        avg_jrn = {}
        for join_id, task_vals in jrn_values.items():
            vals = [task_vals[t] for t in tasks if t in task_vals and not math.isnan(task_vals.get(t, float("nan")))]
            if vals:
                avg_jrn[join_id] = np.mean(vals)

        if not avg_jrn:
            return {}

        # Adaptive thresholds: use percentiles
        values = list(avg_jrn.values())
        if len(values) < 3:
            # All HIGH if fewer than 3
            return {jid: "HIGH" for jid in avg_jrn}

        p33 = np.percentile(values, 33)
        p67 = np.percentile(values, 67)

        classifications = {}
        for jid, v in avg_jrn.items():
            if v >= p67:
                classifications[jid] = "HIGH"
            elif v >= p33:
                classifications[jid] = "CRITICAL"
            else:
                classifications[jid] = "LOW"

        return classifications

    # Classify by probe JRN
    probe_classifications = classify_joins(probe_jrn_values, ALL_TASKS)
    logger.info(f"Probe classifications: {probe_classifications}")

    # Classify by entropy
    entropy_classifications = classify_joins(entropy_values, ALL_TASKS)
    logger.info(f"Entropy classifications: {entropy_classifications}")

    # Evaluate 4 configs
    config_results: Dict[str, Dict[str, Dict]] = {}  # task -> config -> scores
    configs = {
        "probe_guided": probe_classifications,
        "entropy_guided": entropy_classifications,
        "uniform_mean": {fk["id"]: "HIGH" for fk in FK_DEFS},  # All included, mean only
        "uniform_rich": {fk["id"]: "CRITICAL" for fk in FK_DEFS},  # All included, rich agg
    }

    for task_name, td in task_datasets.items():
        entity_col = td["entity_col"]
        logger.info(f"\nTask: {task_name}")
        config_results[task_name] = {}

        for config_name, join_class in configs.items():
            t0 = time.time()

            # Build features
            X_train_cfg = build_config_features(
                config_name, join_class, td["X_train"], entity_col, tables, table_pk_cols
            )
            X_val_cfg = build_config_features(
                config_name, join_class, td["X_val"], entity_col, tables, table_pk_cols
            )

            # Align columns
            common = sorted(set(X_train_cfg.columns) & set(X_val_cfg.columns))
            numeric = [c for c in common if X_train_cfg[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            X_train_cfg = X_train_cfg[numeric].fillna(0)
            X_val_cfg = X_val_cfg[numeric].fillna(0)

            scores = []
            for seed in SEEDS:
                s = train_eval_lgbm(
                    X_train_cfg, td["y_train"],
                    X_val_cfg, td["y_val"],
                    td["task_type"], seed,
                    n_estimators=300, max_depth=8,
                )
                if not math.isnan(s):
                    scores.append(s)

            elapsed = time.time() - t0
            mean_score = float(np.mean(scores)) if scores else float("nan")
            std_score = float(np.std(scores)) if len(scores) > 1 else 0.0

            config_results[task_name][config_name] = {
                "mean": mean_score,
                "std": std_score,
                "scores": [float(s) for s in scores],
                "n_features": X_train_cfg.shape[1],
            }
            logger.info(f"  {config_name}: mean={mean_score:.4f} +/- {std_score:.4f} "
                        f"(n_feats={X_train_cfg.shape[1]}) [{elapsed:.1f}s]")

    # Count wins
    wins: Dict[str, int] = {c: 0 for c in configs}
    best_config_per_task = {}
    for task_name, task_cfg in config_results.items():
        best_config = max(task_cfg, key=lambda c: task_cfg[c]["mean"] if not math.isnan(task_cfg[c]["mean"]) else -999)
        wins[best_config] += 1
        best_config_per_task[task_name] = best_config
        logger.info(f"  {task_name}: best={best_config} ({task_cfg[best_config]['mean']:.4f})")

    logger.info(f"\n=== ARCHITECTURE WINS ===")
    for cfg, n in wins.items():
        logger.info(f"  {cfg}: {n} wins")

    # ══════════════════════════════════════════════════════════════════════
    # COMPILE OUTPUT: method_out.json in exp_gen_sol_out schema
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("Compiling results into method_out.json")
    logger.info("=" * 70)

    # Build examples for exp_gen_sol_out schema
    examples = []

    # Example 1: Part A summary - Individual JRN results (task-specific + local)
    part_a_individual = {}
    for fk_def in FK_DEFS:
        join_id = fk_def["id"]
        entry = {"source": fk_def["source"], "target": fk_def["target"]}
        # Task-specific JRN
        for task_name in ALL_TASKS:
            if join_id in individual_jrn and task_name in individual_jrn[join_id]:
                r = individual_jrn[join_id][task_name]
                entry[task_name] = {
                    "jrn": r.get("jrn", None),
                    "baseline": r.get("baseline", None),
                    "augmented": r.get("augmented", None),
                }
        # Local JRN
        if join_id in local_jrn:
            lj = local_jrn[join_id]
            entry["local_jrn"] = lj.get("local_jrn", None)
            entry["local_proxy_col"] = lj.get("proxy_col", None)
        part_a_individual[join_id] = entry

    examples.append({
        "input": json.dumps({
            "analysis": "Part A - Individual JRN values for all 13 FK joins across 3 driver tasks + local proxy",
            "method": "GBM probe (LightGBM n_estimators=200, 3 seeds) + local proxy-target JRN",
            "joins_tested": len(FK_DEFS),
            "tasks_tested": len(ALL_TASKS),
        }),
        "output": json.dumps(part_a_individual),
        "metadata_result_type": "individual_jrn",
        "predict_jrn_probe": json.dumps(part_a_individual),
    })

    # Example 2: Part A - Chain compounding results
    compounding_summary = {
        "chains": [{
            "chain_id": r["chain_id"],
            "chain_desc": r["chain_desc"],
            "task": r["task"],
            "predicted_jrn": r["predicted_jrn"],
            "measured_jrn": r["measured_jrn"],
        } for r in chain_results],
        "r_squared": float(r2) if not math.isnan(r2) else None,
        "pearson_r": float(pearson_r) if not math.isnan(pearson_r) else None,
        "spearman_rho": float(spearman_rho) if not math.isnan(spearman_rho) else None,
        "num_chains_evaluated": len(chain_results),
        "comparison": {"rel_stack_r2": 0.83, "rel_f1_r2": float(r2) if not math.isnan(r2) else None},
    }

    examples.append({
        "input": json.dumps({
            "analysis": "Part A - Multiplicative compounding of JRN along multi-hop chains",
            "hypothesis": "Chain JRN = product of individual JRNs",
            "metric": "R^2 between predicted (product) and measured chain JRN",
            "reference": "rel-stack R^2 = 0.83",
        }),
        "output": json.dumps(compounding_summary),
        "metadata_result_type": "chain_compounding",
        "predict_compounding": json.dumps(compounding_summary),
    })

    # Example 3: Part B - Entropy vs Probe correlation
    entropy_probe_summary = {
        "per_task_spearman": {k: v for k, v in per_task_spearman.items()},
        "overall_spearman_rho": float(overall_rho) if not math.isnan(overall_rho) else None,
        "entropy_values": {k: v for k, v in entropy_values.items()},
        "probe_jrn_values": {k: {tk: float(tv) if not math.isnan(tv) else None for tk, tv in v.items()} for k, v in probe_jrn_values.items()},
    }

    examples.append({
        "input": json.dumps({
            "analysis": "Part B - Entropy-based JRN proxy vs GBM probe correlation",
            "method": "Conditional entropy reduction H(Y)-H(Y|agg_features) with 10-bin discretization",
            "metric": "Spearman rho between entropy ranking and probe ranking",
        }),
        "output": json.dumps(entropy_probe_summary),
        "metadata_result_type": "entropy_vs_probe",
        "predict_entropy_proxy": json.dumps(entropy_probe_summary),
    })

    # Example 4: Part B - Architecture comparison
    arch_summary = {
        "config_results": {t: {c: v for c, v in cvs.items()} for t, cvs in config_results.items()},
        "wins_summary": wins,
        "best_config_per_task": best_config_per_task,
        "join_classifications": {
            "probe_based": probe_classifications,
            "entropy_based": entropy_classifications,
        },
    }

    # Compute deltas
    probe_vs_uniform_mean = {}
    probe_vs_entropy = {}
    for task_name in config_results:
        probe_score = config_results[task_name].get("probe_guided", {}).get("mean", float("nan"))
        entropy_score = config_results[task_name].get("entropy_guided", {}).get("mean", float("nan"))
        uniform_mean_score = config_results[task_name].get("uniform_mean", {}).get("mean", float("nan"))
        if not math.isnan(probe_score) and not math.isnan(uniform_mean_score):
            probe_vs_uniform_mean[task_name] = float(probe_score - uniform_mean_score)
        if not math.isnan(probe_score) and not math.isnan(entropy_score):
            probe_vs_entropy[task_name] = float(probe_score - entropy_score)

    arch_summary["probe_vs_uniform_mean_delta"] = probe_vs_uniform_mean
    arch_summary["probe_vs_entropy_delta"] = probe_vs_entropy

    examples.append({
        "input": json.dumps({
            "analysis": "Part B - Architecture comparison: 4 configs x 3 tasks x 3 seeds",
            "configs": ["probe_guided", "entropy_guided", "uniform_mean", "uniform_rich"],
            "model": "LightGBM n_estimators=300, max_depth=8",
        }),
        "output": json.dumps(arch_summary),
        "metadata_result_type": "architecture_comparison",
        "predict_architecture": json.dumps(arch_summary),
    })

    # Example 5: Overall summary
    overall_summary = {
        "title": "JRN Multiplicative Compounding on rel-f1 + Entropy vs Probe Architecture",
        "dataset": "rel-f1 (Formula 1)",
        "num_tables": 9,
        "num_joins": 13,
        "num_chains_tested": len(chain_results),
        "tasks_used": ALL_TASKS,
        "part_a_r_squared": float(r2) if not math.isnan(r2) else None,
        "part_a_pearson_r": float(pearson_r) if not math.isnan(pearson_r) else None,
        "part_a_spearman_rho": float(spearman_rho) if not math.isnan(spearman_rho) else None,
        "part_b_entropy_spearman": float(overall_rho) if not math.isnan(overall_rho) else None,
        "part_b_best_config": max(wins, key=wins.get) if wins else None,
        "part_b_wins": wins,
        "runtime_seconds": time.time() - t_start,
    }

    examples.append({
        "input": json.dumps({
            "analysis": "Overall experiment summary",
            "experiment": "JRN Multiplicative Compounding + Entropy vs Probe",
        }),
        "output": json.dumps(overall_summary),
        "metadata_result_type": "summary",
        "predict_summary": json.dumps(overall_summary),
    })

    # Build final output in exp_gen_sol_out schema
    output = {
        "metadata": {
            "method_name": "JRN Multiplicative Compounding + Entropy vs Probe Architecture",
            "description": "Tests multiplicative compounding of JRN along multi-hop chains and compares entropy-based vs GBM-probe-based architecture guidance on rel-f1",
            "dataset": "rel-f1",
            "seeds": SEEDS,
            "n_joins": 13,
            "n_chains": len(chain_results),
            "n_tasks": len(ALL_TASKS),
        },
        "datasets": [
            {
                "dataset": "rel-f1",
                "examples": examples,
            }
        ],
    }

    # Write output
    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {output_path} ({file_size_mb:.2f} MB)")
    logger.info(f"Total runtime: {time.time() - t_start:.1f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()
