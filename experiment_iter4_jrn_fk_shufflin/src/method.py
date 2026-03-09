#!/usr/bin/env python3
"""JRN FK-Shuffling Confound Control on rel-f1.

Tests whether JRN (Join Reproduction Number) captures genuine relational join
structure or merely reflects child-table feature quality. For each FK join x task,
compute normal JRN via LightGBM probes, then compute shuffled JRN (FK column
permuted to break true parent-child associations). Decompose JRN into structural
component (destroyed by shuffling) and feature component (survives shuffling).
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
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

# Set memory limit: use at most 80% of container RAM
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1024**3)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError):
    pass

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
INPUT_FILE = DATA_DIR / "full_data_out.json"

SEEDS = [42, 123, 777]
N_SHUFFLES = 5
LGB_PARAMS_BASE = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "verbose": -1,
    "n_jobs": 1,  # single-threaded per model; we parallelize at job level
}
TASKS = ["rel-f1/driver-dnf", "rel-f1/driver-top3", "rel-f1/driver-position"]
TASK_TYPES = {
    "rel-f1/driver-dnf": "binary_classification",
    "rel-f1/driver-top3": "binary_classification",
    "rel-f1/driver-position": "regression",
}

# FK join classification (from data exploration)
# Group A: source table has driverId
# Direct joins (source -> drivers): #4, #8, #11
# Indirect joins (source -> non-driver target): #3, #7, #9, #10, #12
# Group B: source table has NO driverId: #0, #1, #2, #5, #6

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_and_parse_data(input_file: Path) -> Tuple[
    Dict[str, pd.DataFrame],
    List[Dict[str, Any]],
    Dict[str, pd.DataFrame],
]:
    """Load full_data_out.json and reconstruct DataFrames, FK joins, and task samples."""
    logger.info(f"Loading data from {input_file}")
    raw = json.loads(input_file.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples")

    # Separate by row type
    table_rows: Dict[str, List[Dict]] = defaultdict(list)
    fk_joins: List[Dict[str, Any]] = []
    task_samples: Dict[str, List[Dict]] = defaultdict(list)
    table_pk_cols: Dict[str, str] = {}

    for ex in examples:
        rt = ex.get("metadata_row_type")
        if rt == "table_row":
            tname = ex["metadata_table_name"]
            features = json.loads(ex["input"])
            pk_col = ex["metadata_primary_key_col"]
            pk_val = ex["metadata_primary_key_value"]
            table_pk_cols[tname] = pk_col
            # Add PK back into features
            try:
                features[pk_col] = int(pk_val)
            except (ValueError, TypeError):
                features[pk_col] = pk_val
            table_rows[tname].append(features)
        elif rt == "fk_join_metadata":
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])
            fk_joins.append({**inp, **out, "join_idx": ex["metadata_row_index"]})
        elif rt == "task_sample":
            task_name = ex.get("metadata_task_name", "")
            if task_name in TASKS:
                sample = json.loads(ex["input"])
                output_val = ex["output"]
                if output_val != "masked":
                    try:
                        sample["target"] = float(output_val)
                    except (ValueError, TypeError):
                        continue
                    sample["fold"] = ex.get("metadata_fold_name", "train")
                    try:
                        sample["driverId"] = int(sample["driverId"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    task_samples[task_name].append(sample)

    # Convert to DataFrames
    tables = {}
    for tname, rows in table_rows.items():
        df = pd.DataFrame(rows)
        # Convert numeric-looking columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except (ValueError, TypeError):
                pass
        tables[tname] = df
        logger.info(f"  Table '{tname}': {df.shape[0]} rows, {df.shape[1]} cols, pk={table_pk_cols.get(tname)}")

    # Sort FK joins by index
    fk_joins.sort(key=lambda x: x["join_idx"])
    logger.info(f"  FK joins: {len(fk_joins)}")
    for fk in fk_joins:
        logger.info(f"    #{fk['join_idx']}: {fk['source_table']}.{fk['source_fk_col']} -> {fk['target_table']}.{fk['target_pk_col']}")

    # Convert task samples to DataFrames
    task_dfs = {}
    for task_name, samples in task_samples.items():
        df = pd.DataFrame(samples)
        task_dfs[task_name] = df
        fold_counts = df["fold"].value_counts().to_dict()
        logger.info(f"  Task '{task_name}': {len(df)} samples, folds={fold_counts}")

    del raw, examples
    gc.collect()

    return tables, fk_joins, task_dfs


# ── Driver Feature Preparation ──────────────────────────────────────────────

def prepare_driver_features(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create baseline driver features from drivers table."""
    df = tables["drivers"].copy()

    # The PK is driverId
    if "driverId" not in df.columns:
        logger.error("driverId not in drivers table!")
        return pd.DataFrame()

    feature_cols = []

    # nationality -> label encode
    if "nationality" in df.columns:
        le = LabelEncoder()
        mask = df["nationality"].notna()
        df.loc[mask, "nationality_enc"] = le.fit_transform(df.loc[mask, "nationality"].astype(str))
        df["nationality_enc"] = df["nationality_enc"].fillna(-1).astype(float)
        feature_cols.append("nationality_enc")

    # dob -> birth_year
    if "dob" in df.columns:
        try:
            dob_dt = pd.to_datetime(df["dob"], errors="coerce")
            df["birth_year"] = dob_dt.dt.year.astype(float)
            df["birth_year"] = df["birth_year"].fillna(df["birth_year"].median())
            feature_cols.append("birth_year")
        except Exception:
            logger.warning("Could not parse dob column")

    # code -> label encode
    if "code" in df.columns:
        le2 = LabelEncoder()
        mask = df["code"].notna()
        if mask.any():
            df.loc[mask, "code_enc"] = le2.fit_transform(df.loc[mask, "code"].astype(str))
            df["code_enc"] = df["code_enc"].fillna(-1).astype(float)
            feature_cols.append("code_enc")

    result = df[["driverId"] + feature_cols].copy()
    result = result.set_index("driverId")
    logger.info(f"Driver features: {result.shape[1]} features for {result.shape[0]} drivers")
    return result


# ── Join Feature Computation ────────────────────────────────────────────────

def _get_numeric_cols(df: pd.DataFrame, exclude: set) -> List[str]:
    """Get numeric columns excluding specified ones."""
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def compute_direct_join_features(
    source_df: pd.DataFrame,
    join_idx: int,
    pk_col: str,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> pd.DataFrame:
    """Compute features for direct joins (source -> drivers).

    For direct joins, the source table has driverId. We aggregate source
    numeric features by driverId.
    """
    src = source_df.copy()
    if shuffle:
        rng = np.random.RandomState(shuffle_seed)
        src["driverId"] = rng.permutation(src["driverId"].values)

    exclude = {"driverId", "raceId", "constructorId", pk_col, "date"}
    feat_cols = _get_numeric_cols(src, exclude)

    if not feat_cols:
        return pd.DataFrame()

    agg = src.groupby("driverId")[feat_cols].agg(["mean", "std"])
    agg.columns = [f"j{join_idx}_{c[0]}_{c[1]}" for c in agg.columns]
    return agg.fillna(0)


def compute_indirect_join_features(
    source_df: pd.DataFrame,
    fk_col: str,
    target_df: pd.DataFrame,
    target_pk_col: str,
    join_idx: int,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> pd.DataFrame:
    """Compute features for indirect joins (source -> non-driver target).

    Source has driverId. We join source to target via FK, then aggregate
    target-side numeric features by driverId.
    """
    src = source_df.copy()
    if shuffle:
        rng = np.random.RandomState(shuffle_seed)
        src[fk_col] = rng.permutation(src[fk_col].values)

    # Merge source to target
    merged = src.merge(
        target_df,
        left_on=fk_col,
        right_on=target_pk_col,
        how="left",
        suffixes=("", "_tgt"),
    )

    # Get target-side numeric features
    tgt_feat_cols = [
        c for c in merged.columns
        if c.endswith("_tgt") and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    # Also get target numeric cols that didn't get _tgt suffix (when names don't collide)
    target_numeric = _get_numeric_cols(target_df, {target_pk_col, "date"})
    for c in target_numeric:
        if c in merged.columns and c not in tgt_feat_cols and not c.endswith("_tgt"):
            # Only if this column came from target (check if it's NOT in source)
            if c not in source_df.columns:
                tgt_feat_cols.append(c)

    if not tgt_feat_cols or "driverId" not in merged.columns:
        return pd.DataFrame()

    agg = merged.groupby("driverId")[tgt_feat_cols].agg(["mean", "std"])
    agg.columns = [f"j{join_idx}_{c[0]}_{c[1]}" for c in agg.columns]
    return agg.fillna(0)


def compute_bridge_join_features(
    source_df: pd.DataFrame,
    fk_col: str,
    target_df: pd.DataFrame,
    target_pk_col: str,
    bridge_df: pd.DataFrame,
    bridge_fk_col: str,
    join_idx: int,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> pd.DataFrame:
    """Compute features for Group B joins (source has no driverId).

    Uses a bridge table (e.g. results) that has both the relevant FK and driverId
    to connect to driver-level predictions.
    """
    src = source_df.copy()
    if shuffle:
        rng = np.random.RandomState(shuffle_seed)
        src[fk_col] = rng.permutation(src[fk_col].values)

    # Join source to target
    merged = src.merge(
        target_df,
        left_on=fk_col,
        right_on=target_pk_col,
        how="left",
        suffixes=("_src", "_tgt"),
    )

    # Get all numeric features from the merged result
    exclude = {fk_col, target_pk_col, "date", "date_src", "date_tgt"}
    all_numeric = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in exclude]

    if not all_numeric:
        return pd.DataFrame()

    # Get the key column from source that links to bridge
    # For races->circuits: bridge on raceId
    # For constructor_*->races: bridge on raceId
    # For constructor_*->constructors: bridge on constructorId
    source_pk = None
    for col in ["raceId", "constructorId"]:
        if col in src.columns and col != fk_col:
            source_pk = col
            break
    if source_pk is None:
        # Use the source table's natural key
        for col in src.columns:
            if col.endswith("Id") and col != fk_col:
                source_pk = col
                break

    if source_pk is None:
        return pd.DataFrame()

    # Bridge: find driverIds associated with each source key
    if bridge_fk_col not in bridge_df.columns or "driverId" not in bridge_df.columns:
        return pd.DataFrame()

    # Merge the enriched source with bridge to get driverIds
    # merged has source features + target features per source row
    # We need to link source rows to drivers via bridge table
    bridge_link = bridge_df[[bridge_fk_col, "driverId"]].drop_duplicates()

    # The bridge links via the source's identifying column
    if source_pk in merged.columns:
        link_col = source_pk
    elif source_pk + "_src" in merged.columns:
        link_col = source_pk + "_src"
    else:
        return pd.DataFrame()

    final = merged.merge(bridge_link, left_on=link_col, right_on=bridge_fk_col, how="inner")

    if "driverId" not in final.columns or final.empty:
        return pd.DataFrame()

    # Filter numeric cols that exist in final
    feat_cols = [c for c in all_numeric if c in final.columns]
    if not feat_cols:
        return pd.DataFrame()

    agg = final.groupby("driverId")[feat_cols].agg(["mean", "std"])
    agg.columns = [f"j{join_idx}_{c[0]}_{c[1]}" for c in agg.columns]
    return agg.fillna(0)


def compute_join_features(
    join_idx: int,
    fk_join: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> pd.DataFrame:
    """Dispatch to the right feature computation based on join type."""
    source_table = fk_join["source_table"]
    target_table = fk_join["target_table"]
    source_fk_col = fk_join["source_fk_col"]
    target_pk_col = fk_join["target_pk_col"]

    src_df = tables.get(source_table)
    tgt_df = tables.get(target_table)

    if src_df is None or tgt_df is None:
        logger.warning(f"Missing table for join #{join_idx}: {source_table} or {target_table}")
        return pd.DataFrame()

    has_driver_id = "driverId" in src_df.columns

    if has_driver_id and target_table == "drivers":
        # Direct join: source -> drivers
        pk_col = {
            "standings": "standingsId",
            "qualifying": "qualifyId",
            "results": "resultId",
        }.get(source_table, source_fk_col)
        return compute_direct_join_features(
            src_df, join_idx, pk_col, shuffle=shuffle, shuffle_seed=shuffle_seed
        )
    elif has_driver_id:
        # Indirect join: source -> non-driver target
        return compute_indirect_join_features(
            src_df, source_fk_col, tgt_df, target_pk_col, join_idx,
            shuffle=shuffle, shuffle_seed=shuffle_seed,
        )
    else:
        # Group B: bridge needed
        bridge_df = tables.get("results")
        if bridge_df is None:
            return pd.DataFrame()

        # Determine bridge FK column based on source table
        if source_table == "races":
            bridge_fk_col = "raceId"
        elif source_table in ("constructor_standings", "constructor_results"):
            if target_table == "races":
                bridge_fk_col = "raceId"
            else:  # target is constructors
                bridge_fk_col = "constructorId"
        else:
            bridge_fk_col = "raceId"

        return compute_bridge_join_features(
            src_df, source_fk_col, tgt_df, target_pk_col,
            bridge_df, bridge_fk_col, join_idx,
            shuffle=shuffle, shuffle_seed=shuffle_seed,
        )


# ── Model Training ──────────────────────────────────────────────────────────

def train_and_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: str,
    seed: int,
) -> float:
    """Train LightGBM and evaluate. Returns score (higher is better)."""
    params = {**LGB_PARAMS_BASE, "random_state": seed}

    try:
        if task_type == "binary_classification":
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val)) < 2:
                return 0.5
            return float(roc_auc_score(y_val, pred))
        else:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            return 1.0 / (mae + 1e-8)
    except Exception as e:
        logger.warning(f"Model training failed (seed={seed}): {e}")
        return float("nan")


# ── Single Job Function (for parallel execution) ────────────────────────────

def run_single_job(args: Tuple) -> Dict[str, Any]:
    """Run a single (join_idx, task, seed, shuffle_idx) training job.

    Returns a dict with results.
    """
    (
        join_idx, task_name, seed, shuffle_idx,
        driver_features_arr, driver_features_cols, driver_features_index,
        join_features_arr, join_features_cols, join_features_index,
        train_driverIds, train_targets,
        val_driverIds, val_targets,
        task_type, is_baseline,
    ) = args

    # Reconstruct DataFrames
    driver_feat_df = pd.DataFrame(
        driver_features_arr, columns=driver_features_cols, index=driver_features_index
    )

    # Prepare train/val
    train_df = pd.DataFrame({"driverId": train_driverIds, "target": train_targets})
    val_df = pd.DataFrame({"driverId": val_driverIds, "target": val_targets})

    X_train = train_df.merge(driver_feat_df, left_on="driverId", right_index=True, how="left")
    X_val = val_df.merge(driver_feat_df, left_on="driverId", right_index=True, how="left")

    if not is_baseline and join_features_arr is not None:
        join_feat_df = pd.DataFrame(
            join_features_arr, columns=join_features_cols, index=join_features_index
        )
        X_train = X_train.merge(join_feat_df, left_on="driverId", right_index=True, how="left")
        X_val = X_val.merge(join_feat_df, left_on="driverId", right_index=True, how="left")

    feat_cols = [c for c in X_train.columns if c not in ("driverId", "target")]
    X_tr = X_train[feat_cols].fillna(0)
    X_va = X_val[feat_cols].fillna(0)
    y_tr = X_train["target"]
    y_va = X_val["target"]

    score = train_and_eval(X_tr, y_tr, X_va, y_va, task_type, seed)

    return {
        "join_idx": join_idx,
        "task_name": task_name,
        "seed": seed,
        "shuffle_idx": shuffle_idx,
        "score": score,
        "is_baseline": is_baseline,
        "n_features": len(feat_cols),
    }


# ── Main Experiment ─────────────────────────────────────────────────────────

@logger.catch
def main():
    t_start = time.time()

    # ── STEP 1: Load data ────────────────────────────────────────────────
    tables, fk_joins, task_dfs = load_and_parse_data(INPUT_FILE)

    # ── STEP 2: Prepare driver baseline features ─────────────────────────
    driver_features_df = prepare_driver_features(tables)
    drv_feat_arr = driver_features_df.values
    drv_feat_cols = list(driver_features_df.columns)
    drv_feat_idx = driver_features_df.index.tolist()

    # Prepare task train/val data
    task_data = {}
    for task_name in TASKS:
        if task_name not in task_dfs:
            logger.warning(f"Task {task_name} not found in data!")
            continue
        tdf = task_dfs[task_name]
        train_mask = tdf["fold"] == "train"
        val_mask = tdf["fold"] == "val"
        train_df = tdf[train_mask]
        val_df = tdf[val_mask]
        if len(val_df) < 10:
            logger.warning(f"Task {task_name}: only {len(val_df)} val samples, skipping")
            continue
        task_data[task_name] = {
            "train_driverIds": train_df["driverId"].values,
            "train_targets": train_df["target"].values,
            "val_driverIds": val_df["driverId"].values,
            "val_targets": val_df["target"].values,
        }
        logger.info(f"Task {task_name}: train={len(train_df)}, val={len(val_df)}")

    if not task_data:
        logger.error("No valid tasks found!")
        return

    # ── STEP 3: Compute all join features (normal + shuffled) ────────────
    logger.info("Computing join features for all joins...")
    join_features_cache = {}  # (join_idx, shuffle_idx) -> (arr, cols, idx)

    for ji, fk_join in enumerate(fk_joins):
        # Normal features
        jf = compute_join_features(ji, fk_join, tables, shuffle=False)
        if jf.empty:
            logger.warning(f"Join #{ji} produced empty features (normal)")
            join_features_cache[(ji, -1)] = (None, None, None)
        else:
            join_features_cache[(ji, -1)] = (jf.values, list(jf.columns), jf.index.tolist())
            logger.info(f"  Join #{ji} normal: {jf.shape[1]} features, {jf.shape[0]} drivers")

        # Shuffled features
        for si in range(N_SHUFFLES):
            jf_s = compute_join_features(ji, fk_join, tables, shuffle=True, shuffle_seed=si * 1000 + 1)
            if jf_s.empty:
                join_features_cache[(ji, si)] = (None, None, None)
            else:
                join_features_cache[(ji, si)] = (jf_s.values, list(jf_s.columns), jf_s.index.tolist())

    logger.info(f"Join feature computation done in {time.time()-t_start:.1f}s")

    # ── STEP 4-6: Build all jobs and run in parallel ─────────────────────
    all_jobs = []

    for task_name, td in task_data.items():
        task_type = TASK_TYPES[task_name]

        # Baseline jobs (no join features)
        for seed in SEEDS:
            all_jobs.append((
                -1, task_name, seed, -1,
                drv_feat_arr, drv_feat_cols, drv_feat_idx,
                None, None, None,
                td["train_driverIds"], td["train_targets"],
                td["val_driverIds"], td["val_targets"],
                task_type, True,
            ))

        # Normal join jobs
        for ji in range(len(fk_joins)):
            jf_arr, jf_cols, jf_idx = join_features_cache.get((ji, -1), (None, None, None))
            if jf_arr is None:
                continue
            for seed in SEEDS:
                all_jobs.append((
                    ji, task_name, seed, -1,
                    drv_feat_arr, drv_feat_cols, drv_feat_idx,
                    jf_arr, jf_cols, jf_idx,
                    td["train_driverIds"], td["train_targets"],
                    td["val_driverIds"], td["val_targets"],
                    task_type, False,
                ))

        # Shuffled join jobs
        for ji in range(len(fk_joins)):
            for si in range(N_SHUFFLES):
                jf_arr, jf_cols, jf_idx = join_features_cache.get((ji, si), (None, None, None))
                if jf_arr is None:
                    continue
                for seed in SEEDS:
                    all_jobs.append((
                        ji, task_name, seed, si,
                        drv_feat_arr, drv_feat_cols, drv_feat_idx,
                        jf_arr, jf_cols, jf_idx,
                        td["train_driverIds"], td["train_targets"],
                        td["val_driverIds"], td["val_targets"],
                        task_type, False,
                    ))

    logger.info(f"Total jobs to run: {len(all_jobs)}")

    # Run with ProcessPoolExecutor
    n_workers = max(1, NUM_CPUS)
    logger.info(f"Running with {n_workers} parallel workers")
    all_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_single_job, job): i for i, job in enumerate(all_jobs)}
        done_count = 0
        for fut in as_completed(futures):
            try:
                result = fut.result()
                all_results.append(result)
            except Exception as e:
                logger.exception(f"Job failed: {e}")
            done_count += 1
            if done_count % 50 == 0:
                elapsed = time.time() - t_start
                logger.info(f"Progress: {done_count}/{len(all_jobs)} jobs ({elapsed:.0f}s)")

    logger.info(f"All jobs done in {time.time()-t_start:.1f}s, got {len(all_results)} results")

    # ── STEP 7: Aggregate results ────────────────────────────────────────
    # Baseline scores
    baseline_scores: Dict[str, List[float]] = defaultdict(list)
    for r in all_results:
        if r["is_baseline"] and not math.isnan(r["score"]):
            baseline_scores[r["task_name"]].append(r["score"])

    baseline_mean: Dict[str, float] = {}
    baseline_std: Dict[str, float] = {}
    for task_name, scores in baseline_scores.items():
        baseline_mean[task_name] = float(np.mean(scores))
        baseline_std[task_name] = float(np.std(scores))
        logger.info(f"Baseline {task_name}: {baseline_mean[task_name]:.4f} +/- {baseline_std[task_name]:.4f}")

    # Normal JRN
    normal_scores: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for r in all_results:
        if not r["is_baseline"] and r["shuffle_idx"] == -1 and not math.isnan(r["score"]):
            normal_scores[(r["join_idx"], r["task_name"])].append(r["score"])

    results_normal = {}
    for (ji, task_name), scores in normal_scores.items():
        mean_score = float(np.mean(scores))
        bm = baseline_mean.get(task_name, 1e-8)
        jrn = mean_score / bm if bm > 1e-10 else 1.0
        results_normal[(ji, task_name)] = {
            "scores": [float(s) for s in scores],
            "mean": mean_score,
            "std": float(np.std(scores)),
            "jrn": jrn,
        }

    # Shuffled JRN
    shuffled_scores: Dict[Tuple[int, str, int], List[float]] = defaultdict(list)
    for r in all_results:
        if not r["is_baseline"] and r["shuffle_idx"] >= 0 and not math.isnan(r["score"]):
            shuffled_scores[(r["join_idx"], r["task_name"], r["shuffle_idx"])].append(r["score"])

    results_shuffled = {}
    for (ji, task_name) in set((k[0], k[1]) for k in shuffled_scores.keys()):
        shuffle_jrns = []
        per_shuffle = []
        for si in range(N_SHUFFLES):
            key = (ji, task_name, si)
            if key in shuffled_scores:
                scores = shuffled_scores[key]
                mean_score = float(np.mean(scores))
                bm = baseline_mean.get(task_name, 1e-8)
                shuf_jrn = mean_score / bm if bm > 1e-10 else 1.0
                shuffle_jrns.append(shuf_jrn)
                per_shuffle.append({
                    "shuffle_idx": si,
                    "scores": [float(s) for s in scores],
                    "mean_score": mean_score,
                    "jrn": shuf_jrn,
                })
        if shuffle_jrns:
            results_shuffled[(ji, task_name)] = {
                "per_shuffle_jrn": shuffle_jrns,
                "mean_shuffled_jrn": float(np.mean(shuffle_jrns)),
                "std_shuffled_jrn": float(np.std(shuffle_jrns)),
                "per_shuffle": per_shuffle,
            }

    # ── STEP 7 cont: Decompose JRN ──────────────────────────────────────
    decomposition = {}
    for (ji, task_name) in results_normal:
        normal_jrn = results_normal[(ji, task_name)]["jrn"]
        if (ji, task_name) in results_shuffled:
            shuffled_jrn_mean = results_shuffled[(ji, task_name)]["mean_shuffled_jrn"]
        else:
            shuffled_jrn_mean = 1.0

        jrn_structural = normal_jrn - shuffled_jrn_mean
        jrn_feature = shuffled_jrn_mean - 1.0
        structural_fraction = jrn_structural / (abs(jrn_structural) + abs(jrn_feature) + 1e-8)

        decomposition[(ji, task_name)] = {
            "normal_jrn": normal_jrn,
            "shuffled_jrn_mean": shuffled_jrn_mean,
            "jrn_structural": jrn_structural,
            "jrn_feature": jrn_feature,
            "structural_fraction": structural_fraction,
        }

    # ── STEP 7 cont: Statistical tests ──────────────────────────────────
    normal_jrns_list = []
    shuffled_jrns_list = []
    for key in sorted(decomposition.keys()):
        normal_jrns_list.append(decomposition[key]["normal_jrn"])
        shuffled_jrns_list.append(decomposition[key]["shuffled_jrn_mean"])

    normal_arr = np.array(normal_jrns_list)
    shuffled_arr = np.array(shuffled_jrns_list)

    if len(normal_arr) >= 3:
        t_stat, p_ttest = stats.ttest_rel(normal_arr, shuffled_arr)
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(normal_arr - shuffled_arr)
        except ValueError:
            w_stat, p_wilcoxon = float("nan"), float("nan")

        diffs = normal_arr - shuffled_arr
        cohens_d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-8))
    else:
        t_stat = p_ttest = w_stat = p_wilcoxon = cohens_d = float("nan")

    structural_dominant_fraction = (
        sum(1 for d in decomposition.values() if d["structural_fraction"] > 0.5)
        / max(len(decomposition), 1)
    )

    logger.info(f"Statistical tests: t={t_stat:.3f}, p_ttest={p_ttest:.4f}, "
                f"p_wilcoxon={p_wilcoxon:.4f}, Cohen's d={cohens_d:.3f}")
    logger.info(f"Structural dominant fraction: {structural_dominant_fraction:.3f}")

    # ── STEP 8: Correlation analysis ─────────────────────────────────────
    per_join_structural = defaultdict(list)
    for (ji, task_name), dec in decomposition.items():
        per_join_structural[ji].append(dec["jrn_structural"])

    avg_structural_jrns = []
    fanout_means = []
    fanout_maxes = []
    cardinality_ratios = []
    join_coverages = []

    for ji, fk in enumerate(fk_joins):
        if ji in per_join_structural:
            avg_s = float(np.mean(per_join_structural[ji]))
            avg_structural_jrns.append(avg_s)
            fanout_means.append(fk.get("fanout_mean", 0))
            fanout_maxes.append(fk.get("fanout_max", 0))
            cardinality_ratios.append(fk.get("cardinality_ratio", 0))
            join_coverages.append(fk.get("join_coverage", 0))

    correlation_analysis = {}
    if len(avg_structural_jrns) >= 3:
        for label, vals in [
            ("structural_vs_fanout_mean", fanout_means),
            ("structural_vs_fanout_max", fanout_maxes),
            ("structural_vs_cardinality", cardinality_ratios),
            ("structural_vs_coverage", join_coverages),
        ]:
            try:
                rho, p = stats.spearmanr(avg_structural_jrns, vals)
                correlation_analysis[label] = {"rho": float(rho), "p": float(p)}
            except Exception:
                correlation_analysis[label] = {"rho": float("nan"), "p": float("nan")}

    # ── STEP 9: Per-join results ─────────────────────────────────────────
    per_join_results = []
    jrn_matrix_normal = []
    jrn_matrix_shuffled = []

    for ji, fk in enumerate(fk_joins):
        per_task = {}
        normal_jrns_join = []
        shuffled_jrns_join = []

        for task_name in TASKS:
            key = (ji, task_name)
            if key in decomposition:
                dec = decomposition[key]
                short_task = task_name.split("/")[-1]
                per_task[short_task] = {
                    "normal_jrn": round(dec["normal_jrn"], 4),
                    "shuffled_jrn": round(dec["shuffled_jrn_mean"], 4),
                    "jrn_structural": round(dec["jrn_structural"], 4),
                    "jrn_feature": round(dec["jrn_feature"], 4),
                    "structural_fraction": round(dec["structural_fraction"], 4),
                }
                normal_jrns_join.append(dec["normal_jrn"])
                shuffled_jrns_join.append(dec["shuffled_jrn_mean"])
            else:
                normal_jrns_join.append(None)
                shuffled_jrns_join.append(None)

        valid_normal = [v for v in normal_jrns_join if v is not None]
        valid_shuffled = [v for v in shuffled_jrns_join if v is not None]

        avg_normal = float(np.mean(valid_normal)) if valid_normal else None
        avg_shuffled = float(np.mean(valid_shuffled)) if valid_shuffled else None
        avg_structural = (avg_normal - avg_shuffled) if (avg_normal is not None and avg_shuffled is not None) else None
        avg_feature = (avg_shuffled - 1.0) if avg_shuffled is not None else None

        # Interpretation
        interp = _interpret_join(ji, fk, avg_structural, avg_feature, avg_normal)

        per_join_results.append({
            "join_idx": ji,
            "source_table": fk["source_table"],
            "target_table": fk["target_table"],
            "source_fk_col": fk["source_fk_col"],
            "fanout_mean": fk.get("fanout_mean"),
            "fanout_max": fk.get("fanout_max"),
            "per_task": per_task,
            "avg_normal_jrn": round(avg_normal, 4) if avg_normal is not None else None,
            "avg_shuffled_jrn": round(avg_shuffled, 4) if avg_shuffled is not None else None,
            "avg_structural": round(avg_structural, 4) if avg_structural is not None else None,
            "avg_feature": round(avg_feature, 4) if avg_feature is not None else None,
            "interpretation": interp,
        })

        jrn_matrix_normal.append([round(v, 4) if v is not None else None for v in normal_jrns_join])
        jrn_matrix_shuffled.append([round(v, 4) if v is not None else None for v in shuffled_jrns_join])

    # ── STEP 10: Conclusion ──────────────────────────────────────────────
    mean_struct = float(np.mean(normal_arr - shuffled_arr))
    mean_feat = float(np.mean(shuffled_arr - 1.0))

    if not math.isnan(p_ttest) and p_ttest < 0.05:
        if structural_dominant_fraction > 0.5:
            conclusion = (
                f"Structural component DOMINATES JRN signal (paired t-test p={p_ttest:.6f}, "
                f"Cohen's d={cohens_d:.2f}, structural dominant in {structural_dominant_fraction*100:.0f}% of pairs). "
                f"Normal JRN is significantly higher than shuffled JRN, confirming that JRN captures "
                f"genuine relational join structure beyond child-table feature quality. "
                f"Mean structural JRN={mean_struct:.4f}, mean feature JRN={mean_feat:.4f}."
            )
        else:
            conclusion = (
                f"Structural component is STATISTICALLY SIGNIFICANT but feature component is larger in magnitude "
                f"(paired t-test p={p_ttest:.6f}, Cohen's d={cohens_d:.2f}, structural dominant in only "
                f"{structural_dominant_fraction*100:.0f}% of pairs). Normal JRN is reliably higher than shuffled JRN "
                f"(mean structural={mean_struct:.4f}), confirming genuine structural signal exists. However, "
                f"the feature component (mean={mean_feat:.4f}) — improvement from child-table features regardless "
                f"of FK correctness — is larger for most joins. JRN captures BOTH structural and feature components, "
                f"with the structural component being a significant but secondary contributor."
            )
    elif not math.isnan(p_ttest):
        conclusion = (
            f"Structural component does NOT significantly differ from zero (paired t-test p={p_ttest:.4f}). "
            f"JRN may primarily reflect feature quality rather than join structure. "
            f"Mean structural={mean_struct:.4f}, mean feature={mean_feat:.4f}."
        )
    else:
        conclusion = "Insufficient data for statistical tests."

    # ── Build output ─────────────────────────────────────────────────────
    output_results = {
        "summary_statistics": {
            "n_joins": len(fk_joins),
            "n_tasks": len(task_data),
            "n_valid_pairs": len(decomposition),
            "n_seeds": len(SEEDS),
            "n_shuffles": N_SHUFFLES,
            "paired_ttest": {
                "t_stat": round(t_stat, 4) if not math.isnan(t_stat) else None,
                "p_value": round(p_ttest, 6) if not math.isnan(p_ttest) else None,
            },
            "wilcoxon_test": {
                "w_stat": round(w_stat, 4) if not math.isnan(w_stat) else None,
                "p_value": round(p_wilcoxon, 6) if not math.isnan(p_wilcoxon) else None,
            },
            "cohens_d": round(cohens_d, 4) if not math.isnan(cohens_d) else None,
            "mean_normal_jrn": round(float(np.mean(normal_arr)), 4),
            "mean_shuffled_jrn": round(float(np.mean(shuffled_arr)), 4),
            "mean_structural_jrn": round(float(np.mean(normal_arr - shuffled_arr)), 4),
            "mean_feature_jrn": round(float(np.mean(shuffled_arr - 1.0)), 4),
            "structural_dominant_fraction": round(structural_dominant_fraction, 4),
        },
        "baseline_performance": {
            task_name: {
                "mean": round(baseline_mean[task_name], 4),
                "std": round(baseline_std[task_name], 4),
                "metric": "AUROC" if TASK_TYPES[task_name] == "binary_classification" else "1/MAE",
            }
            for task_name in task_data
        },
        "per_join_results": per_join_results,
        "jrn_matrix_normal": jrn_matrix_normal,
        "jrn_matrix_shuffled": jrn_matrix_shuffled,
        "correlation_analysis": correlation_analysis,
        "conclusion": conclusion,
    }

    # ── Build exp_gen_sol_out format ──────────────────────────────────────
    # Create examples: one per (join, task) pair with JRN decomposition
    output_examples = []
    for ji, fk in enumerate(fk_joins):
        for task_name in TASKS:
            key = (ji, task_name)
            short_task = task_name.split("/")[-1]

            input_str = json.dumps({
                "join_idx": ji,
                "source_table": fk["source_table"],
                "target_table": fk["target_table"],
                "source_fk_col": fk["source_fk_col"],
                "target_pk_col": fk["target_pk_col"],
                "task": short_task,
                "fanout_mean": fk.get("fanout_mean"),
                "fanout_max": fk.get("fanout_max"),
            })

            if key in decomposition:
                dec = decomposition[key]
                output_str = json.dumps({
                    "normal_jrn": round(dec["normal_jrn"], 4),
                    "shuffled_jrn_mean": round(dec["shuffled_jrn_mean"], 4),
                    "jrn_structural": round(dec["jrn_structural"], 4),
                    "jrn_feature": round(dec["jrn_feature"], 4),
                    "structural_fraction": round(dec["structural_fraction"], 4),
                })

                example = {
                    "input": input_str,
                    "output": output_str,
                    "metadata_join_idx": ji,
                    "metadata_task_name": task_name,
                    "metadata_source_table": fk["source_table"],
                    "metadata_target_table": fk["target_table"],
                    "predict_normal_jrn": str(round(dec["normal_jrn"], 4)),
                    "predict_shuffled_jrn": str(round(dec["shuffled_jrn_mean"], 4)),
                    "predict_structural_fraction": str(round(dec["structural_fraction"], 4)),
                }
            else:
                output_str = json.dumps({"status": "skipped", "reason": "empty_features"})
                example = {
                    "input": input_str,
                    "output": output_str,
                    "metadata_join_idx": ji,
                    "metadata_task_name": task_name,
                    "metadata_source_table": fk["source_table"],
                    "metadata_target_table": fk["target_table"],
                    "predict_normal_jrn": "N/A",
                    "predict_shuffled_jrn": "N/A",
                    "predict_structural_fraction": "N/A",
                }

            output_examples.append(example)

    method_out = {
        "metadata": {
            "title": "JRN FK-Shuffling Confound Control on rel-f1",
            "description": (
                "Tests whether JRN captures genuine relational join structure or merely reflects "
                "child-table feature quality. For each of 13 FK joins x 3 driver tasks in the "
                "rel-f1 dataset, compute normal JRN via LightGBM probes, then compute shuffled "
                "JRN (FK column permuted to break true parent-child associations). Decompose "
                "JRN into structural component (destroyed by shuffling) and feature component "
                "(survives shuffling). Statistical tests confirm whether the structural component "
                "dominates."
            ),
            "method": "JRN FK-Shuffling Confound Control",
            "dataset": "rel-f1",
            "n_joins": len(fk_joins),
            "n_tasks": len(task_data),
            "n_seeds": len(SEEDS),
            "n_shuffles": N_SHUFFLES,
            "model": "LightGBM",
            "results": output_results,
        },
        "datasets": [
            {
                "dataset": "rel-f1",
                "examples": output_examples,
            }
        ],
    }

    # ── Write output ─────────────────────────────────────────────────────
    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(method_out, indent=2, default=str))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output: {output_path} ({file_size_mb:.1f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.1f}s")
    logger.info(f"Conclusion: {conclusion}")


def _interpret_join(
    ji: int,
    fk: Dict[str, Any],
    avg_structural: Optional[float],
    avg_feature: Optional[float],
    avg_normal: Optional[float],
) -> str:
    """Generate a domain interpretation for a join's decomposition."""
    src = fk["source_table"]
    tgt = fk["target_table"]

    if avg_structural is None:
        return f"Join #{ji} ({src}->{tgt}): Could not compute features."

    if avg_structural > 0.05:
        level = "HIGH"
    elif avg_structural > 0.01:
        level = "MODERATE"
    elif avg_structural > -0.01:
        level = "LOW"
    else:
        level = "NEGATIVE"

    # Domain-specific interpretations
    domain_notes = {
        ("results", "drivers"): "Race results are highly driver-specific; position/points directly tied to individual driver performance.",
        ("standings", "drivers"): "Championship standings are driver-specific accumulations; strong structural signal expected.",
        ("qualifying", "drivers"): "Qualifying positions directly reflect individual driver pace.",
        ("results", "races"): "Race features (circuit, weather) affect all drivers equally; moderate structural signal.",
        ("standings", "races"): "Race context for standings provides temporal signal about championship progression.",
        ("qualifying", "races"): "Race context for qualifying provides circuit-specific qualifying patterns.",
        ("qualifying", "constructors"): "Constructor affects qualifying through car performance; moderate structural signal.",
        ("results", "constructors"): "Constructor affects results through car competitiveness; structural but shared across team drivers.",
        ("races", "circuits"): "Circuit characteristics affect race outcomes; indirect driver connection via bridge table.",
        ("constructor_standings", "races"): "Constructor standings by race; indirect connection to drivers.",
        ("constructor_standings", "constructors"): "Constructor standings for constructor; indirect driver connection.",
        ("constructor_results", "races"): "Constructor race results; indirect driver connection.",
        ("constructor_results", "constructors"): "Constructor-level results; indirect driver connection.",
    }

    note = domain_notes.get((src, tgt), f"{src}->{tgt} join.")

    return (
        f"{level} structural component (structural={avg_structural:.4f}, feature={avg_feature:.4f}, "
        f"normal_jrn={avg_normal:.4f}). {note}"
    )


if __name__ == "__main__":
    main()
