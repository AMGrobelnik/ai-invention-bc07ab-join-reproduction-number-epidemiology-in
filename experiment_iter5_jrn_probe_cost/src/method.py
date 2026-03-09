#!/usr/bin/env python3
"""JRN Probe Cost-Efficiency & Ranking Stability on rel-f1.

Measures:
  (A) JRN ranking stability across probe budgets (LightGBM hyperparams x data subsample)
  (B) Wall-clock cost comparison of JRN probes vs greedy forward selection,
      exhaustive search, and random search for join subset selection.
"""

import gc
import itertools
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

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
TOTAL_RAM_GB = _container_ram_gb() or 16.0
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.70 * 1e9)  # 70% of container RAM

# Set memory limit
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError):
    pass

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DATA_FILE = DATA_DIR / "full_data_out.json"

TASKS = {
    "driver-dnf":      {"type": "binary_classification", "metric": "auroc"},
    "driver-top3":     {"type": "binary_classification", "metric": "auroc"},
    "driver-position": {"type": "regression",            "metric": "mae"},
}

VAL_TIMESTAMP = pd.Timestamp("2005-01-01")
TEST_TIMESTAMP = pd.Timestamp("2010-01-01")

FULL_BUDGET = {"n_estimators": 100, "max_depth": 5, "subsample_frac": 1.0}
SEEDS = [42, 123, 456]

# ── Data loading ─────────────────────────────────────────────────────────────
def load_data_from_json(data_file: Path) -> tuple:
    """Load tables, FK joins, and task data from full_data_out.json."""
    logger.info(f"Loading data from {data_file} ({data_file.stat().st_size / 1e6:.1f} MB)")
    raw = json.loads(data_file.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples")

    # ── Reconstruct tables ───────────────────────────────────────────────
    table_rows = defaultdict(list)
    table_meta = {}
    for ex in examples:
        if ex.get("metadata_row_type") != "table_row":
            continue
        tname = ex["metadata_table_name"]
        row = json.loads(ex["input"])
        # Add primary key from output (format: "tablename/pk_value")
        pk_col = ex["metadata_primary_key_col"]
        pk_val = ex["metadata_primary_key_value"]
        try:
            pk_val = int(pk_val)
        except (ValueError, TypeError):
            pass
        row[pk_col] = pk_val
        table_rows[tname].append(row)
        if tname not in table_meta:
            fkeys = json.loads(ex.get("metadata_foreign_keys_json", "{}"))
            table_meta[tname] = {
                "pkey_col": pk_col,
                "fkey_col_to_pkey_table": fkeys,
                "time_col": ex.get("metadata_time_col"),
            }

    tables = {}
    for tname, rows in table_rows.items():
        df = pd.DataFrame(rows)
        # Parse time column
        tc = table_meta[tname]["time_col"]
        if tc and tc in df.columns:
            df[tc] = pd.to_datetime(df[tc], errors="coerce")
        # Convert numeric-looking columns
        for c in df.columns:
            if c == tc:
                continue
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
        tables[tname] = {
            "df": df,
            "pkey_col": table_meta[tname]["pkey_col"],
            "fkey_col_to_pkey_table": table_meta[tname]["fkey_col_to_pkey_table"],
            "time_col": tc,
        }
        logger.info(f"  Table '{tname}': {df.shape[0]} rows, {df.shape[1]} cols, pkey={table_meta[tname]['pkey_col']}, time={tc}")

    # ── Reconstruct FK joins ─────────────────────────────────────────────
    fk_joins = []
    for ex in examples:
        if ex.get("metadata_row_type") != "fk_join_metadata":
            continue
        src = ex["metadata_source_table"]
        tgt = ex["metadata_target_table"]
        fk_col = ex["metadata_source_fk_col"]
        tgt_pk = tables[tgt]["pkey_col"]
        fk_joins.append((src, fk_col, tgt, tgt_pk))
    logger.info(f"Extracted {len(fk_joins)} FK joins")
    for i, (s, f, t, p) in enumerate(fk_joins):
        logger.debug(f"  Join {i}: {s}.{f} -> {t}.{p}")

    # ── Reconstruct task data ────────────────────────────────────────────
    task_data = {}
    for ex in examples:
        if ex.get("metadata_row_type") != "task_sample":
            continue
        # Map full task name to short name
        full_name = ex["metadata_task_name"]  # e.g. "rel-f1/driver-dnf"
        short_name = full_name.split("/")[-1]  # "driver-dnf"
        if short_name not in TASKS:
            continue

        if short_name not in task_data:
            task_data[short_name] = {"train": [], "val": [], "test": [],
                                     "target_col": ex.get("metadata_target_col", "label"),
                                     "entity_col": ex.get("metadata_entity_col", "driverId"),
                                     "task_type": ex.get("metadata_task_type", TASKS[short_name]["type"])}

        inp = json.loads(ex["input"])
        label = ex["output"]
        try:
            label = float(label)
        except (ValueError, TypeError):
            pass
        fold = ex.get("metadata_fold_name", "train")
        task_data[short_name][fold].append({**inp, "__label__": label})

    for tname, tinfo in task_data.items():
        for fold in ["train", "val", "test"]:
            n = len(tinfo[fold])
            logger.info(f"  Task '{tname}' {fold}: {n} samples")

    # Free raw data
    del examples, raw
    gc.collect()

    return tables, fk_joins, task_data


# ── Feature engineering ──────────────────────────────────────────────────────
def get_numeric_cols(df: pd.DataFrame, exclude_cols: set) -> list:
    """Get numeric columns excluding specified cols."""
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols]


def build_split_features(tables: dict, fk_joins: list,
                         samples_df: pd.DataFrame, entity_col: str,
                         time_col: str, cutoff_ts: pd.Timestamp) -> tuple:
    """Build feature matrix using split-level temporal cutoff.

    For each entity, aggregate all source rows before cutoff_ts.
    All samples in this split share the same features per entity.
    Returns: (base_features, {join_idx: features_array})
    """
    n_samples = len(samples_df)
    results_df = tables["results"]["df"].copy()
    drivers_df = tables["drivers"]["df"]
    driver_pkey = tables["drivers"]["pkey_col"]

    # ── Base features: driver table's own numeric features ────────────
    driver_exclude = {driver_pkey} | set(tables["drivers"]["fkey_col_to_pkey_table"].keys())
    tc = tables["drivers"]["time_col"]
    if tc:
        driver_exclude.add(tc)
    driver_numeric = get_numeric_cols(drivers_df, driver_exclude)

    if driver_numeric:
        base_feat = samples_df.merge(
            drivers_df[[driver_pkey] + driver_numeric],
            left_on=entity_col, right_on=driver_pkey, how="left"
        )[driver_numeric].fillna(0).values
    else:
        base_feat = np.zeros((n_samples, 1))

    logger.debug(f"Base features shape: {base_feat.shape}")

    # ── Join features ────────────────────────────────────────────────────
    join_features = {}
    for j_idx, (src, fk_col, tgt, pk_col) in enumerate(fk_joins):
        src_df = tables[src]["df"].copy()
        src_time = tables[src]["time_col"]

        # Filter source by temporal cutoff
        if src_time and src_time in src_df.columns:
            src_df = src_df[src_df[src_time] < cutoff_ts].copy()

        # Determine numeric feature columns from source
        exclude = {tables[src]["pkey_col"]} | set(tables[src]["fkey_col_to_pkey_table"].keys())
        if src_time and src_time in src_df.columns:
            exclude.add(src_time)
        src_numeric = get_numeric_cols(src_df, exclude)

        if len(src_numeric) == 0 or len(src_df) == 0:
            join_features[j_idx] = np.zeros((n_samples, 1))
            continue

        # ROUTING: get features to entity-level (driverId)
        if entity_col in src_df.columns:
            # Source table directly has driverId
            if fk_col == entity_col:
                # Direct join to entity (e.g., results.driverId->drivers)
                feat_cols = src_numeric
                agg = src_df.groupby(entity_col)[feat_cols].mean()
            else:
                # FK points elsewhere (e.g., results.raceId->races)
                tgt_df = tables[tgt]["df"]
                tgt_exclude = {pk_col} | set(tables[tgt]["fkey_col_to_pkey_table"].keys())
                tgt_tc = tables[tgt]["time_col"]
                if tgt_tc:
                    tgt_exclude.add(tgt_tc)
                tgt_numeric = get_numeric_cols(tgt_df, tgt_exclude)

                if tgt_numeric:
                    renamed = {c: f"j{j_idx}_{c}" for c in tgt_numeric}
                    tgt_sub = tgt_df[[pk_col] + tgt_numeric].copy()
                    tgt_sub.rename(columns=renamed, inplace=True)
                    src_df = src_df.merge(tgt_sub, left_on=fk_col, right_on=pk_col, how="left")
                    feat_cols = list(renamed.values())
                else:
                    feat_cols = src_numeric

                agg = src_df.groupby(entity_col)[feat_cols].mean()
        else:
            # Source doesn't have driverId — route through results table
            agg_src = src_df.groupby(fk_col)[src_numeric].mean().reset_index()
            renamed = {c: f"j{j_idx}_{c}" for c in src_numeric}
            agg_src.rename(columns=renamed, inplace=True)
            feat_cols = list(renamed.values())

            # Find merge column to results
            res_df_filtered = results_df.copy()
            res_time = tables["results"]["time_col"]
            if res_time and res_time in res_df_filtered.columns:
                res_df_filtered = res_df_filtered[res_df_filtered[res_time] < cutoff_ts].copy()

            if fk_col in res_df_filtered.columns:
                merge_col = fk_col
            elif pk_col in res_df_filtered.columns:
                merge_col = pk_col
            else:
                join_features[j_idx] = np.zeros((n_samples, 1))
                continue

            enriched = res_df_filtered.merge(agg_src, on=merge_col, how="left")
            if entity_col not in enriched.columns:
                join_features[j_idx] = np.zeros((n_samples, 1))
                continue
            agg = enriched.groupby(entity_col)[feat_cols].mean()

        # Map features to samples
        feat_matrix = samples_df[[entity_col]].merge(
            agg, left_on=entity_col, right_index=True, how="left"
        ).drop(columns=[entity_col]).fillna(0).values

        if feat_matrix.shape[1] == 0:
            feat_matrix = np.zeros((n_samples, 1))

        join_features[j_idx] = feat_matrix
        logger.debug(f"  Join {j_idx} ({src}.{fk_col}->{tgt}): {feat_matrix.shape[1]} features")

    return base_feat, join_features


# ── Model training & evaluation ─────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_val, y_val, task_type,
                       n_estimators=500, max_depth=8, seed=42,
                       n_jobs=1):
    """Train GradientBoosting and return (performance, wall_time)."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    params = dict(n_estimators=n_estimators, max_depth=max_depth,
                  random_state=seed, min_samples_leaf=5,
                  subsample=0.8, learning_rate=0.1)

    t0 = time.time()
    if task_type == "binary_classification":
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        perf = roc_auc_score(y_val, proba)
    else:
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        perf = 1.0 / (mae + 1e-8)  # Invert so higher=better
    elapsed = time.time() - t0
    return perf, elapsed


def compute_jrn(base_X_tr, join_X_tr, y_tr, base_X_val, join_X_val, y_val,
                task_type, n_estimators=500, max_depth=8, seed=42,
                subsample_frac=1.0):
    """Compute JRN = perf(base+join) / perf(base) for one join.

    Returns: (jrn, perf_base, perf_aug, time_base, time_aug)
    """
    # Subsample training data
    n = len(y_tr)
    if subsample_frac < 1.0:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max(int(n * subsample_frac), 10), replace=False)
        bX, jX, ys = base_X_tr[idx], join_X_tr[idx], y_tr[idx]
    else:
        bX, jX, ys = base_X_tr, join_X_tr, y_tr

    # Ensure valid labels for classification
    if task_type == "binary_classification":
        unique = np.unique(ys)
        if len(unique) < 2:
            return 1.0, 0.5, 0.5, 0.0, 0.0

    X_base_tr = bX
    X_aug_tr = np.hstack([bX, jX])
    X_base_val = base_X_val
    X_aug_val = np.hstack([base_X_val, join_X_val])

    perf_base, t_base = train_and_evaluate(
        X_base_tr, ys, X_base_val, y_val, task_type,
        n_estimators=n_estimators, max_depth=max_depth, seed=seed, n_jobs=1)

    perf_aug, t_aug = train_and_evaluate(
        X_aug_tr, ys, X_aug_val, y_val, task_type,
        n_estimators=n_estimators, max_depth=max_depth, seed=seed, n_jobs=1)

    jrn = perf_aug / max(perf_base, 1e-8)
    return jrn, perf_base, perf_aug, t_base, t_aug


# ── Parallel JRN computation ────────────────────────────────────────────────
def _compute_jrn_worker(args):
    """Worker for parallel JRN computation."""
    (base_X_tr, join_X_tr, y_tr, base_X_val, join_X_val, y_val,
     task_type, n_estimators, max_depth, seed, subsample_frac, j_idx) = args
    try:
        jrn, pb, pa, tb, ta = compute_jrn(
            base_X_tr, join_X_tr, y_tr, base_X_val, join_X_val, y_val,
            task_type, n_estimators, max_depth, seed, subsample_frac)
        return j_idx, jrn, pb, pa, tb, ta
    except Exception as e:
        logger.warning(f"JRN worker failed for join {j_idx}: {e}")
        return j_idx, 1.0, 0.5, 0.5, 0.0, 0.0


# ── Main ─────────────────────────────────────────────────────────────────────
@logger.catch
def main():
    t_start = time.time()

    # ── Step 1: Load data ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data from JSON dependency")
    logger.info("=" * 60)
    tables, fk_joins, task_data = load_data_from_json(DATA_FILE)
    N_JOINS = len(fk_joins)
    logger.info(f"N_JOINS = {N_JOINS}, Tasks = {list(task_data.keys())}")

    # ── Step 2: Build features per task ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Feature engineering (split-level cutoffs)")
    logger.info("=" * 60)

    task_features = {}
    for task_name in TASKS:
        if task_name not in task_data:
            logger.warning(f"Task '{task_name}' not found in data, skipping")
            continue

        td = task_data[task_name]
        entity_col = td["entity_col"]
        task_type = td["task_type"]

        # Build train/val DataFrames
        train_rows = td["train"]
        val_rows = td["val"]
        if len(train_rows) == 0 or len(val_rows) == 0:
            logger.warning(f"Task '{task_name}' has empty train/val, skipping")
            continue

        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows)

        # Parse dates
        time_col = "date"
        for df in [train_df, val_df]:
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        y_train = train_df["__label__"].values.astype(float)
        y_val = val_df["__label__"].values.astype(float)

        # For binary classification, ensure 0/1
        if task_type == "binary_classification":
            y_train = y_train.astype(int)
            y_val = y_val.astype(int)

        logger.info(f"Task '{task_name}': {len(y_train)} train, {len(y_val)} val samples")

        # Build features with split-level cutoffs
        # Train features: use data before val_timestamp (2005-01-01)
        base_X_tr, join_X_tr = build_split_features(
            tables, fk_joins, train_df, entity_col, time_col, VAL_TIMESTAMP)
        # Val features: use data before test_timestamp (2010-01-01)
        base_X_vl, join_X_vl = build_split_features(
            tables, fk_joins, val_df, entity_col, time_col, TEST_TIMESTAMP)

        logger.info(f"  Base features: train={base_X_tr.shape}, val={base_X_vl.shape}")
        logger.info(f"  Join feature blocks: {N_JOINS} joins")

        task_features[task_name] = {
            "base_X_tr": base_X_tr, "base_X_vl": base_X_vl,
            "join_X_tr": join_X_tr, "join_X_vl": join_X_vl,
            "y_train": y_train, "y_val": y_val,
            "task_type": task_type,
        }

    active_tasks = list(task_features.keys())
    logger.info(f"Active tasks: {active_tasks}")

    # ── Step 3: Full-budget reference JRN ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Computing full-budget reference JRN (3 seeds)")
    logger.info("=" * 60)

    t_ref_start = time.time()
    ref_jrn = {}           # {task: np.array(N_JOINS,)}
    ref_jrn_details = {}   # {task: list of dicts per join}

    for task_name in active_tasks:
        tf = task_features[task_name]
        jrn_per_join = []
        details_per_join = []

        for j_idx in range(N_JOINS):
            jrn_vals, pb_vals, pa_vals = [], [], []
            for seed in SEEDS:
                jrn, pb, pa, tb, ta = compute_jrn(
                    tf["base_X_tr"], tf["join_X_tr"][j_idx],
                    tf["y_train"],
                    tf["base_X_vl"], tf["join_X_vl"][j_idx],
                    tf["y_val"],
                    tf["task_type"],
                    seed=seed, **FULL_BUDGET)
                jrn_vals.append(jrn)
                pb_vals.append(pb)
                pa_vals.append(pa)

            mean_jrn = float(np.mean(jrn_vals))
            jrn_per_join.append(mean_jrn)
            details_per_join.append({
                "join_idx": j_idx,
                "source": fk_joins[j_idx][0],
                "fk_col": fk_joins[j_idx][1],
                "target": fk_joins[j_idx][2],
                "jrn_mean": mean_jrn,
                "jrn_std": float(np.std(jrn_vals)),
                "jrn_values": [float(v) for v in jrn_vals],
                "perf_base_mean": float(np.mean(pb_vals)),
                "perf_aug_mean": float(np.mean(pa_vals)),
            })
            logger.info(f"  {task_name} join {j_idx} ({fk_joins[j_idx][0]}.{fk_joins[j_idx][1]}->{fk_joins[j_idx][2]}): "
                        f"JRN={mean_jrn:.4f} (std={np.std(jrn_vals):.4f})")

        ref_jrn[task_name] = np.array(jrn_per_join)
        ref_jrn_details[task_name] = details_per_join

    t_ref = time.time() - t_ref_start
    logger.info(f"Reference JRN computed in {t_ref:.1f}s")

    # ── Step 4: Part A — Convergence Analysis ────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Part A — Convergence analysis (budget sweep)")
    logger.info("=" * 60)

    # Adaptive budget grid based on remaining time
    elapsed_so_far = time.time() - t_start
    remaining_time = 3300 - elapsed_so_far  # ~55 min max
    logger.info(f"Elapsed so far: {elapsed_so_far:.0f}s, remaining budget: {remaining_time:.0f}s")

    # Estimate time per JRN call from reference computation
    n_ref_calls = len(active_tasks) * N_JOINS * len(SEEDS)
    time_per_jrn_call = t_ref / max(n_ref_calls, 1)
    logger.info(f"Estimated time per JRN call: {time_per_jrn_call:.2f}s")

    # Budget grid — adapt based on available time
    # Full grid: 4*2*2*4 = 64 combos. Each combo = N_JOINS * n_tasks * n_seeds calls
    BUDGET_GRID = {
        "n_estimators": [25, 50, 100, 200],
        "max_depth": [3, 6],
        "n_seeds": [1, 3],
        "subsample_frac": [0.1, 0.3, 0.5, 1.0],
    }
    all_combos = list(itertools.product(
        BUDGET_GRID["n_estimators"], BUDGET_GRID["max_depth"],
        BUDGET_GRID["n_seeds"], BUDGET_GRID["subsample_frac"]))

    # Estimate total time for full grid
    total_calls = sum(ns * N_JOINS * len(active_tasks) for _, _, ns, _ in all_combos)
    # Budget configs use cheaper models (fewer trees, lower depth), estimate 30% of full
    est_time = total_calls * time_per_jrn_call * 0.3
    logger.info(f"Estimated convergence sweep time: {est_time:.0f}s for {len(all_combos)} configs, {total_calls} calls")

    # If estimated time exceeds budget, reduce grid
    time_for_convergence = remaining_time * 0.55  # allocate 55% for convergence
    if est_time > time_for_convergence:
        # Reduce grid
        BUDGET_GRID = {
            "n_estimators": [50, 200],
            "max_depth": [3, 6],
            "n_seeds": [1, 3],
            "subsample_frac": [0.1, 0.3, 1.0],
        }
        all_combos = list(itertools.product(
            BUDGET_GRID["n_estimators"], BUDGET_GRID["max_depth"],
            BUDGET_GRID["n_seeds"], BUDGET_GRID["subsample_frac"]))
        total_calls = sum(ns * N_JOINS * len(active_tasks) for _, _, ns, _ in all_combos)
        est_time = total_calls * time_per_jrn_call * 0.3
        logger.info(f"Reduced grid: {len(all_combos)} configs, est {est_time:.0f}s")

    if est_time > time_for_convergence:
        # Further reduce
        BUDGET_GRID = {
            "n_estimators": [50, 200],
            "max_depth": [3, 6],
            "n_seeds": [1],
            "subsample_frac": [0.3, 1.0],
        }
        all_combos = list(itertools.product(
            BUDGET_GRID["n_estimators"], BUDGET_GRID["max_depth"],
            BUDGET_GRID["n_seeds"], BUDGET_GRID["subsample_frac"]))
        logger.info(f"Further reduced grid: {len(all_combos)} configs")

    convergence_results = []
    t_conv_start = time.time()

    for combo_idx, (ne, md, ns, sf) in enumerate(all_combos):
        budget_seeds = SEEDS[:ns]
        budget_jrn = {}
        budget_time = 0.0

        for task_name in active_tasks:
            tf = task_features[task_name]
            jrn_per_join = []

            for j_idx in range(N_JOINS):
                jrn_vals = []
                for seed in budget_seeds:
                    t0 = time.time()
                    jrn, _, _, _, _ = compute_jrn(
                        tf["base_X_tr"], tf["join_X_tr"][j_idx],
                        tf["y_train"],
                        tf["base_X_vl"], tf["join_X_vl"][j_idx],
                        tf["y_val"],
                        tf["task_type"],
                        n_estimators=ne, max_depth=md, seed=seed,
                        subsample_frac=sf)
                    budget_time += time.time() - t0
                    jrn_vals.append(jrn)
                jrn_per_join.append(float(np.mean(jrn_vals)))

            budget_jrn[task_name] = np.array(jrn_per_join)

        # Spearman and Kendall correlations vs reference
        rhos = {}
        taus = {}
        for task_name in active_tasks:
            rho, rho_p = spearmanr(ref_jrn[task_name], budget_jrn[task_name])
            tau, tau_p = kendalltau(ref_jrn[task_name], budget_jrn[task_name])
            if np.isnan(rho):
                rho = 0.0
            if np.isnan(tau):
                tau = 0.0
            rhos[task_name] = {"rho": float(rho), "pval": float(rho_p) if not np.isnan(rho_p) else 1.0}
            taus[task_name] = {"tau": float(tau), "pval": float(tau_p) if not np.isnan(tau_p) else 1.0}

        mean_rho = float(np.mean([r["rho"] for r in rhos.values()]))
        mean_tau = float(np.mean([t["tau"] for t in taus.values()]))

        convergence_results.append({
            "n_estimators": ne, "max_depth": md, "n_seeds": ns,
            "subsample_frac": sf, "wall_clock_seconds": round(budget_time, 3),
            "spearman_rho_per_task": rhos,
            "kendall_tau_per_task": taus,
            "mean_rho": round(mean_rho, 4),
            "mean_tau": round(mean_tau, 4),
            "budget_jrn": {t: budget_jrn[t].tolist() for t in active_tasks},
        })

        logger.info(f"  Config {combo_idx+1}/{len(all_combos)} "
                    f"(ne={ne}, md={md}, ns={ns}, sf={sf}): "
                    f"ρ={mean_rho:.3f}, τ={mean_tau:.3f}, time={budget_time:.1f}s")

        # Time check
        if time.time() - t_start > 2700:  # 45 min
            logger.warning("Time budget tight, stopping convergence sweep")
            break

    t_conv = time.time() - t_conv_start
    logger.info(f"Convergence sweep done: {len(convergence_results)} configs in {t_conv:.1f}s")

    # Find minimum-cost config with mean_rho > 0.9 (or 0.8, or best)
    passing_90 = [r for r in convergence_results if r["mean_rho"] > 0.9]
    passing_80 = [r for r in convergence_results if r["mean_rho"] > 0.8]
    passing_70 = [r for r in convergence_results if r["mean_rho"] > 0.7]

    if passing_90:
        min_cost_config = min(passing_90, key=lambda r: r["wall_clock_seconds"])
        threshold_used = 0.9
    elif passing_80:
        min_cost_config = min(passing_80, key=lambda r: r["wall_clock_seconds"])
        threshold_used = 0.8
    elif passing_70:
        min_cost_config = min(passing_70, key=lambda r: r["wall_clock_seconds"])
        threshold_used = 0.7
    else:
        min_cost_config = max(convergence_results, key=lambda r: r["mean_rho"]) if convergence_results else None
        threshold_used = 0.0

    if min_cost_config:
        logger.info(f"Best cheap config (ρ>{threshold_used}): "
                    f"ne={min_cost_config['n_estimators']}, md={min_cost_config['max_depth']}, "
                    f"ns={min_cost_config['n_seeds']}, sf={min_cost_config['subsample_frac']} "
                    f"→ ρ={min_cost_config['mean_rho']:.3f}, time={min_cost_config['wall_clock_seconds']:.1f}s")

    # ── Step 5: Part B — Cost Comparison ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Part B — Cost comparison (JRN vs greedy vs random vs exhaustive)")
    logger.info("=" * 60)

    cost_results = {}
    for task_name in active_tasks:
        logger.info(f"Cost comparison for task: {task_name}")
        tf = task_features[task_name]
        base_X_tr = tf["base_X_tr"]
        base_X_vl = tf["base_X_vl"]
        join_Xs_tr = tf["join_X_tr"]
        join_Xs_vl = tf["join_X_vl"]
        y_tr = tf["y_train"]
        y_vl = tf["y_val"]
        task_type = tf["task_type"]

        # ── (a) JRN Probe ────────────────────────────────────────────────
        t_jrn_start = time.time()
        # Train baseline once
        perf_base, _ = train_and_evaluate(
            base_X_tr, y_tr, base_X_vl, y_vl, task_type,
            **{k: v for k, v in FULL_BUDGET.items() if k != "subsample_frac"},
            seed=42)

        jrn_values = []
        jrn_perfs_aug = []
        for j in range(N_JOINS):
            X_aug_tr = np.hstack([base_X_tr, join_Xs_tr[j]])
            X_aug_vl = np.hstack([base_X_vl, join_Xs_vl[j]])
            perf_aug, _ = train_and_evaluate(
                X_aug_tr, y_tr, X_aug_vl, y_vl, task_type,
                n_estimators=FULL_BUDGET["n_estimators"],
                max_depth=FULL_BUDGET["max_depth"],
                seed=42)
            jrn_val = perf_aug / max(perf_base, 1e-8)
            jrn_values.append(jrn_val)
            jrn_perfs_aug.append(perf_aug)

        # JRN-guided selection: include joins with JRN > 1.0
        jrn_selected = [j for j in range(N_JOINS) if jrn_values[j] > 1.0]
        if len(jrn_selected) == 0:
            jrn_selected = [int(np.argmax(jrn_values))]

        X_jrn_tr = np.hstack([base_X_tr] + [join_Xs_tr[j] for j in jrn_selected])
        X_jrn_vl = np.hstack([base_X_vl] + [join_Xs_vl[j] for j in jrn_selected])
        perf_jrn, _ = train_and_evaluate(
            X_jrn_tr, y_tr, X_jrn_vl, y_vl, task_type,
            n_estimators=FULL_BUDGET["n_estimators"],
            max_depth=FULL_BUDGET["max_depth"],
            seed=42)
        t_jrn = time.time() - t_jrn_start
        logger.info(f"  JRN probe: {t_jrn:.1f}s, selected {len(jrn_selected)} joins, perf={perf_jrn:.4f}")

        # ── (b) Greedy Forward Selection ─────────────────────────────────
        t_greedy_start = time.time()
        selected = []
        remaining = list(range(N_JOINS))
        greedy_perfs = []
        greedy_order = []

        # Time check — limit greedy to avoid timeout
        max_greedy_steps = N_JOINS
        remaining_time_check = 3300 - (time.time() - t_start)
        if remaining_time_check < 600:
            max_greedy_steps = min(5, N_JOINS)
            logger.warning(f"Limited greedy to {max_greedy_steps} steps due to time")

        for step in range(min(max_greedy_steps, N_JOINS)):
            best_j, best_perf = None, -np.inf
            for j in remaining:
                candidate = selected + [j]
                X_tr = np.hstack([base_X_tr] + [join_Xs_tr[jj] for jj in candidate])
                X_vl = np.hstack([base_X_vl] + [join_Xs_vl[jj] for jj in candidate])
                perf, _ = train_and_evaluate(
                    X_tr, y_tr, X_vl, y_vl, task_type,
                    n_estimators=FULL_BUDGET["n_estimators"],
                    max_depth=FULL_BUDGET["max_depth"],
                    seed=42)
                if perf > best_perf:
                    best_perf, best_j = perf, j
            selected.append(best_j)
            remaining.remove(best_j)
            greedy_perfs.append(float(best_perf))
            greedy_order.append(best_j)
            logger.debug(f"    Greedy step {step+1}: added join {best_j}, perf={best_perf:.4f}")

            # Time check within greedy
            if time.time() - t_start > 3000:
                logger.warning("Stopping greedy early due to time")
                break

        t_greedy = time.time() - t_greedy_start
        perf_greedy = max(greedy_perfs) if greedy_perfs else perf_base
        greedy_best_k = int(np.argmax(greedy_perfs) + 1) if greedy_perfs else 0
        n_greedy_models = sum(range(len(remaining) + 1, N_JOINS + 1)) if len(selected) == N_JOINS else sum(N_JOINS - i for i in range(len(selected)))
        logger.info(f"  Greedy: {t_greedy:.1f}s, {len(selected)} steps, best_k={greedy_best_k}, perf={perf_greedy:.4f}")

        # ── (c) Exhaustive Search (sampled) ──────────────────────────────
        t_exhaust_start = time.time()
        n_subsets = min(50, 2**N_JOINS)
        rng = np.random.RandomState(42)
        exhaust_perfs = []

        for _ in range(n_subsets):
            k = rng.randint(1, N_JOINS + 1)
            subset = sorted(rng.choice(N_JOINS, k, replace=False).tolist())
            X_tr = np.hstack([base_X_tr] + [join_Xs_tr[j] for j in subset])
            X_vl = np.hstack([base_X_vl] + [join_Xs_vl[j] for j in subset])
            perf, _ = train_and_evaluate(
                X_tr, y_tr, X_vl, y_vl, task_type,
                n_estimators=FULL_BUDGET["n_estimators"],
                max_depth=FULL_BUDGET["max_depth"],
                seed=42)
            exhaust_perfs.append({"subset": subset, "perf": float(perf)})

            if time.time() - t_start > 3100:
                logger.warning("Stopping exhaustive early due to time")
                break

        t_exhaust_sample = time.time() - t_exhaust_start
        t_exhaust_full = t_exhaust_sample * (2**N_JOINS / max(len(exhaust_perfs), 1))
        perf_exhaust = max(e["perf"] for e in exhaust_perfs) if exhaust_perfs else perf_base
        best_exhaust_subset = max(exhaust_perfs, key=lambda x: x["perf"])["subset"] if exhaust_perfs else []
        logger.info(f"  Exhaustive ({len(exhaust_perfs)} subsets): {t_exhaust_sample:.1f}s, "
                    f"extrapolated full={t_exhaust_full:.1f}s, perf={perf_exhaust:.4f}")

        # ── (d) Random Search ────────────────────────────────────────────
        t_random_start = time.time()
        n_random = 25
        rng2 = np.random.RandomState(123)
        random_perfs = []

        for _ in range(n_random):
            k = rng2.randint(1, N_JOINS + 1)
            subset = sorted(rng2.choice(N_JOINS, k, replace=False).tolist())
            X_tr = np.hstack([base_X_tr] + [join_Xs_tr[j] for j in subset])
            X_vl = np.hstack([base_X_vl] + [join_Xs_vl[j] for j in subset])
            perf, _ = train_and_evaluate(
                X_tr, y_tr, X_vl, y_vl, task_type,
                n_estimators=FULL_BUDGET["n_estimators"],
                max_depth=FULL_BUDGET["max_depth"],
                seed=42)
            random_perfs.append({"subset": subset, "perf": float(perf)})

            if time.time() - t_start > 3200:
                logger.warning("Stopping random early due to time")
                break

        t_random = time.time() - t_random_start
        perf_random = max(r["perf"] for r in random_perfs) if random_perfs else perf_base
        logger.info(f"  Random ({len(random_perfs)} trials): {t_random:.1f}s, perf={perf_random:.4f}")

        # ── Record results ───────────────────────────────────────────────
        n_greedy_models_actual = sum(N_JOINS - i for i in range(len(selected)))
        cost_results[task_name] = {
            "jrn_time": round(t_jrn, 3),
            "greedy_time": round(t_greedy, 3),
            "exhaust_time_sampled": round(t_exhaust_sample, 3),
            "exhaust_time_extrapolated": round(t_exhaust_full, 3),
            "random_time": round(t_random, 3),
            "jrn_perf": round(float(perf_jrn), 6),
            "greedy_perf": round(float(perf_greedy), 6),
            "exhaust_perf": round(float(perf_exhaust), 6),
            "random_perf": round(float(perf_random), 6),
            "base_perf": round(float(perf_base), 6),
            "cost_ratio_jrn_vs_greedy": round(t_jrn / max(t_greedy, 1e-6), 4),
            "cost_ratio_jrn_vs_exhaust": round(t_jrn / max(t_exhaust_full, 1e-6), 6),
            "perf_per_second_jrn": round(float(perf_jrn) / max(t_jrn, 1e-6), 6),
            "perf_per_second_greedy": round(float(perf_greedy) / max(t_greedy, 1e-6), 6),
            "jrn_selected_joins": jrn_selected,
            "jrn_values": [round(float(v), 6) for v in jrn_values],
            "greedy_order": greedy_order,
            "greedy_perfs": greedy_perfs,
            "greedy_best_k": greedy_best_k,
            "best_exhaust_subset": best_exhaust_subset,
            "n_models_jrn": N_JOINS + 1,
            "n_models_greedy": n_greedy_models_actual,
            "n_models_exhaust_sampled": len(exhaust_perfs),
            "n_models_exhaust_full": 2**N_JOINS,
            "n_models_random": len(random_perfs),
        }

    # ── Step 6: Breakeven Analysis ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Breakeven analysis")
    logger.info("=" * 60)

    # Empirical per-model training time
    if cost_results:
        first_task = active_tasks[0]
        cr = cost_results[first_task]
        t_per_model = cr["greedy_time"] / max(cr["n_models_greedy"], 1)
    else:
        t_per_model = 1.0

    breakeven_analysis = {
        "jrn_models_per_J": "J+1",
        "greedy_models_per_J": "J*(J+1)/2",
        "exhaust_models_per_J": "2^J",
        "t_per_model_seconds": round(t_per_model, 4),
        "scaling_table": [],
    }

    for J in [3, 5, 8, 10, 13, 20, 30, 50]:
        jrn_models = J + 1
        greedy_models = J * (J + 1) // 2
        exhaust_models = 2**J
        random_models = 50

        breakeven_analysis["scaling_table"].append({
            "J": J,
            "jrn_models": jrn_models,
            "greedy_models": greedy_models,
            "exhaust_models": exhaust_models,
            "random_models": random_models,
            "speedup_jrn_vs_greedy": round(greedy_models / jrn_models, 2),
            "speedup_jrn_vs_exhaust": round(exhaust_models / jrn_models, 2),
            "est_jrn_time": round(jrn_models * t_per_model, 2),
            "est_greedy_time": round(greedy_models * t_per_model, 2),
        })

    # Performance comparison
    perf_comparison = {}
    for task_name in active_tasks:
        cr = cost_results[task_name]
        perf_comparison[task_name] = {
            "jrn_beats_random": cr["jrn_perf"] >= cr["random_perf"],
            "jrn_beats_greedy": cr["jrn_perf"] >= cr["greedy_perf"],
            "jrn_perf_vs_greedy_ratio": round(cr["jrn_perf"] / max(cr["greedy_perf"], 1e-8), 4),
            "jrn_cost_vs_greedy_ratio": round(cr["jrn_time"] / max(cr["greedy_time"], 1e-6), 4),
        }
    breakeven_analysis["perf_comparison"] = perf_comparison

    logger.info(f"Breakeven analysis: JRN is {91/14:.1f}x cheaper than greedy at J=13")

    # ── Step 7: Build output ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Building output")
    logger.info("=" * 60)

    # Summary statistics
    summary_stats = {}
    if cost_results:
        summary_stats = {
            "mean_jrn_time": round(float(np.mean([r["jrn_time"] for r in cost_results.values()])), 3),
            "mean_greedy_time": round(float(np.mean([r["greedy_time"] for r in cost_results.values()])), 3),
            "mean_random_time": round(float(np.mean([r["random_time"] for r in cost_results.values()])), 3),
            "mean_speedup_vs_greedy": round(float(np.mean([r["cost_ratio_jrn_vs_greedy"] for r in cost_results.values()])), 4),
            "jrn_wins_vs_random": sum(1 for t in active_tasks if cost_results[t]["jrn_perf"] >= cost_results[t]["random_perf"]),
            "jrn_wins_vs_greedy": sum(1 for t in active_tasks if cost_results[t]["jrn_perf"] >= cost_results[t]["greedy_perf"]),
            "n_tasks": len(active_tasks),
        }

    # Full-budget reference time
    full_budget_time = t_ref / len(active_tasks) if active_tasks else 0

    convergence_summary = {
        "threshold_rho_used": threshold_used,
        "n_configs_total": len(convergence_results),
        "n_configs_passing_90": len(passing_90),
        "n_configs_passing_80": len(passing_80),
        "n_configs_passing_70": len(passing_70),
        "full_budget_time_per_task": round(full_budget_time, 3),
        "budget_grid": BUDGET_GRID,
    }
    if min_cost_config:
        convergence_summary["cheapest_passing_time"] = min_cost_config["wall_clock_seconds"]
        convergence_summary["cheapest_config"] = {
            "n_estimators": min_cost_config["n_estimators"],
            "max_depth": min_cost_config["max_depth"],
            "n_seeds": min_cost_config["n_seeds"],
            "subsample_frac": min_cost_config["subsample_frac"],
        }
        convergence_summary["speedup_ratio"] = round(
            full_budget_time * len(active_tasks) / max(min_cost_config["wall_clock_seconds"], 1e-6), 2)

    # Build output in exp_gen_sol_out.json format
    # Each example = one experimental measurement
    output_examples = []

    # Part A examples: convergence results
    for cr in convergence_results:
        example = {
            "input": json.dumps({
                "experiment": "convergence_analysis",
                "n_estimators": cr["n_estimators"],
                "max_depth": cr["max_depth"],
                "n_seeds": cr["n_seeds"],
                "subsample_frac": cr["subsample_frac"],
            }),
            "output": json.dumps({
                "mean_spearman_rho": cr["mean_rho"],
                "mean_kendall_tau": cr["mean_tau"],
                "wall_clock_seconds": cr["wall_clock_seconds"],
            }),
            "metadata_experiment_part": "A_convergence",
            "metadata_n_estimators": cr["n_estimators"],
            "metadata_max_depth": cr["max_depth"],
            "metadata_n_seeds": cr["n_seeds"],
            "metadata_subsample_frac": cr["subsample_frac"],
            "metadata_mean_rho": cr["mean_rho"],
            "metadata_mean_tau": cr["mean_tau"],
            "predict_jrn_probe": json.dumps(cr["spearman_rho_per_task"]),
            "predict_baseline_full_budget": json.dumps({t: 1.0 for t in active_tasks}),
        }
        output_examples.append(example)

    # Part B examples: cost comparison per task
    for task_name in active_tasks:
        cr = cost_results[task_name]
        example = {
            "input": json.dumps({
                "experiment": "cost_comparison",
                "task": task_name,
                "n_joins": N_JOINS,
            }),
            "output": json.dumps({
                "jrn_perf": cr["jrn_perf"],
                "greedy_perf": cr["greedy_perf"],
                "random_perf": cr["random_perf"],
                "exhaust_perf": cr["exhaust_perf"],
                "jrn_time": cr["jrn_time"],
                "greedy_time": cr["greedy_time"],
            }),
            "metadata_experiment_part": "B_cost_comparison",
            "metadata_task_name": task_name,
            "metadata_jrn_time": cr["jrn_time"],
            "metadata_greedy_time": cr["greedy_time"],
            "metadata_cost_ratio": cr["cost_ratio_jrn_vs_greedy"],
            "predict_jrn_probe": json.dumps({
                "perf": cr["jrn_perf"],
                "time": cr["jrn_time"],
                "selected_joins": cr["jrn_selected_joins"],
            }),
            "predict_baseline_greedy": json.dumps({
                "perf": cr["greedy_perf"],
                "time": cr["greedy_time"],
                "order": cr["greedy_order"],
            }),
            "predict_baseline_random": json.dumps({
                "perf": cr["random_perf"],
                "time": cr["random_time"],
            }),
            "predict_baseline_exhaustive": json.dumps({
                "perf": cr["exhaust_perf"],
                "time": cr["exhaust_time_sampled"],
                "time_extrapolated": cr["exhaust_time_extrapolated"],
            }),
        }
        output_examples.append(example)

    # Reference JRN examples (one per task x join)
    for task_name in active_tasks:
        for detail in ref_jrn_details[task_name]:
            example = {
                "input": json.dumps({
                    "experiment": "reference_jrn",
                    "task": task_name,
                    "join_idx": detail["join_idx"],
                    "source": detail["source"],
                    "fk_col": detail["fk_col"],
                    "target": detail["target"],
                }),
                "output": json.dumps({
                    "jrn_mean": detail["jrn_mean"],
                    "jrn_std": detail["jrn_std"],
                    "perf_base": detail["perf_base_mean"],
                    "perf_aug": detail["perf_aug_mean"],
                }),
                "metadata_experiment_part": "reference_jrn",
                "metadata_task_name": task_name,
                "metadata_join_idx": detail["join_idx"],
                "metadata_jrn_mean": detail["jrn_mean"],
                "predict_jrn_probe": str(detail["jrn_mean"]),
                "predict_baseline_null": "1.0",
            }
            output_examples.append(example)

    # Breakeven examples
    for row in breakeven_analysis["scaling_table"]:
        example = {
            "input": json.dumps({
                "experiment": "breakeven_analysis",
                "n_joins": row["J"],
            }),
            "output": json.dumps({
                "jrn_models": row["jrn_models"],
                "greedy_models": row["greedy_models"],
                "speedup_vs_greedy": row["speedup_jrn_vs_greedy"],
            }),
            "metadata_experiment_part": "breakeven",
            "metadata_n_joins": row["J"],
            "metadata_speedup_vs_greedy": row["speedup_jrn_vs_greedy"],
            "predict_jrn_probe": json.dumps(row),
            "predict_baseline_greedy": json.dumps({
                "models": row["greedy_models"],
                "time": row["est_greedy_time"],
            }),
        }
        output_examples.append(example)

    full_output = {
        "metadata": {
            "title": "JRN Probe Cost-Efficiency and Ranking Stability on rel-f1",
            "dataset": "rel-f1",
            "n_joins": N_JOINS,
            "n_tasks": len(active_tasks),
            "tasks": active_tasks,
            "fk_joins": [{"idx": i, "source": s, "fk_col": f, "target": t}
                         for i, (s, f, t, _) in enumerate(fk_joins)],
            "part_a_convergence": {
                "reference_budget": FULL_BUDGET,
                "reference_jrn": {t: ref_jrn[t].tolist() for t in active_tasks},
                "convergence_summary": convergence_summary,
                "minimum_cost_config": {
                    k: v for k, v in (min_cost_config or {}).items()
                    if k not in ("budget_jrn",)
                } if min_cost_config else None,
            },
            "part_b_cost_comparison": cost_results,
            "breakeven_analysis": breakeven_analysis,
            "summary_statistics": summary_stats,
            "total_runtime_seconds": round(time.time() - t_start, 1),
        },
        "datasets": [{
            "dataset": "rel-f1",
            "examples": output_examples,
        }],
    }

    # Write output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(full_output, indent=2, default=str))
    logger.info(f"Output written to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Total examples: {len(output_examples)}")
    logger.info(f"Total runtime: {time.time() - t_start:.1f}s")

    # Print key results
    logger.info("=" * 60)
    logger.info("KEY RESULTS")
    logger.info("=" * 60)
    for task_name in active_tasks:
        cr = cost_results[task_name]
        logger.info(f"  {task_name}:")
        logger.info(f"    JRN probe: perf={cr['jrn_perf']:.4f}, time={cr['jrn_time']:.1f}s")
        logger.info(f"    Greedy:    perf={cr['greedy_perf']:.4f}, time={cr['greedy_time']:.1f}s")
        logger.info(f"    Random:    perf={cr['random_perf']:.4f}, time={cr['random_time']:.1f}s")
        logger.info(f"    Exhaust:   perf={cr['exhaust_perf']:.4f}, time={cr['exhaust_time_sampled']:.1f}s (sampled)")
        logger.info(f"    Speedup JRN vs greedy: {cr['cost_ratio_jrn_vs_greedy']:.2f}x")

    if summary_stats:
        logger.info(f"  Mean speedup vs greedy: {summary_stats['mean_speedup_vs_greedy']:.2f}x")
        logger.info(f"  JRN wins vs random: {summary_stats['jrn_wins_vs_random']}/{summary_stats['n_tasks']}")


if __name__ == "__main__":
    main()
