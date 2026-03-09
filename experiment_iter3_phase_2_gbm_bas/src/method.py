#!/usr/bin/env python3
"""Phase 2 GBM-Based Aggregation Sensitivity and JRN Estimation on rel-f1.

Re-run Phase 2 aggregation sensitivity using GBM probes (LightGBM) instead of
failed MLP probes. For each join-task pair, compute JRN via GBM performance
ratios and measure aggregation strategy sensitivity (CoV across 5 statistical
aggregation types). Test whether sensitivity peaks non-monotonically near
JRN ~ 1 (inverted-U shape) via quadratic and piecewise regression.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from loguru import logger
from scipy.stats import kruskal, spearmanr
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
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DATA_FILE = DATA_DIR / "full_data_out.json"

# ── Constants ────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456]
AGG_TYPES = ["mean", "sum", "max", "std", "all_combined"]
VAL_TIMESTAMP = pd.Timestamp("2005-01-01")
TEST_TIMESTAMP = pd.Timestamp("2010-01-01")

TASK_NAMES = [
    "rel-f1/driver-dnf",
    "rel-f1/driver-top3",
    "rel-f1/driver-position",
    "rel-f1/results-position",
    "rel-f1/qualifying-position",
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Parse Dataset
# ══════════════════════════════════════════════════════════════════════════════
def parse_dataset(data: dict) -> Tuple[Dict[str, pd.DataFrame], list, dict, dict]:
    """Parse full_data_out.json into tables, FK joins, task samples, schema info."""
    examples = data["datasets"][0]["examples"]
    logger.info(f"Parsing {len(examples)} examples")

    table_rows: Dict[str, list] = defaultdict(list)
    table_pk_cols: Dict[str, str] = {}
    table_fk_maps: Dict[str, dict] = {}
    table_time_cols: Dict[str, Optional[str]] = {}
    fk_joins: list = []
    task_samples: Dict[str, list] = defaultdict(list)
    task_meta: Dict[str, dict] = {}

    for ex in examples:
        row_type = ex.get("metadata_row_type")
        if row_type == "table_row":
            tbl = ex["metadata_table_name"]
            inp = json.loads(ex["input"])
            pk_col = ex["metadata_primary_key_col"]
            pk_val = ex["metadata_primary_key_value"]
            inp[pk_col] = pk_val
            table_rows[tbl].append(inp)
            table_pk_cols[tbl] = pk_col
            table_time_cols[tbl] = ex.get("metadata_time_col")
            if "metadata_foreign_keys_json" in ex:
                table_fk_maps[tbl] = json.loads(ex["metadata_foreign_keys_json"])
        elif row_type == "fk_join_metadata":
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])
            fk_joins.append({
                "idx": ex["metadata_row_index"],
                "child": inp["source_table"],
                "parent": inp["target_table"],
                "fk_col": inp["source_fk_col"],
                "parent_pk_col": inp["target_pk_col"],
                "fanout_mean": out.get("fanout_mean", 0),
                "num_edges": out.get("num_edges", 0),
            })
        elif row_type == "task_sample":
            task_name = ex["metadata_task_name"]
            fold = ex.get("metadata_fold_name", "unknown")
            inp = json.loads(ex["input"])
            inp["__fold__"] = fold
            inp["__target__"] = ex["output"]
            task_samples[task_name].append(inp)
            if task_name not in task_meta:
                task_meta[task_name] = {
                    "task_type": ex.get("metadata_task_type"),
                    "entity_table": ex.get("metadata_entity_table"),
                    "entity_col": ex.get("metadata_entity_col"),
                    "target_col": ex.get("metadata_target_col"),
                }

    # Convert to DataFrames
    tables: Dict[str, pd.DataFrame] = {}
    for tbl, rows in table_rows.items():
        df = pd.DataFrame(rows)
        pk_col = table_pk_cols[tbl]
        try:
            df[pk_col] = pd.to_numeric(df[pk_col], errors="coerce").fillna(-1).astype(int)
        except Exception:
            pass
        for col in df.columns:
            if col == pk_col:
                continue
            if col in ("date", "dob"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                try:
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if converted.notna().sum() > 0:
                        df[col] = converted
                except Exception:
                    pass
        tables[tbl] = df
        logger.info(f"  Table '{tbl}': {len(df)} rows, cols={list(df.columns)}")

    fk_joins.sort(key=lambda x: x["idx"])
    logger.info(f"  {len(fk_joins)} FK joins parsed")

    task_dfs = {}
    for task_name, samples in task_samples.items():
        tdf = pd.DataFrame(samples)
        if "date" in tdf.columns:
            tdf["date"] = pd.to_datetime(tdf["date"], errors="coerce")
        meta = task_meta[task_name]
        if meta["task_type"] == "binary_classification":
            tdf["__target__"] = pd.to_numeric(tdf["__target__"], errors="coerce").fillna(0).astype(int)
        elif meta["task_type"] == "regression":
            tdf["__target__"] = pd.to_numeric(tdf["__target__"], errors="coerce")
        ecol = meta["entity_col"]
        if ecol in tdf.columns:
            try:
                tdf[ecol] = pd.to_numeric(tdf[ecol], errors="coerce").fillna(-1).astype(int)
            except Exception:
                pass
        task_dfs[task_name] = tdf
        logger.info(f"  Task '{task_name}': {len(tdf)} samples, folds={tdf['__fold__'].value_counts().to_dict()}")

    schema_info = {
        "table_pk_cols": table_pk_cols,
        "table_fk_maps": table_fk_maps,
        "table_time_cols": table_time_cols,
        "task_meta": task_meta,
    }
    return tables, fk_joins, task_dfs, schema_info


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Connection finder
# ══════════════════════════════════════════════════════════════════════════════
def find_connection(entity_table: str, join_info: dict, table_fk_maps: dict) -> Optional[dict]:
    """Determine how to connect a join (child->parent) to an entity table."""
    child = join_info["child"]
    parent = join_info["parent"]
    fk_col = join_info["fk_col"]

    if parent == entity_table:
        return {"type": "direct", "child": child, "parent": parent, "fk_col": fk_col}

    entity_fks = table_fk_maps.get(entity_table, {})
    if parent in entity_fks.values():
        bridge_fk = [k for k, v in entity_fks.items() if v == parent][0]
        return {"type": "entity_fk", "child": child, "parent": parent,
                "fk_col": fk_col, "bridge_fk": bridge_fk}

    if child in entity_fks.values():
        bridge_fk = [k for k, v in entity_fks.items() if v == child][0]
        return {"type": "entity_fk_to_child", "child": child, "parent": parent,
                "fk_col": fk_col, "bridge_fk": bridge_fk}

    if entity_table == "drivers":
        bridge_tables_with_driver = []
        for tbl, fks in table_fk_maps.items():
            if "drivers" in fks.values():
                bridge_tables_with_driver.append(tbl)

        if parent == "races":
            for bridge in ["results", "standings", "qualifying"]:
                if bridge in bridge_tables_with_driver:
                    bridge_fks = table_fk_maps.get(bridge, {})
                    if "races" in bridge_fks.values():
                        return {"type": "bridge", "child": child, "parent": parent,
                                "fk_col": fk_col, "bridge_table": bridge,
                                "entity_col_in_bridge": "driverId",
                                "parent_col_in_bridge": "raceId"}
        elif parent == "constructors":
            for bridge in ["results", "qualifying"]:
                if bridge in bridge_tables_with_driver:
                    bridge_fks = table_fk_maps.get(bridge, {})
                    if "constructors" in bridge_fks.values():
                        return {"type": "bridge", "child": child, "parent": parent,
                                "fk_col": fk_col, "bridge_table": bridge,
                                "entity_col_in_bridge": "driverId",
                                "parent_col_in_bridge": "constructorId"}
        elif parent == "circuits":
            return {"type": "bridge_circuit", "child": child, "parent": parent,
                    "fk_col": fk_col, "bridge_table": "results",
                    "entity_col_in_bridge": "driverId",
                    "intermediate_table": "races",
                    "intermediate_col": "raceId",
                    "circuit_col": "circuitId"}

        if child in ("constructor_standings", "constructor_results"):
            for bridge in ["results", "qualifying"]:
                if bridge in bridge_tables_with_driver:
                    bridge_fks = table_fk_maps.get(bridge, {})
                    if "constructors" in bridge_fks.values():
                        return {"type": "bridge_constructor", "child": child, "parent": parent,
                                "fk_col": fk_col, "bridge_table": bridge,
                                "entity_col_in_bridge": "driverId",
                                "constructor_col_in_bridge": "constructorId"}
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Vectorized Feature Aggregation Engine
# ══════════════════════════════════════════════════════════════════════════════
def get_numeric_feature_cols(df: pd.DataFrame, exclude_cols: list) -> list:
    """Get numeric feature columns, excluding IDs and specified columns."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_set = set(exclude_cols)
    return [c for c in numeric if c not in exclude_set and not c.endswith("Id")]


def compute_aggregated_features(
    child_df: pd.DataFrame, fk_col: str, time_cutoff: pd.Timestamp,
    agg_type: str, exclude_cols: Optional[list] = None,
) -> Optional[pd.DataFrame]:
    """Aggregate child table's numeric features grouped by FK column (vectorized)."""
    if exclude_cols is None:
        exclude_cols = []
    if "date" in child_df.columns:
        filtered = child_df.loc[child_df["date"] < time_cutoff]
    else:
        filtered = child_df
    if len(filtered) == 0:
        return None
    feature_cols = get_numeric_feature_cols(filtered, [fk_col] + exclude_cols)
    if len(feature_cols) == 0:
        return None
    grouped = filtered.groupby(fk_col)[feature_cols]
    if agg_type == "mean":
        result = grouped.mean().add_prefix("agg_mean_")
    elif agg_type == "sum":
        result = grouped.sum().add_prefix("agg_sum_")
    elif agg_type == "max":
        result = grouped.max().add_prefix("agg_max_")
    elif agg_type == "std":
        result = grouped.std().fillna(0).add_prefix("agg_std_")
    elif agg_type == "all_combined":
        parts = [
            grouped.mean().add_prefix("agg_mean_"),
            grouped.sum().add_prefix("agg_sum_"),
            grouped.max().add_prefix("agg_max_"),
            grouped.std().fillna(0).add_prefix("agg_std_"),
            grouped.min().add_prefix("agg_min_"),
        ]
        result = pd.concat(parts, axis=1)
    else:
        return None
    counts = filtered.groupby(fk_col).size().rename("agg_count")
    result = pd.concat([result, counts], axis=1)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: VECTORIZED Feature Builder (the key fix from previous attempt)
# ══════════════════════════════════════════════════════════════════════════════
def build_features_vectorized(
    task_df: pd.DataFrame, fold: str, task_meta: dict,
    join_info: dict, connection: dict,
    tables: Dict[str, pd.DataFrame], schema_info: dict,
    agg_type: Optional[str], time_cutoff: pd.Timestamp,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Build feature matrix using vectorized pandas merges (no per-entity loops)."""
    entity_table = task_meta["entity_table"]
    entity_col = task_meta["entity_col"]
    entity_pk = schema_info["table_pk_cols"].get(entity_table, entity_col)

    fold_df = task_df[task_df["__fold__"] == fold].reset_index(drop=True)
    if len(fold_df) == 0:
        return None, None
    y = fold_df["__target__"].values
    if entity_col not in fold_df.columns:
        return None, None

    entity_ids = fold_df[[entity_col]].copy()  # keep as DataFrame for merging
    entity_ids.columns = ["_eid"]

    # ── Baseline features from entity table ──
    entity_df = tables.get(entity_table)
    if entity_df is None:
        return None, None
    entity_feature_cols = get_numeric_feature_cols(entity_df, [entity_pk])
    if entity_feature_cols:
        entity_lookup = entity_df[[entity_pk] + entity_feature_cols].copy()
        entity_lookup = entity_lookup.rename(columns={entity_pk: "_eid"})
        # Handle duplicates: take first
        entity_lookup = entity_lookup.drop_duplicates(subset=["_eid"], keep="first")
        X_base = entity_ids.merge(entity_lookup, on="_eid", how="left")[entity_feature_cols]
    else:
        X_base = pd.DataFrame({"_dummy": np.zeros(len(entity_ids))})
    X_base = X_base.apply(pd.to_numeric, errors="coerce").fillna(0)

    if agg_type is None:
        return X_base, y

    # ── With-join features (vectorized) ──
    conn_type = connection["type"]
    child = connection["child"]
    fk_col_conn = connection["fk_col"]
    child_df = tables.get(child)
    if child_df is None:
        return X_base, y

    agg_X = None

    if conn_type == "direct":
        # Parent == entity_table: aggregate child per entity via fk_col
        agg_df = compute_aggregated_features(child_df, fk_col_conn, time_cutoff, agg_type)
        if agg_df is not None:
            agg_df = agg_df.reset_index().rename(columns={fk_col_conn: "_eid"})
            agg_X = entity_ids.merge(agg_df, on="_eid", how="left").drop(columns=["_eid"])

    elif conn_type == "entity_fk":
        # Entity has FK to parent; aggregate child at parent level, look up via entity's FK
        bridge_fk = connection["bridge_fk"]
        agg_df = compute_aggregated_features(child_df, fk_col_conn, time_cutoff, agg_type)
        if agg_df is not None:
            # Get the entity's FK value for each sample
            entity_fk_lookup = entity_df[[entity_pk, bridge_fk]].copy()
            entity_fk_lookup = entity_fk_lookup.rename(columns={entity_pk: "_eid"})
            entity_fk_lookup = entity_fk_lookup.drop_duplicates(subset=["_eid"], keep="first")
            # Merge to get bridge_fk for each sample
            samples_with_fk = entity_ids.merge(entity_fk_lookup, on="_eid", how="left")
            # Merge with agg_df on the FK value
            agg_df_reset = agg_df.reset_index().rename(columns={fk_col_conn: bridge_fk})
            agg_cols = [c for c in agg_df_reset.columns if c != bridge_fk]
            merged = samples_with_fk.merge(agg_df_reset, on=bridge_fk, how="left")
            agg_X = merged[agg_cols]

    elif conn_type == "entity_fk_to_child":
        # Entity has FK to child table (e.g. results→races for races→circuits join)
        bridge_fk = connection["bridge_fk"]
        parent = connection["parent"]
        parent_pk = schema_info["table_pk_cols"].get(parent, fk_col_conn)
        parent_df = tables.get(parent)
        if parent_df is not None:
            parent_num_cols = get_numeric_feature_cols(parent_df, [parent_pk])
            if parent_num_cols:
                # entity → child (via bridge_fk) → parent (via fk_col)
                # Get entity's FK to child
                entity_fk_lookup = entity_df[[entity_pk, bridge_fk]].copy()
                entity_fk_lookup = entity_fk_lookup.rename(columns={entity_pk: "_eid"})
                entity_fk_lookup = entity_fk_lookup.drop_duplicates(subset=["_eid"], keep="first")
                samples_with_child_fk = entity_ids.merge(entity_fk_lookup, on="_eid", how="left")
                # child table: get child_pk → parent FK mapping
                child_pk = schema_info["table_pk_cols"].get(child, bridge_fk)
                if fk_col_conn in child_df.columns:
                    child_to_parent = child_df[[child_pk, fk_col_conn]].drop_duplicates(subset=[child_pk], keep="first")
                    child_to_parent = child_to_parent.rename(columns={child_pk: bridge_fk})
                    samples_with_parent_fk = samples_with_child_fk.merge(child_to_parent, on=bridge_fk, how="left")
                    # Look up parent features
                    parent_lookup = parent_df[[parent_pk] + parent_num_cols].copy()
                    parent_lookup = parent_lookup.rename(columns={parent_pk: fk_col_conn})
                    parent_lookup = parent_lookup.drop_duplicates(subset=[fk_col_conn], keep="first")
                    merged = samples_with_parent_fk.merge(parent_lookup, on=fk_col_conn, how="left")
                    agg_X = merged[parent_num_cols].add_prefix("agg_parent_")

    elif conn_type == "bridge":
        # Driver→bridge_table→parent: vectorized double-aggregation
        bridge_table = connection["bridge_table"]
        entity_col_in_bridge = connection["entity_col_in_bridge"]
        parent_col_in_bridge = connection["parent_col_in_bridge"]
        bridge_df = tables.get(bridge_table)
        if bridge_df is not None:
            # 1) Aggregate child at parent level
            agg_at_parent = compute_aggregated_features(child_df, fk_col_conn, time_cutoff, agg_type)
            if agg_at_parent is not None:
                # 2) Filter bridge by time
                if "date" in bridge_df.columns:
                    bridge_filt = bridge_df.loc[bridge_df["date"] < time_cutoff,
                                                [entity_col_in_bridge, parent_col_in_bridge]]
                else:
                    bridge_filt = bridge_df[[entity_col_in_bridge, parent_col_in_bridge]]
                # 3) Merge bridge with agg_at_parent
                agg_reset = agg_at_parent.reset_index().rename(
                    columns={fk_col_conn: parent_col_in_bridge})
                agg_cols = [c for c in agg_reset.columns if c != parent_col_in_bridge]
                merged = bridge_filt.merge(agg_reset, on=parent_col_in_bridge, how="inner")
                if len(merged) > 0:
                    # 4) Re-aggregate at entity level (mean of parent-level agg features)
                    driver_agg = merged.groupby(entity_col_in_bridge)[agg_cols].mean()
                    driver_agg = driver_agg.reset_index().rename(
                        columns={entity_col_in_bridge: "_eid"})
                    agg_X = entity_ids.merge(driver_agg, on="_eid", how="left").drop(
                        columns=["_eid"])

    elif conn_type == "bridge_circuit":
        # 2-hop: driver → results → races → circuits (VECTORIZED)
        bridge_table = connection["bridge_table"]
        entity_col_in_bridge = connection["entity_col_in_bridge"]
        intermediate_col = connection["intermediate_col"]
        circuit_col = connection["circuit_col"]
        bridge_df = tables.get(bridge_table)
        races_df = tables.get("races")
        circuits_df = tables.get("circuits")
        if bridge_df is not None and races_df is not None and circuits_df is not None:
            circuit_pk = schema_info["table_pk_cols"].get("circuits", "circuitId")
            circuit_num_cols = get_numeric_feature_cols(circuits_df, [circuit_pk])
            if circuit_num_cols:
                # 1) Filter bridge by time
                if "date" in bridge_df.columns:
                    br = bridge_df.loc[bridge_df["date"] < time_cutoff,
                                       [entity_col_in_bridge, intermediate_col]]
                else:
                    br = bridge_df[[entity_col_in_bridge, intermediate_col]]
                # 2) Merge bridge → races to get circuitId
                race_pk = schema_info["table_pk_cols"].get("races", "raceId")
                race_circuit = races_df[[race_pk, circuit_col]].drop_duplicates(
                    subset=[race_pk], keep="first")
                race_circuit = race_circuit.rename(columns={race_pk: intermediate_col})
                br_with_circuit = br.merge(race_circuit, on=intermediate_col, how="inner")
                # 3) Merge with circuit features
                circuit_feats = circuits_df[[circuit_pk] + circuit_num_cols].copy()
                circuit_feats = circuit_feats.rename(columns={circuit_pk: circuit_col})
                circuit_feats = circuit_feats.drop_duplicates(subset=[circuit_col], keep="first")
                br_full = br_with_circuit.merge(circuit_feats, on=circuit_col, how="inner")
                # 4) Aggregate circuit features per driver
                if len(br_full) > 0:
                    driver_circuit_agg = br_full.groupby(entity_col_in_bridge)[circuit_num_cols].mean()
                    driver_circuit_agg.columns = [f"agg_circuit_{c}" for c in circuit_num_cols]
                    driver_circuit_agg = driver_circuit_agg.reset_index().rename(
                        columns={entity_col_in_bridge: "_eid"})
                    agg_X = entity_ids.merge(driver_circuit_agg, on="_eid", how="left").drop(
                        columns=["_eid"])

    elif conn_type == "bridge_constructor":
        # Driver→results/qualifying→constructor_standings/results
        bridge_table = connection["bridge_table"]
        entity_col_in_bridge = connection["entity_col_in_bridge"]
        constructor_col = connection["constructor_col_in_bridge"]
        bridge_df = tables.get(bridge_table)
        if bridge_df is not None:
            agg_at_constructor = compute_aggregated_features(
                child_df, fk_col_conn, time_cutoff, agg_type)
            if agg_at_constructor is not None:
                if "date" in bridge_df.columns:
                    bridge_filt = bridge_df.loc[bridge_df["date"] < time_cutoff,
                                                [entity_col_in_bridge, constructor_col]]
                else:
                    bridge_filt = bridge_df[[entity_col_in_bridge, constructor_col]]
                agg_reset = agg_at_constructor.reset_index().rename(
                    columns={fk_col_conn: constructor_col})
                agg_cols = [c for c in agg_reset.columns if c != constructor_col]
                merged = bridge_filt.merge(agg_reset, on=constructor_col, how="inner")
                if len(merged) > 0:
                    driver_agg = merged.groupby(entity_col_in_bridge)[agg_cols].mean()
                    driver_agg = driver_agg.reset_index().rename(
                        columns={entity_col_in_bridge: "_eid"})
                    agg_X = entity_ids.merge(driver_agg, on="_eid", how="left").drop(
                        columns=["_eid"])

    if agg_X is None or len(agg_X.columns) == 0:
        return X_base, y

    # Combine baseline + join features
    X_combined = pd.concat([X_base.reset_index(drop=True), agg_X.reset_index(drop=True)], axis=1)
    X_combined = X_combined.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_combined = X_combined.replace([np.inf, -np.inf], 0)
    return X_combined, y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: GBM Training
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    task_type: str, seed: int,
    n_estimators: int = 200, max_depth: int = 6,
) -> float:
    """Train a GBM and return performance metric."""
    try:
        import lightgbm as lgb
        if task_type == "binary_classification":
            if len(np.unique(y_val)) < 2:
                return 0.5
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=seed, verbose=-1, n_jobs=1,
                min_child_samples=5, learning_rate=0.1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred)
        else:
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=seed, verbose=-1, n_jobs=1,
                min_child_samples=5, learning_rate=0.1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            return 1.0 / max(mae, 1e-8)
    except Exception as e:
        logger.warning(f"LightGBM failed (seed={seed}): {e}, falling back to sklearn")
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        if task_type == "binary_classification":
            if len(np.unique(y_val)) < 2:
                return 0.5
            model = GradientBoostingClassifier(
                n_estimators=min(n_estimators, 100), max_depth=max_depth, random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred)
        else:
            model = GradientBoostingRegressor(
                n_estimators=min(n_estimators, 100), max_depth=max_depth, random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            return 1.0 / max(mae, 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Construct task data for results/qualifying
# ══════════════════════════════════════════════════════════════════════════════
def construct_task_data_from_table(
    tables: Dict[str, pd.DataFrame], schema_info: dict,
    task_name: str, task_meta: dict,
) -> Optional[pd.DataFrame]:
    """For results-position and qualifying-position, construct train/val from table."""
    entity_table = task_meta["entity_table"]
    entity_col = task_meta["entity_col"]
    target_col = task_meta["target_col"]
    entity_pk = schema_info["table_pk_cols"].get(entity_table, entity_col)
    tbl_df = tables.get(entity_table)
    if tbl_df is None or target_col not in tbl_df.columns or "date" not in tbl_df.columns:
        return None
    df = tbl_df[[entity_pk, "date", target_col]].copy()
    df = df.rename(columns={entity_pk: entity_col, target_col: "__target__"})
    df["__target__"] = pd.to_numeric(df["__target__"], errors="coerce")
    df = df.dropna(subset=["__target__", "date"])
    df["__fold__"] = "test"
    df.loc[df["date"] < VAL_TIMESTAMP, "__fold__"] = "train"
    df.loc[(df["date"] >= VAL_TIMESTAMP) & (df["date"] < TEST_TIMESTAMP), "__fold__"] = "val"
    logger.info(f"  Constructed task data for '{task_name}': {len(df)} samples, "
                f"folds={df['__fold__'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════
def _safe_float(val) -> float:
    if val is None:
        return 0.0
    v = float(val)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return round(v, 6)


def _write_minimal_output(total_models: int):
    out = {
        "metadata": {
            "method_name": "Phase 2 GBM-Based Aggregation Sensitivity and JRN Estimation",
            "error": "No successful join-task pairs",
            "num_total_models_trained": total_models,
        },
        "datasets": [{"dataset": "rel-f1", "examples": [
            {"input": "none", "output": "no successful pairs"}
        ]}],
    }
    (WORKSPACE / "method_out.json").write_text(json.dumps(out, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
@logger.catch
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Phase 2 GBM-Based Aggregation Sensitivity and JRN Estimation")
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info(f"Loading data from {DATA_FILE}")
    data = json.loads(DATA_FILE.read_text())
    tables, fk_joins, task_dfs, schema_info = parse_dataset(data)
    del data
    gc.collect()

    logger.info(f"Tables: {list(tables.keys())}")
    logger.info(f"FK joins: {len(fk_joins)}")
    logger.info(f"Tasks: {list(task_dfs.keys())}")

    table_fk_maps = schema_info["table_fk_maps"]
    task_meta_all = schema_info["task_meta"]

    # ── Construct train/val for results-position and qualifying-position ──
    for tn in ["rel-f1/results-position", "rel-f1/qualifying-position"]:
        if tn in task_dfs:
            existing = task_dfs[tn]
            if "train" not in existing["__fold__"].values:
                constructed = construct_task_data_from_table(tables, schema_info, tn, task_meta_all[tn])
                if constructed is not None:
                    task_dfs[tn] = constructed

    # ── Enumerate testable (join, task) pairs ────────────────────────────
    testable_pairs = []
    connections = {}
    for task_name in TASK_NAMES:
        if task_name not in task_meta_all:
            logger.warning(f"Task {task_name} not found in data, skipping")
            continue
        meta = task_meta_all[task_name]
        entity_table = meta["entity_table"]
        for join_info in fk_joins:
            conn = find_connection(entity_table, join_info, table_fk_maps)
            if conn is not None:
                pair_key = (join_info["idx"], task_name)
                testable_pairs.append(pair_key)
                connections[pair_key] = conn

    logger.info(f"Found {len(testable_pairs)} testable (join, task) pairs")

    # ── Main training loop ───────────────────────────────────────────────
    results_matrix: Dict[tuple, float] = {}
    gt_results: Dict[tuple, float] = {}
    total_models = 0
    failed_pairs = []

    for pair_idx, (join_idx, task_name) in enumerate(testable_pairs):
        pair_t0 = time.time()
        join_info = fk_joins[join_idx]
        conn = connections[(join_idx, task_name)]
        meta = task_meta_all[task_name]
        task_df = task_dfs.get(task_name)

        if task_df is None:
            logger.warning(f"No task data for {task_name}, skipping")
            failed_pairs.append((join_idx, task_name))
            continue

        join_label = f"{join_info['child']}→{join_info['parent']} ({join_info['fk_col']})"
        short_task = task_name.split("/")[-1]
        logger.info(f"[{pair_idx+1}/{len(testable_pairs)}] Join {join_idx}: {join_label} | Task: {short_task} | conn={conn['type']}")

        # Build baseline features
        try:
            X_train_base, y_train = build_features_vectorized(
                task_df, "train", meta, join_info, conn, tables, schema_info,
                agg_type=None, time_cutoff=VAL_TIMESTAMP)
            X_val_base, y_val = build_features_vectorized(
                task_df, "val", meta, join_info, conn, tables, schema_info,
                agg_type=None, time_cutoff=TEST_TIMESTAMP)
        except Exception as e:
            logger.warning(f"  Failed to build baseline features: {e}")
            failed_pairs.append((join_idx, task_name))
            continue

        if X_train_base is None or X_val_base is None or len(X_train_base) < 10 or len(X_val_base) < 5:
            logger.warning(f"  Insufficient data: train={X_train_base.shape if X_train_base is not None else 0}, "
                           f"val={X_val_base.shape if X_val_base is not None else 0}")
            failed_pairs.append((join_idx, task_name))
            continue

        # ── Baseline models ──
        for seed in SEEDS:
            perf = train_and_evaluate(X_train_base, y_train, X_val_base, y_val, meta["task_type"], seed)
            results_matrix[(join_idx, task_name, "baseline", seed)] = perf
            total_models += 1

        base_mean = np.mean([results_matrix[(join_idx, task_name, "baseline", s)] for s in SEEDS])

        # ── With-join models for each aggregation type ──
        for agg_type in AGG_TYPES:
            try:
                X_train_join, _ = build_features_vectorized(
                    task_df, "train", meta, join_info, conn, tables, schema_info,
                    agg_type=agg_type, time_cutoff=VAL_TIMESTAMP)
                X_val_join, _ = build_features_vectorized(
                    task_df, "val", meta, join_info, conn, tables, schema_info,
                    agg_type=agg_type, time_cutoff=TEST_TIMESTAMP)
            except Exception as e:
                logger.warning(f"  Failed agg={agg_type}: {e}")
                for seed in SEEDS:
                    results_matrix[(join_idx, task_name, agg_type, seed)] = base_mean
                continue

            if X_train_join is None or X_val_join is None:
                for seed in SEEDS:
                    results_matrix[(join_idx, task_name, agg_type, seed)] = base_mean
                continue

            for seed in SEEDS:
                perf = train_and_evaluate(X_train_join, y_train, X_val_join, y_val, meta["task_type"], seed)
                results_matrix[(join_idx, task_name, agg_type, seed)] = perf
                total_models += 1

        # ── Ground truth (larger GBM) ──
        try:
            X_train_gt, _ = build_features_vectorized(
                task_df, "train", meta, join_info, conn, tables, schema_info,
                agg_type="all_combined", time_cutoff=VAL_TIMESTAMP)
            X_val_gt, _ = build_features_vectorized(
                task_df, "val", meta, join_info, conn, tables, schema_info,
                agg_type="all_combined", time_cutoff=TEST_TIMESTAMP)
            if X_train_gt is not None and X_val_gt is not None:
                for seed in SEEDS:
                    perf_gt = train_and_evaluate(
                        X_train_gt, y_train, X_val_gt, y_val, meta["task_type"],
                        seed, n_estimators=500, max_depth=8)
                    gt_results[(join_idx, task_name, seed)] = perf_gt
                    total_models += 1
        except Exception as e:
            logger.warning(f"  GT model failed: {e}")

        elapsed = time.time() - pair_t0
        logger.info(f"  Pair done in {elapsed:.1f}s | baseline={base_mean:.4f} | models={total_models}")

        # Time check - abort if running too long
        total_elapsed = time.time() - t0
        if total_elapsed > 3000:  # 50 min safety
            logger.warning(f"Time limit approaching ({total_elapsed:.0f}s), stopping at pair {pair_idx+1}")
            break

    logger.info(f"Total models trained: {total_models}")
    logger.info(f"Failed pairs: {len(failed_pairs)}")

    successful_pairs = [p for p in testable_pairs if p not in failed_pairs
                        and (p[0], p[1], "baseline", 42) in results_matrix]
    logger.info(f"Successful pairs: {len(successful_pairs)}")

    if len(successful_pairs) == 0:
        logger.error("No successful pairs!")
        _write_minimal_output(total_models)
        return

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: Compute JRN and Aggregation Sensitivity
    # ══════════════════════════════════════════════════════════════════════
    jrn_matrix: Dict[tuple, float] = {}
    sensitivity_matrix: Dict[tuple, float] = {}
    gt_jrn_matrix: Dict[tuple, float] = {}
    winning_agg: Dict[tuple, str] = {}
    per_pair_details: list = []

    for join_idx, task_name in successful_pairs:
        base_perfs = [results_matrix.get((join_idx, task_name, "baseline", s), 0.5) for s in SEEDS]
        base_mean = float(np.mean(base_perfs))

        agg_perfs = {}
        for at in AGG_TYPES:
            perfs = [results_matrix.get((join_idx, task_name, at, s), base_mean) for s in SEEDS]
            agg_perfs[at] = float(np.mean(perfs))

        best_agg_type = max(agg_perfs, key=agg_perfs.get)
        best_agg_perf = agg_perfs[best_agg_type]
        jrn = best_agg_perf / base_mean if base_mean > 0 else 1.0
        jrn_matrix[(join_idx, task_name)] = jrn

        perf_values = list(agg_perfs.values())
        mean_perf = np.mean(perf_values)
        cov = float(np.std(perf_values) / mean_perf) if mean_perf > 0 else 0.0
        sensitivity_matrix[(join_idx, task_name)] = cov
        winning_agg[(join_idx, task_name)] = best_agg_type

        gt_perfs = [gt_results.get((join_idx, task_name, s)) for s in SEEDS]
        gt_perfs = [p for p in gt_perfs if p is not None]
        gt_jrn_val = float("nan")
        if gt_perfs:
            gt_jrn_val = float(np.mean(gt_perfs)) / base_mean if base_mean > 0 else 1.0
            gt_jrn_matrix[(join_idx, task_name)] = gt_jrn_val

        ji = fk_joins[join_idx]
        per_pair_details.append({
            "join_idx": join_idx,
            "join_name": f"{ji['child']}->{ji['parent']}",
            "fk_col": ji["fk_col"],
            "task_name": task_name.split("/")[-1],
            "baseline_perf": round(base_mean, 4),
            "best_agg_perf": round(best_agg_perf, 4),
            "jrn": round(jrn, 4),
            "sensitivity_cov": round(cov, 4),
            "winning_agg": best_agg_type,
            "agg_perfs": {k: round(v, 4) for k, v in agg_perfs.items()},
            "gt_jrn": round(gt_jrn_val, 4) if not math.isnan(gt_jrn_val) else None,
        })

    logger.info(f"Computed JRN for {len(jrn_matrix)} pairs")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: Probe-to-GT Correlation
    # ══════════════════════════════════════════════════════════════════════
    common_pairs = [k for k in successful_pairs if k in gt_jrn_matrix]
    if len(common_pairs) >= 5:
        probe_jrns = [jrn_matrix[k] for k in common_pairs]
        gt_jrns = [gt_jrn_matrix[k] for k in common_pairs]
        rho, rho_pval = spearmanr(probe_jrns, gt_jrns)
        logger.info(f"Probe-GT Spearman rho={rho:.3f}, p={rho_pval:.4f} (n={len(common_pairs)})")
    else:
        rho, rho_pval = float("nan"), float("nan")
        probe_jrns, gt_jrns = [], []
        logger.warning("Not enough common pairs for probe-GT correlation")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 9: Statistical Analysis
    # ══════════════════════════════════════════════════════════════════════
    jrn_values = np.array([jrn_matrix[k] for k in successful_pairs])
    sens_values = np.array([sensitivity_matrix[k] for k in successful_pairs])

    logger.info(f"JRN stats: mean={np.mean(jrn_values):.3f}, std={np.std(jrn_values):.3f}, "
                f"min={np.min(jrn_values):.3f}, max={np.max(jrn_values):.3f}")
    logger.info(f"Sensitivity stats: mean={np.mean(sens_values):.3f}, std={np.std(sens_values):.3f}")

    # 9a: Quadratic Model
    beta0, beta1, beta2, p_beta2, r_sq_quad, peak_jrn = 0.0, 0.0, 0.0, 1.0, 0.0, None
    try:
        X_quad = np.column_stack([jrn_values, jrn_values ** 2])
        X_quad = sm.add_constant(X_quad)
        model_quad = sm.OLS(sens_values, X_quad).fit()
        beta0, beta1, beta2 = model_quad.params
        p_beta2 = model_quad.pvalues[2] if len(model_quad.pvalues) > 2 else 1.0
        r_sq_quad = model_quad.rsquared
        if beta2 < 0:
            peak_jrn = -beta1 / (2 * beta2)
        logger.info(f"Quadratic: beta0={beta0:.4f}, beta1={beta1:.4f}, beta2={beta2:.4f}, p={p_beta2:.4f}, R2={r_sq_quad:.4f}")
        if peak_jrn is not None:
            logger.info(f"  Inverted-U peak at JRN={peak_jrn:.3f}")
    except Exception as e:
        logger.warning(f"Quadratic fit failed: {e}")

    # 9b: Piecewise Linear Model
    best_bic, best_breakpoint = np.inf, 1.0
    best_slope_before, best_slope_after = 0.0, 0.0
    best_models_pw = (None, None)
    try:
        for b in np.arange(0.85, 1.25, 0.01):
            mask_lo = jrn_values < b
            mask_hi = jrn_values >= b
            if mask_lo.sum() < 3 or mask_hi.sum() < 3:
                continue
            try:
                X_lo = sm.add_constant(jrn_values[mask_lo])
                X_hi = sm.add_constant(jrn_values[mask_hi])
                m_lo = sm.OLS(sens_values[mask_lo], X_lo).fit()
                m_hi = sm.OLS(sens_values[mask_hi], X_hi).fit()
                total_bic = m_lo.bic + m_hi.bic
                if total_bic < best_bic:
                    best_bic = total_bic
                    best_breakpoint = float(b)
                    best_slope_before = float(m_lo.params[1]) if len(m_lo.params) > 1 else 0.0
                    best_slope_after = float(m_hi.params[1]) if len(m_hi.params) > 1 else 0.0
                    best_models_pw = (m_lo, m_hi)
            except Exception:
                continue
        logger.info(f"Piecewise: breakpoint={best_breakpoint:.2f}, slope_before={best_slope_before:.4f}, "
                    f"slope_after={best_slope_after:.4f}")
    except Exception as e:
        logger.warning(f"Piecewise fit failed: {e}")

    # 9c: Kruskal-Wallis test
    kw_stat, kw_pval = float("nan"), float("nan")
    try:
        tertile_edges = np.percentile(jrn_values, [33.3, 66.7])
        bins = np.digitize(jrn_values, tertile_edges)
        groups = [sens_values[bins == i] for i in range(3)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            kw_stat, kw_pval = kruskal(*groups)
            logger.info(f"Kruskal-Wallis: H={kw_stat:.3f}, p={kw_pval:.4f}")
    except Exception as e:
        logger.warning(f"Kruskal-Wallis failed: {e}")

    # 9d: Monotonic check
    mono_rho, mono_pval = float("nan"), float("nan")
    try:
        mono_rho, mono_pval = spearmanr(jrn_values, sens_values)
        logger.info(f"JRN-sensitivity Spearman: rho={mono_rho:.3f}, p={mono_pval:.4f}")
    except Exception as e:
        logger.warning(f"Monotonic check failed: {e}")

    # 9e: Winning agg by JRN bin
    winning_agg_by_bin = {"low": {}, "mid": {}, "high": {}}
    try:
        tertile_edges = np.percentile(jrn_values, [33.3, 66.7])
        for ji, tn in successful_pairs:
            jrn_val = jrn_matrix[(ji, tn)]
            wagg = winning_agg[(ji, tn)]
            if jrn_val < tertile_edges[0]:
                bn = "low"
            elif jrn_val < tertile_edges[1]:
                bn = "mid"
            else:
                bn = "high"
            winning_agg_by_bin[bn][wagg] = winning_agg_by_bin[bn].get(wagg, 0) + 1
    except Exception as e:
        logger.warning(f"Winning agg by bin failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 10: Visualization
    # ══════════════════════════════════════════════════════════════════════
    logger.info("Generating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: JRN heatmap
    try:
        join_names = [f"{fk_joins[j]['child']}->{fk_joins[j]['parent']}"
                      for j in sorted(set(j for j, _ in successful_pairs))]
        task_short = sorted(set(t.split("/")[-1] for _, t in successful_pairs))
        jrn_2d = np.full((len(join_names), len(task_short)), np.nan)
        join_idx_map = {name: i for i, name in enumerate(join_names)}
        task_idx_map = {name: i for i, name in enumerate(task_short)}
        for (ji, tn), val in jrn_matrix.items():
            jname = f"{fk_joins[ji]['child']}->{fk_joins[ji]['parent']}"
            tname = tn.split("/")[-1]
            if jname in join_idx_map and tname in task_idx_map:
                jrn_2d[join_idx_map[jname], task_idx_map[tname]] = val
        jrn_heatmap_df = pd.DataFrame(jrn_2d, index=join_names, columns=task_short)
        ax1 = axes[0, 0]
        sns.heatmap(jrn_heatmap_df, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0, ax=ax1,
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax1.set_title("JRN Matrix", fontsize=11)
        ax1.tick_params(axis="both", labelsize=7)
    except Exception as e:
        logger.warning(f"JRN heatmap failed: {e}")

    # Plot 2: Sensitivity heatmap
    try:
        sens_2d = np.full((len(join_names), len(task_short)), np.nan)
        for (ji, tn), val in sensitivity_matrix.items():
            jname = f"{fk_joins[ji]['child']}->{fk_joins[ji]['parent']}"
            tname = tn.split("/")[-1]
            if jname in join_idx_map and tname in task_idx_map:
                sens_2d[join_idx_map[jname], task_idx_map[tname]] = val
        sens_heatmap_df = pd.DataFrame(sens_2d, index=join_names, columns=task_short)
        ax2 = axes[0, 1]
        sns.heatmap(sens_heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax2,
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax2.set_title("Aggregation Sensitivity (CoV)", fontsize=11)
        ax2.tick_params(axis="both", labelsize=7)
    except Exception as e:
        logger.warning(f"Sensitivity heatmap failed: {e}")

    # Plot 3: JRN vs Sensitivity scatter + quadratic
    try:
        ax3 = axes[0, 2]
        ax3.scatter(jrn_values, sens_values, alpha=0.6, c="steelblue", edgecolors="k", s=40)
        jrn_sorted = np.linspace(jrn_values.min() - 0.05, jrn_values.max() + 0.05, 100)
        quad_pred = beta0 + beta1 * jrn_sorted + beta2 * jrn_sorted ** 2
        ax3.plot(jrn_sorted, quad_pred, "r-", lw=2,
                 label=f"Quad (beta2={beta2:.3f}, p={p_beta2:.3f})")
        ax3.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="JRN=1")
        if peak_jrn is not None:
            ax3.axvline(x=peak_jrn, color="orange", linestyle=":", alpha=0.7,
                        label=f"Peak={peak_jrn:.2f}")
        ax3.set_xlabel("JRN")
        ax3.set_ylabel("Sensitivity (CoV)")
        ax3.set_title("Phase 2: Threshold Test", fontsize=11)
        ax3.legend(fontsize=8)
    except Exception as e:
        logger.warning(f"Scatter plot failed: {e}")

    # Plot 4: Probe vs GT JRN
    try:
        ax4 = axes[1, 0]
        if probe_jrns and gt_jrns:
            ax4.scatter(probe_jrns, gt_jrns, alpha=0.6, c="steelblue", edgecolors="k", s=40)
            lims = [min(min(probe_jrns), min(gt_jrns)), max(max(probe_jrns), max(gt_jrns))]
            ax4.plot(lims, lims, "r--", alpha=0.5)
            ax4.set_xlabel("Probe JRN")
            ax4.set_ylabel("GT JRN")
            ax4.set_title(f"Probe-GT (rho={rho:.3f})", fontsize=11)
        else:
            ax4.text(0.5, 0.5, "No GT data", ha="center", va="center", transform=ax4.transAxes)
    except Exception as e:
        logger.warning(f"Probe-GT plot failed: {e}")

    # Plot 5: Winning agg by JRN bin
    try:
        ax5 = axes[1, 1]
        bin_labels = ["low", "mid", "high"]
        bar_data = [[winning_agg_by_bin.get(bl, {}).get(a, 0) for a in AGG_TYPES] for bl in bin_labels]
        bar_df = pd.DataFrame(bar_data, index=bin_labels, columns=AGG_TYPES)
        bar_df.plot(kind="bar", ax=ax5, colormap="Set2")
        ax5.set_title("Winning Agg by JRN Bin", fontsize=11)
        ax5.set_ylabel("Count")
        ax5.legend(fontsize=7, ncol=2)
        ax5.tick_params(axis="x", rotation=0)
    except Exception as e:
        logger.warning(f"Winning agg plot failed: {e}")

    # Plot 6: Piecewise linear fit
    try:
        ax6 = axes[1, 2]
        ax6.scatter(jrn_values, sens_values, alpha=0.6, c="steelblue", edgecolors="k", s=40)
        ax6.axvline(x=best_breakpoint, color="red", linestyle="--", lw=2,
                    label=f"Break={best_breakpoint:.2f}")
        if best_models_pw[0] is not None and best_models_pw[1] is not None:
            jrn_lo = np.linspace(jrn_values.min(), best_breakpoint, 50)
            jrn_hi = np.linspace(best_breakpoint, jrn_values.max(), 50)
            pred_lo = best_models_pw[0].predict(sm.add_constant(jrn_lo))
            pred_hi = best_models_pw[1].predict(sm.add_constant(jrn_hi))
            ax6.plot(jrn_lo, pred_lo, "g-", lw=2, label=f"slope={best_slope_before:.3f}")
            ax6.plot(jrn_hi, pred_hi, "m-", lw=2, label=f"slope={best_slope_after:.3f}")
        ax6.set_xlabel("JRN")
        ax6.set_ylabel("Sensitivity (CoV)")
        ax6.set_title("Piecewise Linear Fit", fontsize=11)
        ax6.legend(fontsize=8)
    except Exception as e:
        logger.warning(f"Piecewise plot failed: {e}")

    plt.tight_layout()
    fig_path = WORKSPACE / "jrn_phase2_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {fig_path}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 11: Per-join summary
    # ══════════════════════════════════════════════════════════════════════
    per_join_summary = []
    for ji in sorted(set(j for j, _ in successful_pairs)):
        join_info_j = fk_joins[ji]
        pair_jrns_j = [jrn_matrix[(ji, tn)] for ji2, tn in successful_pairs if ji2 == ji]
        pair_sens_j = [sensitivity_matrix[(ji, tn)] for ji2, tn in successful_pairs if ji2 == ji]
        per_join_summary.append({
            "join_idx": ji,
            "join_name": f"{join_info_j['child']}->{join_info_j['parent']}",
            "fk_col": join_info_j["fk_col"],
            "fanout_mean": join_info_j.get("fanout_mean", 0),
            "mean_jrn": round(float(np.mean(pair_jrns_j)), 4),
            "std_jrn": round(float(np.std(pair_jrns_j)), 4),
            "mean_sensitivity": round(float(np.mean(pair_sens_j)), 4),
            "n_tasks": len(pair_jrns_j),
        })

    # ══════════════════════════════════════════════════════════════════════
    # STEP 12: Conclusion
    # ══════════════════════════════════════════════════════════════════════
    inverted_u_confirmed = bool(beta2 < 0 and p_beta2 < 0.05)
    inverted_v_confirmed = bool(best_slope_before > 0 and best_slope_after < 0)

    if inverted_u_confirmed or inverted_v_confirmed:
        conclusion = (
            f"Phase 2 analysis on {len(successful_pairs)} join-task pairs CONFIRMS non-monotonic "
            f"(inverted-U/V) relationship between JRN and aggregation sensitivity. "
            f"Quadratic beta2={beta2:.4f} (p={p_beta2:.4f}), piecewise breakpoint at JRN={best_breakpoint:.2f}. "
            f"This supports the hypothesis that aggregation choice matters most near JRN~1."
        )
    else:
        conclusion = (
            f"Phase 2 analysis on {len(successful_pairs)} join-task pairs does NOT confirm inverted-U shape. "
            f"Quadratic beta2={beta2:.4f} (p={p_beta2:.4f}), monotonic rho={mono_rho:.3f}. "
            f"The JRN-sensitivity relationship appears {'monotonic' if abs(mono_rho) > 0.3 else 'flat/weak'}. "
            f"Probe-GT correlation: rho={rho:.3f}."
        )
    logger.info(f"Conclusion: {conclusion[:200]}...")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 13: Build output (exp_gen_sol_out schema)
    # ══════════════════════════════════════════════════════════════════════
    examples = []
    for detail in per_pair_details:
        example = {
            "input": json.dumps({
                "join_idx": detail["join_idx"],
                "join_name": detail["join_name"],
                "fk_col": detail["fk_col"],
                "task_name": detail["task_name"],
            }),
            "output": json.dumps({
                "jrn": detail["jrn"],
                "sensitivity_cov": detail["sensitivity_cov"],
                "baseline_perf": detail["baseline_perf"],
                "best_agg_perf": detail["best_agg_perf"],
                "winning_agg": detail["winning_agg"],
                "gt_jrn": detail["gt_jrn"],
                "agg_perfs": detail["agg_perfs"],
            }),
            "predict_jrn": str(detail["jrn"]),
            "predict_sensitivity": str(detail["sensitivity_cov"]),
            "predict_winning_agg": detail["winning_agg"],
            "metadata_join_idx": str(detail["join_idx"]),
            "metadata_task_name": detail["task_name"],
            "metadata_jrn": str(detail["jrn"]),
            "metadata_sensitivity": str(detail["sensitivity_cov"]),
            "metadata_winning_agg": detail["winning_agg"],
        }
        examples.append(example)

    method_out = {
        "metadata": {
            "method_name": "Phase 2 GBM-Based Aggregation Sensitivity and JRN Estimation",
            "dataset": "rel-f1",
            "description": (
                "GBM probe-based estimation of Join Reproduction Number (JRN) and "
                "aggregation sensitivity analysis across 5 statistical aggregation strategies "
                "on the RelBench rel-f1 (Formula 1) relational database."
            ),
            "num_testable_pairs": len(successful_pairs),
            "num_total_models_trained": total_models,
            "num_failed_pairs": len(failed_pairs),
            "seeds": SEEDS,
            "agg_types": AGG_TYPES,
            "tasks": [t.split("/")[-1] for t in TASK_NAMES],
            "quadratic_fit": {
                "beta0": _safe_float(beta0),
                "beta1": _safe_float(beta1),
                "beta2": _safe_float(beta2),
                "beta2_pvalue": _safe_float(p_beta2),
                "r_squared": _safe_float(r_sq_quad),
                "peak_jrn": _safe_float(peak_jrn) if peak_jrn is not None else None,
                "inverted_u_confirmed": inverted_u_confirmed,
            },
            "piecewise_fit": {
                "best_breakpoint": _safe_float(best_breakpoint),
                "slope_before": _safe_float(best_slope_before),
                "slope_after": _safe_float(best_slope_after),
                "inverted_v_confirmed": inverted_v_confirmed,
            },
            "probe_gt_correlation": {
                "spearman_rho": _safe_float(rho),
                "spearman_pvalue": _safe_float(rho_pval),
                "n_pairs": len(common_pairs),
            },
            "jrn_sensitivity_correlation": {
                "spearman_rho": _safe_float(mono_rho),
                "spearman_pvalue": _safe_float(mono_pval),
            },
            "kruskal_wallis": {
                "statistic": _safe_float(kw_stat),
                "pvalue": _safe_float(kw_pval),
            },
            "per_join_summary": per_join_summary,
            "winning_agg_by_jrn_bin": winning_agg_by_bin,
            "figures": ["jrn_phase2_results.png"],
            "conclusion": conclusion,
        },
        "datasets": [
            {
                "dataset": "rel-f1",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2, default=str))
    logger.info(f"Wrote {len(examples)} examples to {out_path}")
    logger.info(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    total_time = time.time() - t0
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
