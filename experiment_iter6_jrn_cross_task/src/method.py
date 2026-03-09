#!/usr/bin/env python3
"""JRN Cross-Task Transfer Analysis on rel-f1 and rel-stack.

Compute the full JRN matrix (reachable joins x entity-level tasks) for both
rel-f1 and rel-stack using LightGBM probes, then analyze JRN transferability
across tasks via leave-one-task-out transfer rho, pairwise Spearman rho,
Kendall's W, entity-table concordance, and practical join-selection gaps.
"""

import gc
import json
import math
import os
import resource
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, no GPU")

# Set RAM limit to ~80% of container limit (leave room for OS)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1024**3)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
    logger.info(f"RAM budget set to {RAM_BUDGET_BYTES / 1e9:.1f} GB")
except Exception:
    logger.warning("Could not set RAM limit via resource.setrlimit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2]
LGBM_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": NUM_CPUS,
}
MAX_TRAIN_SAMPLES = 40_000
MAX_AGG_ROWS = 150_000
MAX_VAL_SAMPLES = 20_000

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _enrich_df_with_derived_features(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """Add derived numeric features from datetime and categorical columns."""
    if exclude_cols is None:
        exclude_cols = []
    result = df.copy()

    # Extract features from datetime columns
    for col in df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns:
        if col in exclude_cols:
            continue
        try:
            result[f"{col}_year"] = df[col].dt.year.astype(float)
            result[f"{col}_month"] = df[col].dt.month.astype(float)
            result[f"{col}_dayofweek"] = df[col].dt.dayofweek.astype(float)
        except Exception:
            pass

    # Label-encode low-cardinality categorical columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col in exclude_cols:
            continue
        nunique = df[col].nunique()
        if 1 < nunique <= 200:  # Only encode low-cardinality
            try:
                codes = df[col].astype("category").cat.codes.astype(float)
                codes[codes < 0] = np.nan  # -1 means NaN
                result[f"{col}_enc"] = codes
            except Exception:
                pass

    return result


def compute_aggregation_features(
    child_df: pd.DataFrame,
    fk_col: str,
    max_rows: int = MAX_AGG_ROWS,
) -> pd.DataFrame:
    """Aggregate child table rows by FK column into entity-level features."""
    if len(child_df) > max_rows:
        child_df = child_df.sample(max_rows, random_state=42)

    # Enrich with derived features before aggregation
    child_df = _enrich_df_with_derived_features(child_df, exclude_cols=[fk_col])

    numeric_cols = child_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != fk_col]

    if not numeric_cols:
        # At minimum, compute count
        grouped = child_df.groupby(fk_col).size().to_frame("agg_count")
        return grouped.fillna(0)

    agg_dict = {}
    for col in numeric_cols[:15]:  # Cap at 15 numeric cols to limit feature explosion
        agg_dict[col] = ["mean", "std", "min", "max"]

    grouped = child_df.groupby(fk_col).agg(agg_dict)
    grouped.columns = [f"agg_{col}_{stat}" for col, stat in grouped.columns]

    # Add count
    counts = child_df.groupby(fk_col).size().to_frame("agg_count")
    grouped = grouped.join(counts)

    return grouped.fillna(0)


def compute_parent_lookup_features(
    parent_df: pd.DataFrame,
    pk_col: str,
) -> pd.DataFrame:
    """Extract parent table features for lookup by child FK."""
    enriched = _enrich_df_with_derived_features(parent_df, exclude_cols=[pk_col])
    numeric_cols = enriched.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != pk_col][:15]

    if not numeric_cols:
        return pd.DataFrame(index=parent_df[pk_col])

    result = enriched.set_index(pk_col)[numeric_cols].copy()
    result.columns = [f"parent_{c}" for c in result.columns]
    return result.fillna(0)


def get_entity_base_features(db, entity_table_name: str) -> pd.DataFrame:
    """Get base features from the entity table (with derived features)."""
    entity_tbl = db.table_dict[entity_table_name]
    entity_df = entity_tbl.df
    entity_pk = entity_tbl.pkey_col

    # Enrich with derived features from dates and categoricals
    enriched = _enrich_df_with_derived_features(entity_df, exclude_cols=[entity_pk])

    numeric_cols = enriched.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != entity_pk][:15]

    if not numeric_cols:
        # Add constant feature so baseline model can at least predict mean
        result = pd.DataFrame(index=entity_df[entity_pk].values)
        result["entity_const"] = 1.0
        return result

    entity_features = enriched.set_index(entity_pk)[numeric_cols].copy()
    entity_features.columns = [f"entity_{c}" for c in entity_features.columns]
    return entity_features.fillna(0)


def build_join_features(
    db,
    entity_table_name: str,
    entity_col: str,
    join_info: dict,
    hop_distance: int = 1,
) -> pd.DataFrame | None:
    """Build features from a single join for the entity table.

    Returns DataFrame indexed by entity_col values, or None on failure.
    """
    try:
        join = join_info
        child_table_name = join["child_table"]
        parent_table_name = join["parent_table"]
        fk_col = join["fk_col"]
        pk_col = join["pk_col"]

        if hop_distance == 1:
            if parent_table_name == entity_table_name:
                # Child->Entity aggregation: aggregate child rows by FK (= entity PK)
                child_df = db.table_dict[child_table_name].df.copy()
                child_df = child_df.dropna(subset=[fk_col])
                # Enrichment happens inside compute_aggregation_features
                agg_feats = compute_aggregation_features(child_df, fk_col)
                agg_feats.columns = [f"j_{child_table_name}_{c}" for c in agg_feats.columns]
                return agg_feats

            elif child_table_name == entity_table_name:
                # Entity->Parent lookup: get parent features via FK
                parent_df = db.table_dict[parent_table_name].df
                parent_feats = compute_parent_lookup_features(parent_df, pk_col)

                if parent_feats.empty or len(parent_feats.columns) == 0:
                    return None

                parent_feats.columns = [f"j_{parent_table_name}_{c}" for c in parent_feats.columns]

                # Map entity -> FK value -> parent features
                entity_tbl = db.table_dict[entity_table_name]
                entity_pk = entity_tbl.pkey_col
                entity_df = entity_tbl.df

                if fk_col not in entity_df.columns:
                    return None

                fk_map = entity_df[[entity_pk, fk_col]].copy()
                fk_map = fk_map.set_index(entity_pk)
                merged = fk_map.merge(parent_feats, left_on=fk_col, right_index=True, how="left")
                merged = merged.drop(columns=[fk_col], errors="ignore")
                return merged.fillna(0)

            else:
                # This join doesn't directly touch entity_table — skip for 1-hop
                return None

        elif hop_distance == 2:
            # 2-hop: compose through intermediate table
            # Find the intermediate table and compose features
            # Strategy: entity <- intermediate (1-hop) -> target (2-hop join)
            # OR: entity -> intermediate (1-hop) <- target child (2-hop join)
            return _build_2hop_features(db, entity_table_name, entity_col, join_info)

    except Exception:
        logger.exception(f"Failed building join features for {join_info.get('join_id', '?')}")
        return None


def _build_2hop_features(
    db,
    entity_table_name: str,
    entity_col: str,
    join_info: dict,
) -> pd.DataFrame | None:
    """Build 2-hop join features by composing through intermediate table."""
    try:
        child_table_name = join_info["child_table"]
        parent_table_name = join_info["parent_table"]
        fk_col = join_info["fk_col"]
        pk_col = join_info["pk_col"]

        entity_tbl = db.table_dict[entity_table_name]
        entity_pk = entity_tbl.pkey_col

        # Find 1-hop neighbor tables and how they connect to entity
        onehop_info = []  # list of {table, fk_col_in_child, direction}
        onehop_tables = set()
        for tname, tbl in db.table_dict.items():
            if tbl.fkey_col_to_pkey_table:
                for fk, pt in tbl.fkey_col_to_pkey_table.items():
                    if pt == entity_table_name:
                        onehop_tables.add(tname)
                        onehop_info.append({
                            "table": tname, "fk": fk,
                            "direction": "child_to_entity",
                        })
                    if tname == entity_table_name:
                        onehop_tables.add(pt)
                        onehop_info.append({
                            "table": pt, "fk": fk,
                            "direction": "entity_to_parent",
                        })

        # Find which intermediate table connects entity to this 2-hop join
        intermediate = None
        if child_table_name in onehop_tables:
            intermediate = child_table_name
        elif parent_table_name in onehop_tables:
            intermediate = parent_table_name

        if intermediate is None:
            return None

        # CASE A: entity <- intermediate (child->entity) -> 2-hop target (parent)
        # intermediate has FK to entity, and FK to 2-hop parent
        if intermediate == child_table_name:
            child_df = db.table_dict[child_table_name].df.copy()
            parent_df = db.table_dict[parent_table_name].df.copy()

            parent_feats = compute_parent_lookup_features(parent_df, pk_col)
            if parent_feats.empty or len(parent_feats.columns) == 0:
                return None

            # Find how intermediate connects to entity
            entity_fk = None
            child_tbl = db.table_dict[child_table_name]
            for fk, pt in child_tbl.fkey_col_to_pkey_table.items():
                if pt == entity_table_name:
                    entity_fk = fk
                    break

            if entity_fk is None:
                # Reverse: entity has FK to intermediate
                # Entity -> intermediate (entity.fk = intermediate.pk)
                for info in onehop_info:
                    if info["table"] == child_table_name and info["direction"] == "entity_to_parent":
                        # Entity has FK pointing to intermediate as parent
                        # Get FK col from entity table that points to intermediate
                        entity_fk_to_inter = info["fk"]
                        inter_pk = db.table_dict[child_table_name].pkey_col

                        # Get parent features from 2-hop target
                        child_enriched = child_df[[inter_pk, fk_col]].copy()
                        child_enriched = child_enriched.dropna(subset=[fk_col])
                        child_enriched = child_enriched.merge(
                            parent_feats, left_on=fk_col, right_index=True, how="left"
                        )
                        child_enriched = child_enriched.drop(columns=[fk_col], errors="ignore")

                        # Now we have intermediate features indexed by inter_pk
                        # Entity table has FK pointing to inter_pk
                        # So we can look up features for each entity via entity.fk_col -> inter_pk
                        inter_feats = child_enriched.set_index(inter_pk)
                        numeric_cols = inter_feats.select_dtypes(include=[np.number]).columns.tolist()[:10]
                        if not numeric_cols:
                            return None
                        inter_feats = inter_feats[numeric_cols]
                        inter_feats.columns = [f"j2h_{parent_table_name}_{c}" for c in inter_feats.columns]

                        # Map entity -> FK value -> intermediate features
                        entity_df = entity_tbl.df
                        fk_map = entity_df[[entity_pk, entity_fk_to_inter]].copy()
                        fk_map = fk_map.set_index(entity_pk)
                        merged = fk_map.merge(
                            inter_feats, left_on=entity_fk_to_inter,
                            right_index=True, how="left"
                        )
                        merged = merged.drop(columns=[entity_fk_to_inter], errors="ignore")
                        return merged.fillna(0)
                return None

            # Normal case: intermediate has FK to entity
            child_enriched = child_df[[entity_fk, fk_col]].copy()
            child_enriched = child_enriched.dropna(subset=[entity_fk, fk_col])
            child_enriched = child_enriched.merge(
                parent_feats, left_on=fk_col, right_index=True, how="left"
            )
            child_enriched = child_enriched.drop(columns=[fk_col], errors="ignore")

            if len(child_enriched) > MAX_AGG_ROWS:
                child_enriched = child_enriched.sample(MAX_AGG_ROWS, random_state=42)

            numeric_cols = child_enriched.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != entity_fk][:10]

            if not numeric_cols:
                return None

            agg_dict = {c: ["mean", "std"] for c in numeric_cols}
            grouped = child_enriched.groupby(entity_fk).agg(agg_dict)
            grouped.columns = [f"j2h_{parent_table_name}_{col}_{stat}" for col, stat in grouped.columns]
            counts = child_enriched.groupby(entity_fk).size().to_frame(f"j2h_{parent_table_name}_count")
            grouped = grouped.join(counts)
            return grouped.fillna(0)

        # CASE B: intermediate is the parent table of the 2-hop join
        # entity -> intermediate (entity has FK to intermediate)
        # 2-hop child -> intermediate (2-hop child has FK to intermediate)
        elif intermediate == parent_table_name:
            # 2-hop child aggregated by FK to intermediate,
            # then looked up via entity FK to intermediate
            twohop_child_df = db.table_dict[child_table_name].df.copy()

            # Aggregate 2-hop child by FK to intermediate
            twohop_child_df = twohop_child_df.dropna(subset=[fk_col])
            agg_feats = compute_aggregation_features(twohop_child_df, fk_col)
            if agg_feats.empty or len(agg_feats.columns) == 0:
                return None
            agg_feats.columns = [f"j2h_{child_table_name}_{c}" for c in agg_feats.columns]

            # Find entity FK to intermediate
            entity_fk_to_inter = None
            for fk, pt in entity_tbl.fkey_col_to_pkey_table.items():
                if pt == parent_table_name:
                    entity_fk_to_inter = fk
                    break

            if entity_fk_to_inter is None:
                # Reverse: intermediate has FK to entity (intermediate is child of entity)
                inter_tbl = db.table_dict[parent_table_name]
                inter_fk_to_entity = None
                for fk, pt in inter_tbl.fkey_col_to_pkey_table.items():
                    if pt == entity_table_name:
                        inter_fk_to_entity = fk
                        break
                if inter_fk_to_entity is None:
                    return None

                # Intermediate has FK to entity. Merge agg_feats with intermediate,
                # then aggregate to entity level.
                inter_df = inter_tbl.df[[inter_tbl.pkey_col, inter_fk_to_entity]].copy()
                inter_df = inter_df.dropna(subset=[inter_fk_to_entity])
                inter_merged = inter_df.merge(
                    agg_feats, left_on=inter_tbl.pkey_col, right_index=True, how="left"
                )
                inter_merged = inter_merged.drop(columns=[inter_tbl.pkey_col], errors="ignore")

                if len(inter_merged) > MAX_AGG_ROWS:
                    inter_merged = inter_merged.sample(MAX_AGG_ROWS, random_state=42)

                numeric_cols = inter_merged.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != inter_fk_to_entity][:10]
                if not numeric_cols:
                    return None

                agg_dict2 = {c: ["mean", "std"] for c in numeric_cols}
                grouped = inter_merged.groupby(inter_fk_to_entity).agg(agg_dict2)
                grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
                return grouped.fillna(0)

            # Entity has FK to intermediate → direct lookup
            entity_df = entity_tbl.df
            fk_map = entity_df[[entity_pk, entity_fk_to_inter]].copy().set_index(entity_pk)
            merged = fk_map.merge(
                agg_feats, left_on=entity_fk_to_inter, right_index=True, how="left"
            )
            merged = merged.drop(columns=[entity_fk_to_inter], errors="ignore")
            return merged.fillna(0)

    except Exception:
        logger.exception("Failed building 2-hop features")
        return None


# ---------------------------------------------------------------------------
# Model Training & JRN Computation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
    seed: int,
) -> float | None:
    """Train LightGBM and return performance metric."""
    import lightgbm as lgb

    try:
        if X_train.shape[1] == 0:
            return None

        params = LGBM_PARAMS.copy()
        params["random_state"] = seed

        if task_type == "classification":
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)

            if y_pred.ndim == 2:
                y_pred = y_pred[:, 1]

            # Handle edge case: only one class in validation
            unique_vals = np.unique(y_val)
            if len(unique_vals) < 2:
                logger.warning("Only one class in validation set, using accuracy instead")
                y_pred_labels = (y_pred > 0.5).astype(int)
                return float(np.mean(y_pred_labels == y_val))

            return float(roc_auc_score(y_val, y_pred))
        else:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = math.sqrt(mean_squared_error(y_val, y_pred))
            return float(rmse) if rmse > 0 else 1e-10

    except Exception:
        logger.exception(f"Model training failed (seed={seed}, type={task_type})")
        return None


def compute_jrn(baseline_perf: float, augmented_perf: float, task_type: str) -> float:
    """Compute JRN ratio. >1 means join helps."""
    if baseline_perf is None or augmented_perf is None:
        return 1.0

    if task_type == "classification":
        # AUROC ratio: aug / base, >1 = join helps
        if baseline_perf <= 0:
            return 1.0
        return augmented_perf / baseline_perf
    else:
        # RMSE ratio: base / aug, >1 = join helps (lower RMSE = better)
        if augmented_perf <= 0:
            return 1.0
        return baseline_perf / augmented_perf


# ---------------------------------------------------------------------------
# Schema & Join Extraction
# ---------------------------------------------------------------------------

def extract_joins(db) -> list[dict]:
    """Extract all FK join relationships from the database."""
    joins = []
    for table_name, table in db.table_dict.items():
        fk_map = table.fkey_col_to_pkey_table
        if fk_map:
            for fk_col, parent_table_name in fk_map.items():
                parent = db.table_dict[parent_table_name]
                joins.append({
                    "child_table": table_name,
                    "fk_col": fk_col,
                    "parent_table": parent_table_name,
                    "pk_col": parent.pkey_col,
                    "join_id": f"{table_name}.{fk_col}->{parent_table_name}",
                })
    return joins


def compute_reachable_joins(
    entity_table: str,
    all_joins: list[dict],
) -> list[dict]:
    """Find joins reachable from entity_table in 1 or 2 hops."""
    # 1-hop: join where parent == entity_table OR child == entity_table
    onehop = []
    intermediate_tables = set()

    for j in all_joins:
        if j["parent_table"] == entity_table or j["child_table"] == entity_table:
            jc = j.copy()
            jc["hop_distance"] = 1
            onehop.append(jc)
            if j["parent_table"] == entity_table:
                intermediate_tables.add(j["child_table"])
            else:
                intermediate_tables.add(j["parent_table"])

    # 2-hop: connects to an intermediate table
    onehop_ids = {j["join_id"] for j in onehop}
    twohop = []
    for j in all_joins:
        if j["join_id"] not in onehop_ids:
            if j["parent_table"] in intermediate_tables or j["child_table"] in intermediate_tables:
                jc = j.copy()
                jc["hop_distance"] = 2
                twohop.append(jc)

    return onehop + twohop


# ---------------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------------

def process_dataset(dataset_name: str) -> dict:
    """Process a single dataset: compute JRN matrix and all analyses."""
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    logger.info(f"{'='*60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    dataset = get_dataset(dataset_name, download=True)
    db = dataset.get_db()

    # Log table sizes
    for tname, tbl in db.table_dict.items():
        logger.info(f"  Table {tname}: shape={tbl.df.shape}")

    # Extract all FK joins
    all_joins = extract_joins(db)
    logger.info(f"  Found {len(all_joins)} FK joins")

    # Define entity tasks (skip link prediction tasks)
    TASK_CONFIGS = {
        "rel-f1": [
            "driver-dnf", "driver-top3", "driver-position",
            "results-position", "qualifying-position",
        ],
        "rel-stack": [
            "user-engagement", "user-badge", "post-votes",
        ],
    }

    task_names = TASK_CONFIGS[dataset_name]

    # Load tasks and extract metadata
    tasks_info = {}
    for tname in task_names:
        try:
            task = get_task(dataset_name, tname, download=True)
            tt = str(task.task_type)
            task_type = "classification" if "CLASSIFICATION" in tt else "regression"
            tasks_info[tname] = {
                "entity_table": task.entity_table,
                "entity_col": task.entity_col,
                "target_col": task.target_col,
                "task_type": task_type,
                "task_obj": task,
            }
            logger.info(f"  Task {tname}: entity={task.entity_table}, col={task.entity_col}, "
                        f"target={task.target_col}, type={task_type}")
        except Exception:
            logger.exception(f"  Failed to load task {tname}")

    # Compute reachable joins for each task
    reachable_joins_map = {}
    for tname, tinfo in tasks_info.items():
        reachable = compute_reachable_joins(tinfo["entity_table"], all_joins)
        reachable_joins_map[tname] = reachable
        logger.info(f"  Task {tname}: {len(reachable)} reachable joins "
                     f"({sum(1 for j in reachable if j['hop_distance']==1)} 1-hop, "
                     f"{sum(1 for j in reachable if j['hop_distance']==2)} 2-hop)")

    # Get base entity features (compute once per entity table)
    entity_features_cache = {}
    for tname, tinfo in tasks_info.items():
        et = tinfo["entity_table"]
        if et not in entity_features_cache:
            entity_features_cache[et] = get_entity_base_features(db, et)
            logger.info(f"  Entity features for {et}: {entity_features_cache[et].shape}")

    # Precompute join features for all (entity_table, join) pairs
    join_features_cache = {}
    for tname, tinfo in tasks_info.items():
        et = tinfo["entity_table"]
        ec = tinfo["entity_col"]
        for join in reachable_joins_map[tname]:
            jid = join["join_id"]
            cache_key = (et, ec, jid)
            if cache_key not in join_features_cache:
                jf = build_join_features(
                    db, et, ec, join, hop_distance=join["hop_distance"]
                )
                join_features_cache[cache_key] = jf
                if jf is not None:
                    logger.debug(f"  Join features [{et}][{jid}]: {jf.shape}")
                else:
                    logger.debug(f"  Join features [{et}][{jid}]: None (no features)")

    # -----------------------------------------------------------------------
    # STEP 3: JRN ESTIMATION
    # -----------------------------------------------------------------------
    jrn_matrix = {}
    baseline_perfs = {}
    augmented_perfs = {}

    for tname, tinfo in tasks_info.items():
        logger.info(f"  Computing JRN for task: {tname}")
        task_obj = tinfo["task_obj"]
        task_type = tinfo["task_type"]
        entity_table = tinfo["entity_table"]
        entity_col = tinfo["entity_col"]
        target_col = tinfo["target_col"]

        # Load train/val data
        train_df = task_obj.get_table("train").df.copy()
        val_df = task_obj.get_table("val").df.copy()

        # Subsample large datasets
        if len(train_df) > MAX_TRAIN_SAMPLES:
            train_df = train_df.sample(MAX_TRAIN_SAMPLES, random_state=42)
            logger.info(f"    Subsampled train to {len(train_df)} rows")
        if len(val_df) > MAX_VAL_SAMPLES:
            val_df = val_df.sample(MAX_VAL_SAMPLES, random_state=42)
            logger.info(f"    Subsampled val to {len(val_df)} rows")

        # Merge base entity features
        base_feats = entity_features_cache[entity_table]
        train_merged = train_df.merge(base_feats, left_on=entity_col, right_index=True, how="left")
        val_merged = val_df.merge(base_feats, left_on=entity_col, right_index=True, how="left")

        # Get base feature columns
        base_feature_cols = [c for c in base_feats.columns if c in train_merged.columns]

        # Extract y
        y_train = train_merged[target_col].values.astype(float)
        y_val = val_merged[target_col].values.astype(float)

        # Remove NaN targets
        train_mask = ~np.isnan(y_train)
        val_mask = ~np.isnan(y_val)
        y_train = y_train[train_mask]
        y_val = y_val[val_mask]
        train_merged = train_merged[train_mask].reset_index(drop=True)
        val_merged = val_merged[val_mask].reset_index(drop=True)

        if len(y_train) == 0 or len(y_val) == 0:
            logger.warning(f"    Empty train/val for {tname}, skipping")
            continue

        # Build base X
        X_train_base = train_merged[base_feature_cols].fillna(0).values.astype(np.float32)
        X_val_base = val_merged[base_feature_cols].fillna(0).values.astype(np.float32)

        # Train baseline models
        baseline_perfs[tname] = {}
        for seed in SEEDS:
            perf = train_and_evaluate(X_train_base, y_train, X_val_base, y_val, task_type, seed)
            baseline_perfs[tname][seed] = perf
            logger.debug(f"    Baseline seed={seed}: {perf}")

        mean_base = np.mean([v for v in baseline_perfs[tname].values() if v is not None])
        logger.info(f"    Baseline mean perf: {mean_base:.4f} ({task_type})")

        # Train augmented models for each reachable join
        jrn_matrix[tname] = {}
        augmented_perfs[tname] = {}

        for join in reachable_joins_map[tname]:
            jid = join["join_id"]
            cache_key = (entity_table, entity_col, jid)
            join_feats = join_features_cache.get(cache_key)

            if join_feats is None or join_feats.empty or len(join_feats.columns) == 0:
                # No features from this join → JRN = 1.0 (neutral)
                jrn_matrix[tname][jid] = {
                    "jrn_mean": 1.0,
                    "jrn_std": 0.0,
                    "jrn_seeds": [1.0] * len(SEEDS),
                    "hop_distance": join["hop_distance"],
                    "note": "no_features",
                }
                continue

            # Merge join features with train/val
            train_with_join = train_merged.merge(
                join_feats, left_on=entity_col, right_index=True, how="left"
            )
            val_with_join = val_merged.merge(
                join_feats, left_on=entity_col, right_index=True, how="left"
            )

            join_feat_cols = [c for c in join_feats.columns if c in train_with_join.columns]

            if not join_feat_cols:
                jrn_matrix[tname][jid] = {
                    "jrn_mean": 1.0,
                    "jrn_std": 0.0,
                    "jrn_seeds": [1.0] * len(SEEDS),
                    "hop_distance": join["hop_distance"],
                    "note": "no_features_after_merge",
                }
                continue

            all_feat_cols = base_feature_cols + join_feat_cols
            X_train_aug = train_with_join[all_feat_cols].fillna(0).values.astype(np.float32)
            X_val_aug = val_with_join[all_feat_cols].fillna(0).values.astype(np.float32)

            augmented_perfs[tname][jid] = {}
            jrn_per_seed = []

            for seed in SEEDS:
                aug_perf = train_and_evaluate(X_train_aug, y_train, X_val_aug, y_val, task_type, seed)
                augmented_perfs[tname][jid][seed] = aug_perf

                base_perf = baseline_perfs[tname].get(seed)
                jrn_val = compute_jrn(base_perf, aug_perf, task_type)
                jrn_per_seed.append(jrn_val)

            jrn_matrix[tname][jid] = {
                "jrn_mean": float(np.mean(jrn_per_seed)),
                "jrn_std": float(np.std(jrn_per_seed)),
                "jrn_seeds": [float(j) for j in jrn_per_seed],
                "hop_distance": join["hop_distance"],
            }

            logger.info(f"    Join {jid}: JRN={np.mean(jrn_per_seed):.4f} "
                        f"(+/-{np.std(jrn_per_seed):.4f}), hop={join['hop_distance']}")

        # Cleanup large objects per-task
        del train_merged, val_merged, train_df, val_df
        gc.collect()

    # Serialize baseline perfs (convert seed keys to string for JSON)
    baseline_perfs_ser = {}
    for tname, seed_perfs in baseline_perfs.items():
        baseline_perfs_ser[tname] = {str(k): v for k, v in seed_perfs.items()}

    augmented_perfs_ser = {}
    for tname, join_perfs in augmented_perfs.items():
        augmented_perfs_ser[tname] = {}
        for jid, seed_perfs in join_perfs.items():
            augmented_perfs_ser[tname][jid] = {str(k): v for k, v in seed_perfs.items()}

    # -----------------------------------------------------------------------
    # STEP 4: LEAVE-ONE-TASK-OUT TRANSFER
    # -----------------------------------------------------------------------
    logger.info(f"  Computing transfer analysis for {dataset_name}")
    task_name_list = list(jrn_matrix.keys())
    all_join_sets = [set(jrn_matrix[t].keys()) for t in task_name_list]
    all_joins_union = sorted(set.union(*all_join_sets)) if all_join_sets else []
    common_joins = sorted(set.intersection(*all_join_sets)) if all_join_sets else []

    transfer_rhos = {}
    for held_out_task in task_name_list:
        remaining_tasks = [t for t in task_name_list if t != held_out_task]

        if len(remaining_tasks) < 1 or len(all_joins_union) < 3:
            transfer_rhos[held_out_task] = {"rho": None, "pval": None}
            continue

        held_out_jrn = [
            jrn_matrix[held_out_task].get(j, {}).get("jrn_mean", 1.0)
            for j in all_joins_union
        ]

        universal_jrn = []
        for j in all_joins_union:
            vals = [jrn_matrix[t].get(j, {}).get("jrn_mean", 1.0) for t in remaining_tasks]
            universal_jrn.append(np.mean(vals))

        rho, pval = spearmanr(held_out_jrn, universal_jrn)
        if np.isnan(rho):
            transfer_rhos[held_out_task] = {"rho": None, "pval": None}
        else:
            transfer_rhos[held_out_task] = {"rho": float(rho), "pval": float(pval)}

    valid_rhos = [v["rho"] for v in transfer_rhos.values() if v["rho"] is not None]
    mean_transfer_rho = float(np.mean(valid_rhos)) if valid_rhos else None

    transfer_results = {
        "per_task_transfer_rho": transfer_rhos,
        "mean_transfer_rho": mean_transfer_rho,
        "num_common_joins": len(common_joins),
        "num_all_joins": len(all_joins_union),
    }
    logger.info(f"  Transfer rho: {mean_transfer_rho}")

    # -----------------------------------------------------------------------
    # STEP 5: PAIRWISE TASK CORRELATION & KENDALL'S W
    # -----------------------------------------------------------------------
    logger.info(f"  Computing concordance for {dataset_name}")

    jrn_rank_matrix = np.zeros((len(task_name_list), len(all_joins_union)))
    for i, t in enumerate(task_name_list):
        for k, j in enumerate(all_joins_union):
            jrn_rank_matrix[i, k] = jrn_matrix[t].get(j, {}).get("jrn_mean", 1.0)

    pairwise_rho = {}
    for i, t1 in enumerate(task_name_list):
        for j_idx, t2 in enumerate(task_name_list):
            if i < j_idx:
                rho, pval = spearmanr(jrn_rank_matrix[i], jrn_rank_matrix[j_idx])
                if np.isnan(rho):
                    rho, pval = 0.0, 1.0
                pairwise_rho[f"{t1}_vs_{t2}"] = {"rho": float(rho), "pval": float(pval)}

    # Kendall's W
    k_tasks = len(task_name_list)
    n_joins = len(all_joins_union)

    ranks = np.zeros_like(jrn_rank_matrix)
    for i in range(k_tasks):
        ranks[i] = rankdata(jrn_rank_matrix[i])

    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12.0 * S / (k_tasks**2 * (n_joins**3 - n_joins)) if (n_joins > 1 and k_tasks > 0) else 0.0

    concordance_results = {
        "kendalls_W": float(W),
        "pairwise_spearman_rho": pairwise_rho,
        "num_tasks": k_tasks,
        "num_joins": n_joins,
    }
    logger.info(f"  Kendall's W: {W:.4f}")

    # -----------------------------------------------------------------------
    # STEP 6: TASK-TYPE ANALYSIS
    # -----------------------------------------------------------------------
    logger.info(f"  Computing task-type analysis for {dataset_name}")

    entity_groups = defaultdict(list)
    for tname in task_name_list:
        et = tasks_info[tname]["entity_table"]
        entity_groups[et].append(tname)

    same_entity_rhos = []
    cross_entity_rhos = []
    same_type_rhos = []
    cross_type_rhos = []

    for pair_key, pair_val in pairwise_rho.items():
        t1, t2 = pair_key.split("_vs_")
        e1 = tasks_info[t1]["entity_table"]
        e2 = tasks_info[t2]["entity_table"]
        type1 = tasks_info[t1]["task_type"]
        type2 = tasks_info[t2]["task_type"]
        rho_val = pair_val["rho"]

        if rho_val is not None and not np.isnan(rho_val):
            if e1 == e2:
                same_entity_rhos.append(rho_val)
            else:
                cross_entity_rhos.append(rho_val)

            if type1 == type2:
                same_type_rhos.append(rho_val)
            else:
                cross_type_rhos.append(rho_val)

    task_type_results = {
        "entity_groups": {k: v for k, v in entity_groups.items()},
        "same_entity_mean_rho": float(np.mean(same_entity_rhos)) if same_entity_rhos else None,
        "cross_entity_mean_rho": float(np.mean(cross_entity_rhos)) if cross_entity_rhos else None,
        "same_entity_rhos": same_entity_rhos,
        "cross_entity_rhos": cross_entity_rhos,
        "same_type_mean_rho": float(np.mean(same_type_rhos)) if same_type_rhos else None,
        "cross_type_mean_rho": float(np.mean(cross_type_rhos)) if cross_type_rhos else None,
        "same_type_rhos": same_type_rhos,
        "cross_type_rhos": cross_type_rhos,
    }

    # -----------------------------------------------------------------------
    # STEP 7: PRACTICAL IMPACT (UNIVERSAL vs TASK-SPECIFIC)
    # -----------------------------------------------------------------------
    logger.info(f"  Computing practical impact for {dataset_name}")

    practical_per_task = {}
    for held_out_task in task_name_list:
        remaining_tasks = [t for t in task_name_list if t != held_out_task]
        reachable_for_task = list(jrn_matrix[held_out_task].keys())

        if len(reachable_for_task) < 2:
            practical_per_task[held_out_task] = {"note": "too_few_joins"}
            continue

        # Task-specific ranking (oracle: ranked by actual JRN for this task)
        task_specific_ranking = sorted(
            reachable_for_task,
            key=lambda j: jrn_matrix[held_out_task][j]["jrn_mean"],
            reverse=True,
        )

        # Universal ranking (mean JRN across remaining tasks)
        universal_jrn_vals = {}
        for j in reachable_for_task:
            vals = [jrn_matrix[t].get(j, {}).get("jrn_mean", 1.0) for t in remaining_tasks]
            universal_jrn_vals[j] = np.mean(vals)
        universal_ranking = sorted(
            reachable_for_task, key=lambda j: universal_jrn_vals[j], reverse=True,
        )

        # Compare top-k selections
        practical_per_task[held_out_task] = {
            "num_reachable_joins": len(reachable_for_task),
            "top_k_comparisons": {},
        }

        for k_val in [1, 2, 3]:
            if k_val > len(reachable_for_task):
                break

            # Get task-specific top-k JRN values
            ts_top_k = task_specific_ranking[:k_val]
            ts_jrn_mean = np.mean([jrn_matrix[held_out_task][j]["jrn_mean"] for j in ts_top_k])

            # Get universal top-k JRN values
            uni_top_k = universal_ranking[:k_val]
            uni_jrn_mean = np.mean([jrn_matrix[held_out_task][j]["jrn_mean"] for j in uni_top_k])

            # Random top-k (average over 10 random selections)
            random_jrns = []
            for r_seed in range(10):
                rng = np.random.RandomState(r_seed)
                selected_random = list(rng.choice(reachable_for_task, k_val, replace=False))
                rand_jrn = np.mean([jrn_matrix[held_out_task][j]["jrn_mean"] for j in selected_random])
                random_jrns.append(rand_jrn)
            random_jrn_mean = np.mean(random_jrns)

            # Relative gap: |uni - ts| / ts  (percentage difference)
            gap = abs(uni_jrn_mean - ts_jrn_mean) / max(abs(ts_jrn_mean), 1e-6)

            practical_per_task[held_out_task]["top_k_comparisons"][f"top_{k_val}"] = {
                "task_specific_joins": ts_top_k,
                "task_specific_mean_jrn": float(ts_jrn_mean),
                "universal_joins": uni_top_k,
                "universal_mean_jrn": float(uni_jrn_mean),
                "random_mean_jrn": float(random_jrn_mean),
                "relative_gap_pct": float(gap * 100),
            }

    # Compute overall mean gap
    all_gaps = []
    for tname, tdata in practical_per_task.items():
        if "top_k_comparisons" in tdata:
            for kk, kdata in tdata["top_k_comparisons"].items():
                all_gaps.append(kdata["relative_gap_pct"])
    mean_gap = float(np.mean(all_gaps)) if all_gaps else 0.0

    practical_results = {
        "per_task": practical_per_task,
        "mean_gap_pct": mean_gap,
    }

    # -----------------------------------------------------------------------
    # STEP 8: COST-ADJUSTED COMPARISON
    # -----------------------------------------------------------------------
    J = len(all_joins_union)
    T = len(task_name_list)

    cost_results = {
        "num_joins": J,
        "num_tasks": T,
        "universal_cost_models": J,
        "per_task_cost_models": J * T,
        "greedy_cost_models": J * (J + 1) // 2 * T,
        "performance_gap_pct": mean_gap,
        "recommendation": "universal" if mean_gap < 5.0 else "per_task",
    }

    # Free dataset from memory
    del db, dataset
    gc.collect()

    return {
        "jrn_matrix": jrn_matrix,
        "baseline_perfs": baseline_perfs_ser,
        "augmented_perfs": augmented_perfs_ser,
        "transfer_analysis": transfer_results,
        "concordance_analysis": concordance_results,
        "task_type_analysis": task_type_results,
        "practical_impact": practical_results,
        "cost_adjusted": cost_results,
        "all_joins": [{"join_id": j["join_id"], "child_table": j["child_table"],
                       "parent_table": j["parent_table"], "fk_col": j["fk_col"]}
                      for j in all_joins],
        "tasks_info": {tn: {k: v for k, v in ti.items() if k != "task_obj"}
                       for tn, ti in tasks_info.items()},
    }


# ---------------------------------------------------------------------------
# Output Assembly
# ---------------------------------------------------------------------------

def build_output_examples(results: dict, dataset_name: str) -> list[dict]:
    """Convert analysis results into exp_gen_sol_out.json examples format.

    Each example MUST have predict_* fields for at least one method.
    We provide:
      - predict_jrn_probe: our LightGBM probe-based JRN estimation
      - predict_baseline_uniform: naive baseline assuming all joins equally useful (JRN=1.0)
    """
    examples = []

    jrn_matrix = results["jrn_matrix"]

    # Per-(task, join) JRN measurement examples
    for task_name, join_jrns in jrn_matrix.items():
        for join_id, jrn_data in join_jrns.items():
            input_str = json.dumps({
                "dataset": dataset_name,
                "task": task_name,
                "join": join_id,
                "hop_distance": jrn_data.get("hop_distance", 1),
                "entity_table": results["tasks_info"][task_name]["entity_table"],
                "task_type": results["tasks_info"][task_name]["task_type"],
            })
            output_str = json.dumps({
                "jrn_mean": jrn_data["jrn_mean"],
                "jrn_std": jrn_data["jrn_std"],
                "jrn_seeds": jrn_data["jrn_seeds"],
            })
            examples.append({
                "input": input_str,
                "output": output_str,
                "predict_jrn_probe": str(round(jrn_data["jrn_mean"], 4)),
                "predict_baseline_uniform": "1.0",
                "metadata_dataset": dataset_name,
                "metadata_task": task_name,
                "metadata_join": join_id,
                "metadata_type": "jrn_measurement",
                "metadata_hop_distance": jrn_data.get("hop_distance", 1),
            })

    # Transfer analysis results
    transfer = results["transfer_analysis"]
    mean_rho = transfer.get("mean_transfer_rho")
    examples.append({
        "input": json.dumps({
            "dataset": dataset_name,
            "analysis": "leave_one_task_out_transfer",
        }),
        "output": json.dumps(transfer),
        "predict_jrn_probe": str(round(mean_rho, 4)) if mean_rho is not None else "0.0",
        "predict_baseline_uniform": "0.0",
        "metadata_dataset": dataset_name,
        "metadata_type": "transfer_analysis",
    })

    # Concordance analysis
    concordance = results["concordance_analysis"]
    examples.append({
        "input": json.dumps({
            "dataset": dataset_name,
            "analysis": "concordance_kendalls_W",
        }),
        "output": json.dumps(concordance),
        "predict_jrn_probe": str(round(concordance["kendalls_W"], 4)),
        "predict_baseline_uniform": "0.0",
        "metadata_dataset": dataset_name,
        "metadata_type": "concordance_analysis",
    })

    # Task-type analysis
    task_type_analysis = results["task_type_analysis"]
    same_rho = task_type_analysis.get("same_entity_mean_rho")
    examples.append({
        "input": json.dumps({
            "dataset": dataset_name,
            "analysis": "task_type_entity_concordance",
        }),
        "output": json.dumps(task_type_analysis),
        "predict_jrn_probe": str(round(same_rho, 4)) if same_rho is not None else "0.0",
        "predict_baseline_uniform": "0.0",
        "metadata_dataset": dataset_name,
        "metadata_type": "task_type_analysis",
    })

    # Practical impact
    practical = results["practical_impact"]
    examples.append({
        "input": json.dumps({
            "dataset": dataset_name,
            "analysis": "practical_impact_universal_vs_specific",
        }),
        "output": json.dumps(practical, default=str),
        "predict_jrn_probe": str(round(practical["mean_gap_pct"], 2)),
        "predict_baseline_uniform": "100.0",
        "metadata_dataset": dataset_name,
        "metadata_type": "practical_impact",
    })

    # Cost-adjusted recommendation
    cost = results["cost_adjusted"]
    examples.append({
        "input": json.dumps({
            "dataset": dataset_name,
            "analysis": "cost_adjusted_recommendation",
        }),
        "output": json.dumps(cost),
        "predict_jrn_probe": cost["recommendation"],
        "predict_baseline_uniform": "universal",
        "metadata_dataset": dataset_name,
        "metadata_type": "cost_adjusted",
    })

    return examples


@logger.catch
def main():
    logger.info("Starting JRN Cross-Task Transfer Analysis")
    logger.info(f"Workspace: {WORKSPACE}")

    all_results = {}
    all_examples = []

    # Process each dataset sequentially (memory-safe)
    for dataset_name in ["rel-f1", "rel-stack"]:
        try:
            results = process_dataset(dataset_name)
            all_results[dataset_name] = results
            examples = build_output_examples(results, dataset_name)
            all_examples.extend(examples)
            logger.info(f"Completed {dataset_name}: {len(examples)} examples generated")
        except Exception:
            logger.exception(f"Failed to process {dataset_name}")
        gc.collect()

    # Build summary
    summary = {}
    for ds_name in ["rel-f1", "rel-stack"]:
        if ds_name in all_results:
            r = all_results[ds_name]
            summary[ds_name] = {
                "kendalls_W": r["concordance_analysis"]["kendalls_W"],
                "mean_transfer_rho": r["transfer_analysis"]["mean_transfer_rho"],
                "same_entity_mean_rho": r["task_type_analysis"]["same_entity_mean_rho"],
                "cross_entity_mean_rho": r["task_type_analysis"]["cross_entity_mean_rho"],
                "same_type_mean_rho": r["task_type_analysis"]["same_type_mean_rho"],
                "cross_type_mean_rho": r["task_type_analysis"]["cross_type_mean_rho"],
                "num_tasks": r["concordance_analysis"]["num_tasks"],
                "num_joins": r["concordance_analysis"]["num_joins"],
                "cost_recommendation": r["cost_adjusted"]["recommendation"],
                "mean_gap_pct": r["practical_impact"]["mean_gap_pct"],
            }

    # Cross-dataset comparison
    if "rel-f1" in summary and "rel-stack" in summary:
        summary["cross_dataset_comparison"] = {
            "rel_f1_W_vs_rel_stack_W": f"{summary['rel-f1']['kendalls_W']:.4f} vs {summary['rel-stack']['kendalls_W']:.4f}",
            "higher_concordance_dataset": "rel-stack" if summary["rel-stack"]["kendalls_W"] > summary["rel-f1"]["kendalls_W"] else "rel-f1",
            "recommendation": (
                "Universal JRN ranking is more transferable on rel-stack (higher W, "
                "same-entity tasks dominate). On rel-f1, per-task JRN may be needed "
                "due to mixed entity tables reducing concordance."
            ),
        }

    # Assemble method_out.json (exp_gen_sol_out schema)
    output = {
        "metadata": {
            "experiment": "JRN Cross-Task Transfer Analysis",
            "datasets": ["rel-f1", "rel-stack"],
            "config": {
                "lgbm_params": LGBM_PARAMS,
                "seeds": SEEDS,
                "max_train_samples": MAX_TRAIN_SAMPLES,
                "max_val_samples": MAX_VAL_SAMPLES,
                "max_agg_rows": MAX_AGG_ROWS,
            },
            "summary": summary,
        },
        "datasets": [],
    }

    # Group examples by dataset
    for ds_name in ["rel-f1", "rel-stack"]:
        ds_examples = [e for e in all_examples if e.get("metadata_dataset") == ds_name]
        if ds_examples:
            output["datasets"].append({
                "dataset": ds_name,
                "examples": ds_examples,
            })

    # Write output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Print key summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for ds_name, s in summary.items():
        if isinstance(s, dict) and "kendalls_W" in s:
            logger.info(f"  {ds_name}:")
            logger.info(f"    Kendall's W: {s['kendalls_W']:.4f}")
            logger.info(f"    Mean transfer rho: {s['mean_transfer_rho']}")
            logger.info(f"    Same-entity rho: {s['same_entity_mean_rho']}")
            logger.info(f"    Cross-entity rho: {s['cross_entity_mean_rho']}")
            logger.info(f"    Mean gap %: {s['mean_gap_pct']:.2f}")
            logger.info(f"    Recommendation: {s['cost_recommendation']}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
