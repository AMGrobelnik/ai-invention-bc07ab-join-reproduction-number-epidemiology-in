#!/usr/bin/env python3
"""JRN Multiplicative Compounding: 2x2 Probe Grid on rel-f1.

Measures individual JRN for all 13 FK joins using both MLP and GBM probes,
measures chain JRN for 12 multi-hop chains, and tests 4 alternative compounding
models (multiplicative, additive, bottleneck, log-linear) to resolve the
compounding question on the rel-f1 (Formula 1) relational database.
"""

import json
import sys
import os
import gc
import time
import warnings
import math
import resource
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from loguru import logger

warnings.filterwarnings("ignore")

# ============================================================
# SETUP: Paths, hardware, logging, memory
# ============================================================
WORKSPACE = Path(__file__).parent
DEP_DATA = Path(
    "/ai-inventor/aii_pipeline/runs/run__20260309_024817"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)
SEEDS = [42, 123, 7]

LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")


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


def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 29.0


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(14 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# Try LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  GradientBoostingRegressor)

# ============================================================
# FK JOIN DEFINITIONS
# ============================================================
FK_JOINS = [
    ("races", "circuitId", "circuits", "circuitId"),                          # 0
    ("constructor_standings", "raceId", "races", "raceId"),                   # 1
    ("constructor_standings", "constructorId", "constructors", "constructorId"),  # 2
    ("standings", "raceId", "races", "raceId"),                               # 3
    ("standings", "driverId", "drivers", "driverId"),                         # 4
    ("constructor_results", "raceId", "races", "raceId"),                     # 5
    ("constructor_results", "constructorId", "constructors", "constructorId"),# 6
    ("qualifying", "raceId", "races", "raceId"),                              # 7
    ("qualifying", "driverId", "drivers", "driverId"),                        # 8
    ("qualifying", "constructorId", "constructors", "constructorId"),         # 9
    ("results", "raceId", "races", "raceId"),                                 # 10
    ("results", "driverId", "drivers", "driverId"),                           # 11
    ("results", "constructorId", "constructors", "constructorId"),            # 12
]

# Self-supervised tasks per parent table
SS_TASKS = {
    "races":        {"target": "year",        "type": "regression"},
    "circuits":     {"target": "lat",         "type": "regression"},
    "constructors": {"target": "nationality", "type": "classification"},
    "drivers":      {"target": "nationality", "type": "classification"},
}

# Self-supervised tasks at child level (for enrichment JRN)
CHILD_SS_TASKS = {
    "results":                {"target": "points",   "type": "regression"},
    "standings":              {"target": "points",   "type": "regression"},
    "qualifying":             {"target": "position", "type": "regression"},
    "constructor_standings":  {"target": "points",   "type": "regression"},
    "constructor_results":    {"target": "points",   "type": "regression"},
}

DRIVER_TASKS = ["driver-dnf", "driver-top3", "driver-position"]
DRIVER_TASK_TYPES = {
    "driver-dnf": "classification",
    "driver-top3": "classification",
    "driver-position": "regression",
}


# ============================================================
# STEP 1: DATA LOADING
# ============================================================
def load_data(data_path: Path):
    """Load and reconstruct tables, FK metadata, and task samples."""
    logger.info(f"Loading data from {data_path}")
    t0 = time.time()

    with open(data_path, "r") as f:
        raw = json.load(f)
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples in {time.time()-t0:.1f}s")

    table_rows: dict[str, list] = defaultdict(list)
    fk_metadata: list = []
    task_samples: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for ex in examples:
        rt = ex.get("metadata_row_type")
        if rt == "table_row":
            table_rows[ex["metadata_table_name"]].append(ex)
        elif rt == "fk_join_metadata":
            fk_metadata.append(ex)
        elif rt == "task_sample":
            task_name = ex["metadata_task_name"].split("/")[-1]
            fold = ex["metadata_fold_name"]
            task_samples[task_name][fold].append(ex)

    del raw, examples
    gc.collect()

    # Reconstruct tables
    tables: dict[str, pd.DataFrame] = {}
    table_pk_cols: dict[str, str] = {}
    for tname, rows in table_rows.items():
        pk_col = rows[0]["metadata_primary_key_col"]
        table_pk_cols[tname] = pk_col
        records = []
        for r in rows:
            feats = json.loads(r["input"])
            pk_val = r["metadata_primary_key_value"]
            try:
                feats[pk_col] = int(pk_val)
            except ValueError:
                feats[pk_col] = pk_val
            records.append(feats)
        df = pd.DataFrame(records)
        tables[tname] = df
        logger.info(f"  Table {tname}: {len(df)} rows, cols={list(df.columns)[:6]}...")

    del table_rows
    gc.collect()

    # Parse FK metadata
    fk_info = []
    for ex in fk_metadata:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        fk_info.append({**inp, **out})
    logger.info(f"  FK joins: {len(fk_info)}")

    # Reconstruct task datasets
    tasks: dict[str, dict[str, pd.DataFrame]] = {}
    for task_name, folds in task_samples.items():
        task_dfs = {}
        for fold_name, samples in folds.items():
            records = []
            for s in samples:
                feats = json.loads(s["input"])
                label = s["output"]
                try:
                    label = float(label)
                except ValueError:
                    pass
                feats["__label__"] = label
                feats["__task_type__"] = s.get("metadata_task_type", "unknown")
                feats["__entity_table__"] = s.get("metadata_entity_table", "unknown")
                feats["__entity_col__"] = s.get("metadata_entity_col", "unknown")
                records.append(feats)
            task_dfs[fold_name] = pd.DataFrame(records)
        tasks[task_name] = task_dfs
        total = sum(len(df) for df in task_dfs.values())
        logger.info(f"  Task {task_name}: {total} samples, folds={list(task_dfs.keys())}")

    del task_samples
    gc.collect()

    return tables, table_pk_cols, fk_info, tasks


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
def encode_features(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """Convert DataFrame columns to numeric features."""
    if exclude_cols is None:
        exclude_cols = []

    numeric_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in exclude_cols:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_df[col] = series.fillna(0).astype(float)
        elif any(kw in col.lower() for kw in ["date", "dob"]):
            try:
                ts = pd.to_datetime(series, errors="coerce")
                numeric_df[col + "_ts"] = (ts.astype("int64") // 10**9).fillna(0).astype(float)
            except Exception:
                pass
        elif col.lower() == "time":
            # Time strings like "00:00:00" or race times - skip or encode
            pass
        else:
            le = LabelEncoder()
            vals = series.fillna("__MISSING__").astype(str)
            numeric_df[col + "_enc"] = le.fit_transform(vals).astype(float)

    return numeric_df


# ============================================================
# STEP 3: AGGREGATION & ENRICHMENT
# ============================================================
def aggregate_child_to_parent(
    child_df: pd.DataFrame,
    child_features: pd.DataFrame,
    child_fk_col: str,
    parent_df: pd.DataFrame,
    parent_pk_col: str,
    prefix: str = "agg",
) -> pd.DataFrame:
    """Aggregate child features to parent level using mean."""
    child_with_fk = child_features.copy()
    child_with_fk["__fk__"] = child_df[child_fk_col].values

    # Group by FK and compute mean + count
    agg = child_with_fk.groupby("__fk__").mean()
    counts = child_with_fk.groupby("__fk__").size()

    # Align with parent PKs
    parent_pks = parent_df[parent_pk_col]
    result = agg.reindex(parent_pks.values).fillna(0).reset_index(drop=True)
    result.columns = [f"{prefix}_{c}" for c in result.columns]
    result[f"{prefix}_count"] = counts.reindex(parent_pks.values).fillna(0).values

    return result


def aggregate_to_entity_lookup(
    child_df: pd.DataFrame,
    child_features: pd.DataFrame,
    child_fk_col: str,
    prefix: str = "agg",
) -> pd.DataFrame:
    """Aggregate child features grouped by FK col. Returns DataFrame indexed by FK value."""
    child_with_fk = child_features.copy()
    child_with_fk["__fk__"] = child_df[child_fk_col].values
    agg = child_with_fk.groupby("__fk__").mean()
    counts = child_with_fk.groupby("__fk__").size()
    agg.columns = [f"{prefix}_{c}" for c in agg.columns]
    agg[f"{prefix}_count"] = counts
    return agg


def enrich_child_with_parent(
    child_df: pd.DataFrame,
    child_fk_col: str,
    parent_df: pd.DataFrame,
    parent_pk_col: str,
    parent_features: pd.DataFrame,
    prefix: str = "enr",
) -> pd.DataFrame:
    """Enrich each child row with parent features (1:1 lookup)."""
    parent_lookup = parent_features.copy()
    parent_lookup.index = parent_df[parent_pk_col].values
    child_fk_values = child_df[child_fk_col].values
    enriched = parent_lookup.reindex(child_fk_values).fillna(0).reset_index(drop=True)
    enriched.columns = [f"{prefix}_{c}" for c in enriched.columns]
    return enriched


# ============================================================
# STEP 4: PROBE TRAINING
# ============================================================
def train_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe_type: str,
    task_type: str,
    seed: int,
    n_jobs: int = 1,
) -> float:
    """Train probe and return metric (higher = better).

    Classification: AUROC.  Regression: 1/MAE (capped at 100).
    """
    if X_train.shape[0] < 5 or X_test.shape[0] < 5:
        return 0.5 if task_type == "classification" else 0.01
    if X_train.shape[1] == 0:
        return 0.5 if task_type == "classification" else 0.01

    # Standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

    if task_type == "classification":
        y_tr = np.array(y_train, dtype=float).astype(int)
        y_te = np.array(y_test, dtype=float).astype(int)
        n_classes = len(np.unique(y_tr))

        if n_classes < 2:
            return 0.5

        if probe_type == "MLP":
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=80,
                random_state=seed,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
            )
        else:
            if HAS_LGB:
                model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, random_state=seed,
                    n_jobs=n_jobs, verbose=-1, force_col_wise=True,
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, random_state=seed,
                )

        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            logger.debug(f"Probe fit failed: {e}")
            return 0.5

        try:
            y_prob = model.predict_proba(X_te)
            if n_classes == 2:
                return roc_auc_score(y_te, y_prob[:, 1])
            else:
                return roc_auc_score(
                    y_te, y_prob, multi_class="ovr", average="macro",
                    labels=np.arange(y_prob.shape[1]),
                )
        except (ValueError, IndexError):
            # Fallback to accuracy
            y_pred = model.predict(X_te)
            return float(np.mean(y_pred == y_te))

    else:  # regression
        y_tr = np.array(y_train, dtype=float)
        y_te = np.array(y_test, dtype=float)

        if np.std(y_tr) < 1e-8:
            return 0.01

        if probe_type == "MLP":
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=80,
                random_state=seed,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
            )
        else:
            if HAS_LGB:
                model = lgb.LGBMRegressor(
                    n_estimators=200, max_depth=6, random_state=seed,
                    n_jobs=n_jobs, verbose=-1, force_col_wise=True,
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, random_state=seed,
                )

        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            logger.debug(f"Probe fit failed: {e}")
            return 0.01

        y_pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        if mae < 1e-8:
            return 100.0
        return min(1.0 / mae, 100.0)


def compute_jrn_multi_seed(
    X_base_train, X_base_test,
    X_aug_train, X_aug_test,
    y_train, y_test,
    probe_type, task_type, seeds,
    n_jobs=1,
) -> dict:
    """Compute JRN across multiple seeds."""
    m_bases, m_joins, jrns = [], [], []
    for seed in seeds:
        mb = train_and_eval_probe(X_base_train, y_train, X_base_test, y_test,
                                  probe_type, task_type, seed, n_jobs)
        mj = train_and_eval_probe(X_aug_train, y_train, X_aug_test, y_test,
                                  probe_type, task_type, seed, n_jobs)
        jrn = mj / mb if mb > 1e-8 else 1.0
        m_bases.append(mb)
        m_joins.append(mj)
        jrns.append(jrn)

    return {
        "M_base_mean": float(np.mean(m_bases)),
        "M_base_std": float(np.std(m_bases)),
        "M_join_mean": float(np.mean(m_joins)),
        "M_join_std": float(np.std(m_joins)),
        "JRN_mean": float(np.mean(jrns)),
        "JRN_std": float(np.std(jrns)),
        "JRN_values": [float(j) for j in jrns],
    }


# ============================================================
# STEP 5: INDIVIDUAL JRN ESTIMATION
# ============================================================
def prepare_ss_split(
    table_features: pd.DataFrame,
    table_df: pd.DataFrame,
    target_col: str,
    task_type: str,
    seed: int = 42,
):
    """Prepare self-supervised train/test split at table level."""
    if task_type == "classification":
        enc_col = target_col + "_enc"
        if enc_col in table_features.columns:
            y = table_features[enc_col].values.astype(int)
            X = table_features.drop(columns=[enc_col], errors="ignore")
        elif target_col in table_df.columns:
            le = LabelEncoder()
            y = le.fit_transform(table_df[target_col].fillna("__MISSING__").astype(str))
            X = table_features.copy()
        else:
            return None
    else:
        if target_col in table_features.columns:
            y = table_features[target_col].values.astype(float)
            X = table_features.drop(columns=[target_col], errors="ignore")
        elif target_col in table_df.columns:
            y = pd.to_numeric(table_df[target_col], errors="coerce").fillna(0).values
            X = table_features.copy()
        else:
            return None

    if len(y) < 10:
        return None

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=seed)
    return X.values, y, tr_idx, te_idx


def measure_all_individual_jrn(tables, pk_cols, table_features, tasks):
    """Measure individual JRN for all 13 FK joins x tasks x probes."""
    all_results = {}

    for join_idx, (child_t, fk_col, parent_t, pk_col) in enumerate(FK_JOINS):
        join_label = f"{child_t}.{fk_col}->{parent_t}.{pk_col}"
        logger.info(f"--- Join {join_idx}: {join_label} ---")

        if child_t not in tables or parent_t not in tables:
            logger.warning(f"  Missing table(s), skipping join {join_idx}")
            continue

        child_df = tables[child_t]
        parent_df = tables[parent_t]
        child_feats = table_features[child_t]
        parent_feats = table_features[parent_t]

        # Aggregate child -> parent
        agg_feats = aggregate_child_to_parent(
            child_df, child_feats, fk_col, parent_df, pk_col,
            prefix=f"agg_{child_t}",
        )

        # --- Self-supervised task at parent ---
        if parent_t in SS_TASKS:
            ss = SS_TASKS[parent_t]
            split = prepare_ss_split(parent_feats, parent_df, ss["target"], ss["type"])
            if split is not None:
                X_parent, y, tr_idx, te_idx = split
                X_aug = np.hstack([X_parent, agg_feats.values])

                for probe in ["MLP", "GBM"]:
                    key = f"join{join_idx}_ss_{parent_t}_{ss['target']}_{probe}"
                    res = compute_jrn_multi_seed(
                        X_parent[tr_idx], X_parent[te_idx],
                        X_aug[tr_idx], X_aug[te_idx],
                        y[tr_idx], y[te_idx],
                        probe, ss["type"], SEEDS, n_jobs=NUM_CPUS,
                    )
                    all_results[key] = {
                        **res,
                        "join_idx": join_idx,
                        "join_label": join_label,
                        "task": f"ss_{parent_t}_{ss['target']}",
                        "task_type": ss["type"],
                        "probe": probe,
                        "parent_table": parent_t,
                    }
                    logger.info(
                        f"  {probe} ss_{parent_t}_{ss['target']}: "
                        f"JRN={res['JRN_mean']:.4f}+/-{res['JRN_std']:.4f} "
                        f"(base={res['M_base_mean']:.4f}, join={res['M_join_mean']:.4f})"
                    )

        # --- Driver tasks (only for joins ending at drivers) ---
        if parent_t == "drivers":
            driver_feat_lookup = parent_feats.copy()
            driver_feat_lookup.index = parent_df["driverId"].values
            agg_feat_lookup = agg_feats.copy()
            agg_feat_lookup.index = parent_df["driverId"].values

            for task_name in DRIVER_TASKS:
                if task_name not in tasks:
                    continue
                task_folds = tasks[task_name]
                tt = DRIVER_TASK_TYPES[task_name]

                # Use train fold; prefer val over test (test often has masked labels)
                train_fold = task_folds.get("train", pd.DataFrame())
                eval_fold = task_folds.get("val", task_folds.get("test", pd.DataFrame()))
                if len(train_fold) < 10 or len(eval_fold) < 5:
                    continue

                def _build_feats(fold_df, include_agg):
                    dids = fold_df["driverId"].values if "driverId" in fold_df.columns else np.array([])
                    base = driver_feat_lookup.reindex(dids).fillna(0).values
                    # Add date feature
                    if "date" in fold_df.columns:
                        try:
                            dts = pd.to_datetime(fold_df["date"], errors="coerce")
                            date_feat = (dts.astype("int64") // 10**9).fillna(0).values.reshape(-1, 1)
                        except Exception:
                            date_feat = np.zeros((len(fold_df), 1))
                    else:
                        date_feat = np.zeros((len(fold_df), 1))
                    base = np.hstack([base, date_feat])
                    if include_agg:
                        agg = agg_feat_lookup.reindex(dids).fillna(0).values
                        return np.hstack([base, agg])
                    return base

                # Filter out non-numeric labels ('masked', etc.)
                def _clean_fold(fold_df):
                    labels = fold_df["__label__"]
                    numeric_mask = pd.to_numeric(labels, errors="coerce").notna()
                    return fold_df[numeric_mask.values].reset_index(drop=True)

                train_clean = _clean_fold(train_fold)
                test_clean = _clean_fold(eval_fold)
                if len(train_clean) < 10 or len(test_clean) < 5:
                    logger.warning(f"    Skipping {task_name}: train={len(train_clean)}, test={len(test_clean)} after cleaning")
                    continue

                y_tr = train_clean["__label__"].values.astype(float)
                y_te = test_clean["__label__"].values.astype(float)
                X_base_tr = _build_feats(train_clean, False)
                X_base_te = _build_feats(test_clean, False)
                X_aug_tr = _build_feats(train_clean, True)
                X_aug_te = _build_feats(test_clean, True)

                for probe in ["MLP", "GBM"]:
                    key = f"join{join_idx}_{task_name}_{probe}"
                    res = compute_jrn_multi_seed(
                        X_base_tr, X_base_te, X_aug_tr, X_aug_te,
                        y_tr, y_te, probe, tt, SEEDS, n_jobs=NUM_CPUS,
                    )
                    all_results[key] = {
                        **res,
                        "join_idx": join_idx,
                        "join_label": join_label,
                        "task": task_name,
                        "task_type": tt,
                        "probe": probe,
                        "parent_table": parent_t,
                    }
                    logger.info(
                        f"  {probe} {task_name}: "
                        f"JRN={res['JRN_mean']:.4f}+/-{res['JRN_std']:.4f} "
                        f"(base={res['M_base_mean']:.4f}, join={res['M_join_mean']:.4f})"
                    )

    return all_results


# ============================================================
# ENRICHMENT JRN (at child level)
# ============================================================
def measure_enrichment_jrn(tables, pk_cols, table_features):
    """Measure enrichment JRN: at child level, does adding parent info help?"""
    enrichment_results = {}

    # Define enrichment joins needed for chains
    enrichment_joins = [
        # (child_table, fk_col, parent_table, pk_col, join_idx_of_standard)
        ("results", "raceId", "races", "raceId", 10),
        ("standings", "raceId", "races", "raceId", 3),
        ("qualifying", "raceId", "races", "raceId", 7),
        ("results", "constructorId", "constructors", "constructorId", 12),
        ("qualifying", "constructorId", "constructors", "constructorId", 9),
        ("constructor_standings", "raceId", "races", "raceId", 1),
        ("constructor_results", "raceId", "races", "raceId", 5),
    ]

    for child_t, fk_col, parent_t, pk_col, std_join_idx in enrichment_joins:
        logger.info(f"--- Enrichment: {child_t} + {parent_t} (via {fk_col}) ---")

        if child_t not in tables or parent_t not in tables:
            continue
        if child_t not in CHILD_SS_TASKS:
            continue

        child_df = tables[child_t]
        parent_df = tables[parent_t]
        child_feats = table_features[child_t]
        parent_feats = table_features[parent_t]

        # Enrich child rows with parent features
        enriched = enrich_child_with_parent(
            child_df, fk_col, parent_df, pk_col, parent_feats,
            prefix=f"enr_{parent_t}",
        )

        # Self-supervised task at child level
        ss = CHILD_SS_TASKS[child_t]
        split = prepare_ss_split(child_feats, child_df, ss["target"], ss["type"])
        if split is None:
            continue
        X_child, y, tr_idx, te_idx = split
        X_aug = np.hstack([X_child, enriched.values])

        for probe in ["MLP", "GBM"]:
            key = f"enrich_{child_t}_{parent_t}_{fk_col}_{probe}"
            res = compute_jrn_multi_seed(
                X_child[tr_idx], X_child[te_idx],
                X_aug[tr_idx], X_aug[te_idx],
                y[tr_idx], y[te_idx],
                probe, ss["type"], SEEDS, n_jobs=NUM_CPUS,
            )
            enrichment_results[key] = {
                **res,
                "join_idx": std_join_idx,
                "join_label": f"enrich_{child_t}_with_{parent_t}",
                "task": f"ss_{child_t}_{ss['target']}",
                "task_type": ss["type"],
                "probe": probe,
                "child_table": child_t,
                "parent_table": parent_t,
            }
            logger.info(
                f"  {probe} enrich {child_t}+{parent_t}: "
                f"JRN={res['JRN_mean']:.4f}+/-{res['JRN_std']:.4f}"
            )

    return enrichment_results


# ============================================================
# STEP 6: CHAIN DEFINITION & MEASUREMENT
# ============================================================
def define_chains():
    """Define 12 multi-hop chains."""
    chains = []

    # Chains 1-5: aggregate chains ending at circuits
    # pattern: child -> races -> circuits
    agg_to_circuits = [
        (1, "results", "raceId", 10),
        (2, "qualifying", "raceId", 7),
        (3, "standings", "raceId", 3),
        (4, "constructor_standings", "raceId", 1),
        (5, "constructor_results", "raceId", 5),
    ]
    for cid, child_t, fk_col, hop1_join_idx in agg_to_circuits:
        chains.append({
            "chain_id": cid,
            "chain_type": "aggregate",
            "description": f"{child_t} -> races -> circuits",
            "hops": [
                {"child": child_t, "fk_col": fk_col, "parent": "races",
                 "pk_col": "raceId", "type": "aggregate", "join_idx": hop1_join_idx},
                {"child": "races", "fk_col": "circuitId", "parent": "circuits",
                 "pk_col": "circuitId", "type": "aggregate", "join_idx": 0},
            ],
            "end_table": "circuits",
            "end_task": "ss_circuits_lat",
            "end_task_type": "regression",
        })

    # Chains 6-8: enrichment chains ending at drivers (enrich child with races)
    enrich_races_to_drivers = [
        (6, "results", "raceId", "driverId", 11),
        (7, "standings", "raceId", "driverId", 4),
        (8, "qualifying", "raceId", "driverId", 8),
    ]
    for cid, child_t, enrich_fk, agg_fk, hop2_join_idx in enrich_races_to_drivers:
        chains.append({
            "chain_id": cid,
            "chain_type": "enrichment_to_drivers",
            "description": f"races enrich {child_t} -> drivers",
            "hops": [
                {"child": child_t, "enrich_parent": "races",
                 "enrich_fk": enrich_fk, "enrich_pk": "raceId",
                 "type": "enrich", "source_join_idx": -1},
                {"child": child_t, "fk_col": agg_fk, "parent": "drivers",
                 "pk_col": "driverId", "type": "aggregate", "join_idx": hop2_join_idx},
            ],
            "end_table": "drivers",
            "end_tasks": DRIVER_TASKS,
            "end_task_types": DRIVER_TASK_TYPES,
        })

    # Chains 9-10: enrichment with constructors -> drivers
    enrich_constr_to_drivers = [
        (9, "results", "constructorId", "driverId", 11),
        (10, "qualifying", "constructorId", "driverId", 8),
    ]
    for cid, child_t, enrich_fk, agg_fk, hop2_join_idx in enrich_constr_to_drivers:
        chains.append({
            "chain_id": cid,
            "chain_type": "enrichment_to_drivers",
            "description": f"constructors enrich {child_t} -> drivers",
            "hops": [
                {"child": child_t, "enrich_parent": "constructors",
                 "enrich_fk": enrich_fk, "enrich_pk": "constructorId",
                 "type": "enrich", "source_join_idx": -1},
                {"child": child_t, "fk_col": agg_fk, "parent": "drivers",
                 "pk_col": "driverId", "type": "aggregate", "join_idx": hop2_join_idx},
            ],
            "end_table": "drivers",
            "end_tasks": DRIVER_TASKS,
            "end_task_types": DRIVER_TASK_TYPES,
        })

    # Chains 11-12: enrichment ending at constructors
    enrich_to_constructors = [
        (11, "constructor_standings", "raceId", "constructorId", 2),
        (12, "constructor_results", "raceId", "constructorId", 6),
    ]
    for cid, child_t, enrich_fk, agg_fk, hop2_join_idx in enrich_to_constructors:
        chains.append({
            "chain_id": cid,
            "chain_type": "enrichment_to_constructors",
            "description": f"races enrich {child_t} -> constructors",
            "hops": [
                {"child": child_t, "enrich_parent": "races",
                 "enrich_fk": enrich_fk, "enrich_pk": "raceId",
                 "type": "enrich", "source_join_idx": -1},
                {"child": child_t, "fk_col": agg_fk, "parent": "constructors",
                 "pk_col": "constructorId", "type": "aggregate",
                 "join_idx": hop2_join_idx},
            ],
            "end_table": "constructors",
            "end_task": "ss_constructors_nationality",
            "end_task_type": "classification",
        })

    return chains


def measure_chain_jrn_all(chains, tables, pk_cols, table_features, tasks):
    """Measure chain JRN for all chains."""
    chain_results = {}

    for chain in chains:
        cid = chain["chain_id"]
        logger.info(f"=== Chain {cid}: {chain['description']} ===")

        try:
            if chain["chain_type"] == "aggregate":
                _measure_aggregate_chain(chain, tables, pk_cols, table_features, chain_results)
            elif chain["chain_type"] in ("enrichment_to_drivers", "enrichment_to_constructors"):
                _measure_enrichment_chain(chain, tables, pk_cols, table_features, tasks, chain_results)
        except Exception:
            logger.exception(f"  Failed chain {cid}")
            continue

    return chain_results


def _measure_aggregate_chain(chain, tables, pk_cols, table_features, results_out):
    """Measure an aggregate chain: child -> intermediate -> end_table."""
    cid = chain["chain_id"]
    hop1 = chain["hops"][0]
    hop2 = chain["hops"][1]

    child_t = hop1["child"]
    inter_t = hop1["parent"]  # intermediate (races)
    end_t = hop2["parent"]  # end (circuits)

    child_df = tables[child_t]
    inter_df = tables[inter_t]
    end_df = tables[end_t]

    child_feats = table_features[child_t]
    inter_feats = table_features[inter_t]
    end_feats = table_features[end_t]

    # Step 1: Aggregate child -> intermediate
    agg_child_to_inter = aggregate_child_to_parent(
        child_df, child_feats, hop1["fk_col"], inter_df, hop1["pk_col"],
        prefix=f"agg_{child_t}",
    )

    # Enriched intermediate = inter_feats + agg_child
    enriched_inter = pd.concat([inter_feats, agg_child_to_inter], axis=1)

    # Step 2: Aggregate enriched intermediate -> end
    chain_feats = aggregate_child_to_parent(
        inter_df, enriched_inter, hop2["fk_col"], end_df, hop2["pk_col"],
        prefix=f"chain_{child_t}",
    )

    # Also: 1-hop only (just intermediate -> end, without child enrichment)
    onehop_feats = aggregate_child_to_parent(
        inter_df, inter_feats, hop2["fk_col"], end_df, hop2["pk_col"],
        prefix=f"hop2_{inter_t}",
    )

    # Prepare task at end table
    ss = SS_TASKS.get(end_t)
    if ss is None:
        return
    split = prepare_ss_split(end_feats, end_df, ss["target"], ss["type"])
    if split is None:
        return
    X_end, y, tr_idx, te_idx = split

    X_base = X_end
    X_1hop = np.hstack([X_end, onehop_feats.values])
    X_2hop = np.hstack([X_end, chain_feats.values])

    task_label = chain["end_task"]
    task_type = chain["end_task_type"]

    for probe in ["MLP", "GBM"]:
        key = f"chain{cid}_{task_label}_{probe}"
        res = compute_jrn_multi_seed(
            X_base[tr_idx], X_base[te_idx],
            X_2hop[tr_idx], X_2hop[te_idx],
            y[tr_idx], y[te_idx],
            probe, task_type, SEEDS, n_jobs=NUM_CPUS,
        )
        # Also measure 1-hop
        res_1hop = compute_jrn_multi_seed(
            X_base[tr_idx], X_base[te_idx],
            X_1hop[tr_idx], X_1hop[te_idx],
            y[tr_idx], y[te_idx],
            probe, task_type, SEEDS, n_jobs=NUM_CPUS,
        )
        results_out[key] = {
            **res,
            "chain_id": cid,
            "chain_desc": chain["description"],
            "task": task_label,
            "task_type": task_type,
            "probe": probe,
            "end_table": chain["end_table"],
            "hop1_join_idx": chain["hops"][0]["join_idx"],
            "hop2_join_idx": chain["hops"][1]["join_idx"],
            "onehop_JRN_mean": res_1hop["JRN_mean"],
            "onehop_JRN_std": res_1hop["JRN_std"],
        }
        logger.info(
            f"  {probe} {task_label}: chain_JRN={res['JRN_mean']:.4f}, "
            f"1hop_JRN={res_1hop['JRN_mean']:.4f}"
        )


def _measure_enrichment_chain(chain, tables, pk_cols, table_features, tasks, results_out):
    """Measure enrichment chain: enrich child with parent, then aggregate to end."""
    cid = chain["chain_id"]
    hop1 = chain["hops"][0]
    hop2 = chain["hops"][1]

    child_t = hop1["child"]
    enrich_parent = hop1["enrich_parent"]
    enrich_fk = hop1["enrich_fk"]
    enrich_pk = hop1["enrich_pk"]
    agg_fk = hop2["fk_col"]
    end_t = hop2["parent"]

    child_df = tables[child_t]
    enrich_parent_df = tables[enrich_parent]
    end_df = tables[end_t]

    child_feats = table_features[child_t]
    enrich_parent_feats = table_features[enrich_parent]
    end_feats = table_features[end_t]

    # Step 1: Enrich child with parent features
    enriched_child_extra = enrich_child_with_parent(
        child_df, enrich_fk, enrich_parent_df, enrich_pk, enrich_parent_feats,
        prefix=f"enr_{enrich_parent}",
    )
    enriched_child = pd.concat([child_feats, enriched_child_extra], axis=1)

    # Step 2: Aggregate enriched child -> end table
    chain_feats = aggregate_child_to_parent(
        child_df, enriched_child, agg_fk, end_df, hop2["pk_col"],
        prefix=f"chain_{child_t}_{enrich_parent}",
    )

    # 1-hop: just aggregate child (without enrichment) -> end
    onehop_feats = aggregate_child_to_parent(
        child_df, child_feats, agg_fk, end_df, hop2["pk_col"],
        prefix=f"hop2_{child_t}",
    )

    # Determine tasks
    if end_t == "drivers":
        task_list = chain.get("end_tasks", DRIVER_TASKS)
        task_types = chain.get("end_task_types", DRIVER_TASK_TYPES)
    elif end_t == "constructors":
        task_list = [chain.get("end_task", "ss_constructors_nationality")]
        task_types = {task_list[0]: chain.get("end_task_type", "classification")}
    else:
        return

    for task_label in task_list:
        task_type = task_types.get(task_label, "regression")

        if task_label.startswith("ss_"):
            # Self-supervised at end table
            ss = SS_TASKS.get(end_t)
            if ss is None:
                continue
            split = prepare_ss_split(end_feats, end_df, ss["target"], ss["type"])
            if split is None:
                continue
            X_end, y, tr_idx, te_idx = split
            task_type = ss["type"]
        else:
            # Driver task
            if task_label not in tasks:
                continue
            task_folds = tasks[task_label]
            # Use val fold (test often has masked labels)
            train_fold = task_folds.get("train", pd.DataFrame())
            test_fold = task_folds.get("val", task_folds.get("test", pd.DataFrame()))
            if len(train_fold) < 10 or len(test_fold) < 5:
                continue

            # Filter out non-numeric labels ('masked', etc.)
            def _clean_fold2(fold_df):
                labels = fold_df["__label__"]
                numeric_mask = pd.to_numeric(labels, errors="coerce").notna()
                return fold_df[numeric_mask.values].reset_index(drop=True)

            train_fold = _clean_fold2(train_fold)
            test_fold = _clean_fold2(test_fold)
            if len(train_fold) < 10 or len(test_fold) < 5:
                logger.warning(f"    Chain {cid} skipping {task_label}: insufficient clean samples")
                continue

            driver_df = tables["drivers"]
            driver_feats = table_features["drivers"]
            driver_lookup = driver_feats.copy()
            driver_lookup.index = driver_df["driverId"].values

            chain_lookup = chain_feats.copy()
            chain_lookup.index = end_df["driverId"].values
            onehop_lookup = onehop_feats.copy()
            onehop_lookup.index = end_df["driverId"].values

            def _build(fold_df, extra_lookup=None):
                dids = fold_df["driverId"].values
                base = driver_lookup.reindex(dids).fillna(0).values
                if "date" in fold_df.columns:
                    try:
                        dts = pd.to_datetime(fold_df["date"], errors="coerce")
                        dfeat = (dts.astype("int64") // 10**9).fillna(0).values.reshape(-1, 1)
                    except Exception:
                        dfeat = np.zeros((len(fold_df), 1))
                else:
                    dfeat = np.zeros((len(fold_df), 1))
                base = np.hstack([base, dfeat])
                if extra_lookup is not None:
                    extra = extra_lookup.reindex(dids).fillna(0).values
                    return np.hstack([base, extra])
                return base

            # Filter out non-numeric labels ('masked', etc.)
            def _clean_fold2(fold_df):
                labels = fold_df["__label__"]
                numeric_mask = pd.to_numeric(labels, errors="coerce").notna()
                return fold_df[numeric_mask.values].reset_index(drop=True)

            train_fold = _clean_fold2(train_fold)
            test_fold = _clean_fold2(test_fold)
            if len(train_fold) < 10 or len(test_fold) < 5:
                continue

            y_tr = train_fold["__label__"].values.astype(float)
            y_te = test_fold["__label__"].values.astype(float)

            X_base_tr = _build(train_fold, None)
            X_base_te = _build(test_fold, None)
            X_1hop_tr = _build(train_fold, onehop_lookup)
            X_1hop_te = _build(test_fold, onehop_lookup)
            X_2hop_tr = _build(train_fold, chain_lookup)
            X_2hop_te = _build(test_fold, chain_lookup)

            for probe in ["MLP", "GBM"]:
                key = f"chain{cid}_{task_label}_{probe}"
                res = compute_jrn_multi_seed(
                    X_base_tr, X_base_te, X_2hop_tr, X_2hop_te,
                    y_tr, y_te, probe, task_type, SEEDS, n_jobs=NUM_CPUS,
                )
                res_1hop = compute_jrn_multi_seed(
                    X_base_tr, X_base_te, X_1hop_tr, X_1hop_te,
                    y_tr, y_te, probe, task_type, SEEDS, n_jobs=NUM_CPUS,
                )
                results_out[key] = {
                    **res,
                    "chain_id": cid,
                    "chain_desc": chain["description"],
                    "task": task_label,
                    "task_type": task_type,
                    "probe": probe,
                    "end_table": end_t,
                    "hop1_join_idx": -1,  # enrichment
                    "hop2_join_idx": chain["hops"][1]["join_idx"],
                    "onehop_JRN_mean": res_1hop["JRN_mean"],
                    "onehop_JRN_std": res_1hop["JRN_std"],
                }
                logger.info(
                    f"  {probe} {task_label}: chain_JRN={res['JRN_mean']:.4f}, "
                    f"1hop_JRN={res_1hop['JRN_mean']:.4f}"
                )
            continue  # handled inline

        # Self-supervised path (for constructors end table)
        X_base = X_end  # noqa: F821 - set in ss branch above
        X_1hop = np.hstack([X_end, onehop_feats.values])
        X_2hop = np.hstack([X_end, chain_feats.values])

        for probe in ["MLP", "GBM"]:
            key = f"chain{cid}_{task_label}_{probe}"
            res = compute_jrn_multi_seed(
                X_base[tr_idx], X_base[te_idx],
                X_2hop[tr_idx], X_2hop[te_idx],
                y[tr_idx], y[te_idx],
                probe, task_type, SEEDS, n_jobs=NUM_CPUS,
            )
            res_1hop = compute_jrn_multi_seed(
                X_base[tr_idx], X_base[te_idx],
                X_1hop[tr_idx], X_1hop[te_idx],
                y[tr_idx], y[te_idx],
                probe, task_type, SEEDS, n_jobs=NUM_CPUS,
            )
            results_out[key] = {
                **res,
                "chain_id": cid,
                "chain_desc": chain["description"],
                "task": task_label,
                "task_type": task_type,
                "probe": probe,
                "end_table": end_t,
                "hop1_join_idx": -1,
                "hop2_join_idx": chain["hops"][1]["join_idx"],
                "onehop_JRN_mean": res_1hop["JRN_mean"],
                "onehop_JRN_std": res_1hop["JRN_std"],
            }
            logger.info(
                f"  {probe} {task_label}: chain_JRN={res['JRN_mean']:.4f}, "
                f"1hop={res_1hop['JRN_mean']:.4f}"
            )


# ============================================================
# STEP 7: COMPOUNDING MODEL COMPARISON
# ============================================================
def get_hop_jrn_for_chain(chain_res, individual_jrn, enrichment_jrn):
    """Look up individual JRN values for each hop of a chain."""
    hop1_idx = chain_res.get("hop1_join_idx", -1)
    hop2_idx = chain_res.get("hop2_join_idx", -1)
    probe = chain_res["probe"]
    end_table = chain_res["end_table"]

    # Hop 2 JRN: standard individual JRN at end table
    hop2_jrn = None
    if end_table == "circuits":
        hop2_key = f"join0_ss_circuits_lat_{probe}"
        if hop2_key in individual_jrn:
            hop2_jrn = individual_jrn[hop2_key]["JRN_mean"]
    elif end_table == "constructors":
        hop2_key = f"join{hop2_idx}_ss_constructors_nationality_{probe}"
        if hop2_key in individual_jrn:
            hop2_jrn = individual_jrn[hop2_key]["JRN_mean"]
    elif end_table == "drivers":
        task = chain_res["task"]
        hop2_key = f"join{hop2_idx}_{task}_{probe}"
        if hop2_key in individual_jrn:
            hop2_jrn = individual_jrn[hop2_key]["JRN_mean"]

    # Hop 1 JRN: either standard join or enrichment
    hop1_jrn = None
    if hop1_idx >= 0:
        # Standard aggregate chain
        parent_t = FK_JOINS[hop1_idx][2]
        ss = SS_TASKS.get(parent_t, {})
        tgt = ss.get("target", "")
        hop1_key = f"join{hop1_idx}_ss_{parent_t}_{tgt}_{probe}"
        if hop1_key in individual_jrn:
            hop1_jrn = individual_jrn[hop1_key]["JRN_mean"]
    else:
        # Enrichment chain - look up enrichment JRN
        for ek, ev in enrichment_jrn.items():
            if ev["probe"] == probe:
                hop1_jrn = ev["JRN_mean"]
                break  # Use first matching enrichment

    if hop1_jrn is None:
        hop1_jrn = 1.0
    if hop2_jrn is None:
        hop2_jrn = 1.0

    return hop1_jrn, hop2_jrn


def compare_compounding_models(chain_jrn, individual_jrn, enrichment_jrn):
    """Compare 4 compounding models against measured chain JRN."""
    compounding_results = {}

    for probe in ["MLP", "GBM"]:
        measured = []
        hop1_list = []
        hop2_list = []
        chain_keys = []

        for key, cres in chain_jrn.items():
            if cres["probe"] != probe:
                continue
            h1, h2 = get_hop_jrn_for_chain(cres, individual_jrn, enrichment_jrn)
            measured.append(cres["JRN_mean"])
            hop1_list.append(h1)
            hop2_list.append(h2)
            chain_keys.append(key)

        if len(measured) < 3:
            logger.warning(f"  {probe}: only {len(measured)} chain measurements, skipping")
            continue

        measured = np.array(measured)
        hop1 = np.array(hop1_list)
        hop2 = np.array(hop2_list)

        models = {}

        # (a) Multiplicative
        pred_mult = hop1 * hop2
        models["multiplicative"] = pred_mult

        # (b) Additive
        pred_add = 1.0 + (hop1 - 1.0) + (hop2 - 1.0)
        models["additive"] = pred_add

        # (c) Bottleneck
        pred_bneck = np.minimum(hop1, hop2)
        models["bottleneck"] = pred_bneck

        # (d) Log-linear regression
        try:
            X_log = np.column_stack([
                np.log(np.clip(hop1, 1e-6, None)),
                np.log(np.clip(hop2, 1e-6, None)),
            ])
            y_log = np.log(np.clip(measured, 1e-6, None))
            reg = LinearRegression().fit(X_log, y_log)
            pred_ll = np.exp(reg.predict(X_log))
            models["log_linear"] = pred_ll
        except Exception:
            models["log_linear"] = pred_mult  # fallback

        probe_results = {}
        for model_name, pred in models.items():
            try:
                r2 = 1.0 - np.sum((measured - pred) ** 2) / np.sum((measured - np.mean(measured)) ** 2)
            except (ZeroDivisionError, FloatingPointError):
                r2 = 0.0
            try:
                sp_r, sp_p = stats.spearmanr(measured, pred)
            except Exception:
                sp_r, sp_p = 0.0, 1.0
            mae = float(np.mean(np.abs(measured - pred)))
            rmse = float(np.sqrt(np.mean((measured - pred) ** 2)))

            probe_results[model_name] = {
                "R2": float(r2) if np.isfinite(r2) else 0.0,
                "spearman_r": float(sp_r) if np.isfinite(sp_r) else 0.0,
                "spearman_p": float(sp_p) if np.isfinite(sp_p) else 1.0,
                "MAE": mae,
                "RMSE": rmse,
            }
            logger.info(
                f"  {probe} {model_name}: R2={r2:.4f}, "
                f"Spearman={sp_r:.4f}, MAE={mae:.4f}"
            )

        compounding_results[probe] = {
            "model_comparison": probe_results,
            "chain_details": {
                ck: {
                    "measured": float(m),
                    "hop1_jrn": float(h1),
                    "hop2_jrn": float(h2),
                    "pred_mult": float(h1 * h2),
                    "pred_add": float(1.0 + (h1 - 1.0) + (h2 - 1.0)),
                    "pred_bneck": float(min(h1, h2)),
                }
                for ck, m, h1, h2 in zip(chain_keys, measured, hop1, hop2)
            },
        }

    return compounding_results


# ============================================================
# STEP 8: DIAGNOSTICS
# ============================================================
def compute_diagnostics(chain_jrn, individual_jrn, enrichment_jrn, fk_info):
    """Diagnose top deviating chains."""
    diagnostics = []

    for probe in ["MLP", "GBM"]:
        deviations = []
        for key, cres in chain_jrn.items():
            if cres["probe"] != probe:
                continue
            h1, h2 = get_hop_jrn_for_chain(cres, individual_jrn, enrichment_jrn)
            pred = h1 * h2
            meas = cres["JRN_mean"]
            dev = abs(meas - pred)
            deviations.append((key, dev, meas, pred, h1, h2, cres))

        deviations.sort(key=lambda x: -x[1])

        for key, dev, meas, pred, h1, h2, cres in deviations[:3]:
            # Fanout analysis
            hop2_idx = cres.get("hop2_join_idx", -1)
            fanout_info = {}
            if 0 <= hop2_idx < len(fk_info):
                fi = fk_info[hop2_idx]
                fanout_info = {
                    "fanout_mean": fi.get("fanout_mean", 0),
                    "fanout_std": fi.get("fanout_std", 0),
                    "coverage": fi.get("join_coverage", 0),
                }

            direction = "super-multiplicative" if meas > pred else "sub-multiplicative"
            if dev < 0.02:
                interpretation = "Near-multiplicative: deviation < 0.02"
            elif direction == "super-multiplicative":
                interpretation = (
                    "Super-multiplicative: information from 2nd hop synergizes with 1st hop, "
                    "suggesting conditional dependence (information from hop1 helps decode hop2)."
                )
            else:
                interpretation = (
                    "Sub-multiplicative: information loss in aggregation chain, "
                    "possibly due to high fanout variance or feature overlap."
                )

            diagnostics.append({
                "chain_key": key,
                "chain_id": cres["chain_id"],
                "chain_desc": cres["chain_desc"],
                "probe": probe,
                "task": cres["task"],
                "measured_jrn": meas,
                "predicted_jrn": pred,
                "deviation": dev,
                "direction": direction,
                "hop1_jrn": h1,
                "hop2_jrn": h2,
                "fanout_analysis": fanout_info,
                "interpretation": interpretation,
            })

    return diagnostics


# ============================================================
# STEP 9: CROSS-DATASET COMPARISON
# ============================================================
def compute_cross_dataset(compounding_results):
    """Compare rel-f1 results to prior rel-stack results."""
    prior_rel_stack = {
        "MLP_R2": 0.83,
        "GBM_spearman_r": 0.68,
    }

    rel_f1 = {}
    for probe in ["MLP", "GBM"]:
        if probe not in compounding_results:
            continue
        mc = compounding_results[probe]["model_comparison"]
        mult = mc.get("multiplicative", {})
        rel_f1[f"{probe}_R2"] = mult.get("R2", 0.0)
        rel_f1[f"{probe}_spearman_r"] = mult.get("spearman_r", 0.0)

    # Deltas
    dataset_delta_mlp_r2 = rel_f1.get("MLP_R2", 0) - prior_rel_stack["MLP_R2"]
    dataset_delta_gbm_sp = rel_f1.get("GBM_spearman_r", 0) - prior_rel_stack["GBM_spearman_r"]
    probe_delta_r2 = rel_f1.get("MLP_R2", 0) - rel_f1.get("GBM_R2", 0)
    probe_delta_sp = rel_f1.get("MLP_spearman_r", 0) - rel_f1.get("GBM_spearman_r", 0)

    if abs(dataset_delta_mlp_r2) > 0.2 and abs(probe_delta_r2) > 0.1:
        analysis = "Both dataset-driven and probe-type-driven differences observed."
    elif abs(dataset_delta_mlp_r2) > 0.2:
        analysis = "Primarily dataset-driven: rel-f1 behaves differently from rel-stack."
    elif abs(probe_delta_r2) > 0.1:
        analysis = "Primarily probe-type-driven: MLP and GBM disagree systematically."
    else:
        analysis = "Results are broadly consistent across datasets and probe types."

    return {
        "rel_f1": rel_f1,
        "rel_stack_prior": prior_rel_stack,
        "dataset_delta_mlp_r2": float(dataset_delta_mlp_r2),
        "dataset_delta_gbm_spearman": float(dataset_delta_gbm_sp),
        "probe_delta_r2": float(probe_delta_r2),
        "probe_delta_spearman": float(probe_delta_sp),
        "discrepancy_analysis": analysis,
    }


# ============================================================
# STEP 10: BUILD OUTPUT
# ============================================================
def build_output(
    individual_jrn, enrichment_jrn, chain_jrn,
    compounding, diagnostics, cross_dataset, fk_info,
):
    """Build output in exp_gen_sol_out.json schema format."""
    examples = []

    # --- Individual JRN examples ---
    for key, res in individual_jrn.items():
        inp_desc = (
            f"Measure individual JRN for FK join: {res['join_label']}, "
            f"task: {res['task']}, probe: {res['probe']}"
        )
        out_desc = (
            f"JRN = {res['JRN_mean']:.4f} +/- {res['JRN_std']:.4f} "
            f"(M_base={res['M_base_mean']:.4f}, M_join={res['M_join_mean']:.4f})"
        )
        examples.append({
            "input": inp_desc,
            "output": out_desc,
            "metadata_measurement_type": "individual_jrn",
            "metadata_join_idx": res["join_idx"],
            "metadata_join_label": res["join_label"],
            "metadata_task": res["task"],
            "metadata_task_type": res["task_type"],
            "metadata_probe": res["probe"],
            "metadata_parent_table": res["parent_table"],
            "metadata_jrn_mean": res["JRN_mean"],
            "metadata_jrn_std": res["JRN_std"],
            "metadata_m_base_mean": res["M_base_mean"],
            "metadata_m_join_mean": res["M_join_mean"],
            "predict_baseline": f"{res['M_base_mean']:.6f}",
            "predict_our_method": f"{res['M_join_mean']:.6f}",
        })

    # --- Enrichment JRN examples ---
    for key, res in enrichment_jrn.items():
        inp_desc = (
            f"Measure enrichment JRN: {res['join_label']}, "
            f"task: {res['task']}, probe: {res['probe']}"
        )
        out_desc = (
            f"Enrichment JRN = {res['JRN_mean']:.4f} +/- {res['JRN_std']:.4f}"
        )
        examples.append({
            "input": inp_desc,
            "output": out_desc,
            "metadata_measurement_type": "enrichment_jrn",
            "metadata_join_label": res["join_label"],
            "metadata_task": res["task"],
            "metadata_probe": res["probe"],
            "metadata_jrn_mean": res["JRN_mean"],
            "metadata_jrn_std": res["JRN_std"],
            "predict_baseline": f"{res['M_base_mean']:.6f}",
            "predict_our_method": f"{res['M_join_mean']:.6f}",
        })

    # --- Chain JRN examples ---
    for key, cres in chain_jrn.items():
        probe = cres["probe"]

        # Get predicted values from compounding
        chain_detail = {}
        if probe in compounding:
            chain_detail = compounding[probe].get("chain_details", {}).get(key, {})

        inp_desc = (
            f"Chain JRN: {cres['chain_desc']} (chain {cres['chain_id']}), "
            f"task: {cres['task']}, probe: {probe}"
        )
        out_desc = (
            f"Measured chain JRN = {cres['JRN_mean']:.4f}, "
            f"1-hop JRN = {cres['onehop_JRN_mean']:.4f}"
        )

        ex = {
            "input": inp_desc,
            "output": out_desc,
            "metadata_measurement_type": "chain_jrn",
            "metadata_chain_id": cres["chain_id"],
            "metadata_chain_desc": cres["chain_desc"],
            "metadata_task": cres["task"],
            "metadata_probe": probe,
            "metadata_end_table": cres["end_table"],
            "metadata_measured_jrn": cres["JRN_mean"],
            "metadata_onehop_jrn": cres["onehop_JRN_mean"],
            "predict_baseline": f"{cres['onehop_JRN_mean']:.6f}",
            "predict_our_method": f"{cres['JRN_mean']:.6f}",
        }
        if chain_detail:
            ex["predict_multiplicative"] = f"{chain_detail.get('pred_mult', 0):.6f}"
            ex["predict_additive"] = f"{chain_detail.get('pred_add', 0):.6f}"
            ex["predict_bottleneck"] = f"{chain_detail.get('pred_bneck', 0):.6f}"
        examples.append(ex)

    # --- Compounding model summaries ---
    for probe, pdata in compounding.items():
        for model_name, mdata in pdata.get("model_comparison", {}).items():
            inp_desc = f"Compounding model comparison: {model_name}, probe: {probe}"
            out_desc = (
                f"R2={mdata['R2']:.4f}, Spearman_r={mdata['spearman_r']:.4f}, "
                f"MAE={mdata['MAE']:.4f}, RMSE={mdata['RMSE']:.4f}"
            )
            examples.append({
                "input": inp_desc,
                "output": out_desc,
                "metadata_measurement_type": "compounding_summary",
                "metadata_model": model_name,
                "metadata_probe": probe,
                "metadata_r2": mdata["R2"],
                "metadata_spearman_r": mdata["spearman_r"],
                "metadata_mae": mdata["MAE"],
                "metadata_rmse": mdata["RMSE"],
                "predict_baseline": "1.000000",
                "predict_our_method": f"{mdata['R2']:.6f}",
            })

    # --- 2x2 Grid summary ---
    grid_data = {"R2": {}, "Spearman_r": {}}
    for probe in ["MLP", "GBM"]:
        if probe in compounding:
            mc = compounding[probe].get("model_comparison", {})
            mult = mc.get("multiplicative", {})
            grid_data["R2"][probe] = mult.get("R2", 0.0)
            grid_data["Spearman_r"][probe] = mult.get("spearman_r", 0.0)

    examples.append({
        "input": "2x2 Grid Summary: multiplicative compounding R2 and Spearman by probe type",
        "output": json.dumps(grid_data),
        "metadata_measurement_type": "two_by_two_grid",
        "predict_baseline": "0.000000",
        "predict_our_method": json.dumps(grid_data),
    })

    # --- Diagnostics ---
    for diag in diagnostics:
        inp_desc = (
            f"Diagnostic: chain {diag['chain_id']} ({diag['chain_desc']}), "
            f"probe: {diag['probe']}, task: {diag['task']}"
        )
        out_desc = (
            f"{diag['direction']}: measured={diag['measured_jrn']:.4f}, "
            f"predicted={diag['predicted_jrn']:.4f}, deviation={diag['deviation']:.4f}. "
            f"{diag['interpretation']}"
        )
        examples.append({
            "input": inp_desc,
            "output": out_desc,
            "metadata_measurement_type": "chain_diagnostic",
            "metadata_chain_id": diag["chain_id"],
            "metadata_probe": diag["probe"],
            "metadata_deviation": diag["deviation"],
            "predict_baseline": f"{diag['predicted_jrn']:.6f}",
            "predict_our_method": f"{diag['measured_jrn']:.6f}",
        })

    # --- Cross-dataset ---
    examples.append({
        "input": "Cross-dataset comparison: rel-f1 vs rel-stack (prior work)",
        "output": json.dumps(cross_dataset),
        "metadata_measurement_type": "cross_dataset_comparison",
        "predict_baseline": json.dumps(cross_dataset.get("rel_stack_prior", {})),
        "predict_our_method": json.dumps(cross_dataset.get("rel_f1", {})),
    })

    # --- Conclusions ---
    best_model = "multiplicative"
    compounding_holds = False
    probe_matters = False

    for probe in ["MLP", "GBM"]:
        if probe in compounding:
            mc = compounding[probe]["model_comparison"]
            best_r2 = -999
            for mn, md in mc.items():
                if md["R2"] > best_r2:
                    best_r2 = md["R2"]
                    best_model = mn
            if best_r2 > 0.3:
                compounding_holds = True

    mlp_r2 = grid_data["R2"].get("MLP", 0)
    gbm_r2 = grid_data["R2"].get("GBM", 0)
    if abs(mlp_r2 - gbm_r2) > 0.15:
        probe_matters = True

    conclusion_text = (
        f"Compounding {'holds' if compounding_holds else 'does NOT hold'} on rel-f1. "
        f"Best compounding model: {best_model}. "
        f"Probe type {'matters' if probe_matters else 'does NOT matter significantly'}. "
        f"MLP R2={mlp_r2:.3f}, GBM R2={gbm_r2:.3f}. "
        f"{cross_dataset.get('discrepancy_analysis', '')}"
    )

    examples.append({
        "input": "Definitive conclusion on JRN multiplicative compounding for rel-f1",
        "output": conclusion_text,
        "metadata_measurement_type": "conclusion",
        "metadata_compounding_holds": compounding_holds,
        "metadata_best_model": best_model,
        "metadata_probe_matters": probe_matters,
        "predict_baseline": "No compounding model (JRN=1.0 everywhere)",
        "predict_our_method": conclusion_text[:200],
    })

    # Build final output
    output = {
        "metadata": {
            "title": "JRN Multiplicative Compounding: 2x2 Probe Grid on rel-f1",
            "description": (
                "Self-contained experiment measuring individual JRN for 13 FK joins "
                "using MLP and GBM probes, chain JRN for 12 multi-hop chains, and "
                "comparing 4 compounding models on the rel-f1 Formula 1 dataset."
            ),
            "dataset": "rel-f1",
            "num_fk_joins": 13,
            "num_chains": 12,
            "probes": ["MLP", "GBM"],
            "seeds": SEEDS,
            "compounding_models": ["multiplicative", "additive", "bottleneck", "log_linear"],
        },
        "datasets": [{
            "dataset": "rel-f1",
            "examples": examples,
        }],
    }

    return output


# ============================================================
# MAIN
# ============================================================
@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("JRN Multiplicative Compounding: 2x2 Probe Grid on rel-f1")
    logger.info(f"CPUs={NUM_CPUS}, RAM={TOTAL_RAM_GB:.1f}GB, LightGBM={HAS_LGB}")
    logger.info("=" * 60)

    # Phase 1: Load data
    data_path = DEP_DATA / "full_data_out.json"
    tables, pk_cols, fk_info, tasks = load_data(data_path)

    # Phase 2: Feature engineering
    logger.info("--- Feature Engineering ---")
    table_features = {}
    for tname, df in tables.items():
        pk = pk_cols[tname]
        table_features[tname] = encode_features(df, exclude_cols=[pk])
        logger.info(f"  {tname}: {table_features[tname].shape[1]} features")

    # Phase 3: Individual JRN (self-supervised + driver tasks)
    logger.info("=" * 60)
    logger.info("PHASE 3: Individual JRN Estimation")
    logger.info("=" * 60)
    individual_jrn = measure_all_individual_jrn(
        tables, pk_cols, table_features, tasks,
    )
    logger.info(f"  Total individual JRN measurements: {len(individual_jrn)}")

    # Phase 4: Enrichment JRN
    logger.info("=" * 60)
    logger.info("PHASE 4: Enrichment JRN")
    logger.info("=" * 60)
    enrichment_jrn = measure_enrichment_jrn(tables, pk_cols, table_features)
    logger.info(f"  Total enrichment JRN measurements: {len(enrichment_jrn)}")

    # Phase 5: Chain JRN
    logger.info("=" * 60)
    logger.info("PHASE 5: Chain JRN Measurement")
    logger.info("=" * 60)
    chains = define_chains()
    chain_jrn = measure_chain_jrn_all(chains, tables, pk_cols, table_features, tasks)
    logger.info(f"  Total chain JRN measurements: {len(chain_jrn)}")

    # Phase 6: Compounding model comparison
    logger.info("=" * 60)
    logger.info("PHASE 6: Compounding Model Comparison")
    logger.info("=" * 60)
    compounding = compare_compounding_models(chain_jrn, individual_jrn, enrichment_jrn)

    # Phase 7: Diagnostics
    logger.info("=" * 60)
    logger.info("PHASE 7: Diagnostics")
    logger.info("=" * 60)
    diagnostics = compute_diagnostics(chain_jrn, individual_jrn, enrichment_jrn, fk_info)
    for d in diagnostics:
        logger.info(f"  Chain {d['chain_id']} {d['probe']}: {d['direction']} dev={d['deviation']:.4f}")

    # Phase 8: Cross-dataset comparison
    logger.info("=" * 60)
    logger.info("PHASE 8: Cross-dataset Comparison")
    logger.info("=" * 60)
    cross_dataset = compute_cross_dataset(compounding)
    logger.info(f"  Analysis: {cross_dataset['discrepancy_analysis']}")

    # Phase 9: Build output
    logger.info("=" * 60)
    logger.info("PHASE 9: Building Output")
    logger.info("=" * 60)
    output = build_output(
        individual_jrn, enrichment_jrn, chain_jrn,
        compounding, diagnostics, cross_dataset, fk_info,
    )

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    n_examples = len(output["datasets"][0]["examples"])
    logger.info(f"Output saved to {out_path} ({n_examples} examples)")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
