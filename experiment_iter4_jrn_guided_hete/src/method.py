#!/usr/bin/env python3
"""JRN-Guided Heterogeneous Architecture: GBM Probe Estimation on rel-stack.

Computes GBM-based JRN for all reachable FK joins across 3 rel-stack tasks
(user-engagement, user-badge, post-votes), then compares 5 architecture
configurations (JRN-guided, uniform-mean, uniform-rich, top-K, oracle).
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from loguru import logger
from scipy import stats
from sklearn.metrics import average_precision_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ============================================================
# Workspace & Logging
# ============================================================
WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/runs/run__20260309_024817"
    "/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus"
)
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================
# Hardware Detection & Memory Limits
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


def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return psutil.virtual_memory().total / 1e9


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
_avail = psutil.virtual_memory().available
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.70 * 1e9, _avail * 0.85))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
            f"budget={RAM_BUDGET / 1e9:.1f}GB")

# ============================================================
# Constants & Schema
# ============================================================

SCHEMA = {
    "users":       {"pkey": "Id", "time_col": "CreationDate", "fkeys": {}},
    "posts":       {"pkey": "Id", "time_col": "CreationDate",
                    "fkeys": {"OwnerUserId": "users", "ParentId": "posts"}},
    "comments":    {"pkey": "Id", "time_col": "CreationDate",
                    "fkeys": {"UserId": "users", "PostId": "posts"}},
    "votes":       {"pkey": "Id", "time_col": "CreationDate",
                    "fkeys": {"PostId": "posts", "UserId": "users"}},
    "badges":      {"pkey": "Id", "time_col": "Date",
                    "fkeys": {"UserId": "users"}},
    "postLinks":   {"pkey": "Id", "time_col": "CreationDate",
                    "fkeys": {"PostId": "posts", "RelatedPostId": "posts"}},
    "postHistory": {"pkey": "Id", "time_col": "CreationDate",
                    "fkeys": {"PostId": "posts", "UserId": "users"}},
}

ALL_JOINS = [
    ("comments",    "UserId",        "users"),   # J1
    ("comments",    "PostId",        "posts"),   # J2
    ("badges",      "UserId",        "users"),   # J3
    ("postLinks",   "PostId",        "posts"),   # J4
    ("postLinks",   "RelatedPostId", "posts"),   # J5
    ("postHistory", "PostId",        "posts"),   # J6
    ("postHistory", "UserId",        "users"),   # J7
    ("votes",       "PostId",        "posts"),   # J8
    ("votes",       "UserId",        "users"),   # J9
    ("posts",       "OwnerUserId",   "users"),   # J10
    ("posts",       "ParentId",      "posts"),   # J11
]


def _jk(child: str, fk: str, parent: str) -> str:
    return f"{child}.{fk} -> {parent}"


# Reachable *child-aggregation* joins for each entity table
REACHABLE_CHILD_JOINS: dict[str, list[tuple[str, str, str]]] = {
    "users": [
        ("comments",    "UserId",      "users"),
        ("badges",      "UserId",      "users"),
        ("postHistory", "UserId",      "users"),
        ("votes",       "UserId",      "users"),
        ("posts",       "OwnerUserId", "users"),
    ],
    "posts": [
        ("comments",    "PostId",        "posts"),
        ("postLinks",   "PostId",        "posts"),
        ("postLinks",   "RelatedPostId", "posts"),
        ("postHistory", "PostId",        "posts"),
        ("votes",       "PostId",        "posts"),
        ("posts",       "ParentId",      "posts"),
    ],
}

# Parent-lookup joins (entity table has an outgoing FK to parent)
REACHABLE_PARENT_JOINS: dict[str, list[tuple[str, str, str]]] = {
    "users": [],
    "posts": [
        ("posts", "OwnerUserId", "users"),
    ],
}

TASKS = {
    "user-engagement": {
        "entity_table": "users",
        "entity_col": "OwnerUserId",
        "target_col": "contribution",
        "task_type": "classification",
        "metric": "average_precision",
    },
    "user-badge": {
        "entity_table": "users",
        "entity_col": "UserId",
        "target_col": "WillGetBadge",
        "task_type": "classification",
        "metric": "average_precision",
    },
    "post-votes": {
        "entity_table": "posts",
        "entity_col": "PostId",
        "target_col": "popularity",
        "task_type": "regression",
        "metric": "neg_mae",
    },
}

PROBE_PARAMS: dict = {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": NUM_CPUS,
    "max_bin": 127,
}

FINAL_PARAMS: dict = {
    "n_estimators": 250,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": NUM_CPUS,
    "max_bin": 255,
}

SEEDS = [42, 123, 456]
VAL_TS = pd.Timestamp("2019-01-01")
TEST_TS = pd.Timestamp("2021-01-01")

MAX_TRAIN_SAMPLES = 150_000
MAX_VAL_SAMPLES = 50_000
MAX_TEST_SAMPLES = 50_000
AGG_TYPES = ["mean", "sum", "max", "std"]

JRN_HIGH = 1.15
JRN_LOW = 0.85

# ============================================================
# Data Loading
# ============================================================

def load_relbench_data() -> tuple[dict[str, pd.DataFrame],
                                   dict[str, dict[str, pd.DataFrame]]]:
    """Load rel-stack dataset and all three tasks from relbench."""
    logger.info("Loading rel-stack dataset from relbench …")
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset = get_dataset("rel-stack", download=True)
    db = dataset.get_db()

    tables: dict[str, pd.DataFrame] = {}
    for name, table in db.table_dict.items():
        tables[name] = table.df
        logger.info(f"  {name}: {tables[name].shape}")

    tasks_data: dict[str, dict[str, pd.DataFrame]] = {}
    for task_name in TASKS:
        logger.info(f"  Loading task {task_name} …")
        task = get_task("rel-stack", task_name, download=True)
        tasks_data[task_name] = {
            "train": task.get_table("train").df,
            "val":   task.get_table("val").df,
            "test":  task.get_table("test").df,
        }
        for sp in ("train", "val", "test"):
            logger.info(f"    {sp}: {len(tasks_data[task_name][sp])} rows, "
                        f"cols={list(tasks_data[task_name][sp].columns)}")

    return tables, tasks_data


# ============================================================
# Feature Engineering
# ============================================================

def _numeric_features(df: pd.DataFrame,
                      exclude: set[str],
                      prefix: str = "") -> pd.DataFrame:
    """Extract numeric / engineered features from a DataFrame."""
    parts: list[pd.Series] = []

    for col in df.columns:
        if col in exclude:
            continue

        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            parts.append(s.astype(np.float32).rename(f"{prefix}{col}"))
        elif pd.api.types.is_numeric_dtype(s):
            parts.append(s.astype(np.float32).rename(f"{prefix}{col}"))
        elif pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce")
            parts.append(dt.dt.year.astype(np.float32).rename(f"{prefix}{col}_year"))
            parts.append(dt.dt.month.astype(np.float32).rename(f"{prefix}{col}_month"))
            parts.append(dt.dt.dayofweek.astype(np.float32).rename(f"{prefix}{col}_dow"))
        elif s.dtype == object or str(s.dtype) == "string":
            parts.append(
                s.fillna("").astype(str).str.len()
                 .astype(np.float32).rename(f"{prefix}{col}_len")
            )
            parts.append(
                s.notna().astype(np.float32).rename(f"{prefix}{col}_nn")
            )

    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def entity_base_features(entity_table_df: pd.DataFrame,
                         entity_ids: set,
                         pkey: str,
                         table_name: str) -> pd.DataFrame:
    """Base features for entity rows from their own table."""
    exclude = {pkey}
    schema = SCHEMA.get(table_name, {})
    exclude.update(schema.get("fkeys", {}).keys())
    exclude.add(schema.get("time_col", ""))

    sub = (entity_table_df[entity_table_df[pkey].isin(entity_ids)]
           .drop_duplicates(pkey)
           .set_index(pkey))

    feats = _numeric_features(sub, exclude, prefix="base_")
    return feats


def agg_child_features(child_df: pd.DataFrame,
                       child_name: str,
                       fk_col: str,
                       entity_ids: set,
                       cutoff: pd.Timestamp,
                       agg_types: list[str]) -> pd.DataFrame:
    """Aggregate child-table rows grouped by FK with temporal filtering."""
    schema = SCHEMA[child_name]
    tc = schema["time_col"]

    # temporal filter
    tmp = child_df
    if tc in tmp.columns:
        try:
            ts = pd.to_datetime(tmp[tc], errors="coerce")
            tmp = tmp[ts < cutoff]
        except Exception:
            pass

    # FK filter
    fk_vals = tmp[fk_col]
    mask = fk_vals.isin(entity_ids) & fk_vals.notna()
    tmp = tmp[mask]

    if tmp.empty:
        return pd.DataFrame(index=pd.Index(list(entity_ids)))

    # exclude meta cols
    exclude = {schema["pkey"], tc}
    exclude.update(schema["fkeys"].keys())

    feat = _numeric_features(tmp, exclude,
                             prefix=f"{child_name}_{fk_col}_")
    feat = feat.copy()
    feat["__fk"] = tmp[fk_col].values

    num_cols = [c for c in feat.columns if c != "__fk"]

    frames: list[pd.DataFrame] = []

    # row count
    cnt = feat.groupby("__fk").size().rename(
        f"{child_name}_{fk_col}_rowcount")
    frames.append(cnt.to_frame())

    if num_cols:
        for agg in agg_types:
            try:
                g = feat.groupby("__fk")[num_cols].agg(agg)
                g.columns = [f"{c}_{agg}" for c in g.columns]
                frames.append(g)
            except Exception:
                pass

    result = pd.concat(frames, axis=1)
    idx = pd.Index(list(entity_ids))
    return result.reindex(idx).fillna(0).astype(np.float32)


def parent_lookup_features(split_entity_ids: np.ndarray,
                           entity_table_df: pd.DataFrame,
                           entity_pkey: str,
                           fk_col: str,
                           parent_table_df: pd.DataFrame,
                           parent_pkey: str,
                           parent_table_name: str) -> pd.DataFrame:
    """Lookup parent-table features through an outgoing FK."""
    # entity row -> FK value
    sub = (entity_table_df[entity_table_df[entity_pkey].isin(split_entity_ids)]
           .drop_duplicates(entity_pkey)
           [[entity_pkey, fk_col]])

    parent_ids = sub[fk_col].dropna().unique()

    # parent features
    pschema = SCHEMA.get(parent_table_name, {})
    exclude = {parent_pkey}
    exclude.update(pschema.get("fkeys", {}).keys())
    exclude.add(pschema.get("time_col", ""))

    p_sub = (parent_table_df[parent_table_df[parent_pkey].isin(parent_ids)]
             .drop_duplicates(parent_pkey)
             .set_index(parent_pkey))
    p_feat = _numeric_features(p_sub, exclude,
                               prefix=f"plookup_{fk_col}_")

    # map entity -> FK -> parent features
    mapping = sub.set_index(entity_pkey)
    merged = mapping.merge(p_feat, left_on=fk_col, right_index=True,
                           how="left").drop(columns=[fk_col])

    # reindex to split order
    idx = pd.Index(split_entity_ids)
    return merged.reindex(idx).fillna(0).astype(np.float32)


# ============================================================
# Model Training & Evaluation
# ============================================================

def _train_lgbm(X_tr, y_tr, X_va, y_va, seed, task_info, params=None):
    import lightgbm as lgb

    p = dict(params or PROBE_PARAMS)
    p["random_state"] = seed

    # clean col names
    cols = [f"f{i}" for i in range(X_tr.shape[1])]
    Xt = X_tr.copy(); Xt.columns = cols
    Xv = X_va.copy(); Xv.columns = cols

    if task_info["task_type"] == "classification":
        m = lgb.LGBMClassifier(**p)
    else:
        m = lgb.LGBMRegressor(**p)

    m.fit(Xt, y_tr,
          eval_set=[(Xv, y_va)],
          callbacks=[lgb.early_stopping(20, verbose=False),
                     lgb.log_evaluation(-1)])
    return m, cols


def _eval_model(model, X_te, y_te, task_info, cols):
    Xt = X_te.copy(); Xt.columns = cols
    if task_info["task_type"] == "classification":
        yp = model.predict_proba(Xt)[:, 1]
        return float(average_precision_score(y_te, yp))
    else:
        yp = model.predict(Xt)
        return float(-mean_absolute_error(y_te, yp))


def multi_seed_eval(X_tr, y_tr, X_va, y_va, X_te, y_te,
                    task_info, seeds=SEEDS, params=None) -> dict:
    scores = []
    for s in seeds:
        try:
            m, cols = _train_lgbm(X_tr, y_tr, X_va, y_va, s,
                                  task_info, params)
            sc = _eval_model(m, X_te, y_te, task_info, cols)
            scores.append(sc)
            del m
        except Exception as exc:
            logger.warning(f"  seed {s} failed: {exc}")
    if not scores:
        return {"scores": [], "mean": float("nan"), "std": float("nan")}
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


# ============================================================
# JRN Estimation for One Task
# ============================================================

def _merge_agg(split_df: pd.DataFrame, entity_col: str,
               agg_feats: pd.DataFrame) -> pd.DataFrame:
    merged = (split_df[[entity_col]]
              .merge(agg_feats, left_on=entity_col,
                     right_index=True, how="left")
              .drop(columns=[entity_col], errors="ignore")
              .fillna(0)
              .astype(np.float32))
    merged = merged.reset_index(drop=True)
    return merged


def estimate_jrn_for_task(
    task_name: str,
    task_info: dict,
    tables: dict[str, pd.DataFrame],
    tasks_data: dict[str, dict[str, pd.DataFrame]],
) -> dict:
    """Estimate JRN for every reachable join, build all agg caches."""
    t0 = time.time()
    entity_table = task_info["entity_table"]
    entity_col = task_info["entity_col"]
    target_col = task_info["target_col"]
    pkey = SCHEMA[entity_table]["pkey"]

    # ----- splits -----
    raw_train = tasks_data[task_name]["train"].copy()
    raw_val = tasks_data[task_name]["val"].copy()
    # NOTE: relbench test sets have NO labels — use val as test,
    #       split train into train_sub + val_sub
    logger.info(f"  Raw sizes: train={len(raw_train)}, val={len(raw_val)} "
                f"(val used as test; test set has no labels)")

    # Subsample raw_train first to keep things tractable
    if len(raw_train) > MAX_TRAIN_SAMPLES * 2:
        raw_train = raw_train.sample(MAX_TRAIN_SAMPLES * 2, random_state=42)

    # 80/20 split of raw_train → train / val  (stratify for clf)
    from sklearn.model_selection import train_test_split
    if task_info["task_type"] == "classification":
        try:
            train_df, val_df = train_test_split(
                raw_train, test_size=0.2, random_state=42,
                stratify=raw_train[target_col])
        except ValueError:
            train_df, val_df = train_test_split(
                raw_train, test_size=0.2, random_state=42)
    else:
        train_df, val_df = train_test_split(
            raw_train, test_size=0.2, random_state=42)

    # Use original val as test (it has labels)
    test_df = raw_val.copy()
    del raw_train, raw_val

    if len(train_df) > MAX_TRAIN_SAMPLES:
        train_df = train_df.sample(MAX_TRAIN_SAMPLES, random_state=42)
    if len(val_df) > MAX_VAL_SAMPLES:
        val_df = val_df.sample(MAX_VAL_SAMPLES, random_state=42)
    if len(test_df) > MAX_TEST_SAMPLES:
        test_df = test_df.sample(MAX_TEST_SAMPLES, random_state=42)

    logger.info(f"  After split/subsample: train={len(train_df)}, "
                f"val={len(val_df)}, test={len(test_df)}")

    # reset index for clean concat later
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_tr = train_df[target_col].values.astype(np.float32)
    y_va = val_df[target_col].values.astype(np.float32)
    y_te = test_df[target_col].values.astype(np.float32)

    tr_ids = set(train_df[entity_col].dropna())
    va_ids = set(val_df[entity_col].dropna())
    te_ids = set(test_df[entity_col].dropna())
    all_ids = tr_ids | va_ids | te_ids

    # ----- base features -----
    logger.info(f"  Computing base features ({len(all_ids)} entities) …")
    base_feat = entity_base_features(tables[entity_table], all_ids,
                                     pkey, entity_table)
    logger.info(f"  Base feature cols: {base_feat.shape[1]}")

    X_tr_base = _merge_agg(train_df, entity_col, base_feat)
    X_va_base = _merge_agg(val_df, entity_col, base_feat)
    X_te_base = _merge_agg(test_df, entity_col, base_feat)

    # ----- baseline: entity features only -----
    logger.info("  Training BASELINE (entity-only) …")
    baseline = multi_seed_eval(X_tr_base, y_tr, X_va_base, y_va,
                               X_te_base, y_te, task_info)
    logger.info(f"  BASELINE: {baseline['mean']:.6f} ± {baseline['std']:.6f}")

    # ----- per-join JRN probes -----
    jrn_raw: dict[str, dict[str, dict]] = {}      # jk -> agg -> result
    agg_cache: dict[tuple[str, str], dict] = {}    # (jk, agg) -> split feats

    for child, fk, par in REACHABLE_CHILD_JOINS.get(entity_table, []):
        jk = _jk(child, fk, par)
        logger.info(f"  Join: {jk}")
        jrn_raw[jk] = {}

        for ag in AGG_TYPES + ["all"]:
            agg_list = ["mean", "sum", "max", "std", "min"] if ag == "all" else [ag]
            try:
                a_tr = agg_child_features(tables[child], child, fk,
                                          tr_ids, VAL_TS, agg_list)
                a_va = agg_child_features(tables[child], child, fk,
                                          va_ids, VAL_TS, agg_list)
                a_te = agg_child_features(tables[child], child, fk,
                                          te_ids, VAL_TS, agg_list)

                Xa_tr = _merge_agg(train_df, entity_col, a_tr)
                Xa_va = _merge_agg(val_df, entity_col, a_va)
                Xa_te = _merge_agg(test_df, entity_col, a_te)

                X_tr_w = pd.concat([X_tr_base, Xa_tr], axis=1)
                X_va_w = pd.concat([X_va_base, Xa_va], axis=1)
                X_te_w = pd.concat([X_te_base, Xa_te], axis=1)

                res = multi_seed_eval(X_tr_w, y_tr, X_va_w, y_va,
                                      X_te_w, y_te, task_info)
                jrn_raw[jk][ag] = res

                agg_cache[(jk, ag)] = {
                    "train": Xa_tr, "val": Xa_va, "test": Xa_te
                }
                logger.info(f"    [{ag:>4s}] {res['mean']:.6f} ± {res['std']:.6f}  "
                            f"({X_tr_w.shape[1]} feats)")

                del X_tr_w, X_va_w, X_te_w, a_tr, a_va, a_te
            except Exception:
                logger.exception(f"    [{ag}] FAILED")
                jrn_raw[jk][ag] = {"scores": [], "mean": float("nan"),
                                   "std": float("nan")}

    # ----- parent lookup joins -----
    for child, fk, par in REACHABLE_PARENT_JOINS.get(entity_table, []):
        jk = f"{child}.{fk} -> {par} (lookup)"
        logger.info(f"  Parent lookup: {jk}")
        jrn_raw[jk] = {}
        try:
            p_pkey = SCHEMA[par]["pkey"]

            lk_tr = parent_lookup_features(
                train_df[entity_col].values,
                tables[entity_table], pkey, fk,
                tables[par], p_pkey, par)
            lk_va = parent_lookup_features(
                val_df[entity_col].values,
                tables[entity_table], pkey, fk,
                tables[par], p_pkey, par)
            lk_te = parent_lookup_features(
                test_df[entity_col].values,
                tables[entity_table], pkey, fk,
                tables[par], p_pkey, par)

            # Reset index alignment
            lk_tr = lk_tr.reset_index(drop=True)
            lk_va = lk_va.reset_index(drop=True)
            lk_te = lk_te.reset_index(drop=True)

            X_tr_w = pd.concat([X_tr_base, lk_tr], axis=1)
            X_va_w = pd.concat([X_va_base, lk_va], axis=1)
            X_te_w = pd.concat([X_te_base, lk_te], axis=1)

            res = multi_seed_eval(X_tr_w, y_tr, X_va_w, y_va,
                                  X_te_w, y_te, task_info)
            jrn_raw[jk]["lookup"] = res
            agg_cache[(jk, "lookup")] = {
                "train": lk_tr, "val": lk_va, "test": lk_te
            }
            logger.info(f"    [look] {res['mean']:.6f} ± {res['std']:.6f}")
            del X_tr_w, X_va_w, X_te_w
        except Exception:
            logger.exception(f"    [lookup] FAILED")
            jrn_raw[jk]["lookup"] = {"scores": [], "mean": float("nan"),
                                     "std": float("nan")}

    # ----- compute JRN values & classify -----
    base_m = baseline["mean"]
    jrn_vals: dict[str, dict] = {}

    for jk, agg_res in jrn_raw.items():
        best_ag, best_m = None, float("-inf")
        for ag, r in agg_res.items():
            if not np.isnan(r["mean"]) and r["mean"] > best_m:
                best_m = r["mean"]
                best_ag = ag

        if best_ag is None or np.isnan(base_m) or base_m == 0:
            jrn_v = None
            cat = "N/A"
        else:
            if task_info["task_type"] == "regression":
                # neg_mae: both negative, higher is better
                # jrn > 1 when join helps  →  base/best (both neg)
                jrn_v = float(base_m / best_m) if best_m != 0 else None
            else:
                jrn_v = float(best_m / base_m)

            if jrn_v is None or np.isnan(jrn_v):
                cat = "N/A"
            elif jrn_v > JRN_HIGH:
                cat = "HIGH"
            elif jrn_v < JRN_LOW:
                cat = "LOW"
            else:
                cat = "CRITICAL"

        jrn_vals[jk] = {
            "jrn": jrn_v,
            "category": cat,
            "best_agg": best_ag,
            "best_score": float(best_m) if not np.isnan(best_m) else None,
            "per_agg": {ag: {"mean": r["mean"], "std": r["std"]}
                        for ag, r in agg_res.items()},
        }
        jrn_str = f"{jrn_v:.4f}" if jrn_v is not None else "N/A"
        logger.info(f"  JRN({jk}) = {jrn_str}  → {cat}")

    elapsed = time.time() - t0
    logger.info(f"  JRN estimation done in {elapsed:.0f}s")

    return {
        "baseline": baseline,
        "jrn_vals": jrn_vals,
        "jrn_raw": jrn_raw,
        "base_feats": {"train": X_tr_base, "val": X_va_base,
                       "test": X_te_base},
        "agg_cache": agg_cache,
        "labels": {"train": y_tr, "val": y_va, "test": y_te},
        "split_dfs": {"train": train_df, "val": val_df, "test": test_df},
    }


# ============================================================
# Configuration Building & Evaluation
# ============================================================

def _build_config_X(config: str,
                    split: str,
                    jrn_vals: dict,
                    base: dict,
                    cache: dict,
                    oracle_subset: list[str] | None = None):
    parts = [base[split]]
    used: list[str] = []

    join_keys = list(jrn_vals.keys())

    if config == "jrn_guided":
        for jk in join_keys:
            c = jrn_vals[jk]["category"]
            if c == "HIGH":
                for k in ["mean", "lookup"]:
                    if (jk, k) in cache:
                        parts.append(cache[(jk, k)][split])
                        used.append(jk); break
            elif c == "CRITICAL":
                for k in ["all", "lookup"]:
                    if (jk, k) in cache:
                        parts.append(cache[(jk, k)][split])
                        used.append(jk); break

    elif config == "uniform_mean":
        for jk in join_keys:
            for k in ["mean", "lookup"]:
                if (jk, k) in cache:
                    parts.append(cache[(jk, k)][split])
                    used.append(jk); break

    elif config == "uniform_rich":
        for jk in join_keys:
            for k in ["all", "lookup"]:
                if (jk, k) in cache:
                    parts.append(cache[(jk, k)][split])
                    used.append(jk); break

    elif config == "top_k":
        K = sum(1 for v in jrn_vals.values() if v["category"] != "LOW")
        K = max(K, 1)
        ranked = sorted(join_keys,
                        key=lambda j: jrn_vals[j]["jrn"]
                        if jrn_vals[j]["jrn"] is not None else -999,
                        reverse=True)[:K]
        for jk in ranked:
            for k in ["mean", "lookup"]:
                if (jk, k) in cache:
                    parts.append(cache[(jk, k)][split])
                    used.append(jk); break

    elif config == "oracle":
        if oracle_subset is None:
            oracle_subset = []
        for jk in oracle_subset:
            for k in ["all", "mean", "lookup"]:
                if (jk, k) in cache:
                    parts.append(cache[(jk, k)][split])
                    used.append(jk); break

    X = pd.concat(parts, axis=1)
    return X, used


def greedy_oracle(jrn_vals, base, cache, labels, task_info):
    """Greedy forward selection as oracle approximation."""
    logger.info("  Greedy forward selection (oracle) …")
    available = list(jrn_vals.keys())
    best_sub: list[str] = []
    # baseline
    res = multi_seed_eval(base["train"], labels["train"],
                          base["val"], labels["val"],
                          base["test"], labels["test"],
                          task_info, seeds=[42])
    best_sc = res["mean"]
    logger.info(f"    baseline oracle score: {best_sc:.6f}")

    improved = True
    while improved and len(best_sub) < len(available):
        improved = False
        cand_jk, cand_sc = None, best_sc

        for jk in available:
            if jk in best_sub:
                continue
            trial = best_sub + [jk]
            parts_tr = [base["train"]]
            parts_va = [base["val"]]
            parts_te = [base["test"]]
            for j in trial:
                for k in ["all", "mean", "lookup"]:
                    if (j, k) in cache:
                        parts_tr.append(cache[(j, k)]["train"])
                        parts_va.append(cache[(j, k)]["val"])
                        parts_te.append(cache[(j, k)]["test"])
                        break
            Xt = pd.concat(parts_tr, axis=1)
            Xv = pd.concat(parts_va, axis=1)
            Xe = pd.concat(parts_te, axis=1)
            r = multi_seed_eval(Xt, labels["train"],
                                Xv, labels["val"],
                                Xe, labels["test"],
                                task_info, seeds=[42])
            if r["mean"] > cand_sc + 0.0005:
                cand_sc = r["mean"]
                cand_jk = jk
            del Xt, Xv, Xe

        if cand_jk is not None:
            best_sub.append(cand_jk)
            best_sc = cand_sc
            improved = True
            logger.info(f"    +{cand_jk}  → {best_sc:.6f}")

    logger.info(f"  Oracle subset ({len(best_sub)} joins): {best_sub}")
    return best_sub


def eval_all_configs(task_name, task_info, task_res):
    """Evaluate 5 configs with FINAL_PARAMS."""
    logger.info(f"  Final config evaluation for {task_name} …")
    jrn_vals = task_res["jrn_vals"]
    base = task_res["base_feats"]
    cache = task_res["agg_cache"]
    labels = task_res["labels"]

    oracle_sub = greedy_oracle(jrn_vals, base, cache, labels, task_info)

    cfg_res: dict[str, dict] = {}
    for cfg in ["jrn_guided", "uniform_mean", "uniform_rich", "top_k"]:
        Xt, used = _build_config_X(cfg, "train", jrn_vals, base, cache)
        Xv, _ = _build_config_X(cfg, "val", jrn_vals, base, cache)
        Xe, _ = _build_config_X(cfg, "test", jrn_vals, base, cache)

        r = multi_seed_eval(Xt, labels["train"], Xv, labels["val"],
                            Xe, labels["test"], task_info,
                            seeds=SEEDS, params=FINAL_PARAMS)
        r["n_features"] = int(Xt.shape[1])
        r["joins_used"] = used
        cfg_res[cfg] = r
        logger.info(f"    {cfg:15s}: {r['mean']:.6f} ± {r['std']:.6f}  "
                     f"({r['n_features']} feats, {len(used)} joins)")
        del Xt, Xv, Xe

    # oracle
    Xt, used = _build_config_X("oracle", "train", jrn_vals, base, cache,
                               oracle_sub)
    Xv, _ = _build_config_X("oracle", "val", jrn_vals, base, cache,
                            oracle_sub)
    Xe, _ = _build_config_X("oracle", "test", jrn_vals, base, cache,
                            oracle_sub)
    r = multi_seed_eval(Xt, labels["train"], Xv, labels["val"],
                        Xe, labels["test"], task_info,
                        seeds=SEEDS, params=FINAL_PARAMS)
    r["n_features"] = int(Xt.shape[1])
    r["joins_used"] = used
    r["oracle_subset"] = oracle_sub
    cfg_res["oracle"] = r
    logger.info(f"    {'oracle':15s}: {r['mean']:.6f} ± {r['std']:.6f}  "
                 f"({r['n_features']} feats, {len(used)} joins)")
    del Xt, Xv, Xe

    return cfg_res


# ============================================================
# Analysis
# ============================================================

def compute_analysis(all_tr: dict, all_cr: dict) -> dict:
    tnames = list(TASKS.keys())
    analysis: dict = {}

    # --- win-rates ---
    wr: dict = {}
    for other in ["uniform_mean", "uniform_rich", "top_k"]:
        w, t, l = 0, 0, 0
        for tn in tnames:
            jg = all_cr[tn]["jrn_guided"]["mean"]
            ot = all_cr[tn][other]["mean"]
            if jg > ot + 1e-4:
                w += 1
            elif ot > jg + 1e-4:
                l += 1
            else:
                t += 1
        wr[f"jrn_vs_{other}"] = {"wins": w, "ties": t, "losses": l,
                                  "rate": w / len(tnames)}
    analysis["win_rates"] = wr

    # --- oracle gap ---
    og: dict = {}
    for tn in tnames:
        o = all_cr[tn]["oracle"]["mean"]
        j = all_cr[tn]["jrn_guided"]["mean"]
        b = all_tr[tn]["baseline"]["mean"]
        gap = (o - j) / (o - b) if (o - b) != 0 else 0.0
        og[tn] = {"gap": round(gap, 4), "oracle": round(o, 6),
                  "jrn_guided": round(j, 6), "baseline": round(b, 6)}
    analysis["oracle_gap"] = og

    # --- category distribution ---
    cd: dict = {}
    for tn in tnames:
        counts = {"HIGH": 0, "CRITICAL": 0, "LOW": 0, "N/A": 0}
        for v in all_tr[tn]["jrn_vals"].values():
            counts[v["category"]] = counts.get(v["category"], 0) + 1
        cd[tn] = counts
    analysis["category_distribution"] = cd

    # --- domain validation ---
    dv: dict = {}
    for tn in tnames:
        checks = []
        jv = all_tr[tn]["jrn_vals"]
        if tn == "post-votes":
            k = _jk("votes", "PostId", "posts")
            if k in jv:
                checks.append({"join": k, "expected": "HIGH",
                               "actual": jv[k]["category"],
                               "jrn": jv[k]["jrn"],
                               "passed": jv[k]["category"] == "HIGH"})
        if tn == "user-badge":
            k = _jk("badges", "UserId", "users")
            if k in jv:
                checks.append({"join": k, "expected": "HIGH",
                               "actual": jv[k]["category"],
                               "jrn": jv[k]["jrn"],
                               "passed": jv[k]["category"] == "HIGH"})
        if tn == "user-engagement":
            k = _jk("comments", "UserId", "users")
            if k in jv:
                checks.append({"join": k, "expected_informative": True,
                               "actual": jv[k]["category"],
                               "jrn": jv[k]["jrn"]})
        dv[tn] = checks
    analysis["domain_validation"] = dv

    # --- aggregation sensitivity ---
    ag_sens: dict = {}
    for tn in tnames:
        raw = all_tr[tn].get("jrn_raw", {})
        jv = all_tr[tn]["jrn_vals"]
        for jk in jv:
            if jk not in raw:
                continue
            vals = []
            for ag in AGG_TYPES:
                if ag in raw[jk] and not np.isnan(raw[jk][ag]["mean"]):
                    vals.append(raw[jk][ag]["mean"])
            if len(vals) >= 2:
                mu = abs(np.mean(vals))
                sens = float(np.std(vals) / mu) if mu > 0 else 0
                ag_sens[f"{tn}/{jk}"] = {
                    "jrn": jv[jk]["jrn"],
                    "sensitivity": round(sens, 6),
                }
    analysis["aggregation_sensitivity"] = ag_sens

    # --- feature efficiency ---
    fe: dict = {}
    for tn in tnames:
        b_m = all_tr[tn]["baseline"]["mean"]
        for cfg in ["jrn_guided", "uniform_mean", "uniform_rich",
                    "top_k", "oracle"]:
            cr = all_cr[tn][cfg]
            gain = cr["mean"] - b_m
            nf = cr["n_features"]
            fe[f"{tn}/{cfg}"] = {
                "n_features": nf,
                "gain": round(gain, 6),
                "efficiency": round(gain / nf, 8) if nf else 0,
            }
    analysis["feature_efficiency"] = fe

    return analysis


# ============================================================
# Output Formatting (exp_gen_sol_out schema)
# ============================================================

def format_output(all_tr, all_cr, analysis):
    jk_ordered = [_jk(c, f, p) for c, f, p in ALL_JOINS]
    tnames = list(TASKS.keys())

    # JRN matrix
    mat_v, mat_c = [], []
    for jk in jk_ordered:
        rv, rc = [], []
        for tn in tnames:
            jv = all_tr[tn]["jrn_vals"]
            if jk in jv:
                rv.append(jv[jk]["jrn"])
                rc.append(jv[jk]["category"])
            else:
                rv.append(None)
                rc.append("N/A")
        mat_v.append(rv)
        mat_c.append(rc)

    per_agg: dict = {}
    for tn in tnames:
        raw = all_tr[tn].get("jrn_raw", {})
        for jk, ares in raw.items():
            for ag, r in ares.items():
                per_agg[f"{tn}/{jk}/{ag}"] = {
                    "mean": r["mean"], "std": r["std"]
                }

    metadata = {
        "description": "JRN-guided heterogeneous architecture experiment "
                       "on rel-stack",
        "dataset": "rel-stack",
        "tasks": tnames,
        "n_joins_total": 11,
        "probe_params": {k: v for k, v in PROBE_PARAMS.items()},
        "final_params": {k: v for k, v in FINAL_PARAMS.items()},
        "seeds": SEEDS,
        "temporal_splits": {"val": str(VAL_TS), "test": str(TEST_TS)},
        "jrn_thresholds": {"high": JRN_HIGH, "low": JRN_LOW},
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "jrn_matrix": {
            "joins": jk_ordered,
            "tasks": tnames,
            "values": mat_v,
            "categories": mat_c,
            "per_agg_scores": per_agg,
        },
        "baseline_scores": {
            tn: {"mean": all_tr[tn]["baseline"]["mean"],
                 "std": all_tr[tn]["baseline"]["std"],
                 "scores": all_tr[tn]["baseline"]["scores"]}
            for tn in tnames
        },
        "config_results": {
            tn: {
                cn: {
                    "mean": cr["mean"], "std": cr["std"],
                    "scores": cr["scores"],
                    "n_features": cr["n_features"],
                    "joins_used": cr["joins_used"],
                }
                for cn, cr in all_cr[tn].items()
            }
            for tn in tnames
        },
        "analysis": analysis,
    }

    # ----- examples -----
    examples: list[dict] = []

    # (A) one per task — config comparison
    for tn in tnames:
        ti = TASKS[tn]
        cfg_summary = {}
        for cn in ["jrn_guided", "uniform_mean", "uniform_rich",
                    "top_k", "oracle"]:
            cr = all_cr[tn][cn]
            cfg_summary[cn] = {
                "mean": round(cr["mean"], 6),
                "std": round(cr["std"], 6),
                "n_features": cr["n_features"],
                "joins_used": cr["joins_used"],
            }

        jrn_summary = {}
        for jk, jv in all_tr[tn]["jrn_vals"].items():
            jrn_summary[jk] = {
                "jrn": round(jv["jrn"], 4) if jv["jrn"] is not None else None,
                "category": jv["category"],
                "best_agg": jv["best_agg"],
            }

        ex = {
            "input": json.dumps({
                "task": tn, "dataset": "rel-stack",
                "metric": ti["metric"],
                "entity_table": ti["entity_table"],
                "task_type": ti["task_type"],
                "description": (f"Evaluate JRN-guided heterogeneous "
                                f"architecture vs baselines on {tn} "
                                f"(rel-stack). Metric: {ti['metric']}."),
            }),
            "output": json.dumps({
                "config_results": cfg_summary,
                "jrn_matrix": jrn_summary,
                "baseline": round(all_tr[tn]["baseline"]["mean"], 6),
            }),
            "predict_jrn_guided": str(round(
                all_cr[tn]["jrn_guided"]["mean"], 6)),
            "predict_uniform_mean": str(round(
                all_cr[tn]["uniform_mean"]["mean"], 6)),
            "predict_uniform_rich": str(round(
                all_cr[tn]["uniform_rich"]["mean"], 6)),
            "predict_top_k": str(round(
                all_cr[tn]["top_k"]["mean"], 6)),
            "predict_oracle": str(round(
                all_cr[tn]["oracle"]["mean"], 6)),
            "metadata_task": tn,
            "metadata_metric": ti["metric"],
            "metadata_entity_table": ti["entity_table"],
            "metadata_task_type": ti["task_type"],
            "metadata_n_joins": len(all_tr[tn]["jrn_vals"]),
        }
        examples.append(ex)

    # (B) one per (task, join) JRN probe
    for tn in tnames:
        bm = all_tr[tn]["baseline"]["mean"]
        for jk, jv in all_tr[tn]["jrn_vals"].items():
            ex = {
                "input": json.dumps({
                    "task": tn, "join": jk,
                    "probe_type": "jrn_estimation",
                    "description": f"JRN probe for {jk} on {tn}",
                }),
                "output": json.dumps({
                    "jrn": (round(jv["jrn"], 4)
                            if jv["jrn"] is not None else None),
                    "category": jv["category"],
                    "best_agg": jv["best_agg"],
                    "best_score": (round(jv["best_score"], 6)
                                   if jv["best_score"] is not None
                                   else None),
                    "baseline": round(bm, 6),
                }),
                "predict_jrn_value": (
                    str(round(jv["jrn"], 4))
                    if jv["jrn"] is not None else "N/A"),
                "predict_category": jv["category"],
                "metadata_task": tn,
                "metadata_join": jk,
                "metadata_category": jv["category"],
            }
            examples.append(ex)

    return {
        "metadata": metadata,
        "datasets": [{"dataset": "rel-stack", "examples": examples}],
    }


# ============================================================
# Main
# ============================================================

@logger.catch
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("JRN-Guided Heterogeneous Architecture Experiment")
    logger.info("=" * 60)

    # ---- load data ----
    tables, tasks_data = load_relbench_data()
    gc.collect()

    # ---- process each task ----
    all_tr: dict = {}
    all_cr: dict = {}

    for task_name, task_info in TASKS.items():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"TASK: {task_name}")
        logger.info(f"{'=' * 50}")

        task_res = estimate_jrn_for_task(task_name, task_info,
                                         tables, tasks_data)
        cfg_res = eval_all_configs(task_name, task_info, task_res)

        all_cr[task_name] = cfg_res

        # keep lightweight copy for analysis
        all_tr[task_name] = {
            "baseline": task_res["baseline"],
            "jrn_vals": task_res["jrn_vals"],
            "jrn_raw": task_res["jrn_raw"],
        }

        # free heavy data
        del task_res["base_feats"]
        del task_res["agg_cache"]
        del task_res["labels"]
        del task_res["split_dfs"]
        del task_res
        gc.collect()

        elapsed = time.time() - t0
        logger.info(f"Elapsed so far: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # ---- analysis ----
    logger.info(f"\n{'=' * 50}")
    logger.info("ANALYSIS")
    logger.info(f"{'=' * 50}")
    analysis = compute_analysis(all_tr, all_cr)

    # ---- output ----
    output = format_output(all_tr, all_cr, analysis)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # ---- summary ----
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 50}")
    logger.info(f"DONE in {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    logger.info(f"{'=' * 50}")
    for tn in TASKS:
        logger.info(f"\n  {tn}:")
        logger.info(f"    baseline : {all_tr[tn]['baseline']['mean']:.6f}")
        for cn in ["jrn_guided", "uniform_mean", "uniform_rich",
                    "top_k", "oracle"]:
            cr = all_cr[tn][cn]
            logger.info(f"    {cn:15s}: {cr['mean']:.6f} ± {cr['std']:.6f}")
    logger.info(f"\n  Win rates: "
                f"{json.dumps(analysis['win_rates'], indent=4)}")
    logger.info(f"  Oracle gaps: "
                f"{json.dumps(analysis['oracle_gap'], indent=4)}")


if __name__ == "__main__":
    main()
