#!/usr/bin/env python3
"""Training-Free Proxy vs GBM-Probe JRN: Cross-Dataset Diagnostic Reliability.

Computes GBM-probe JRN (ground truth) and 5 training-free proxies for every
relevant (join, task) pair on two RelBench datasets:
  - rel-stack: 7 tables, 11 FKs, 3 non-link tasks
  - rel-avito: 8 tables, 11 FKs, 3 non-link tasks

Produces method_out.json with Spearman rho, ROC-AUC for join selection,
cross-dataset consistency, and conditions analysis.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import entropy as scipy_entropy
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_absolute_error, roc_auc_score

warnings.filterwarnings("ignore")

# Try LightGBM, fall back to sklearn
try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

# ── Logging ───────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ────────────────────────────────────────────────────
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
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.7 * 1e9)  # 70% of container
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3400, 3400))  # ~56 min CPU time

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")
logger.info(f"LightGBM available: {HAS_LGB}")

# ── Dependency paths ──────────────────────────────────────────────────────
STACK_DATA_PATH = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id4_it1__opus/full_data_out.json")
AVITO_DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_2/gen_art/data_id5_it2__opus/data_out")

# ── Task definitions (excluding link prediction) ─────────────────────────
STACK_TASKS = {
    "user-engagement": {"entity_table": "users", "pk_col": "Id", "task_type": "binary_classification"},
    "user-badge": {"entity_table": "users", "pk_col": "Id", "task_type": "binary_classification"},
    "post-votes": {"entity_table": "posts", "pk_col": "Id", "task_type": "regression"},
}

AVITO_TASKS = {
    "ad-ctr": {"entity_table": "AdsInfo", "pk_col": "AdID", "task_type": "regression",
               "relbench_name": "rel-avito/ad-ctr"},
    "user-clicks": {"entity_table": "UserInfo", "pk_col": "UserID", "task_type": "binary_classification",
                    "relbench_name": "rel-avito/user-clicks"},
    "user-visits": {"entity_table": "UserInfo", "pk_col": "UserID", "task_type": "binary_classification",
                    "relbench_name": "rel-avito/user-visits"},
}

# Joins per entity table
STACK_JOINS = {
    "users": [
        ("badges", "UserId", "users", "Id"),
        ("comments", "UserId", "users", "Id"),
        ("posts", "OwnerUserId", "users", "Id"),
        ("postHistory", "UserId", "users", "Id"),
        ("votes", "UserId", "users", "Id"),
    ],
    "posts": [
        ("comments", "PostId", "posts", "Id"),
        ("postLinks", "PostId", "posts", "Id"),
        ("postLinks", "RelatedPostId", "posts", "Id"),
        ("postHistory", "PostId", "posts", "Id"),
        ("votes", "PostId", "posts", "Id"),
    ],
}

AVITO_JOINS = {
    "AdsInfo": [
        ("SearchStream", "AdID", "AdsInfo", "AdID"),
        ("VisitStream", "AdID", "AdsInfo", "AdID"),
        ("PhoneRequestsStream", "AdID", "AdsInfo", "AdID"),
    ],
    "UserInfo": [
        ("SearchInfo", "UserID", "UserInfo", "UserID"),
        ("VisitStream", "UserID", "UserInfo", "UserID"),
        ("PhoneRequestsStream", "UserID", "UserInfo", "UserID"),
    ],
}


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_stack_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """Load rel-stack tables and FK metadata from dependency JSON."""
    logger.info("Loading rel-stack data...")
    raw = json.loads(STACK_DATA_PATH.read_text())

    # Extract FK metadata from the metadata section
    fk_metadata = {}
    join_stats = raw.get("metadata", {}).get("rel_stack", {}).get("join_statistics", {})
    for key, stats in join_stats.items():
        fk_metadata[key] = stats.get("fan_out_stats", {})
        fk_metadata[key]["fanout_mean"] = stats.get("fan_out_stats", {}).get("mean", 0)

    # Reconstruct DataFrames per table
    examples = raw["datasets"][0]["examples"]
    table_rows: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        tbl = ex.get("metadata_table", "unknown")
        row = json.loads(ex["input"])
        table_rows[tbl].append(row)

    tables = {}
    for tbl_name, rows in table_rows.items():
        df = pd.DataFrame(rows)
        # Convert numeric-looking columns
        for col in df.columns:
            if col in ("Text", "Body", "AboutMe", "Comment", "Title", "Tags",
                       "ContentLicense", "UserDisplayName", "OwnerDisplayName",
                       "DisplayName", "Location", "WebsiteUrl", "Name",
                       "RevisionGUID", "ProfileImageUrl"):
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except (ValueError, TypeError):
                pass
        tables[tbl_name] = df
        logger.info(f"  {tbl_name}: {len(df)} rows, {len(df.columns)} cols")

    del raw, examples
    gc.collect()
    return tables, fk_metadata


def load_avito_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict], Dict[str, Dict[str, pd.DataFrame]]]:
    """Load rel-avito tables, FK metadata, and task samples from dependency JSON parts."""
    logger.info("Loading rel-avito data...")
    tables: Dict[str, List[Dict]] = defaultdict(list)
    fk_metadata = {}
    task_samples: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for i in range(1, 7):
        path = AVITO_DATA_DIR / f"full_data_out_{i}.json"
        logger.info(f"  Loading part {i}: {path.name}")
        raw = json.loads(path.read_text())
        examples = raw["datasets"][0]["examples"]

        for ex in examples:
            row_type = ex.get("metadata_row_type", "")

            if row_type == "table_row":
                tbl = ex.get("metadata_table_name", "unknown")
                row = json.loads(ex["input"])
                pk_col = ex.get("metadata_primary_key_col", "")
                pk_val = ex.get("metadata_primary_key_value", "")
                if pk_col and pk_val:
                    try:
                        row[pk_col] = int(pk_val) if pk_val.isdigit() else pk_val
                    except (ValueError, AttributeError):
                        row[pk_col] = pk_val
                tables[tbl].append(row)

            elif row_type == "fk_join_metadata":
                inp = json.loads(ex["input"])
                out = json.loads(ex["output"])
                key = f"{inp['source_table']}.{inp['source_fk_col']} -> {inp['target_table']}.{inp['target_pk_col']}"
                fk_metadata[key] = out

            elif row_type == "task_sample":
                task_name = ex.get("metadata_task_name", "")
                task_type = ex.get("metadata_task_type", "")
                fold = ex.get("metadata_fold_name", "train")
                if task_type == "link_prediction":
                    continue
                inp = json.loads(ex["input"])
                label = ex.get("output", "")
                # Parse label
                if task_type == "binary_classification":
                    if label in ("True", "true", True):
                        label_val = 1
                    elif label in ("False", "false", False):
                        label_val = 0
                    else:
                        try:
                            label_val = int(float(label))
                        except (ValueError, TypeError):
                            continue
                else:
                    try:
                        label_val = float(label)
                    except (ValueError, TypeError):
                        continue
                inp["_target"] = label_val
                task_samples[task_name][fold].append(inp)

        del raw, examples
        gc.collect()

    # Convert to DataFrames
    table_dfs = {}
    for tbl_name, rows in tables.items():
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in ("Title", "SearchQuery", "Params"):
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except (ValueError, TypeError):
                pass
        table_dfs[tbl_name] = df
        logger.info(f"  {tbl_name}: {len(df)} rows, {len(df.columns)} cols")

    task_dfs = {}
    for task_name, folds in task_samples.items():
        task_dfs[task_name] = {}
        for fold_name, rows in folds.items():
            task_dfs[task_name][fold_name] = pd.DataFrame(rows)
            logger.info(f"  Task {task_name}/{fold_name}: {len(rows)} samples")

    del tables, task_samples
    gc.collect()
    return table_dfs, fk_metadata, task_dfs


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: CONSTRUCT PROXY LABELS FOR REL-STACK
# ══════════════════════════════════════════════════════════════════════════

def construct_stack_task_labels(
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Construct proxy task labels for rel-stack since dependency has no task samples."""
    task_dfs = {}
    users = tables["users"].copy()
    posts = tables["posts"].copy()

    # ── user-engagement: binary — has user posted or commented? ──
    post_counts = tables["posts"].groupby("OwnerUserId").size().rename("post_count")
    comment_counts = tables["comments"].groupby("UserId").size().rename("comment_count")
    user_activity = users[["Id"]].copy()
    user_activity = user_activity.merge(post_counts, left_on="Id", right_index=True, how="left")
    user_activity = user_activity.merge(comment_counts, left_on="Id", right_index=True, how="left")
    user_activity["post_count"] = user_activity["post_count"].fillna(0)
    user_activity["comment_count"] = user_activity["comment_count"].fillna(0)
    user_activity["_target"] = ((user_activity["post_count"] + user_activity["comment_count"]) > 2).astype(int)
    user_activity = user_activity.drop(columns=["post_count", "comment_count"])
    # Split 70/30
    n = len(user_activity)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = idx[:int(0.7*n)], idx[int(0.7*n):]
    task_dfs["user-engagement"] = {
        "train": user_activity.iloc[train_idx].reset_index(drop=True),
        "val": user_activity.iloc[val_idx].reset_index(drop=True),
    }
    logger.info(f"  user-engagement: {len(train_idx)} train, {len(val_idx)} val, pos_rate={user_activity['_target'].mean():.3f}")

    # ── user-badge: binary — will user get badge? ──
    badge_counts = tables["badges"].groupby("UserId").size().rename("badge_count")
    user_badge = users[["Id"]].copy()
    user_badge = user_badge.merge(badge_counts, left_on="Id", right_index=True, how="left")
    user_badge["badge_count"] = user_badge["badge_count"].fillna(0)
    user_badge["_target"] = (user_badge["badge_count"] > 1).astype(int)
    user_badge = user_badge.drop(columns=["badge_count"])
    idx = np.random.RandomState(43).permutation(len(user_badge))
    n = len(user_badge)
    task_dfs["user-badge"] = {
        "train": user_badge.iloc[idx[:int(0.7*n)]].reset_index(drop=True),
        "val": user_badge.iloc[idx[int(0.7*n):]].reset_index(drop=True),
    }
    logger.info(f"  user-badge: pos_rate={user_badge['_target'].mean():.3f}")

    # ── post-votes: regression — vote count per post ──
    vote_counts = tables["votes"].groupby("PostId").size().rename("vote_count")
    post_votes = posts[["Id"]].copy()
    post_votes = post_votes.merge(vote_counts, left_on="Id", right_index=True, how="left")
    post_votes["vote_count"] = post_votes["vote_count"].fillna(0)
    post_votes["_target"] = post_votes["vote_count"].astype(float)
    post_votes = post_votes.drop(columns=["vote_count"])
    idx = np.random.RandomState(44).permutation(len(post_votes))
    n = len(post_votes)
    task_dfs["post-votes"] = {
        "train": post_votes.iloc[idx[:int(0.7*n)]].reset_index(drop=True),
        "val": post_votes.iloc[idx[int(0.7*n):]].reset_index(drop=True),
    }
    logger.info(f"  post-votes: mean={post_votes['_target'].mean():.3f}, std={post_votes['_target'].std():.3f}")

    return task_dfs


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING — AGGREGATED JOIN FEATURES
# ══════════════════════════════════════════════════════════════════════════

def compute_aggregated_features(
    child_df: pd.DataFrame,
    fk_col: str,
    max_children_per_parent: int = 10000,
) -> Optional[pd.DataFrame]:
    """Aggregate numeric child features by FK, plus child count."""
    numeric_cols = child_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != fk_col]

    if len(numeric_cols) == 0:
        return None

    valid = child_df[child_df[fk_col].notna()].copy()
    if len(valid) == 0:
        return None

    # Cap children per parent to avoid OOM
    counts = valid[fk_col].value_counts()
    big_parents = counts[counts > max_children_per_parent].index
    if len(big_parents) > 0:
        logger.info(f"  Capping {len(big_parents)} parents with >{max_children_per_parent} children")
        mask_big = valid[fk_col].isin(big_parents)
        sampled = valid[mask_big].groupby(fk_col).apply(
            lambda x: x.sample(min(len(x), max_children_per_parent), random_state=42)
        ).reset_index(drop=True)
        valid = pd.concat([valid[~mask_big], sampled], ignore_index=True)

    agg = valid.groupby(fk_col)[numeric_cols].mean()
    agg.columns = [f"agg_{c}" for c in agg.columns]
    agg["agg_child_count"] = valid.groupby(fk_col).size()

    return agg


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: TRAIN GBM AND COMPUTE JRN
# ══════════════════════════════════════════════════════════════════════════

def _train_and_score(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    task_type: str, seed: int, n_jobs: int,
) -> float:
    """Train a single GBM model and return the metric score."""
    if HAS_LGB:
        params = {
            "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
            "random_state": seed, "verbose": -1, "n_jobs": n_jobs,
            "min_child_samples": 20,
        }
        if task_type == "binary_classification":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
    else:
        params = {
            "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05,
            "random_state": seed,
        }
        if task_type == "binary_classification":
            model = GradientBoostingClassifier(**params)
        else:
            model = GradientBoostingRegressor(**params)

    model.fit(X_train, y_train)

    if task_type == "binary_classification":
        pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, pred)
    else:
        pred = model.predict(X_val)
        return mean_absolute_error(y_val, pred)


def compute_gbm_jrn(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    base_features: List[str], aug_features: List[str],
    target_col: str, task_type: str, n_seeds: int = 3,
) -> Dict[str, float]:
    """Compute GBM-probe JRN: ratio of augmented/baseline metric."""
    results = {"baseline": [], "augmented": [], "jrn": []}

    # Prepare data once
    X_train_base = train_df[base_features].fillna(0).values.astype(np.float32)
    X_val_base = val_df[base_features].fillna(0).values.astype(np.float32)
    all_feats = base_features + aug_features
    X_train_aug = train_df[all_feats].fillna(0).values.astype(np.float32)
    X_val_aug = val_df[all_feats].fillna(0).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float64)
    y_val = val_df[target_col].values.astype(np.float64)

    # Check for degenerate cases
    if task_type == "binary_classification":
        unique_train = np.unique(y_train)
        unique_val = np.unique(y_val)
        if len(unique_train) < 2 or len(unique_val) < 2:
            logger.warning("Degenerate labels — skipping")
            return {"baseline_mean": 0.5, "baseline_std": 0, "augmented_mean": 0.5,
                    "augmented_std": 0, "jrn_mean": 1.0, "jrn_std": 0}

    n_jobs = max(1, NUM_CPUS)

    for seed in range(n_seeds):
        try:
            score_base = _train_and_score(X_train_base, y_train, X_val_base, y_val,
                                           task_type, seed, n_jobs)
            score_aug = _train_and_score(X_train_aug, y_train, X_val_aug, y_val,
                                          task_type, seed, n_jobs)
            results["baseline"].append(score_base)
            results["augmented"].append(score_aug)

            if task_type == "regression":
                jrn = score_base / max(score_aug, 1e-10)
            else:
                jrn = score_aug / max(score_base, 1e-10)
            results["jrn"].append(jrn)
        except Exception:
            logger.exception(f"GBM training failed for seed {seed}")
            continue

    if not results["jrn"]:
        return {"baseline_mean": 0, "baseline_std": 0, "augmented_mean": 0,
                "augmented_std": 0, "jrn_mean": 1.0, "jrn_std": 0}

    return {
        "baseline_mean": float(np.mean(results["baseline"])),
        "baseline_std": float(np.std(results["baseline"])),
        "augmented_mean": float(np.mean(results["augmented"])),
        "augmented_std": float(np.std(results["augmented"])),
        "jrn_mean": float(np.mean(results["jrn"])),
        "jrn_std": float(np.std(results["jrn"])),
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: TRAINING-FREE PROXIES
# ══════════════════════════════════════════════════════════════════════════

def compute_training_free_proxies(
    train_df: pd.DataFrame, target_col: str,
    aug_features: List[str], task_type: str,
    fk_meta: Dict[str, Any],
) -> Dict[str, float]:
    """Compute 5 training-free proxies for a (join, task) pair."""
    y = train_df[target_col].dropna().values.astype(np.float64)
    agg_df = train_df[aug_features].fillna(0)
    proxies: Dict[str, float] = {}

    # Subsample for expensive computations
    max_samples = 20000
    if len(y) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), max_samples, replace=False)
        y_sub = y[idx]
        agg_sub = agg_df.iloc[idx]
    else:
        y_sub = y
        agg_sub = agg_df

    # ── PROXY 1: Conditional Entropy Reduction ──
    try:
        n_bins = min(10, len(np.unique(y_sub)))
        if n_bins < 2:
            n_bins = 2
        y_binned = pd.qcut(y_sub, q=n_bins, labels=False, duplicates="drop")
        H_Y = float(scipy_entropy(np.bincount(y_binned) / len(y_binned) + 1e-12))
        min_cond_H = H_Y
        for col in aug_features:
            x = agg_sub[col].values.astype(float)
            x_valid = ~np.isnan(x)
            if x_valid.sum() < 100:
                continue
            try:
                x_binned = pd.qcut(x[x_valid], q=n_bins, labels=False, duplicates="drop")
            except ValueError:
                continue
            cond_H = 0.0
            for xval in np.unique(x_binned):
                mask = x_binned == xval
                p_x = mask.sum() / len(x_binned)
                y_given_x = y_binned[x_valid][mask]
                counts = np.bincount(y_given_x, minlength=max(y_binned) + 1)
                cond_H += p_x * float(scipy_entropy(counts / max(mask.sum(), 1) + 1e-12))
            min_cond_H = min(min_cond_H, cond_H)
        proxies["entropy_reduction"] = float(H_Y - min_cond_H)
    except Exception:
        logger.exception("Entropy reduction failed")
        proxies["entropy_reduction"] = 0.0

    # ── PROXY 2: Mutual Information ──
    try:
        X_agg = agg_sub.values.astype(np.float64)
        # Replace any remaining nan/inf
        X_agg = np.nan_to_num(X_agg, nan=0.0, posinf=0.0, neginf=0.0)
        if task_type == "binary_classification":
            mi_vals = mutual_info_classif(X_agg, y_sub.astype(int), n_neighbors=5, random_state=42)
        else:
            mi_vals = mutual_info_regression(X_agg, y_sub, n_neighbors=5, random_state=42)
        proxies["mutual_information"] = float(np.max(mi_vals))
    except Exception:
        logger.exception("MI computation failed")
        proxies["mutual_information"] = 0.0

    # ── PROXY 3: Pearson Correlation ──
    try:
        max_r = 0.0
        for col in aug_features:
            x = agg_sub[col].values.astype(float)
            valid = ~(np.isnan(x) | np.isnan(y_sub))
            if valid.sum() < 30:
                continue
            r, _ = pearsonr(x[valid], y_sub[valid])
            if np.isfinite(r):
                max_r = max(max_r, abs(r))
        proxies["pearson_correlation"] = float(max_r)
    except Exception:
        logger.exception("Pearson failed")
        proxies["pearson_correlation"] = 0.0

    # ── PROXY 4: Fan-out Statistics ──
    fanout_mean = fk_meta.get("fanout_mean", fk_meta.get("mean", 0))
    proxies["mean_fanout"] = float(fanout_mean)
    proxies["log_mean_fanout"] = float(np.log1p(fanout_mean))

    # ── PROXY 5: Homophily ──
    try:
        y_median = np.median(y_sub)
        y_above = y_sub > y_median
        max_homophily = 0.5
        for col in aug_features:
            x = agg_sub[col].values.astype(float)
            valid = ~np.isnan(x)
            if valid.sum() < 30:
                continue
            x_median = np.median(x[valid])
            x_above = x > x_median
            agreement = np.mean(x_above[valid] == y_above[valid])
            max_homophily = max(max_homophily, max(agreement, 1 - agreement))
        proxies["homophily"] = float(max_homophily)
    except Exception:
        logger.exception("Homophily failed")
        proxies["homophily"] = 0.5

    return proxies


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: RUN ALL (JOIN, TASK) PAIRS
# ══════════════════════════════════════════════════════════════════════════

def get_entity_base_features(entity_df: pd.DataFrame, pk_col: str) -> List[str]:
    """Get numeric feature columns from entity table."""
    numeric = entity_df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c != pk_col]


def run_pair(
    dataset_name: str, task_name: str, task_type: str,
    pk_col: str, child_table_name: str, fk_col: str,
    parent_table_name: str, parent_pk_col: str,
    entity_df: pd.DataFrame, child_df: pd.DataFrame,
    task_train: pd.DataFrame, task_val: pd.DataFrame,
    fk_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Run one (join, task) pair: compute GBM JRN + 5 proxies."""
    join_key = f"{child_table_name}.{fk_col} -> {parent_table_name}.{parent_pk_col}"
    logger.info(f"  Processing {dataset_name}/{task_name}/{join_key}")

    try:
        # Get base features from entity table
        base_features = get_entity_base_features(entity_df, pk_col)

        # Compute aggregated features from child table
        agg_df = compute_aggregated_features(child_df, fk_col)
        if agg_df is None or agg_df.shape[1] == 0:
            logger.warning(f"  No numeric features for {join_key}, skipping")
            return None

        aug_features = agg_df.columns.tolist()

        # Merge entity features into task data
        entity_feats = entity_df[[pk_col] + base_features].copy()
        train_merged = task_train.merge(entity_feats, left_on=pk_col, right_on=pk_col, how="left")
        val_merged = task_val.merge(entity_feats, left_on=pk_col, right_on=pk_col, how="left")

        # Merge aggregated features
        train_merged = train_merged.merge(agg_df, left_on=pk_col, right_index=True, how="left")
        val_merged = val_merged.merge(agg_df, left_on=pk_col, right_index=True, how="left")

        # Subsample if needed
        if len(train_merged) > 50000:
            train_merged = train_merged.sample(50000, random_state=42)
        if len(val_merged) > 25000:
            val_merged = val_merged.sample(25000, random_state=42)

        # Drop rows with NaN target
        train_merged = train_merged.dropna(subset=["_target"])
        val_merged = val_merged.dropna(subset=["_target"])

        if len(train_merged) < 50 or len(val_merged) < 20:
            logger.warning(f"  Too few samples for {join_key}: train={len(train_merged)}, val={len(val_merged)}")
            return None

        # If base_features is empty, add a dummy constant feature
        if not base_features:
            train_merged["_dummy_base"] = 0.0
            val_merged["_dummy_base"] = 0.0
            base_features = ["_dummy_base"]

        # Compute GBM JRN
        jrn_result = compute_gbm_jrn(
            train_merged, val_merged, base_features, aug_features,
            "_target", task_type, n_seeds=3,
        )

        # Compute training-free proxies
        proxy_result = compute_training_free_proxies(
            train_merged, "_target", aug_features, task_type, fk_meta,
        )

        result = {
            "dataset": dataset_name,
            "task": task_name,
            "task_type": task_type,
            "join": join_key,
            "child_table": child_table_name,
            "parent_table": parent_table_name,
            "fk_col": fk_col,
            "n_train": len(train_merged),
            "n_val": len(val_merged),
            "n_base_features": len(base_features),
            "n_aug_features": len(aug_features),
            **jrn_result,
            **proxy_result,
        }
        logger.info(f"  Done {join_key}: JRN={jrn_result['jrn_mean']:.4f}")
        return result

    except Exception:
        logger.exception(f"  Failed on {join_key}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 7: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def run_analysis(all_results: List[Dict]) -> Dict[str, Any]:
    """Compute Spearman rho, ROC-AUC, cross-dataset consistency."""
    df = pd.DataFrame(all_results)
    proxy_names = ["entropy_reduction", "mutual_information", "pearson_correlation",
                   "log_mean_fanout", "homophily"]

    analysis: Dict[str, Any] = {}

    # 7a. Spearman rho per proxy vs JRN
    spearman_table = {}
    for proxy in proxy_names:
        rho_per_ds = {}
        for ds in df["dataset"].unique():
            subset = df[df["dataset"] == ds]
            if len(subset) >= 4:
                rho, pval = spearmanr(subset["jrn_mean"], subset[proxy])
                if np.isfinite(rho):
                    rho_per_ds[ds] = {"rho": round(float(rho), 4), "pval": round(float(pval), 6),
                                      "n": int(len(subset))}

        # Pooled
        if len(df) >= 4:
            rho_p, pval_p = spearmanr(df["jrn_mean"], df[proxy])
            pooled = {"rho": round(float(rho_p), 4), "pval": round(float(pval_p), 6),
                      "n": int(len(df))}
        else:
            pooled = {"rho": 0, "pval": 1, "n": int(len(df))}

        spearman_table[proxy] = {"per_dataset": rho_per_ds, "pooled": pooled}
    analysis["proxy_spearman_table"] = spearman_table

    # 7b. ROC-AUC for binary join selection (JRN > 1.01 = positive)
    selection_auc = {}
    jrn_positive = (df["jrn_mean"] > 1.01).astype(int)
    if jrn_positive.sum() > 0 and jrn_positive.sum() < len(jrn_positive):
        for proxy in proxy_names:
            try:
                auc = roc_auc_score(jrn_positive, df[proxy])
                selection_auc[proxy] = round(float(auc), 4)
            except ValueError:
                pass
    analysis["proxy_selection_auc_table"] = selection_auc

    # 7c. Cross-dataset consistency
    prior_rho = {"entropy_reduction": 0.945}  # from prior rel-f1 experiment
    consistency = {}
    for proxy in proxy_names:
        rhos = []
        for ds_data in spearman_table[proxy]["per_dataset"].values():
            if np.isfinite(ds_data["rho"]):
                rhos.append(ds_data["rho"])
        if proxy in prior_rho:
            rhos.append(prior_rho[proxy])
        if len(rhos) >= 2:
            mean_rho = np.mean(rhos)
            cv = float(np.std(rhos) / (abs(mean_rho) + 1e-10))
            consistency[proxy] = {
                "cv": round(cv, 4),
                "rhos": [round(r, 4) for r in rhos],
                "mean_rho": round(float(mean_rho), 4),
            }
    analysis["cross_dataset_consistency"] = consistency

    # 7d. Best proxy recommendation
    pooled_rhos = {p: spearman_table[p]["pooled"]["rho"] for p in proxy_names}
    best_proxy = max(pooled_rhos, key=lambda k: abs(pooled_rhos[k]))
    analysis["best_proxy_recommendation"] = (
        f"{best_proxy} (pooled Spearman rho={pooled_rhos[best_proxy]:.3f}) is the strongest "
        f"training-free proxy for predicting GBM-probe JRN across datasets."
    )

    # 7e. Schema property analysis
    analysis["schema_property_analysis"] = {
        "rel-stack": {"num_fk_joins": 11, "num_tables": 7, "max_fanout": 40851,
                      "min_fanout": 1.45, "task_types": ["clf", "clf", "reg"]},
        "rel-avito": {"num_fk_joins": 11, "num_tables": 8, "max_fanout": 114625,
                      "min_fanout": 1.18, "task_types": ["reg", "clf", "clf"]},
        "rel-f1":    {"num_fk_joins": 12, "num_tables": 9, "max_fanout": 300,
                      "min_fanout": 1.1, "task_types": ["clf", "clf", "reg"]},
    }

    # 7f. Conditions for reliable estimation
    conditions = {}
    for proxy in proxy_names:
        strong_ds = [ds for ds, v in spearman_table[proxy]["per_dataset"].items()
                     if abs(v["rho"]) > 0.5]
        weak_ds = [ds for ds, v in spearman_table[proxy]["per_dataset"].items()
                   if abs(v["rho"]) <= 0.5]
        conditions[proxy] = {
            "strong_datasets": strong_ds,
            "weak_datasets": weak_ds,
            "pooled_rho": pooled_rhos[proxy],
        }
    analysis["conditions_for_reliable_estimation"] = conditions

    # 7g. Key findings
    findings = []
    findings.append(f"Evaluated {len(all_results)} (join, task) pairs across "
                    f"{len(df['dataset'].unique())} datasets")
    findings.append(f"Best training-free proxy: {best_proxy} with pooled rho={pooled_rhos[best_proxy]:.3f}")
    jrn_range = (float(df["jrn_mean"].min()), float(df["jrn_mean"].max()))
    findings.append(f"GBM JRN range: [{jrn_range[0]:.4f}, {jrn_range[1]:.4f}]")
    n_beneficial = int((df["jrn_mean"] > 1.01).sum())
    findings.append(f"{n_beneficial}/{len(df)} joins are beneficial (JRN > 1.01)")
    if selection_auc:
        best_auc_proxy = max(selection_auc, key=selection_auc.get)
        findings.append(f"Best join-selection AUC: {best_auc_proxy}={selection_auc[best_auc_proxy]:.3f}")
    analysis["key_findings"] = findings

    return analysis


# ══════════════════════════════════════════════════════════════════════════
# STEP 8: BUILD OUTPUT JSON
# ══════════════════════════════════════════════════════════════════════════

def build_output(all_results: List[Dict], analysis: Dict) -> Dict:
    """Build method_out.json in exp_gen_sol_out schema format."""
    examples = []
    for r in all_results:
        # Input: the join-task pair specification
        input_data = {
            "dataset": r["dataset"],
            "task": r["task"],
            "task_type": r["task_type"],
            "join": r["join"],
            "child_table": r["child_table"],
            "parent_table": r["parent_table"],
            "fk_col": r["fk_col"],
        }
        # Output: the JRN and proxy values
        output_data = {
            "jrn_mean": r["jrn_mean"],
            "jrn_std": r["jrn_std"],
            "baseline_mean": r["baseline_mean"],
            "augmented_mean": r["augmented_mean"],
            "entropy_reduction": r["entropy_reduction"],
            "mutual_information": r["mutual_information"],
            "pearson_correlation": r["pearson_correlation"],
            "log_mean_fanout": r["log_mean_fanout"],
            "homophily": r["homophily"],
        }
        example = {
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_dataset": r["dataset"],
            "metadata_task": r["task"],
            "metadata_task_type": r["task_type"],
            "metadata_join": r["join"],
            "metadata_jrn_mean": r["jrn_mean"],
            "metadata_jrn_std": r["jrn_std"],
            "metadata_baseline_mean": r["baseline_mean"],
            "metadata_baseline_std": r["baseline_std"],
            "metadata_augmented_mean": r["augmented_mean"],
            "metadata_augmented_std": r["augmented_std"],
            "metadata_n_train": r["n_train"],
            "metadata_n_val": r["n_val"],
            "metadata_n_base_features": r["n_base_features"],
            "metadata_n_aug_features": r["n_aug_features"],
            "metadata_entropy_reduction": r["entropy_reduction"],
            "metadata_mutual_information": r["mutual_information"],
            "metadata_pearson_correlation": r["pearson_correlation"],
            "metadata_mean_fanout": r["mean_fanout"],
            "metadata_log_mean_fanout": r["log_mean_fanout"],
            "metadata_homophily": r["homophily"],
            "predict_baseline": json.dumps({"metric": r["baseline_mean"]}),
            "predict_our_method": json.dumps({"jrn": r["jrn_mean"],
                                               "best_proxy": r.get("entropy_reduction", 0)}),
        }
        examples.append(example)

    output = {
        "metadata": {
            "method_name": "Training-Free Proxy vs GBM-Probe JRN",
            "description": "Cross-dataset diagnostic reliability of 5 training-free proxies for estimating JRN",
            "datasets_used": ["rel-stack", "rel-avito"],
            "num_pairs": len(all_results),
            "proxy_names": ["entropy_reduction", "mutual_information", "pearson_correlation",
                           "log_mean_fanout", "homophily"],
            "analysis": analysis,
        },
        "datasets": [
            {
                "dataset": "jrn-proxy-diagnostic",
                "examples": examples,
            }
        ],
    }
    return output


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Starting Training-Free Proxy vs GBM-Probe JRN Experiment")
    logger.info("=" * 60)

    all_results: List[Dict] = []

    # ── Load rel-stack ──
    stack_tables, stack_fk_meta = load_stack_data()
    stack_task_dfs = construct_stack_task_labels(stack_tables)

    for task_name, task_info in STACK_TASKS.items():
        entity_table = task_info["entity_table"]
        pk_col = task_info["pk_col"]
        task_type = task_info["task_type"]
        entity_df = stack_tables[entity_table]
        train_df = stack_task_dfs[task_name]["train"]
        val_df = stack_task_dfs[task_name]["val"]

        joins = STACK_JOINS.get(entity_table, [])
        for (child_tbl, fk_col, parent_tbl, parent_pk) in joins:
            if child_tbl not in stack_tables:
                logger.warning(f"  Child table {child_tbl} not found, skipping")
                continue
            child_df = stack_tables[child_tbl]
            fk_key = f"{child_tbl}.{fk_col} -> {parent_tbl}.{parent_pk}"
            fk_meta = stack_fk_meta.get(fk_key, {"fanout_mean": 1.0, "mean": 1.0})

            result = run_pair(
                dataset_name="rel-stack", task_name=task_name, task_type=task_type,
                pk_col=pk_col, child_table_name=child_tbl, fk_col=fk_col,
                parent_table_name=parent_tbl, parent_pk_col=parent_pk,
                entity_df=entity_df, child_df=child_df,
                task_train=train_df, task_val=val_df,
                fk_meta=fk_meta,
            )
            if result:
                all_results.append(result)

    logger.info(f"Completed rel-stack: {len(all_results)} pairs")

    # Free stack memory
    del stack_tables, stack_task_dfs
    gc.collect()

    # ── Load rel-avito ──
    n_stack = len(all_results)
    avito_tables, avito_fk_meta, avito_task_dfs = load_avito_data()

    for task_name, task_info in AVITO_TASKS.items():
        entity_table = task_info["entity_table"]
        pk_col = task_info["pk_col"]
        task_type = task_info["task_type"]
        relbench_name = task_info["relbench_name"]

        if relbench_name not in avito_task_dfs:
            logger.warning(f"  Task {relbench_name} not found in avito data, skipping")
            continue

        entity_df = avito_tables.get(entity_table)
        if entity_df is None:
            logger.warning(f"  Entity table {entity_table} not found, skipping")
            continue

        task_folds = avito_task_dfs[relbench_name]
        train_df = task_folds.get("train")
        val_df = task_folds.get("val")
        if train_df is None or val_df is None:
            logger.warning(f"  Missing train/val for {relbench_name}, skipping")
            continue

        # Ensure pk_col is in train/val (from task sample input)
        if pk_col not in train_df.columns:
            logger.warning(f"  PK col {pk_col} not in task data for {relbench_name}")
            continue

        joins = AVITO_JOINS.get(entity_table, [])
        for (child_tbl, fk_col, parent_tbl, parent_pk) in joins:
            if child_tbl not in avito_tables:
                logger.warning(f"  Child table {child_tbl} not found, skipping")
                continue
            child_df = avito_tables[child_tbl]
            fk_key = f"{child_tbl}.{fk_col} -> {parent_tbl}.{parent_pk}"
            fk_meta = avito_fk_meta.get(fk_key, {"fanout_mean": 1.0})

            result = run_pair(
                dataset_name="rel-avito", task_name=task_name, task_type=task_type,
                pk_col=pk_col, child_table_name=child_tbl, fk_col=fk_col,
                parent_table_name=parent_tbl, parent_pk_col=parent_pk,
                entity_df=entity_df, child_df=child_df,
                task_train=train_df, task_val=val_df,
                fk_meta=fk_meta,
            )
            if result:
                all_results.append(result)

    logger.info(f"Completed rel-avito: {len(all_results) - n_stack} pairs")
    logger.info(f"Total pairs: {len(all_results)}")

    # Free avito memory
    del avito_tables, avito_task_dfs
    gc.collect()

    # ── Analysis ──
    if len(all_results) < 2:
        logger.error("Too few results for analysis!")
        # Still produce output
        analysis = {"key_findings": ["Too few valid join-task pairs for statistical analysis"]}
    else:
        analysis = run_analysis(all_results)
        logger.info(f"Analysis complete. Key findings:")
        for f in analysis.get("key_findings", []):
            logger.info(f"  - {f}")

    # ── Build and save output ──
    output = build_output(all_results, analysis)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
