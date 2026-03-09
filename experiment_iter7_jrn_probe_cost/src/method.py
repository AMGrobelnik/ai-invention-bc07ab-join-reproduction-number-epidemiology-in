#!/usr/bin/env python3
"""JRN Probe Cost-Efficiency Replication on rel-stack (11 FK Joins, 3 Tasks).

Replicates the JRN probe cost-efficiency analysis on RelBench rel-stack dataset
(Stack Exchange, 7 tables, 11 FK joins, 4.2M rows). Compares JRN-based join
selection vs greedy forward selection vs random baselines across 3 tasks.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, r2_score

warnings.filterwarnings("ignore")

# === LOGGING ===
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# === HARDWARE DETECTION ===
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

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

TOTAL_RAM_GB = _container_ram_gb() or 29.0
NUM_CPUS = _detect_cpus()
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.70 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# === WORKSPACE ===
WORKSPACE = Path(__file__).parent
os.chdir(WORKSPACE)

# === CONSTANTS ===
TASKS = {
    "user-engagement": {"type": "binary", "entity_table": "users", "metric": "average_precision"},
    "user-badge":      {"type": "binary", "entity_table": "users", "metric": "average_precision"},
    "post-votes":      {"type": "regression", "entity_table": "posts", "metric": "r2"},
}

FK_JOINS = [
    ("comments", "UserId", "users"),
    ("comments", "PostId", "posts"),
    ("badges", "UserId", "users"),
    ("postLinks", "PostId", "posts"),
    ("postLinks", "RelatedPostId", "posts"),
    ("postHistory", "PostId", "posts"),
    ("postHistory", "UserId", "users"),
    ("votes", "PostId", "posts"),
    ("votes", "UserId", "users"),
    ("posts", "OwnerUserId", "users"),
    ("posts", "ParentId", "posts"),
]

SEEDS = [42, 123, 456]

# Reduced budget grid (Fallback 1: 12 configs instead of 48) for time efficiency
BUDGET_GRID = {
    "n_estimators": [50, 200],
    "max_depth": [3, 6],
    "subsample": [0.1, 0.5, 1.0],
}  # 2 × 2 × 3 = 12 configs

# Cap training data for speed (Fallback 3)
MAX_TRAIN_ROWS = 50000
MAX_VAL_ROWS = 30000

# Full budget params for greedy/JRN selection models
FULL_BUDGET_PARAMS = {"n_estimators": 200, "max_depth": 6}

# Random baseline orderings (reduced from 20 to 10 for time)
N_RANDOM_ORDERINGS = 10

GLOBAL_START = time.time()


def elapsed_str() -> str:
    """Return elapsed time string."""
    return f"{time.time() - GLOBAL_START:.0f}s"


def log_memory():
    """Log current memory usage from cgroup."""
    try:
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        logger.info(f"Memory: {current / 1e9:.2f} GB / {TOTAL_RAM_GB:.1f} GB")
    except (FileNotFoundError, ValueError):
        pass


def make_json_safe(val):
    """Convert value to JSON-serializable type."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return [make_json_safe(x) for x in val.tolist()]
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


# ============================================================
# PHASE 0: DATA LOADING
# ============================================================
def load_dataset():
    """Load rel-stack dataset and return db object."""
    from relbench.datasets import get_dataset
    logger.info("Loading rel-stack dataset...")
    t0 = time.time()
    dataset = get_dataset("rel-stack", download=True)
    db = dataset.get_db()
    logger.info(f"Dataset loaded in {time.time() - t0:.1f}s")
    for tname, table in db.table_dict.items():
        logger.info(f"  {tname}: {len(table.df)} rows, {len(table.df.columns)} cols")
    log_memory()
    return db


def load_task(task_name: str):
    """Load a relbench task."""
    from relbench.tasks import get_task
    return get_task("rel-stack", task_name, download=True)


# ============================================================
# PHASE 1: FEATURE ENGINEERING
# ============================================================
def build_features_for_entity_table(
    entity_table_name: str,
    db,
    task,
    split: str = "train",
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series]:
    """Build base + per-join features for a task split.

    Returns: (base_df, {join_key: join_feat_df}, labels_series)
    """
    task_table = task.get_table(split)
    entity_col = task.entity_col
    target_col = task.target_col

    task_df = task_table.df
    if max_rows is not None and len(task_df) > max_rows:
        task_df = task_df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    entity_ids = task_df[entity_col]
    labels = task_df[target_col]

    # Base features from entity table
    entity_df = db.table_dict[entity_table_name].df
    pkey_col = db.table_dict[entity_table_name].pkey_col

    base_features = entity_ids.to_frame().merge(
        entity_df, left_on=entity_col, right_on=pkey_col, how="left"
    )
    # Select numeric columns only
    base_numeric = base_features.select_dtypes(include=[np.number]).fillna(0)

    # Drop the entity_col / pkey_col if they ended up in base_numeric to avoid leakage
    drop_cols = [c for c in [entity_col, pkey_col] if c in base_numeric.columns]
    if drop_cols:
        base_numeric = base_numeric.drop(columns=drop_cols, errors="ignore")

    # If base has no numeric features, add a constant column
    if base_numeric.shape[1] == 0:
        base_numeric = pd.DataFrame({"_const": np.ones(len(entity_ids))})

    # Per-join aggregated features
    join_features = {}
    for child_table, fk_col, parent_table in FK_JOINS:
        if parent_table != entity_table_name:
            continue
        if child_table not in db.table_dict:
            continue
        child_df = db.table_dict[child_table].df
        if fk_col not in child_df.columns:
            continue

        child_numeric_cols = child_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove FK column itself from aggregation
        child_numeric_cols = [c for c in child_numeric_cols if c != fk_col]

        if child_numeric_cols:
            agg_dict = {col: ["mean", "count", "std"] for col in child_numeric_cols}
        else:
            # If no numeric columns, just count
            agg_dict = {fk_col: ["count"]}

        try:
            agg_df = child_df.groupby(fk_col).agg(agg_dict)
            agg_df.columns = [f"{child_table}_{c}_{stat}" for c, stat in agg_df.columns]
            agg_df = agg_df.reset_index()

            merged = entity_ids.to_frame().merge(
                agg_df, left_on=entity_col, right_on=fk_col, how="left"
            )
            # Drop join columns to keep only features
            drop = [c for c in [entity_col, fk_col] if c in merged.columns]
            merged = merged.drop(columns=drop, errors="ignore").fillna(0)

            if merged.shape[1] > 0:
                join_key = f"{child_table}.{fk_col}"
                join_features[join_key] = merged
        except Exception as e:
            logger.warning(f"  Skipping join {child_table}.{fk_col}: {e}")
            continue

    return base_numeric, join_features, labels


# ============================================================
# PHASE 2: TRAIN + EVALUATE HELPERS
# ============================================================
def train_and_eval_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: str,
    lgb_params: dict,
) -> float:
    """Train LightGBM and return evaluation score."""
    import lightgbm as lgb

    # Ensure clean column names (no duplicates)
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    X_val.columns = [f"f{i}" for i in range(X_val.shape[1])]

    if task_type == "binary":
        params = {**lgb_params, "objective": "binary"}
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds)
    else:
        params = {**lgb_params, "objective": "regression"}
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)

    del model
    return score


def build_concat_features(
    base: pd.DataFrame,
    join_feats: dict[str, pd.DataFrame],
    selected_joins: list[str],
    idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Concatenate base + selected join features, optionally subsampled by idx."""
    parts = [base.iloc[idx].reset_index(drop=True) if idx is not None else base.reset_index(drop=True)]
    for jk in selected_joins:
        jf = join_feats[jk]
        parts.append(jf.iloc[idx].reset_index(drop=True) if idx is not None else jf.reset_index(drop=True))
    return pd.concat(parts, axis=1)


# ============================================================
# PHASE 2: JRN COMPUTATION
# ============================================================
def compute_jrn_for_task(
    task_name: str,
    task_info: dict,
    budget_config: dict,
    seed: int,
    db,
) -> tuple[dict, float, float]:
    """Compute JRN values for all reachable joins for one task/config/seed.

    Returns: (jrn_dict, score_base, elapsed_seconds)
    """
    t0 = time.time()
    task = load_task(task_name)
    entity_table = task_info["entity_table"]

    # Subsample training data per budget config
    train_max = int(MAX_TRAIN_ROWS * budget_config["subsample"])
    train_max = max(train_max, 1000)  # at least 1000 rows

    base_train, join_feats_train, y_train = build_features_for_entity_table(
        entity_table, db, task, "train", max_rows=train_max
    )
    base_val, join_feats_val, y_val = build_features_for_entity_table(
        entity_table, db, task, "val", max_rows=MAX_VAL_ROWS
    )

    lgb_params = {
        "n_estimators": budget_config["n_estimators"],
        "max_depth": budget_config["max_depth"],
        "learning_rate": 0.1,
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": NUM_CPUS,
    }

    # Baseline: entity features only
    score_base = train_and_eval_lgbm(
        base_train, y_train, base_val, y_val, task_info["type"], lgb_params
    )

    # Per-join: base + join features
    jrn_dict = {}
    for join_key, jf_train in join_feats_train.items():
        try:
            X_train_j = pd.concat(
                [base_train.reset_index(drop=True), jf_train.reset_index(drop=True)],
                axis=1,
            )
            jf_val = join_feats_val.get(join_key)
            if jf_val is None:
                continue
            X_val_j = pd.concat(
                [base_val.reset_index(drop=True), jf_val.reset_index(drop=True)],
                axis=1,
            )

            score_j = train_and_eval_lgbm(
                X_train_j, y_train, X_val_j, y_val, task_info["type"], lgb_params
            )

            # JRN = ratio (handle edge cases)
            if abs(score_base) > 0.01:
                jrn_dict[join_key] = score_j / score_base
            else:
                jrn_dict[join_key] = 1.0 + (score_j - score_base)
        except Exception as e:
            logger.warning(f"  JRN failed for {join_key}: {e}")
            jrn_dict[join_key] = 1.0

    elapsed = time.time() - t0
    return jrn_dict, score_base, elapsed


# ============================================================
# PHASE 3: CONVERGENCE ANALYSIS
# ============================================================
def run_convergence_analysis(db) -> tuple[dict, dict, dict]:
    """Run convergence analysis: compare cheap probes to expensive reference.

    Returns: (all_convergence_results, convergence_summary, reference_jrn)
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: CONVERGENCE ANALYSIS")
    logger.info("=" * 60)

    all_convergence_results = {}
    reference_jrn = {}

    configs = list(itertools.product(
        BUDGET_GRID["n_estimators"],
        BUDGET_GRID["max_depth"],
        BUDGET_GRID["subsample"],
    ))
    logger.info(f"Budget grid: {len(configs)} configs × {len(SEEDS)} seeds × {len(TASKS)} tasks")

    for task_name, task_info in TASKS.items():
        t_task = time.time()
        logger.info(f"\n--- Convergence for task: {task_name} [{elapsed_str()}] ---")

        # Reference: highest budget, averaged over seeds
        ref_config = {"n_estimators": 200, "max_depth": 6, "subsample": 1.0}
        ref_jrns_per_seed = []
        ref_base_scores = []
        for seed in SEEDS:
            jrn, base_score, el = compute_jrn_for_task(
                task_name, task_info, ref_config, seed, db
            )
            ref_jrns_per_seed.append(jrn)
            ref_base_scores.append(base_score)
            logger.debug(f"  Ref seed={seed}: base={base_score:.4f}, {len(jrn)} joins, {el:.1f}s")

        # Average JRN across seeds
        if not ref_jrns_per_seed or not ref_jrns_per_seed[0]:
            logger.warning(f"  No reachable joins for {task_name}, skipping")
            continue

        ref_jrn_avg = {}
        for key in ref_jrns_per_seed[0]:
            vals = [d.get(key, 1.0) for d in ref_jrns_per_seed]
            ref_jrn_avg[key] = float(np.mean(vals))
        reference_jrn[task_name] = ref_jrn_avg

        logger.info(f"  Reference JRN computed: {len(ref_jrn_avg)} joins, "
                     f"base_score={np.mean(ref_base_scores):.4f}")
        for jk, jv in sorted(ref_jrn_avg.items(), key=lambda x: -x[1]):
            logger.info(f"    {jk}: JRN={jv:.4f}")

        # All configs
        task_convergence = []
        for ne, md, ss in configs:
            config = {"n_estimators": ne, "max_depth": md, "subsample": ss}
            config_jrns_per_seed = []
            config_elapsed = 0.0

            for seed in SEEDS:
                jrn, _, el = compute_jrn_for_task(
                    task_name, task_info, config, seed, db
                )
                config_jrns_per_seed.append(jrn)
                config_elapsed += el

            # Average JRN across seeds
            avg_jrn = {}
            for key in config_jrns_per_seed[0]:
                vals = [d.get(key, 1.0) for d in config_jrns_per_seed]
                avg_jrn[key] = float(np.mean(vals))

            # Compare to reference
            joins = sorted(ref_jrn_avg.keys())
            if len(joins) < 3:
                # Not enough joins for meaningful comparison
                task_convergence.append({
                    "config": config,
                    "spearman_rho": None,
                    "kendall_tau": None,
                    "top3_agreement": None,
                    "jrn_values": {k: make_json_safe(v) for k, v in avg_jrn.items()},
                    "wall_clock_seconds": make_json_safe(config_elapsed),
                })
                continue

            ref_vals = [ref_jrn_avg[j] for j in joins]
            cfg_vals = [avg_jrn.get(j, 1.0) for j in joins]

            rho_val, _ = spearmanr(ref_vals, cfg_vals)
            tau_val, _ = kendalltau(ref_vals, cfg_vals)

            # Top-3 agreement
            k_top = min(3, len(joins))
            ref_topk = set(sorted(joins, key=lambda j: ref_jrn_avg[j], reverse=True)[:k_top])
            cfg_topk = set(sorted(joins, key=lambda j: avg_jrn.get(j, 1.0), reverse=True)[:k_top])
            topk_agreement = len(ref_topk & cfg_topk) / k_top

            task_convergence.append({
                "config": config,
                "spearman_rho": make_json_safe(rho_val),
                "kendall_tau": make_json_safe(tau_val),
                "top3_agreement": make_json_safe(topk_agreement),
                "jrn_values": {k: make_json_safe(v) for k, v in avg_jrn.items()},
                "wall_clock_seconds": make_json_safe(config_elapsed),
            })

        all_convergence_results[task_name] = task_convergence

        # Summary stats
        rhos = [r["spearman_rho"] for r in task_convergence if r["spearman_rho"] is not None]
        if rhos:
            logger.info(f"  Task {task_name} convergence: median_rho={np.median(rhos):.3f}, "
                         f"frac>0.9={np.mean([r > 0.9 for r in rhos]):.2f}, "
                         f"elapsed={time.time() - t_task:.0f}s [{elapsed_str()}]")

        gc.collect()

    # Build convergence summary
    convergence_summary = {}
    for task_name, results in all_convergence_results.items():
        rhos = [r["spearman_rho"] for r in results if r["spearman_rho"] is not None]
        if rhos:
            convergence_summary[task_name] = {
                "frac_rho_gt_0.9": make_json_safe(float(np.mean([r > 0.9 for r in rhos]))),
                "frac_rho_gt_0.95": make_json_safe(float(np.mean([r > 0.95 for r in rhos]))),
                "median_rho": make_json_safe(float(np.median(rhos))),
                "min_rho": make_json_safe(float(np.min(rhos))),
                "max_rho": make_json_safe(float(np.max(rhos))),
            }

    return all_convergence_results, convergence_summary, reference_jrn


# ============================================================
# PHASE 4: GREEDY FORWARD SELECTION
# ============================================================
def run_greedy_selection(db, reference_jrn: dict) -> dict:
    """Run greedy forward join selection for each task."""
    logger.info("=" * 60)
    logger.info("PHASE 4: GREEDY FORWARD SELECTION")
    logger.info("=" * 60)

    greedy_results = {}

    for task_name, task_info in TASKS.items():
        if task_name not in reference_jrn:
            continue
        t_task = time.time()
        logger.info(f"\n--- Greedy selection for: {task_name} [{elapsed_str()}] ---")

        task = load_task(task_name)
        entity_table = task_info["entity_table"]

        base_train, join_feats_train, y_train = build_features_for_entity_table(
            entity_table, db, task, "train", max_rows=MAX_TRAIN_ROWS
        )
        base_val, join_feats_val, y_val = build_features_for_entity_table(
            entity_table, db, task, "val", max_rows=MAX_VAL_ROWS
        )

        lgb_params = {
            **FULL_BUDGET_PARAMS,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": NUM_CPUS,
        }

        available_joins = list(join_feats_train.keys())
        selected_joins = []
        greedy_perf_curve = []
        models_trained = 0

        while available_joins:
            best_join = None
            best_score = -np.inf

            for candidate in available_joins:
                try:
                    X_train = build_concat_features(
                        base_train, join_feats_train, selected_joins + [candidate]
                    )
                    X_val = build_concat_features(
                        base_val, join_feats_val, selected_joins + [candidate]
                    )
                    score = train_and_eval_lgbm(
                        X_train, y_train, X_val, y_val, task_info["type"], lgb_params
                    )
                    models_trained += 1
                    if score > best_score:
                        best_score = score
                        best_join = candidate
                except Exception as e:
                    logger.warning(f"  Greedy failed for {candidate}: {e}")
                    models_trained += 1

            if best_join is None:
                break

            selected_joins.append(best_join)
            available_joins.remove(best_join)
            greedy_perf_curve.append({
                "k": len(selected_joins),
                "score": make_json_safe(best_score),
                "join_added": best_join,
            })
            logger.info(f"  k={len(selected_joins)}: +{best_join} → score={best_score:.4f}")

        greedy_results[task_name] = {
            "performance_curve": greedy_perf_curve,
            "total_models_trained": models_trained,
            "best_score": make_json_safe(max(r["score"] for r in greedy_perf_curve) if greedy_perf_curve else 0),
            "join_order": [r["join_added"] for r in greedy_perf_curve],
            "elapsed_seconds": make_json_safe(time.time() - t_task),
        }
        logger.info(f"  Greedy done: {models_trained} models in {time.time() - t_task:.0f}s [{elapsed_str()}]")
        gc.collect()

    return greedy_results


# ============================================================
# PHASE 5: JRN-BASED JOIN SELECTION
# ============================================================
def run_jrn_selection(db, reference_jrn: dict) -> dict:
    """Use JRN ranking to select joins incrementally."""
    logger.info("=" * 60)
    logger.info("PHASE 5: JRN-BASED SELECTION")
    logger.info("=" * 60)

    jrn_results = {}

    for task_name, task_info in TASKS.items():
        if task_name not in reference_jrn:
            continue
        t_task = time.time()
        logger.info(f"\n--- JRN selection for: {task_name} [{elapsed_str()}] ---")

        ref_jrn = reference_jrn[task_name]
        ranked_joins = sorted(ref_jrn.keys(), key=lambda j: ref_jrn[j], reverse=True)

        task = load_task(task_name)
        entity_table = task_info["entity_table"]

        base_train, join_feats_train, y_train = build_features_for_entity_table(
            entity_table, db, task, "train", max_rows=MAX_TRAIN_ROWS
        )
        base_val, join_feats_val, y_val = build_features_for_entity_table(
            entity_table, db, task, "val", max_rows=MAX_VAL_ROWS
        )

        lgb_params = {
            **FULL_BUDGET_PARAMS,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": NUM_CPUS,
        }

        jrn_perf_curve = []
        models_trained = 0

        for k in range(1, len(ranked_joins) + 1):
            selected = ranked_joins[:k]
            try:
                X_train = build_concat_features(base_train, join_feats_train, selected)
                X_val = build_concat_features(base_val, join_feats_val, selected)
                score = train_and_eval_lgbm(
                    X_train, y_train, X_val, y_val, task_info["type"], lgb_params
                )
                models_trained += 1
                jrn_perf_curve.append({
                    "k": k,
                    "score": make_json_safe(score),
                    "joins": list(selected),
                })
                logger.info(f"  k={k}: {selected[-1]} → score={score:.4f}")
            except Exception as e:
                logger.warning(f"  JRN selection failed at k={k}: {e}")
                models_trained += 1

        jrn_results[task_name] = {
            "performance_curve": jrn_perf_curve,
            "total_models_trained": models_trained,
            "best_score": make_json_safe(max(r["score"] for r in jrn_perf_curve) if jrn_perf_curve else 0),
            "join_ranking": ranked_joins,
            "elapsed_seconds": make_json_safe(time.time() - t_task),
        }
        logger.info(f"  JRN done: {models_trained} models in {time.time() - t_task:.0f}s [{elapsed_str()}]")
        gc.collect()

    return jrn_results


# ============================================================
# PHASE 6: RANDOM BASELINE
# ============================================================
def run_random_baseline(db, reference_jrn: dict) -> dict:
    """Run random join orderings as baseline."""
    logger.info("=" * 60)
    logger.info("PHASE 6: RANDOM BASELINE")
    logger.info("=" * 60)

    random_results = {}

    for task_name, task_info in TASKS.items():
        if task_name not in reference_jrn:
            continue
        t_task = time.time()
        logger.info(f"\n--- Random baseline for: {task_name} [{elapsed_str()}] ---")

        task = load_task(task_name)
        entity_table = task_info["entity_table"]

        base_train, join_feats_train, y_train = build_features_for_entity_table(
            entity_table, db, task, "train", max_rows=MAX_TRAIN_ROWS
        )
        base_val, join_feats_val, y_val = build_features_for_entity_table(
            entity_table, db, task, "val", max_rows=MAX_VAL_ROWS
        )

        lgb_params = {
            **FULL_BUDGET_PARAMS,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": NUM_CPUS,
        }

        all_joins = list(join_feats_train.keys())
        all_random_curves = []
        total_models = 0

        for r in range(N_RANDOM_ORDERINGS):
            rng = np.random.RandomState(r)
            random_order = list(rng.permutation(all_joins))
            curve = []

            for k in range(1, len(random_order) + 1):
                selected = random_order[:k]
                try:
                    X_train = build_concat_features(base_train, join_feats_train, selected)
                    X_val = build_concat_features(base_val, join_feats_val, selected)
                    score = train_and_eval_lgbm(
                        X_train, y_train, X_val, y_val, task_info["type"], lgb_params
                    )
                    curve.append(score)
                    total_models += 1
                except Exception as e:
                    logger.warning(f"  Random r={r} k={k} failed: {e}")
                    curve.append(curve[-1] if curve else 0.0)
                    total_models += 1

            all_random_curves.append(curve)
            logger.debug(f"  Random ordering {r}: final_score={curve[-1]:.4f}")

        # Compute mean/std across orderings
        if all_random_curves:
            arr = np.array(all_random_curves)
            random_results[task_name] = {
                "mean_curve": [make_json_safe(v) for v in np.mean(arr, axis=0).tolist()],
                "std_curve": [make_json_safe(v) for v in np.std(arr, axis=0).tolist()],
                "n_orderings": N_RANDOM_ORDERINGS,
                "total_models_trained": total_models,
                "elapsed_seconds": make_json_safe(time.time() - t_task),
            }

        logger.info(f"  Random done: {total_models} models in {time.time() - t_task:.0f}s [{elapsed_str()}]")
        gc.collect()

    return random_results


# ============================================================
# PHASE 7: COST-EFFICIENCY METRICS
# ============================================================
def compute_cost_efficiency(
    reference_jrn: dict,
    greedy_results: dict,
    jrn_results: dict,
) -> tuple[dict, dict]:
    """Compute cost-efficiency metrics and cross-dataset comparison."""
    logger.info("=" * 60)
    logger.info("PHASE 7: COST-EFFICIENCY METRICS")
    logger.info("=" * 60)

    cost_efficiency = {}

    for task_name in TASKS:
        if task_name not in reference_jrn or task_name not in greedy_results or task_name not in jrn_results:
            continue

        J = len(reference_jrn[task_name])
        greedy_models = greedy_results[task_name]["total_models_trained"]
        jrn_sel_models = jrn_results[task_name]["total_models_trained"]
        jrn_probe_models = len(SEEDS) * (1 + J)  # probes: seeds × (base + J joins)
        jrn_total = jrn_probe_models + jrn_sel_models

        greedy_best = greedy_results[task_name]["best_score"] or 0
        jrn_best = jrn_results[task_name]["best_score"] or 0

        cost_ratio = jrn_total / max(greedy_models, 1)
        perf_ratio = (jrn_best / greedy_best * 100) if greedy_best and abs(greedy_best) > 1e-6 else 0
        speedup = greedy_models / max(jrn_total, 1)

        # Breakeven: J*(J+1)/2 > seeds*(1+J) + J
        # J^2/2 + J/2 > seeds + seeds*J + J
        # J^2/2 > seeds + (seeds+0.5)*J → approx J > 2*seeds+1 ≈ 7-8
        breakeven_J = None
        for test_j in range(1, 50):
            greedy_cost = test_j * (test_j + 1) / 2
            jrn_cost = len(SEEDS) * (1 + test_j) + test_j
            if greedy_cost > jrn_cost:
                breakeven_J = test_j
                break

        cost_efficiency[task_name] = {
            "num_reachable_joins": J,
            "greedy_models": greedy_models,
            "greedy_expected_models": J * (J + 1) // 2,
            "jrn_probe_models": jrn_probe_models,
            "jrn_selection_models": jrn_sel_models,
            "jrn_total_models": jrn_total,
            "cost_ratio": make_json_safe(cost_ratio),
            "performance_ratio_pct": make_json_safe(perf_ratio),
            "speedup_factor": make_json_safe(speedup),
            "breakeven_J": breakeven_J,
        }

        logger.info(f"  {task_name}: greedy={greedy_models} models, jrn_total={jrn_total} models, "
                     f"speedup={speedup:.1f}x, perf_ratio={perf_ratio:.1f}%")

    # Cross-dataset comparison with rel-f1 reference
    cross_dataset = {
        "rel_f1_reference": {
            "cost_ratio_range": "0.05-0.11",
            "performance_ratio_range": "96-99%",
            "speedup_range": "10-20x",
        },
        "rel_stack_measured": {
            "cost_ratios": {t: cost_efficiency[t]["cost_ratio"] for t in cost_efficiency},
            "performance_ratios": {t: cost_efficiency[t]["performance_ratio_pct"] for t in cost_efficiency},
            "speedup_factors": {t: cost_efficiency[t]["speedup_factor"] for t in cost_efficiency},
        },
    }

    # Determine if pattern replicates
    speedups = [cost_efficiency[t]["speedup_factor"] for t in cost_efficiency if cost_efficiency[t]["speedup_factor"]]
    perf_ratios = [cost_efficiency[t]["performance_ratio_pct"] for t in cost_efficiency if cost_efficiency[t]["performance_ratio_pct"]]

    if speedups and perf_ratios:
        avg_speedup = np.mean(speedups)
        avg_perf_ratio = np.mean(perf_ratios)
        pattern_replicates = avg_speedup > 2.0 and avg_perf_ratio > 80.0
        cross_dataset["pattern_replicates"] = pattern_replicates
        cross_dataset["avg_speedup"] = make_json_safe(avg_speedup)
        cross_dataset["avg_performance_ratio_pct"] = make_json_safe(avg_perf_ratio)
    else:
        cross_dataset["pattern_replicates"] = "insufficient_data"

    return cost_efficiency, cross_dataset


# ============================================================
# PHASE 8: BUILD OUTPUT IN exp_gen_sol_out.json SCHEMA
# ============================================================
def build_output(
    all_convergence_results: dict,
    convergence_summary: dict,
    reference_jrn: dict,
    greedy_results: dict,
    jrn_results: dict,
    random_results: dict,
    cost_efficiency: dict,
    cross_dataset: dict,
    total_elapsed: float,
) -> dict:
    """Build output conforming to exp_gen_sol_out.json schema."""

    # Build examples: one per task with full results
    examples = []
    for task_name, task_info in TASKS.items():
        # Input: task description
        input_str = json.dumps({
            "task_name": task_name,
            "task_type": task_info["type"],
            "entity_table": task_info["entity_table"],
            "metric": task_info["metric"],
            "dataset": "rel-stack",
            "num_fk_joins": 11,
        })

        # Output: all results for this task
        task_results = {
            "convergence": convergence_summary.get(task_name, {}),
            "reference_jrn": {k: make_json_safe(v) for k, v in reference_jrn.get(task_name, {}).items()},
            "greedy": greedy_results.get(task_name, {}),
            "jrn_selection": jrn_results.get(task_name, {}),
            "random_baseline": random_results.get(task_name, {}),
            "cost_efficiency": cost_efficiency.get(task_name, {}),
        }
        output_str = json.dumps(task_results, default=str)

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_task_name": task_name,
            "metadata_task_type": task_info["type"],
            "metadata_entity_table": task_info["entity_table"],
            "predict_jrn_ranking": json.dumps(
                jrn_results.get(task_name, {}).get("join_ranking", [])
            ),
            "predict_greedy_ranking": json.dumps(
                greedy_results.get(task_name, {}).get("join_order", [])
            ),
        }
        examples.append(example)

    # Add convergence detail examples (one per task × config)
    for task_name, conv_results in all_convergence_results.items():
        for cr in conv_results:
            conv_input = json.dumps({
                "task_name": task_name,
                "probe_config": cr["config"],
                "analysis_type": "convergence_probe",
            })
            conv_output = json.dumps({
                "spearman_rho": cr["spearman_rho"],
                "kendall_tau": cr["kendall_tau"],
                "top3_agreement": cr["top3_agreement"],
                "jrn_values": cr["jrn_values"],
                "wall_clock_seconds": cr["wall_clock_seconds"],
            })
            examples.append({
                "input": conv_input,
                "output": conv_output,
                "metadata_task_name": task_name,
                "metadata_analysis_type": "convergence_probe",
            })

    output = {
        "metadata": {
            "experiment": "jrn_probe_cost_efficiency_rel_stack",
            "dataset": "rel-stack",
            "tasks": list(TASKS.keys()),
            "num_fk_joins": 11,
            "budget_grid": BUDGET_GRID,
            "num_configs": len(list(itertools.product(
                BUDGET_GRID["n_estimators"],
                BUDGET_GRID["max_depth"],
                BUDGET_GRID["subsample"],
            ))),
            "num_seeds": len(SEEDS),
            "max_train_rows": MAX_TRAIN_ROWS,
            "max_val_rows": MAX_VAL_ROWS,
            "total_elapsed_seconds": make_json_safe(total_elapsed),
            "convergence_summary": convergence_summary,
            "cost_efficiency_summary": cost_efficiency,
            "cross_dataset_comparison": cross_dataset,
            "reference_jrn_values": {
                t: {k: make_json_safe(v) for k, v in jrns.items()}
                for t, jrns in reference_jrn.items()
            },
        },
        "datasets": [
            {
                "dataset": "rel-stack",
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
    logger.info("=" * 60)
    logger.info("JRN Probe Cost-Efficiency Replication on rel-stack")
    logger.info("=" * 60)
    log_memory()

    # Phase 0: Load dataset
    db = load_dataset()

    # Phase 3: Convergence analysis (includes computing reference JRN)
    all_convergence_results, convergence_summary, reference_jrn = run_convergence_analysis(db)
    log_memory()
    logger.info(f"Convergence analysis complete [{elapsed_str()}]")

    # Check time budget - if over 40min, skip random baseline
    elapsed_so_far = time.time() - GLOBAL_START
    logger.info(f"Time elapsed: {elapsed_so_far:.0f}s ({elapsed_so_far / 60:.1f}min)")

    # Phase 4: Greedy forward selection
    greedy_results = run_greedy_selection(db, reference_jrn)
    log_memory()
    logger.info(f"Greedy selection complete [{elapsed_str()}]")

    # Phase 5: JRN-based selection
    jrn_results = run_jrn_selection(db, reference_jrn)
    log_memory()
    logger.info(f"JRN selection complete [{elapsed_str()}]")

    # Phase 6: Random baseline (skip if time tight)
    elapsed_so_far = time.time() - GLOBAL_START
    if elapsed_so_far < 2700:  # < 45 minutes
        random_results = run_random_baseline(db, reference_jrn)
        log_memory()
        logger.info(f"Random baseline complete [{elapsed_str()}]")
    else:
        logger.warning(f"Skipping random baseline due to time ({elapsed_so_far / 60:.1f}min elapsed)")
        random_results = {}

    # Phase 7: Cost-efficiency
    cost_efficiency, cross_dataset = compute_cost_efficiency(
        reference_jrn, greedy_results, jrn_results
    )

    total_elapsed = time.time() - GLOBAL_START

    # Phase 8: Build and save output
    output = build_output(
        all_convergence_results,
        convergence_summary,
        reference_jrn,
        greedy_results,
        jrn_results,
        random_results,
        cost_efficiency,
        cross_dataset,
        total_elapsed,
    )

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")
    logger.info("DONE!")


if __name__ == "__main__":
    main()
