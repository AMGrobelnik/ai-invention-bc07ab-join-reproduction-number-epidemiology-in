#!/usr/bin/env python3
"""JRN Estimation & Multiplicative Compounding on rel-stack (Phase 1 + Phase 3).

Estimates Join Reproduction Number (JRN) for all 11 FK joins × 3 entity tasks
in rel-stack using lightweight MLP probes, then validates multiplicative
compounding by comparing measured chain JRN for multi-hop join chains against
the product of individual JRNs.

Baseline: Fan-out heuristic that predicts JRN from join statistics alone.
Our method: Probe-based JRN estimation using M_join/M_base ratio.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

DEVICE = torch.device("cpu")
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# === PATHS ===
WORKSPACE = Path(__file__).parent
DEP_DATA_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/run__20260309_024817/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus"
)
DEP_RESEARCH_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/run__20260309_024817/"
    "3_invention_loop/iter_1/gen_art/research_id2_it1__opus"
)

# === EXPERIMENT CONFIG ===
HIDDEN_DIM = 32
N_LAYERS = 2
EPOCHS = 10
SEEDS = [0, 1, 2]
BATCH_SIZE = 512
LR = 0.001
MAX_TRAIN = 50000
MAX_VAL = 10000

# 11 FK Joins in rel-stack
FK_JOINS = [
    ("comments", "UserId", "users", "Id"),
    ("comments", "PostId", "posts", "Id"),
    ("badges", "UserId", "users", "Id"),
    ("postLinks", "PostId", "posts", "Id"),
    ("postLinks", "RelatedPostId", "posts", "Id"),
    ("postHistory", "PostId", "posts", "Id"),
    ("postHistory", "UserId", "users", "Id"),
    ("votes", "PostId", "posts", "Id"),
    ("votes", "UserId", "users", "Id"),
    ("posts", "OwnerUserId", "users", "Id"),
    ("posts", "ParentId", "posts", "Id"),
]

# Entity tasks (skip link prediction — see fallback plan)
ENTITY_TASKS = [
    ("user-engagement", "users", "contribution", "classification"),
    ("user-badge", "users", "WillGetBadge", "classification"),
    ("post-votes", "posts", "popularity", "regression"),
]

# Multi-hop chains (target = parent of first hop)
CHAINS = [
    # 2-hop chains (target: users)
    [("posts", "OwnerUserId", "users"), ("comments", "PostId", "posts")],
    [("posts", "OwnerUserId", "users"), ("votes", "PostId", "posts")],
    [("posts", "OwnerUserId", "users"), ("postLinks", "PostId", "posts")],
    [("posts", "OwnerUserId", "users"), ("postHistory", "PostId", "posts")],
    [("posts", "OwnerUserId", "users"), ("posts", "ParentId", "posts")],
    # 3-hop chains (target: users)
    [("posts", "OwnerUserId", "users"), ("posts", "ParentId", "posts"),
     ("comments", "PostId", "posts")],
    [("posts", "OwnerUserId", "users"), ("posts", "ParentId", "posts"),
     ("votes", "PostId", "posts")],
]


# =========================================================================
# FEATURE EXTRACTION
# =========================================================================

def extract_numeric_features(
    df: pd.DataFrame, exclude_cols: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Extract numeric features from a DataFrame."""
    exclude = set(exclude_cols or [])
    features: list[np.ndarray] = []
    col_names: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        dtype = str(df[col].dtype)
        if dtype in ("int64", "Int64", "float64"):
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
            features.append(vals.astype(np.float32))
            col_names.append(col)
        elif dtype == "bool":
            features.append(df[col].astype(np.float32).fillna(0).values)
            col_names.append(col)
        elif dtype.startswith("datetime"):
            ts = pd.to_datetime(df[col], errors="coerce")
            numeric = ts.astype(np.int64).values / 1e9
            numeric = np.where(np.isfinite(numeric), numeric, 0).astype(np.float32)
            features.append(numeric)
            col_names.append(col + "_ts")
        # Skip object/text columns
    if not features:
        return np.zeros((len(df), 1), dtype=np.float32), ["dummy"]
    return np.column_stack(features), col_names


def aggregate_child_to_parent(
    child_df: pd.DataFrame,
    child_fk_col: str,
    parent_pks: np.ndarray,
    exclude_child_cols: list[str] | None = None,
) -> np.ndarray:
    """Mean-aggregate child features per FK value, aligned to parent_pks."""
    exc = list(exclude_child_cols or [])
    if child_fk_col not in exc:
        exc.append(child_fk_col)
    child_feats, _ = extract_numeric_features(child_df, exclude_cols=exc)

    fk_values = child_df[child_fk_col].values
    feat_df = pd.DataFrame(child_feats, columns=[f"f{i}" for i in range(child_feats.shape[1])])
    feat_df["__fk__"] = fk_values

    # Drop null FKs
    feat_df = feat_df.dropna(subset=["__fk__"])

    feat_cols = [c for c in feat_df.columns if c != "__fk__"]
    agg_mean = feat_df.groupby("__fk__")[feat_cols].mean()
    counts = feat_df.groupby("__fk__").size().to_frame("__count__")
    agg_combined = agg_mean.join(counts)

    aligned = agg_combined.reindex(parent_pks).fillna(0).values.astype(np.float32)
    return aligned


def build_chain_features(
    chain: list[tuple[str, str, str]],
    db,
    target_pks: np.ndarray,
) -> np.ndarray | None:
    """Build multi-hop chain features by iterative aggregation from deepest hop.

    chain: list of (child_table, fk_col, parent_table)
    target_pks: primary keys of the target entity table (= parent of first hop)

    Returns features array aligned to target_pks, or None on failure.
    """
    accumulated: pd.DataFrame | None = None

    for hop_idx in reversed(range(len(chain))):
        child_tbl, fk_col, parent_tbl = chain[hop_idx]
        child_df = db.table_dict[child_tbl].df
        child_pkey = db.table_dict[child_tbl].pkey_col
        parent_df = db.table_dict[parent_tbl].df
        parent_pkey = db.table_dict[parent_tbl].pkey_col

        # Own features of child table
        exc = [fk_col]
        if child_pkey:
            exc.append(child_pkey)
        own_feats, _ = extract_numeric_features(child_df, exclude_cols=exc)

        if accumulated is not None:
            # Align accumulated features (indexed by this child's pkey values)
            child_pk_vals = child_df[child_pkey].values
            aligned_acc = accumulated.reindex(child_pk_vals).fillna(0).values.astype(np.float32)
            combined = np.hstack([own_feats, aligned_acc])
        else:
            combined = own_feats

        # Aggregate combined features by FK to parent level
        fk_vals = child_df[fk_col].values
        feat_cols = [f"f{i}" for i in range(combined.shape[1])]
        feat_df = pd.DataFrame(combined, columns=feat_cols)
        feat_df["__fk__"] = fk_vals
        feat_df = feat_df.dropna(subset=["__fk__"])

        agg_mean = feat_df.groupby("__fk__")[feat_cols].mean()
        counts = feat_df.groupby("__fk__").size().to_frame("__count__")
        accumulated = agg_mean.join(counts)

        del feat_df, combined
        gc.collect()

    if accumulated is None:
        return None

    aligned = accumulated.reindex(target_pks).fillna(0).values.astype(np.float32)
    return aligned


# =========================================================================
# PROBE MODEL
# =========================================================================

class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 n_layers: int = 2, task_type: str = "classification"):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            in_d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)
        self.task_type = task_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h).squeeze(-1)


def train_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    task_type: str, seed: int,
    hidden_dim: int = HIDDEN_DIM, n_layers: int = N_LAYERS,
    epochs: int = EPOCHS, lr: float = LR, batch_size: int = BATCH_SIZE,
) -> float:
    """Train a probe MLP and return validation metric (higher = better)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ProbeMLP(X_train.shape[1], hidden_dim, n_layers, task_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if task_type == "classification":
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)

    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)

    model.train()
    for _epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_v).cpu().numpy()

    if task_type == "classification":
        val_prob = 1.0 / (1.0 + np.exp(-np.clip(val_pred, -30, 30)))
        try:
            metric = roc_auc_score(y_val, val_prob)
        except ValueError:
            metric = 0.5
    else:
        metric = -mean_absolute_error(y_val, val_pred)

    del model, optimizer, X_t, y_t, X_v, train_ds, train_loader
    gc.collect()
    return float(metric)


# =========================================================================
# BASELINE: FAN-OUT HEURISTIC
# =========================================================================

def fan_out_heuristic_jrn(join_stats: dict, join_key: str) -> float:
    """Predict JRN using join statistics only (no training needed).

    Heuristic: JRN ~ 1 + alpha * coverage * log(1 + fan_out_mean)
    Higher coverage + fan-out → more information → higher JRN.
    """
    stats = join_stats.get(join_key)
    if stats is None:
        return 1.0
    coverage = stats.get("parent_coverage", 0)
    fan_out_mean = stats.get("fan_out_stats", {}).get("mean", 1)
    alpha = 0.05
    return 1.0 + alpha * coverage * np.log1p(fan_out_mean)


def fan_out_heuristic_chain_jrn(
    chain: list[tuple[str, str, str]], join_stats: dict
) -> float:
    """Predict chain JRN as product of individual fan-out heuristics."""
    product = 1.0
    for child, fk, parent in chain:
        jk = f"{child}.{fk} -> {parent}.Id"
        product *= fan_out_heuristic_jrn(join_stats, jk)
    return product


# =========================================================================
# MAIN
# =========================================================================

@logger.catch
def main():
    start_time = time.time()
    os.chdir(WORKSPACE)
    Path("logs").mkdir(exist_ok=True)

    try:
        current_mem = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        logger.info(f"Starting memory: {current_mem / 1e9:.2f} GB / {TOTAL_RAM_GB:.1f} GB")
    except (FileNotFoundError, ValueError):
        pass

    # === LOAD DEPENDENCY DATA ===
    logger.info("Loading dependency data...")
    dep_preview = json.loads((DEP_DATA_PATH / "preview_data_out.json").read_text())
    meta = dep_preview["metadata"]["rel_stack"]
    join_stats = meta["join_statistics"]
    multi_hop_chains_def = meta["multi_hop_chains"]
    task_defs = meta["tasks"]
    schema_info = meta["schema"]
    logger.info(
        f"Metadata: {len(join_stats)} joins, "
        f"{len(multi_hop_chains_def)} chains, {len(task_defs)} tasks"
    )

    research = json.loads((DEP_RESEARCH_PATH / "research_out.json").read_text())
    logger.info(f"Research: {research['title'][:80]}...")

    # === LOAD RELBENCH ===
    logger.info("Loading rel-stack from relbench...")
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset = get_dataset("rel-stack", download=True)
    db = dataset.get_db()
    for tname in schema_info:
        assert tname in db.table_dict, f"Missing table: {tname}"
        logger.info(f"  {tname}: {len(db.table_dict[tname].df)} rows")

    try:
        current_mem = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        logger.info(f"Memory after load: {current_mem / 1e9:.2f} GB / {TOTAL_RAM_GB:.1f} GB")
    except (FileNotFoundError, ValueError):
        pass

    # ==================================================================
    # PHASE 1: INDIVIDUAL JRN ESTIMATION
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Individual JRN Estimation")
    logger.info("=" * 60)

    jrn_results: dict[tuple[str, str], dict] = {}
    base_metrics_cache: dict[str, tuple[float, list[float]]] = {}

    for task_name, entity_table, target_col, task_type in ENTITY_TASKS:
        logger.info(f"=== Task: {task_name} (entity={entity_table}) ===")
        task_start = time.time()

        try:
            task = get_task("rel-stack", task_name, download=True)
            train_table = task.get_table("train")
            val_table = task.get_table("val")
        except Exception:
            logger.exception(f"Failed to load task {task_name}")
            continue

        entity_col = task.entity_col
        train_ids = train_table.df[entity_col].values.copy()
        val_ids = val_table.df[entity_col].values.copy()
        y_train = train_table.df[target_col].values.astype(np.float32).copy()
        y_val = val_table.df[target_col].values.astype(np.float32).copy()
        logger.info(f"  Raw sizes: train={len(y_train)}, val={len(y_val)}")

        # Subsample
        rng = np.random.RandomState(42)
        if len(y_train) > MAX_TRAIN:
            idx = rng.choice(len(y_train), MAX_TRAIN, replace=False)
            train_ids, y_train = train_ids[idx], y_train[idx]
        if len(y_val) > MAX_VAL:
            idx = rng.choice(len(y_val), MAX_VAL, replace=False)
            val_ids, y_val = val_ids[idx], y_val[idx]
        logger.info(f"  After sub-sample: train={len(y_train)}, val={len(y_val)}")

        # Entity table features
        entity_df = db.table_dict[entity_table].df
        entity_pkey = db.table_dict[entity_table].pkey_col

        parent_feats, parent_cols = extract_numeric_features(
            entity_df, exclude_cols=[entity_pkey]
        )
        parent_feat_idx = entity_df[entity_pkey].values
        parent_feat_df = pd.DataFrame(parent_feats, columns=parent_cols,
                                       index=parent_feat_idx)

        X_base_train = parent_feat_df.reindex(train_ids).fillna(0).values.astype(np.float32)
        X_base_val = parent_feat_df.reindex(val_ids).fillna(0).values.astype(np.float32)

        scaler_base = StandardScaler()
        X_base_train = scaler_base.fit_transform(X_base_train).astype(np.float32)
        X_base_val = scaler_base.transform(X_base_val).astype(np.float32)
        logger.info(f"  Base features: {X_base_train.shape[1]} dims")

        # Train baseline probe
        base_metrics: list[float] = []
        for seed in SEEDS:
            m = train_probe(X_base_train, y_train, X_base_val, y_val, task_type, seed)
            base_metrics.append(m)
        M_base = float(np.mean(base_metrics))
        base_metrics_cache[task_name] = (M_base, base_metrics)
        logger.info(f"  M_base = {M_base:.6f} (std={np.std(base_metrics):.6f})")

        # For each relevant join (parent == entity_table)
        for child_table, fk_col, parent_table, pk_col in FK_JOINS:
            if parent_table != entity_table:
                continue

            join_key = f"{child_table}.{fk_col} -> {parent_table}.{pk_col}"
            logger.info(f"  Join: {join_key}")
            join_t0 = time.time()

            try:
                child_df = db.table_dict[child_table].df
                child_pkey = db.table_dict[child_table].pkey_col

                # Aggregate child features to entity level
                X_child_all = aggregate_child_to_parent(
                    child_df=child_df,
                    child_fk_col=fk_col,
                    parent_pks=entity_df[entity_pkey].values,
                    exclude_child_cols=[child_pkey] if child_pkey else None,
                )

                child_feat_df = pd.DataFrame(
                    X_child_all, index=entity_df[entity_pkey].values
                )
                X_child_train = child_feat_df.reindex(train_ids).fillna(0).values.astype(np.float32)
                X_child_val = child_feat_df.reindex(val_ids).fillna(0).values.astype(np.float32)

                child_scaler = StandardScaler()
                X_child_train = child_scaler.fit_transform(X_child_train).astype(np.float32)
                X_child_val = child_scaler.transform(X_child_val).astype(np.float32)

                X_join_train = np.hstack([X_base_train, X_child_train])
                X_join_val = np.hstack([X_base_val, X_child_val])
                logger.info(
                    f"    Features: {X_join_train.shape[1]} "
                    f"({X_base_train.shape[1]} base + {X_child_train.shape[1]} child)"
                )

                join_metrics: list[float] = []
                for seed in SEEDS:
                    m = train_probe(
                        X_join_train, y_train, X_join_val, y_val, task_type, seed
                    )
                    join_metrics.append(m)
                M_join = float(np.mean(join_metrics))

                jrn_ratio = M_join / M_base if abs(M_base) > 1e-10 else 1.0
                rel_imp = (M_join - M_base) / abs(M_base) if abs(M_base) > 1e-10 else 0.0
                # Corrected JRN: always > 1 when join helps
                # For positive metrics (AUROC): same as ratio
                # For negative metrics (neg_MAE): invert so >1 = improvement
                if M_base < 0 and abs(jrn_ratio) > 1e-10:
                    jrn_corrected = 1.0 / jrn_ratio
                else:
                    jrn_corrected = jrn_ratio
                heuristic = fan_out_heuristic_jrn(join_stats, join_key)

                jrn_results[(join_key, task_name)] = {
                    "M_base": M_base,
                    "M_base_std": float(np.std(base_metrics)),
                    "M_join": M_join,
                    "M_join_std": float(np.std(join_metrics)),
                    "JRN_ratio": jrn_ratio,
                    "JRN": jrn_corrected,
                    "relative_improvement": rel_imp,
                    "task_type": task_type,
                    "base_seeds": [float(x) for x in base_metrics],
                    "join_seeds": [float(x) for x in join_metrics],
                    "heuristic_jrn": heuristic,
                    "child_features_dim": int(X_child_train.shape[1]),
                }
                elapsed = time.time() - join_t0
                logger.info(
                    f"    M_join={M_join:.6f} JRN={jrn_corrected:.4f} "
                    f"(ratio={jrn_ratio:.4f}) heuristic={heuristic:.4f} [{elapsed:.1f}s]"
                )

                del X_child_all, child_feat_df, X_child_train, X_child_val
                del X_join_train, X_join_val
                gc.collect()

            except Exception:
                logger.exception(f"    FAILED: {join_key}")
                continue

        del X_base_train, X_base_val, parent_feat_df
        gc.collect()
        logger.info(f"  Task {task_name} done in {time.time() - task_start:.1f}s")

    logger.info(f"Phase 1 complete: {len(jrn_results)} JRN measurements")

    # ==================================================================
    # PHASE 3: MULTIPLICATIVE COMPOUNDING
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Multiplicative Compounding")
    logger.info("=" * 60)

    chain_results: list[dict] = []

    for chain_idx, chain in enumerate(CHAINS):
        target_table = chain[0][2]  # parent of first hop
        chain_desc = " -> ".join(
            f"{c}.{fk} -> {p}" for c, fk, p in chain
        )
        logger.info(f"Chain {chain_idx}: [{chain_desc}] target={target_table}")

        for task_name, entity_table, target_col, task_type in ENTITY_TASKS:
            if entity_table != target_table:
                continue

            logger.info(f"  Task: {task_name}")
            chain_t0 = time.time()

            try:
                # (a) Predicted chain JRN = product of individual corrected JRNs
                individual_jrns: list[float] = []
                for child, fk, parent in chain:
                    jk = f"{child}.{fk} -> {parent}.Id"
                    key = (jk, task_name)
                    if key in jrn_results:
                        individual_jrns.append(jrn_results[key]["JRN"])  # corrected
                    else:
                        individual_jrns.append(1.0)
                predicted_chain_jrn = float(np.prod(individual_jrns))

                # Heuristic chain JRN
                heuristic_chain = fan_out_heuristic_chain_jrn(chain, join_stats)

                # (b) Measure actual chain JRN via multi-hop aggregation
                task_obj = get_task("rel-stack", task_name, download=True)
                train_table = task_obj.get_table("train")
                val_table = task_obj.get_table("val")

                entity_col = task_obj.entity_col
                train_ids = train_table.df[entity_col].values.copy()
                val_ids = val_table.df[entity_col].values.copy()
                y_train = train_table.df[target_col].values.astype(np.float32).copy()
                y_val = val_table.df[target_col].values.astype(np.float32).copy()

                rng = np.random.RandomState(42)
                if len(y_train) > MAX_TRAIN:
                    idx = rng.choice(len(y_train), MAX_TRAIN, replace=False)
                    train_ids, y_train = train_ids[idx], y_train[idx]
                if len(y_val) > MAX_VAL:
                    idx = rng.choice(len(y_val), MAX_VAL, replace=False)
                    val_ids, y_val = val_ids[idx], y_val[idx]

                # Entity base features (reuse cache for M_base)
                entity_df = db.table_dict[entity_table].df
                entity_pkey = db.table_dict[entity_table].pkey_col
                parent_feats, parent_cols = extract_numeric_features(
                    entity_df, exclude_cols=[entity_pkey]
                )
                pfdf = pd.DataFrame(
                    parent_feats, columns=parent_cols,
                    index=entity_df[entity_pkey].values,
                )
                X_base_tr = pfdf.reindex(train_ids).fillna(0).values.astype(np.float32)
                X_base_vl = pfdf.reindex(val_ids).fillna(0).values.astype(np.float32)
                scaler_b = StandardScaler()
                X_base_tr = scaler_b.fit_transform(X_base_tr).astype(np.float32)
                X_base_vl = scaler_b.transform(X_base_vl).astype(np.float32)

                M_base_cached = base_metrics_cache.get(task_name, (None, None))[0]
                if M_base_cached is None:
                    bm = [train_probe(X_base_tr, y_train, X_base_vl, y_val,
                                      task_type, s) for s in SEEDS]
                    M_base_cached = float(np.mean(bm))

                # Build chain features
                target_pks = entity_df[entity_pkey].values
                chain_feats = build_chain_features(chain, db, target_pks)

                if chain_feats is None or chain_feats.shape[1] == 0:
                    logger.warning("    Chain features empty, skipping")
                    continue

                chain_feat_df = pd.DataFrame(chain_feats, index=target_pks)
                X_chain_tr = chain_feat_df.reindex(train_ids).fillna(0).values.astype(np.float32)
                X_chain_vl = chain_feat_df.reindex(val_ids).fillna(0).values.astype(np.float32)

                chain_scaler = StandardScaler()
                X_chain_tr = chain_scaler.fit_transform(X_chain_tr).astype(np.float32)
                X_chain_vl = chain_scaler.transform(X_chain_vl).astype(np.float32)

                X_full_tr = np.hstack([X_base_tr, X_chain_tr])
                X_full_vl = np.hstack([X_base_vl, X_chain_vl])
                logger.info(
                    f"    Chain features: {X_chain_tr.shape[1]} dims, "
                    f"total={X_full_tr.shape[1]}"
                )

                chain_metrics: list[float] = []
                for seed in SEEDS:
                    m = train_probe(
                        X_full_tr, y_train, X_full_vl, y_val, task_type, seed
                    )
                    chain_metrics.append(m)
                M_chain = float(np.mean(chain_metrics))

                actual_chain_ratio = (
                    M_chain / M_base_cached
                    if abs(M_base_cached) > 1e-10 else 1.0
                )
                # Correct for negative metrics
                if M_base_cached < 0 and abs(actual_chain_ratio) > 1e-10:
                    actual_chain_jrn = 1.0 / actual_chain_ratio
                else:
                    actual_chain_jrn = actual_chain_ratio

                deviation = actual_chain_jrn - predicted_chain_jrn
                ratio = (
                    actual_chain_jrn / predicted_chain_jrn
                    if abs(predicted_chain_jrn) > 1e-10 else None
                )

                chain_results.append({
                    "chain": [f"{c}.{fk} -> {p}" for c, fk, p in chain],
                    "depth": len(chain),
                    "task": task_name,
                    "individual_jrns": individual_jrns,
                    "predicted_chain_jrn": predicted_chain_jrn,
                    "actual_chain_jrn": actual_chain_jrn,
                    "heuristic_chain_jrn": heuristic_chain,
                    "M_base": M_base_cached,
                    "M_chain": M_chain,
                    "M_chain_std": float(np.std(chain_metrics)),
                    "chain_seeds": [float(x) for x in chain_metrics],
                    "deviation": deviation,
                    "ratio": ratio,
                    "chain_features_dim": int(X_chain_tr.shape[1]),
                })

                elapsed = time.time() - chain_t0
                logger.info(
                    f"    actual={actual_chain_jrn:.4f} "
                    f"predicted={predicted_chain_jrn:.4f} "
                    f"dev={deviation:.4f} [{elapsed:.1f}s]"
                )

                del chain_feats, chain_feat_df
                del X_chain_tr, X_chain_vl, X_full_tr, X_full_vl
                del X_base_tr, X_base_vl, pfdf
                gc.collect()

            except Exception:
                logger.exception(f"    FAILED chain {chain_idx} x {task_name}")
                continue

    logger.info(f"Phase 3 complete: {len(chain_results)} chain measurements")

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)

    # JRN matrix
    jrn_matrix: dict[str, dict[str, float]] = {}
    for (jk, tn), res in jrn_results.items():
        jrn_matrix.setdefault(jk, {})[tn] = res["JRN"]

    # Classifications
    classifications: dict[str, str] = {}
    for (jk, tn), res in jrn_results.items():
        jrn_val = res["JRN"]
        if jrn_val > 1.05:
            cat = "high_value"
        elif jrn_val < 0.95:
            cat = "low_value"
        else:
            cat = "critical_threshold"
        classifications[f"{jk}__{tn}"] = cat

    n_high = sum(1 for v in classifications.values() if v == "high_value")
    n_crit = sum(1 for v in classifications.values() if v == "critical_threshold")
    n_low = sum(1 for v in classifications.values() if v == "low_value")
    logger.info(f"  Classifications: high={n_high}, critical={n_crit}, low={n_low}")

    # Domain validation (all JRN values are now corrected: > 1 = helpful)
    validations: dict[str, dict] = {}
    vote_post_jrn = jrn_matrix.get("votes.PostId -> posts.Id", {}).get("post-votes")
    validations["votes_posts_for_postvotes"] = {
        "expected": "JRN > 1 (votes directly predict popularity)",
        "actual_jrn": vote_post_jrn,
        "passed": vote_post_jrn is not None and vote_post_jrn > 1.0,
        "note": "corrected JRN (inverted for neg_MAE so >1 = improvement)",
    }
    badges_user_jrn = jrn_matrix.get("badges.UserId -> users.Id", {}).get("user-badge")
    validations["badges_users_for_userbadge"] = {
        "expected": "JRN > 1 (badges predict future badges)",
        "actual_jrn": badges_user_jrn,
        "passed": badges_user_jrn is not None and badges_user_jrn > 1.0,
    }
    posts_user_engage = jrn_matrix.get(
        "posts.OwnerUserId -> users.Id", {}
    ).get("user-engagement")
    validations["posts_users_for_engagement"] = {
        "expected": "JRN > 1 (posting activity predicts engagement)",
        "actual_jrn": posts_user_engage,
        "passed": posts_user_engage is not None and posts_user_engage > 1.0,
    }
    vote_post_jrn_corr = vote_post_jrn
    postlinks_post_jrn = jrn_matrix.get(
        "postLinks.PostId -> posts.Id", {}
    ).get("post-votes")
    validations["votes_stronger_than_postlinks"] = {
        "expected": "votes JRN > postLinks JRN for post-votes",
        "votes_jrn": vote_post_jrn_corr,
        "postlinks_jrn": postlinks_post_jrn,
        "passed": (vote_post_jrn_corr is not None and postlinks_post_jrn is not None
                   and vote_post_jrn_corr > postlinks_post_jrn),
    }

    n_passed = sum(1 for v in validations.values() if v.get("passed"))
    logger.info(f"  Domain validations passed: {n_passed}/{len(validations)}")

    # R² for compounding
    if len(chain_results) >= 2:
        predicted = [r["predicted_chain_jrn"] for r in chain_results]
        actual = [r["actual_chain_jrn"] for r in chain_results]
        mean_actual = float(np.mean(actual))
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        ss_tot = sum((a - mean_actual) ** 2 for a in actual)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        mean_abs_dev = float(np.mean([abs(r["deviation"]) for r in chain_results]))
    else:
        r_squared = 0.0
        mean_abs_dev = 0.0
    logger.info(f"  Compounding R²={r_squared:.4f}, MAD={mean_abs_dev:.4f}")

    # Key findings
    key_findings: list[str] = []
    if len(jrn_results) > 0:
        jrn_vals = [r["JRN"] for r in jrn_results.values()]
        key_findings.append(
            f"JRN range: [{min(jrn_vals):.4f}, {max(jrn_vals):.4f}], "
            f"spread={max(jrn_vals)-min(jrn_vals):.4f}"
        )
    if vote_post_jrn is not None:
        key_findings.append(
            f"Votes->Posts corrected JRN for post-votes = {vote_post_jrn:.4f} "
            f"({'confirms' if vote_post_jrn > 1 else 'contradicts'} domain expectation)"
        )
    if badges_user_jrn is not None:
        key_findings.append(
            f"Badges->Users JRN for user-badge = {badges_user_jrn:.4f} "
            f"({'confirms' if badges_user_jrn > 1 else 'contradicts'} domain expectation)"
        )
    if len(chain_results) >= 2:
        key_findings.append(
            f"Multiplicative compounding: R²={r_squared:.4f} "
            f"across {len(chain_results)} chains"
        )
    for f in key_findings:
        logger.info(f"  Finding: {f}")

    # ==================================================================
    # BUILD OUTPUT (exp_gen_sol_out.json schema)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Building output...")
    logger.info("=" * 60)

    examples: list[dict] = []

    # Phase 1 examples: one per (join, task) pair
    for (jk, tn), res in jrn_results.items():
        input_desc = json.dumps({
            "phase": "phase1_individual_jrn",
            "join": jk,
            "task": tn,
            "entity_table": jk.split(" -> ")[1].split(".")[0],
            "child_table": jk.split(".")[0],
            "question": (
                f"What is the JRN for join '{jk}' on task '{tn}'? "
                f"How much does adding aggregated child features improve "
                f"the probe metric?"
            ),
        })
        output_desc = json.dumps({
            "JRN": res["JRN"],
            "M_base": res["M_base"],
            "M_join": res["M_join"],
            "relative_improvement": res["relative_improvement"],
            "classification": classifications.get(f"{jk}__{tn}", "unknown"),
        })
        predict_baseline = json.dumps({
            "heuristic_jrn": res["heuristic_jrn"],
            "method": "fan_out_heuristic",
            "formula": "1 + 0.05 * coverage * log(1 + fan_out_mean)",
        })
        predict_method = json.dumps({
            "JRN": res["JRN"],
            "M_base": res["M_base"],
            "M_join": res["M_join"],
            "M_base_std": res["M_base_std"],
            "M_join_std": res["M_join_std"],
            "seeds": len(SEEDS),
            "method": "probe_based_jrn",
        })

        examples.append({
            "input": input_desc,
            "output": output_desc,
            "predict_baseline": predict_baseline,
            "predict_our_method": predict_method,
            "metadata_phase": "phase1",
            "metadata_join": jk,
            "metadata_task": tn,
        })

    # Phase 3 examples: one per (chain, task) pair
    for cr in chain_results:
        chain_str = " | ".join(cr["chain"])
        input_desc = json.dumps({
            "phase": "phase3_compounding",
            "chain": cr["chain"],
            "depth": cr["depth"],
            "task": cr["task"],
            "question": (
                f"Does the multiplicative compounding property hold for "
                f"the {cr['depth']}-hop chain [{chain_str}] on task "
                f"'{cr['task']}'? What is the actual vs predicted chain JRN?"
            ),
        })
        output_desc = json.dumps({
            "actual_chain_jrn": cr["actual_chain_jrn"],
            "M_chain": cr["M_chain"],
            "M_base": cr["M_base"],
            "deviation_from_predicted": cr["deviation"],
        })
        predict_baseline = json.dumps({
            "predicted_chain_jrn": cr["heuristic_chain_jrn"],
            "method": "fan_out_heuristic_product",
        })
        predict_method = json.dumps({
            "predicted_chain_jrn": cr["predicted_chain_jrn"],
            "actual_chain_jrn": cr["actual_chain_jrn"],
            "individual_jrns": cr["individual_jrns"],
            "deviation": cr["deviation"],
            "ratio": cr["ratio"],
            "method": "multiplicative_product_of_probe_jrns",
        })

        examples.append({
            "input": input_desc,
            "output": output_desc,
            "predict_baseline": predict_baseline,
            "predict_our_method": predict_method,
            "metadata_phase": "phase3",
            "metadata_chain": chain_str,
            "metadata_task": cr["task"],
        })

    output = {
        "metadata": {
            "title": "JRN Estimation & Multiplicative Compounding on rel-stack",
            "summary": (
                f"Estimated JRN for {len(jrn_results)} (join, task) pairs using "
                f"MLP probe method (hidden={HIDDEN_DIM}, layers={N_LAYERS}, "
                f"epochs={EPOCHS}, seeds={len(SEEDS)}). "
                f"Validated multiplicative compounding on {len(chain_results)} "
                f"multi-hop chains (R²={r_squared:.4f}). "
                f"Join classifications: {n_high} high-value, {n_crit} critical, "
                f"{n_low} low-value."
            ),
            "experiment_config": {
                "dataset": "rel-stack",
                "hidden_dim": HIDDEN_DIM,
                "n_layers": N_LAYERS,
                "epochs": EPOCHS,
                "seeds": SEEDS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "max_train_samples": MAX_TRAIN,
                "max_val_samples": MAX_VAL,
                "aggregation": "mean",
            },
            "phase1_summary": {
                "jrn_matrix": {
                    jk: {tn: float(v) for tn, v in tasks.items()}
                    for jk, tasks in jrn_matrix.items()
                },
                "n_measurements": len(jrn_results),
                "n_high_value": n_high,
                "n_critical": n_crit,
                "n_low_value": n_low,
            },
            "phase3_summary": {
                "n_chains_tested": len(chain_results),
                "r_squared": float(r_squared),
                "mean_absolute_deviation": float(mean_abs_dev),
                "compounding_holds": r_squared > 0.3,
            },
            "domain_validation": {
                k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                    for kk, vv in v.items()}
                for k, v in validations.items()
            },
            "key_findings": key_findings,
            "hypothesis_support": {
                "jrn_is_informative": (
                    max(jrn_vals) - min(jrn_vals) > 0.05
                    if len(jrn_results) > 0 else False
                ),
                "compounding_holds": r_squared > 0.3 if len(chain_results) >= 2 else "inconclusive",
                "domain_expectations_met": n_passed >= 2,
            },
            "runtime_seconds": time.time() - start_time,
        },
        "datasets": [
            {
                "dataset": "rel-stack",
                "examples": examples,
            }
        ],
    }

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    file_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {out_path} ({file_mb:.2f} MB, {len(examples)} examples)")

    total_elapsed = time.time() - start_time
    logger.info(f"TOTAL RUNTIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info("DONE!")


if __name__ == "__main__":
    main()
