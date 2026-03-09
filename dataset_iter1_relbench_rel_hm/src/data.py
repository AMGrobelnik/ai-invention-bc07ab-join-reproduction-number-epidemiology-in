#!/usr/bin/env python3
"""Prepare RelBench rel-hm (H&M) dataset for JRN estimation experiments.

Loads articles + customers from local JSON, transactions from HuggingFace,
computes full join statistics, creates stratified subsample preserving
natural fan-out distribution, and outputs full_data_out.json conforming
to exp_sel_data_out schema.
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
import psutil
from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────
WS = Path(
    "/ai-inventor/aii_pipeline/runs/run__20260309_024817"
    "/3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)
LOGS_DIR = WS / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = WS / "temp" / "datasets"

# ── Logging ──────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "data.log"), rotation="30 MB", level="DEBUG")

# ── Hardware ─────────────────────────────────────────────────────────
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

def _container_ram_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.6 * 1e9)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, (
    f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Constants ────────────────────────────────────────────────────────
FAN_OUT_BUCKETS = [
    (1, 1, "1"), (2, 5, "2-5"), (6, 10, "6-10"), (11, 50, "11-50"),
    (51, 100, "51-100"), (101, 500, "101-500"), (501, 1000, "501-1000"),
    (1001, float("inf"), "1001+"),
]
TARGET_CUSTOMERS = 25000
MAX_TRANSACTIONS = 280000   # keep output under ~290 MB
FOLD_TRAIN, FOLD_VAL, FOLD_TEST = 0, 1, 2


def _ram_gb() -> str:
    return f"{psutil.Process().memory_info().rss / 1e9:.2f} GB"


def assign_bucket(count: int) -> str:
    for lo, hi, label in FAN_OUT_BUCKETS:
        if lo <= count <= hi:
            return label
    return "1001+"


def fan_out_histogram(series: pd.Series) -> dict:
    h = {}
    for lo, hi, label in FAN_OUT_BUCKETS:
        mask = series >= lo if hi == float("inf") else (series >= lo) & (series <= hi)
        h[label] = int(mask.sum())
    return h


# ── Join Statistics ──────────────────────────────────────────────────
def compute_join_stats(
    txn: pd.DataFrame, fk_col: str,
    parent: pd.DataFrame, parent_label: str,
) -> tuple[dict, pd.Series]:
    logger.info(f"Join stats: transactions.{fk_col} → {parent_label}")
    fo = txn.groupby(fk_col).size()
    stats = {
        "join": f"transactions.{fk_col} → {parent_label}",
        "child_rows": int(len(txn)),
        "unique_fk_values": int(len(fo)),
        "parent_rows": int(len(parent)),
        "fan_out": {
            k: round(float(v), 2)
            for k, v in {
                "min": fo.min(), "max": fo.max(), "mean": fo.mean(),
                "median": fo.median(), "std": fo.std(),
                "p25": fo.quantile(.25), "p75": fo.quantile(.75),
                "p90": fo.quantile(.90), "p95": fo.quantile(.95),
                "p99": fo.quantile(.99),
            }.items()
        },
        "coverage": round(float(len(fo) / len(parent)), 4),
        "cardinality_ratio": round(float(len(txn) / len(fo)), 2),
        "fan_out_histogram": fan_out_histogram(fo),
    }
    logger.info(
        f"  coverage={stats['coverage']:.4f}  mean_fo={stats['fan_out']['mean']:.1f}  "
        f"max_fo={stats['fan_out']['max']:.0f}"
    )
    return stats, fo


# ── Stratified Subsample ─────────────────────────────────────────────
def stratified_subsample(
    cust_df: pd.DataFrame, txn_df: pd.DataFrame,
    cust_fo: pd.Series, target: int, max_txn: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    logger.info(f"Stratified subsample: target {target} custs, max {max_txn} txns")
    buckets = cust_fo.apply(assign_bucket)
    bucket_pop = buckets.value_counts()
    total = len(cust_fo)

    sampled_ids: list = []
    for label, pop in bucket_pop.items():
        ids = buckets[buckets == label].index.tolist()
        n = max(1, min(len(ids), int(target * pop / total)))
        sampled_ids.extend(np.random.choice(ids, size=n, replace=False).tolist())

    sampled_set = set(sampled_ids)
    logger.info(f"  {len(sampled_set)} customers selected")

    sub_txn = txn_df[txn_df["customer_id"].isin(sampled_set)].copy()
    logger.info(f"  {len(sub_txn)} transactions")

    # Reduce if too many transactions
    if len(sub_txn) > max_txn:
        ratio = max_txn / len(sub_txn) * 0.9
        keep = set(np.random.choice(
            list(sampled_set), size=int(len(sampled_set) * ratio), replace=False
        ))
        sampled_set = keep
        sub_txn = txn_df[txn_df["customer_id"].isin(sampled_set)].copy()
        logger.info(f"  Reduced → {len(sampled_set)} custs, {len(sub_txn)} txns")

    art_ids = set(sub_txn["article_id"].unique())
    sub_cust = cust_df[cust_df["customer_id"].isin(sampled_set)].copy()
    logger.info(f"  {len(art_ids)} articles in subsample")
    return sub_cust, sub_txn, art_ids


# ── Build Examples ───────────────────────────────────────────────────
def _safe_val(v):
    """Make a value JSON-safe."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def build_customer_examples(
    cust: pd.DataFrame, txn: pd.DataFrame,
    cust_fo: pd.Series, val_start, test_start,
) -> list[dict]:
    logger.info("Building customer examples …")
    dates = pd.to_datetime(txn["t_dat"])
    active_train = set(txn[dates < val_start]["customer_id"].unique())
    active_val = set(txn[(dates >= val_start) & (dates < test_start)]["customer_id"].unique())
    active_test = set(txn[dates >= test_start]["customer_id"].unique())

    cols = list(cust.columns)
    feature_names = [c for c in cols if c != "customer_id"]
    examples = []
    for rec in cust.to_dict("records"):
        cid = rec["customer_id"]
        feats = {k: _safe_val(rec[k]) for k in feature_names}
        fo_val = int(cust_fo.get(cid, 0))

        if cid in active_train:
            fold = FOLD_TRAIN
            churn = int(cid not in active_val)
        elif cid in active_val:
            fold = FOLD_VAL
            churn = int(cid not in active_test)
        elif cid in active_test:
            fold = FOLD_TEST
            churn = 0
        else:
            fold = FOLD_TRAIN
            churn = 1

        examples.append({
            "input": json.dumps({"table": "customers", "row_id": cid, "features": feats}, default=str),
            "output": json.dumps({"user-churn": churn}),
            "metadata_fold": fold,
            "metadata_table": "customers",
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_feature_names": feature_names,
            "metadata_fan_out": fo_val,
            "metadata_fan_out_bucket": assign_bucket(fo_val),
            "metadata_row_index": int(cid.__hash__() % 2**31),
        })
    logger.info(f"  {len(examples)} customer examples")
    return examples


def build_article_examples(
    arts: pd.DataFrame, txn: pd.DataFrame,
    art_fo: pd.Series, val_start, test_start,
) -> list[dict]:
    logger.info("Building article examples …")
    dates = pd.to_datetime(txn["t_dat"])
    train_sales = txn[dates < val_start].groupby("article_id")["price"].sum()
    val_sales = txn[(dates >= val_start) & (dates < test_start)].groupby("article_id")["price"].sum()
    test_sales = txn[dates >= test_start].groupby("article_id")["price"].sum()

    cols = list(arts.columns)
    feature_names = [c for c in cols if c != "article_id"]
    examples = []
    for rec in arts.to_dict("records"):
        aid = int(rec["article_id"])
        feats = {k: _safe_val(rec[k]) for k in feature_names}
        fo_val = int(art_fo.get(aid, 0))

        ts = float(test_sales.get(aid, 0))
        vs = float(val_sales.get(aid, 0))
        trs = float(train_sales.get(aid, 0))
        if ts > 0:
            fold, label = FOLD_TEST, ts
        elif vs > 0:
            fold, label = FOLD_VAL, vs
        else:
            fold, label = FOLD_TRAIN, trs

        examples.append({
            "input": json.dumps({"table": "articles", "row_id": aid, "features": feats}, default=str),
            "output": json.dumps({"item-sales": round(label, 4)}),
            "metadata_fold": fold,
            "metadata_table": "articles",
            "metadata_task_type": "regression",
            "metadata_feature_names": feature_names,
            "metadata_fan_out": fo_val,
            "metadata_fan_out_bucket": assign_bucket(fo_val),
            "metadata_row_index": aid,
        })
    logger.info(f"  {len(examples)} article examples")
    return examples


def build_transaction_examples(
    txn: pd.DataFrame, cust_fo: pd.Series,
    val_start, test_start,
) -> list[dict]:
    logger.info("Building transaction examples …")
    feature_names = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    examples = []
    for idx, rec in enumerate(txn.to_dict("records")):
        t = pd.Timestamp(rec["t_dat"])
        fold = FOLD_TRAIN if t < val_start else (FOLD_VAL if t < test_start else FOLD_TEST)
        cid = rec["customer_id"]
        cfo = int(cust_fo.get(cid, 0))
        price = round(float(rec["price"]), 4)

        feats = {
            "t_dat": str(rec["t_dat"]),
            "customer_id": cid,
            "article_id": int(rec["article_id"]),
            "price": price,
            "sales_channel_id": int(rec["sales_channel_id"]),
        }

        examples.append({
            "input": json.dumps({
                "table": "transactions", "row_id": idx, "features": feats,
                "foreign_keys": {"customer_id": "customers", "article_id": "articles"},
            }),
            "output": json.dumps({"transactions-price": price}),
            "metadata_fold": fold,
            "metadata_table": "transactions",
            "metadata_task_type": "regression",
            "metadata_feature_names": feature_names,
            "metadata_fan_out": cfo,
            "metadata_fan_out_bucket": assign_bucket(cfo),
            "metadata_row_index": idx,
        })
        if idx > 0 and idx % 100_000 == 0:
            logger.info(f"    {idx}/{len(txn)} transactions …")

    logger.info(f"  {len(examples)} transaction examples")
    return examples


# ── Main ─────────────────────────────────────────────────────────────
@logger.catch
def main():
    t0 = time.time()
    np.random.seed(42)
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET/1e9:.1f} GB")
    logger.info(f"Available RAM: {_avail / 1e9:.1f} GB")

    # ── Load articles ────────────────────────────────────────────────
    logger.info("Loading articles …")
    art_path = DATA_DIR / "full_einrafh_hnm-fashion-recommendations-data_articles_train.json"
    articles_df = pd.read_json(art_path)
    logger.info(f"  {len(articles_df)} rows, {len(articles_df.columns)} cols  RAM={_ram_gb()}")

    # ── Load customers ───────────────────────────────────────────────
    logger.info("Loading customers …")
    cust_path = DATA_DIR / "full_einrafh_hnm-fashion-recommendations-data_customers_train.json"
    customers_df = pd.read_json(cust_path)
    logger.info(f"  {len(customers_df)} rows, {len(customers_df.columns)} cols  RAM={_ram_gb()}")

    # ── Load transactions from HuggingFace ───────────────────────────
    logger.info("Loading transactions from HuggingFace …")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    t_load = time.time()
    from datasets import load_dataset
    ds = load_dataset(
        "einrafh/hnm-fashion-recommendations-data", "transactions", split="train"
    )
    logger.info(f"  HF load: {time.time()-t_load:.0f}s  ({len(ds)} rows)")
    txn_df = ds.to_pandas()
    del ds; gc.collect()
    logger.info(f"  {len(txn_df)} rows, {len(txn_df.columns)} cols  RAM={_ram_gb()}")

    num_txn_full = len(txn_df)
    num_cust_full = len(customers_df)
    num_art_full = len(articles_df)

    # ── Full Join Statistics ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Computing FULL join statistics …")
    cust_stats, cust_fo = compute_join_stats(
        txn_df, "customer_id", customers_df, "customers"
    )
    art_stats, art_fo = compute_join_stats(
        txn_df, "article_id", articles_df, "articles"
    )
    full_join_meta = {"customer_join": cust_stats, "article_join": art_stats}
    logger.info(f"RAM after stats: {_ram_gb()}")

    # ── Stratified Subsample ─────────────────────────────────────────
    logger.info("=" * 60)
    sub_cust, sub_txn, sub_art_ids = stratified_subsample(
        customers_df, txn_df, cust_fo,
        target=TARGET_CUSTOMERS, max_txn=MAX_TRANSACTIONS,
    )
    sub_arts = articles_df[articles_df["article_id"].isin(sub_art_ids)].copy()

    sub_cust_fo = sub_txn.groupby("customer_id").size()
    sub_histogram = fan_out_histogram(sub_cust_fo)
    subsample_meta = {
        "customers": int(len(sub_cust)),
        "articles": int(len(sub_arts)),
        "transactions": int(len(sub_txn)),
        "customer_fan_out_histogram": sub_histogram,
    }
    logger.info(f"Subsample: {json.dumps(subsample_meta)}")

    # Free full data
    del txn_df, customers_df, articles_df; gc.collect()
    logger.info(f"RAM after free: {_ram_gb()}")

    # ── Time splits ──────────────────────────────────────────────────
    dates = pd.to_datetime(sub_txn["t_dat"])
    max_date = dates.max()
    test_start = max_date - pd.Timedelta(days=7)
    val_start = test_start - pd.Timedelta(days=7)
    logger.info(f"Splits: train < {val_start} | val < {test_start} | test >= {test_start}")

    # ── Build examples ───────────────────────────────────────────────
    logger.info("=" * 60)
    cust_ex = build_customer_examples(sub_cust, sub_txn, cust_fo, val_start, test_start)
    art_ex = build_article_examples(sub_arts, sub_txn, art_fo, val_start, test_start)
    txn_ex = build_transaction_examples(sub_txn, cust_fo, val_start, test_start)

    all_examples = cust_ex + art_ex + txn_ex
    logger.info(f"Total examples: {len(all_examples)}")
    del cust_ex, art_ex, txn_ex; gc.collect()

    # Fold & table counts
    from collections import Counter
    fold_c = Counter(e["metadata_fold"] for e in all_examples)
    tbl_c = Counter(e["metadata_table"] for e in all_examples)
    logger.info(f"Folds: { {k: fold_c[k] for k in sorted(fold_c)} }")
    logger.info(f"Tables: {dict(tbl_c)}")

    # ── Assemble output ──────────────────────────────────────────────
    metadata = {
        "source": "H&M Personalized Fashion Recommendations (Kaggle / RelBench rel-hm)",
        "huggingface_id": "einrafh/hnm-fashion-recommendations-data",
        "relbench_id": "rel-hm",
        "description": (
            "Relational dataset with 3 tables (customers, articles, transactions) and "
            "2 FK joins exhibiting extreme cardinality variation. Stratified subsample "
            "preserving natural fan-out distribution for JRN estimation experiments."
        ),
        "tables": {
            "customers": {
                "rows_full": num_cust_full, "rows_subsample": int(len(sub_cust)),
                "columns": list(sub_cust.columns), "primary_key": "customer_id",
            },
            "articles": {
                "rows_full": num_art_full, "rows_subsample": int(len(sub_arts)),
                "columns": list(sub_arts.columns), "primary_key": "article_id",
            },
            "transactions": {
                "rows_full": num_txn_full, "rows_subsample": int(len(sub_txn)),
                "columns": list(sub_txn.columns), "primary_key": None,
                "foreign_keys": {"customer_id": "customers", "article_id": "articles"},
                "time_column": "t_dat",
            },
        },
        "join_statistics_full_dataset": full_join_meta,
        "subsample": subsample_meta,
        "time_splits": {
            "val_start": str(val_start), "test_start": str(test_start),
            "max_date": str(max_date),
        },
        "tasks": {
            "user-churn":         {"type": "classification", "table": "customers",    "metric": "AUROC"},
            "item-sales":         {"type": "regression",     "table": "articles",     "metric": "MAE"},
            "transactions-price": {"type": "regression",     "table": "transactions", "metric": "MAE"},
            "user-item-purchase": {"type": "link_prediction", "tables": ["customers","articles"], "metric": "MAP"},
        },
        "fold_map": {"0": "train", "1": "val", "2": "test"},
    }

    # ── Write split files (100 MB limit per file) ─────────────────────
    out_dir = WS / "data_out"
    out_dir.mkdir(exist_ok=True)

    MAX_PART_BYTES = 90 * 1024 * 1024  # 90 MB target per part
    dataset_name = "rel-hm"

    part_num = 1
    part_examples: list[dict] = []
    current_size = 0

    def _flush_part():
        nonlocal part_num, part_examples, current_size
        if not part_examples:
            return
        part_data = {
            "metadata": metadata,
            "datasets": [{"dataset": dataset_name, "examples": part_examples}],
        }
        p = out_dir / f"full_data_out_{part_num}.json"
        with open(p, "w") as f:
            json.dump(part_data, f, ensure_ascii=False, default=str)
        sz = p.stat().st_size / 1e6
        logger.info(f"  Part {part_num}: {len(part_examples)} examples, {sz:.1f} MB")
        part_num += 1
        part_examples = []
        current_size = 0

    logger.info("Writing split files to data_out/ …")
    t_w = time.time()
    for ex in all_examples:
        ex_sz = len(json.dumps(ex, ensure_ascii=False, default=str)) + 2
        if current_size + ex_sz > MAX_PART_BYTES and part_examples:
            _flush_part()
        part_examples.append(ex)
        current_size += ex_sz
    _flush_part()
    logger.info(f"  Write time: {time.time()-t_w:.1f}s")

    # Also write mini + preview for the first part (for quick inspection)
    first_3 = all_examples[:3]
    mini = {"metadata": metadata, "datasets": [{"dataset": dataset_name, "examples": first_3}]}
    (WS / "mini_data_out.json").write_text(json.dumps(mini, indent=2, ensure_ascii=False, default=str))
    # Preview: truncate long strings
    def _trunc(v, mx=200):
        if isinstance(v, str) and len(v) > mx:
            return v[:mx] + "..."
        if isinstance(v, list):
            return [_trunc(i, mx) for i in v[:3]]
        if isinstance(v, dict):
            return {k: _trunc(vv, mx) for k, vv in v.items()}
        return v
    preview = _trunc(mini)
    (WS / "preview_data_out.json").write_text(json.dumps(preview, indent=2, ensure_ascii=False, default=str))
    logger.info("Wrote mini_data_out.json and preview_data_out.json")

    logger.info(f"Total runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    logger.info("✓ Done")


if __name__ == "__main__":
    main()
