#!/usr/bin/env python3
"""Extract RelBench rel-f1 (Formula 1) relational database for JRN estimation experiments.

Extracts all tables with full row data, FK join metadata with statistics,
and all predictive tasks with temporal train/val/test splits.
Outputs structured data_out.json.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
from loguru import logger

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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

# ── Memory limits ────────────────────────────────────────────────────────────
# rel-f1 is small (~74K rows). Budget 14GB for safety (duckdb/relbench uses mmap).
# NOTE: RLIMIT_AS restricts virtual memory which breaks mmap-based libs like duckdb.
# Use RLIMIT_DATA instead, which limits heap only.
RAM_BUDGET = int(14 * 1024**3)  # 14 GB
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
try:
    resource.setrlimit(resource.RLIMIT_DATA, (RAM_BUDGET, RAM_BUDGET))
except (ValueError, OSError):
    logger.warning("Could not set RLIMIT_DATA, continuing without memory limit")

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")


# ── Helper functions ─────────────────────────────────────────────────────────

def safe_json_value(val: Any) -> Any:
    """Convert a value to be JSON-serializable."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val


def df_to_records(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[Dict]:
    """Convert DataFrame to list of dicts with JSON-safe values."""
    if max_rows is not None:
        df = df.head(max_rows)
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            record[col] = safe_json_value(row[col])
        records.append(record)
    return records


def compute_column_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute per-column statistics."""
    stats = {}
    for col in df.columns:
        col_info: Dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "missing_rate": float(df[col].isna().mean()),
            "num_unique": int(df[col].nunique()),
        }
        if df[col].dtype in ["object", "category"]:
            col_info["type"] = "categorical"
            if df[col].nunique() <= 50:
                col_info["value_counts_top10"] = {
                    str(k): int(v)
                    for k, v in df[col].value_counts().head(10).items()
                }
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_info["type"] = "numeric"
            desc = df[col].describe()
            for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                if stat_name in desc.index:
                    col_info[stat_name.replace("%", "pct")] = safe_json_value(desc[stat_name])
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["type"] = "datetime"
            col_info["min"] = safe_json_value(df[col].min())
            col_info["max"] = safe_json_value(df[col].max())
        stats[col] = col_info
    return stats


def compute_fk_join_stats(
    child_table_name: str,
    child_df: pd.DataFrame,
    fk_col: str,
    parent_table_name: str,
    parent_pkey_col: str,
    parent_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute FK join statistics between child and parent tables."""
    non_null_fk = child_df[fk_col].dropna()
    fanout = non_null_fk.groupby(non_null_fk).size()

    num_edges = int(non_null_fk.count())
    num_unique_parents = int(non_null_fk.nunique())
    num_total_parents = len(parent_df)

    stats = {
        "type": "fk_join_metadata",
        "metadata_fold": "full",
        "source_table": child_table_name,
        "source_fk_col": fk_col,
        "target_table": parent_table_name,
        "target_pk_col": parent_pkey_col,
        "num_edges": num_edges,
        "num_unique_parents_referenced": num_unique_parents,
        "num_total_parents": num_total_parents,
        "join_coverage": safe_json_value(num_unique_parents / num_total_parents if num_total_parents > 0 else 0),
        "cardinality_ratio": safe_json_value(len(child_df) / num_total_parents if num_total_parents > 0 else 0),
    }

    if len(fanout) > 0:
        stats.update({
            "fanout_min": int(fanout.min()),
            "fanout_median": safe_json_value(fanout.median()),
            "fanout_mean": safe_json_value(fanout.mean()),
            "fanout_max": int(fanout.max()),
            "fanout_std": safe_json_value(fanout.std()),
            "fanout_p25": safe_json_value(fanout.quantile(0.25)),
            "fanout_p75": safe_json_value(fanout.quantile(0.75)),
            "fanout_p95": safe_json_value(fanout.quantile(0.95)),
        })
    else:
        stats.update({
            "fanout_min": 0, "fanout_median": 0, "fanout_mean": 0,
            "fanout_max": 0, "fanout_std": 0,
            "fanout_p25": 0, "fanout_p75": 0, "fanout_p95": 0,
        })

    return stats


# ── Main extraction ──────────────────────────────────────────────────────────

@logger.catch
def main():
    tic_total = time.time()

    # ── Step 1: Load rel-f1 database ─────────────────────────────────────
    logger.info("Loading rel-f1 dataset via relbench API...")
    from relbench.datasets import get_dataset

    dataset = get_dataset("rel-f1", download=True)
    db = dataset.get_db()
    logger.info(f"Loaded database with {len(db.table_dict)} tables")
    logger.info(f"Val timestamp: {dataset.val_timestamp}, Test timestamp: {dataset.test_timestamp}")

    # ── Step 2: Extract all tables ───────────────────────────────────────
    logger.info("Extracting table data and metadata...")
    output_rows: List[Dict] = []

    table_summary = {}
    for table_name, table in db.table_dict.items():
        df = table.df
        logger.info(f"  Table '{table_name}': {len(df)} rows x {len(df.columns)} cols, "
                     f"pkey={table.pkey_col}, time_col={table.time_col}, "
                     f"fkeys={table.fkey_col_to_pkey_table}")

        table_summary[table_name] = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "pkey": table.pkey_col,
            "time_col": table.time_col,
            "fkeys": table.fkey_col_to_pkey_table,
        }

        # Column statistics
        col_stats = compute_column_stats(df)

        # Convert all data to records
        data_records = df_to_records(df)

        table_row = {
            "type": "table_data",
            "table_name": table_name,
            "metadata_fold": "full",
            "primary_key_col": table.pkey_col,
            "time_col": table.time_col,
            "foreign_keys": dict(table.fkey_col_to_pkey_table),
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns),
            "column_dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_rates": {col: safe_json_value(df[col].isna().mean()) for col in df.columns},
            "column_stats": col_stats,
            "data": data_records,
        }
        output_rows.append(table_row)
        logger.info(f"  → Extracted {len(data_records)} records for '{table_name}'")

        # Free memory
        del data_records, col_stats
        gc.collect()

    # ── Step 3: Compute FK join metadata ─────────────────────────────────
    logger.info("Computing FK join statistics...")
    fk_rows = []
    for table_name, table in db.table_dict.items():
        for fk_col, target_table_name in table.fkey_col_to_pkey_table.items():
            parent_table = db.table_dict[target_table_name]
            parent_pkey = parent_table.pkey_col

            fk_stats = compute_fk_join_stats(
                child_table_name=table_name,
                child_df=table.df,
                fk_col=fk_col,
                parent_table_name=target_table_name,
                parent_pkey_col=parent_pkey,
                parent_df=parent_table.df,
            )
            fk_rows.append(fk_stats)
            logger.info(f"  FK: {table_name}.{fk_col} → {target_table_name}.{parent_pkey} "
                        f"(edges={fk_stats['num_edges']}, coverage={fk_stats['join_coverage']:.3f}, "
                        f"fanout mean={fk_stats['fanout_mean']:.1f})")

    output_rows.extend(fk_rows)
    logger.info(f"Computed {len(fk_rows)} FK join metadata entries")

    # ── Step 4: Extract predictive tasks ─────────────────────────────────
    logger.info("Extracting predictive tasks with splits...")
    from relbench.tasks import get_task

    task_names = ["driver-dnf", "driver-top3", "driver-position",
                  "driver-circuit-compete", "results-position", "qualifying-position"]

    for task_name in task_names:
        full_task_name = f"rel-f1/{task_name}"
        logger.info(f"  Loading task: {full_task_name}")
        try:
            # API: get_task(dataset_name, task_name, download)
            task = get_task("rel-f1", task_name, download=False)
        except Exception:
            logger.exception(f"  Failed to load task {full_task_name}")
            continue

        # Get task metadata
        task_type_str = str(task.task_type).split(".")[-1].lower()
        task_meta = {
            "task_name": full_task_name,
            "task_type": task_type_str,
            "timedelta_days": task.timedelta.days,
            "num_eval_timestamps": task.num_eval_timestamps,
        }

        # Entity-level attributes
        for attr in ["entity_table", "entity_col", "target_col", "time_col"]:
            if hasattr(task, attr):
                task_meta[attr] = getattr(task, attr)

        logger.info(f"  Task meta: {task_meta}")

        # Extract splits
        for split_name in ["train", "val", "test"]:
            try:
                split_table = task.get_table(split_name)
                split_df = split_table.df
                logger.info(f"    Split '{split_name}': {len(split_df)} rows x {len(split_df.columns)} cols")

                # Get target column stats
                target_col = task_meta.get("target_col")
                split_info: Dict[str, Any] = {
                    "type": "task_split",
                    "metadata_fold": split_name,
                    "task_name": full_task_name,
                    "task_type": task_type_str,
                    "num_samples": len(split_df),
                    "columns": list(split_df.columns),
                }

                # Add task-specific info
                for attr in ["entity_table", "entity_col", "target_col", "time_col"]:
                    if attr in task_meta:
                        split_info[attr] = task_meta[attr]

                split_info["timedelta_days"] = task_meta["timedelta_days"]
                split_info["num_eval_timestamps"] = task_meta["num_eval_timestamps"]

                # Compute label statistics
                if target_col and target_col in split_df.columns:
                    target_series = split_df[target_col]
                    if task_type_str == "binary_classification":
                        split_info["positive_rate"] = safe_json_value(target_series.mean())
                        split_info["label_counts"] = {
                            str(k): int(v) for k, v in target_series.value_counts().items()
                        }
                    elif task_type_str == "regression":
                        desc = target_series.describe()
                        split_info["target_stats"] = {
                            k: safe_json_value(v) for k, v in desc.items()
                        }

                # Temporal info
                time_col_name = task_meta.get("time_col")
                if time_col_name and time_col_name in split_df.columns:
                    split_info["timestamp_min"] = safe_json_value(split_df[time_col_name].min())
                    split_info["timestamp_max"] = safe_json_value(split_df[time_col_name].max())

                # Store actual data rows
                split_info["data"] = df_to_records(split_df)

                output_rows.append(split_info)
                logger.info(f"    → Extracted {len(split_df)} task split records")

                del split_df
                gc.collect()

            except Exception:
                logger.exception(f"    Failed to get split '{split_name}' for {full_task_name}")
                continue

    # ── Step 5: Add dataset-level metadata ───────────────────────────────
    dataset_meta = {
        "type": "dataset_metadata",
        "metadata_fold": "full",
        "dataset_name": "rel-f1",
        "source": "RelBench (Stanford, NeurIPS 2024)",
        "license": "CC-BY-4.0",
        "paper": "arXiv:2407.20060",
        "url": "https://relbench.stanford.edu/datasets/rel-f1/",
        "val_timestamp": dataset.val_timestamp.isoformat(),
        "test_timestamp": dataset.test_timestamp.isoformat(),
        "num_tables": len(db.table_dict),
        "table_names": list(db.table_dict.keys()),
        "table_summary": {
            name: {
                "num_rows": info["num_rows"],
                "num_cols": info["num_cols"],
                "pkey": info["pkey"],
                "time_col": info["time_col"],
                "fkeys": info["fkeys"],
            }
            for name, info in table_summary.items()
        },
        "num_fk_relationships": len(fk_rows),
        "task_names": task_names,
        "total_rows_across_tables": sum(info["num_rows"] for info in table_summary.values()),
    }
    output_rows.insert(0, dataset_meta)

    # ── Step 6: Write output ─────────────────────────────────────────────
    output_path = Path("data_out.json")
    logger.info(f"Writing output to {output_path}...")
    output_path.write_text(json.dumps(output_rows, indent=2, default=str))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")
    logger.info(f"Total rows in output: {len(output_rows)}")

    elapsed = time.time() - tic_total
    logger.info(f"Total extraction time: {elapsed:.1f}s")

    # Summary
    type_counts = {}
    for row in output_rows:
        t = row.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"Output row types: {type_counts}")


if __name__ == "__main__":
    main()
