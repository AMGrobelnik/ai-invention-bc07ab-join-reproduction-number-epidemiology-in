#!/usr/bin/env python3
"""Extract RelBench rel-avito (Avito ad marketplace) relational database for JRN experiments.

Extracts all 8 tables with subsampled row data (~200K rows), FK join metadata
with full-data statistics, and all 4 predictive tasks with temporal splits.
Outputs structured data_out.json matching the iter-1 rel-f1 format.
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
# rel-avito is large (~20.7M rows). Budget 22GB for safety on 29GB container.
RAM_BUDGET = int(22 * 1024**3)  # 22 GB
_avail = psutil.virtual_memory().available
if RAM_BUDGET > _avail:
    RAM_BUDGET = int(_avail * 0.85)
    logger.warning(f"Reduced RAM budget to {RAM_BUDGET/1e9:.1f}GB (available: {_avail/1e9:.1f}GB)")

try:
    resource.setrlimit(resource.RLIMIT_DATA, (RAM_BUDGET, RAM_BUDGET))
except (ValueError, OSError):
    logger.warning("Could not set RLIMIT_DATA, continuing without memory limit")

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
OUTPUT_FILE = WORKSPACE / "data_out.json"
CACHE_DIR = WORKSPACE / "temp" / "relbench_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set relbench cache dir to workspace
os.environ["RELBENCH_CACHE_DIR"] = str(CACHE_DIR)

# Subsampling targets
TARGET_TOTAL_ROWS = 200_000
MAX_TASK_SPLIT_ROWS = 50_000

# Small/lookup tables to include in full
SMALL_TABLE_THRESHOLD = 50_000


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


def df_to_records_fast(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of dicts, faster than iterrows for large DataFrames."""
    # Convert problematic types first
    result_df = df.copy()
    for col in result_df.columns:
        if pd.api.types.is_datetime64_any_dtype(result_df[col]):
            result_df[col] = result_df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        elif result_df[col].dtype == 'object':
            result_df[col] = result_df[col].where(result_df[col].notna(), None)
        elif pd.api.types.is_numeric_dtype(result_df[col]):
            # Handle NaN in numeric columns
            if result_df[col].isna().any():
                result_df[col] = result_df[col].where(result_df[col].notna(), None)

    records = result_df.to_dict('records')

    # Clean up numpy types
    cleaned = []
    for rec in records:
        clean_rec = {}
        for k, v in rec.items():
            clean_rec[k] = safe_json_value(v)
        cleaned.append(clean_rec)
    return cleaned


def compute_column_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute per-column statistics on full data."""
    stats = {}
    for col in df.columns:
        col_info: Dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "missing_rate": float(df[col].isna().mean()),
            "num_unique": int(df[col].nunique()),
        }
        if df[col].dtype in ["object", "category"]:
            col_info["type"] = "categorical"
            try:
                vc = df[col].value_counts()
                col_info["value_counts_top10"] = {
                    str(k): int(v)
                    for k, v in vc.head(10).items()
                }
            except Exception:
                col_info["value_counts_top10"] = {}
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_info["type"] = "numeric"
            try:
                desc = df[col].describe()
                for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                    if stat_name in desc.index:
                        col_info[stat_name.replace("%", "pct")] = safe_json_value(desc[stat_name])
            except Exception:
                pass
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["type"] = "datetime"
            try:
                col_info["min"] = safe_json_value(df[col].min())
                col_info["max"] = safe_json_value(df[col].max())
            except Exception:
                pass
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
    """Compute FK join statistics between child and parent tables on FULL data."""
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


def subsample_tables(
    db_table_dict: dict,
    target_total: int = TARGET_TOTAL_ROWS,
    small_threshold: int = SMALL_TABLE_THRESHOLD,
) -> Dict[str, pd.DataFrame]:
    """Subsample tables proportionally, keeping small tables in full.

    Returns dict mapping table_name → subsampled DataFrame.
    """
    table_sizes = {name: len(tbl.df) for name, tbl in db_table_dict.items()}
    total_rows = sum(table_sizes.values())
    logger.info(f"Total rows across all tables: {total_rows:,}")

    # Identify small tables (include in full)
    small_tables = {name: size for name, size in table_sizes.items() if size <= small_threshold}
    large_tables = {name: size for name, size in table_sizes.items() if size > small_threshold}

    small_total = sum(small_tables.values())
    remaining_budget = target_total - small_total

    if remaining_budget <= 0:
        # Even small tables exceed budget; subsample everything proportionally
        logger.warning(f"Small tables ({small_total:,}) exceed budget ({target_total:,})")
        remaining_budget = target_total
        # Treat all as large
        large_tables.update(small_tables)
        small_tables = {}

    logger.info(f"Small tables (full): {list(small_tables.keys())} ({small_total:,} rows)")
    logger.info(f"Large tables (subsample): {list(large_tables.keys())} (budget: {remaining_budget:,} rows)")

    large_total = sum(large_tables.values())
    result = {}

    # Small tables: include in full
    for name in small_tables:
        result[name] = db_table_dict[name].df
        logger.info(f"  {name}: {len(result[name]):,} rows (full)")

    # Large tables: subsample proportionally
    for name, size in large_tables.items():
        fraction = size / large_total
        sample_size = max(1000, int(remaining_budget * fraction))
        sample_size = min(sample_size, size)
        df = db_table_dict[name].df
        if sample_size < size:
            result[name] = df.sample(n=sample_size, random_state=42)
        else:
            result[name] = df
        logger.info(f"  {name}: {len(result[name]):,} rows (from {size:,}, {len(result[name])/size*100:.1f}%)")

    total_sampled = sum(len(df) for df in result.values())
    logger.info(f"Total subsampled rows: {total_sampled:,} (target: {target_total:,})")
    return result


# ── Main extraction ──────────────────────────────────────────────────────────

@logger.catch
def main():
    tic_total = time.time()

    # ── Step 1: Load rel-avito database ───────────────────────────────────
    logger.info("Loading rel-avito dataset via relbench API...")
    logger.info(f"Cache dir: {CACHE_DIR}")
    from relbench.datasets import get_dataset

    dataset = get_dataset("rel-avito", download=True)
    db = dataset.get_db()
    logger.info(f"Loaded database with {len(db.table_dict)} tables")
    logger.info(f"Val timestamp: {dataset.val_timestamp}, Test timestamp: {dataset.test_timestamp}")

    # Log all tables
    table_summary = {}
    for table_name, table in db.table_dict.items():
        df = table.df
        logger.info(f"  Table '{table_name}': {len(df):,} rows x {len(df.columns)} cols, "
                     f"pkey={table.pkey_col}, time_col={table.time_col}, "
                     f"fkeys={table.fkey_col_to_pkey_table}")
        table_summary[table_name] = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "pkey": table.pkey_col,
            "time_col": table.time_col,
            "fkeys": dict(table.fkey_col_to_pkey_table),
        }

    mem_used = psutil.virtual_memory().used / 1e9
    logger.info(f"Memory after loading: {mem_used:.1f}GB used")

    # ── Step 2: Compute FK join statistics on FULL data (critical!) ───────
    logger.info("=" * 60)
    logger.info("Computing FK join statistics on FULL data...")
    fk_rows = []
    for table_name, table in db.table_dict.items():
        for fk_col, target_table_name in table.fkey_col_to_pkey_table.items():
            parent_table = db.table_dict[target_table_name]
            parent_pkey = parent_table.pkey_col

            logger.info(f"  Computing FK: {table_name}.{fk_col} → {target_table_name}.{parent_pkey} ...")
            tic = time.time()

            fk_stats = compute_fk_join_stats(
                child_table_name=table_name,
                child_df=table.df,
                fk_col=fk_col,
                parent_table_name=target_table_name,
                parent_pkey_col=parent_pkey,
                parent_df=parent_table.df,
            )
            elapsed = time.time() - tic
            fk_rows.append(fk_stats)
            logger.info(f"    edges={fk_stats['num_edges']:,}, coverage={fk_stats['join_coverage']:.3f}, "
                        f"fanout mean={fk_stats['fanout_mean']:.1f}, max={fk_stats['fanout_max']:,}, "
                        f"({elapsed:.1f}s)")
            gc.collect()

    logger.info(f"Computed {len(fk_rows)} FK join metadata entries")

    # ── Step 3: Compute column stats on FULL data ────────────────────────
    logger.info("=" * 60)
    logger.info("Computing per-table column statistics on FULL data...")
    all_column_stats = {}
    all_column_dtypes = {}
    all_missing_rates = {}

    for table_name, table in db.table_dict.items():
        logger.info(f"  Computing stats for '{table_name}' ({len(table.df):,} rows)...")
        tic = time.time()
        col_stats = compute_column_stats(table.df)
        all_column_stats[table_name] = col_stats
        all_column_dtypes[table_name] = {col: str(table.df[col].dtype) for col in table.df.columns}
        all_missing_rates[table_name] = {
            col: safe_json_value(table.df[col].isna().mean()) for col in table.df.columns
        }
        elapsed = time.time() - tic
        logger.info(f"    → {len(col_stats)} columns, {elapsed:.1f}s")
        gc.collect()

    mem_used = psutil.virtual_memory().used / 1e9
    logger.info(f"Memory after stats: {mem_used:.1f}GB used")

    # ── Step 4: Subsample tables ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Subsampling tables for row data...")
    subsampled = subsample_tables(db.table_dict, target_total=TARGET_TOTAL_ROWS)

    # ── Step 5: Build table_data records ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("Building table_data records with subsampled row data...")
    output_rows: List[Dict] = []

    for table_name, table in db.table_dict.items():
        sub_df = subsampled[table_name]
        full_size = len(table.df)
        sub_size = len(sub_df)
        is_subsampled = sub_size < full_size

        logger.info(f"  Converting '{table_name}': {sub_size:,} rows (from {full_size:,})...")
        tic = time.time()

        # Use fast conversion for subsampled data
        data_records = df_to_records_fast(sub_df)

        table_row = {
            "type": "table_data",
            "table_name": table_name,
            "metadata_fold": "full",
            "primary_key_col": table.pkey_col,
            "time_col": table.time_col,
            "foreign_keys": dict(table.fkey_col_to_pkey_table),
            "num_rows": full_size,  # Report FULL size
            "num_cols": len(table.df.columns),
            "columns": list(table.df.columns),
            "column_dtypes": all_column_dtypes[table_name],
            "missing_rates": all_missing_rates[table_name],
            "column_stats": all_column_stats[table_name],
            "subsampled": is_subsampled,
            "subsample_size": sub_size,
            "full_size": full_size,
            "data": data_records,
        }
        output_rows.append(table_row)
        elapsed = time.time() - tic
        logger.info(f"    → {len(data_records)} records, {elapsed:.1f}s")

        del data_records, sub_df
        gc.collect()

    # Free subsampled data
    del subsampled
    gc.collect()

    # Add FK join metadata
    output_rows.extend(fk_rows)

    mem_used = psutil.virtual_memory().used / 1e9
    logger.info(f"Memory after table records: {mem_used:.1f}GB used")

    # ── Step 6: Extract predictive tasks ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("Extracting predictive tasks with splits...")
    from relbench.tasks import get_task

    task_configs = {
        "ad-ctr": {"task_type": "regression", "metric": "MAE"},
        "user-clicks": {"task_type": "binary_classification", "metric": "AUROC"},
        "user-visits": {"task_type": "binary_classification", "metric": "AUROC"},
        "user-ad-visit": {"task_type": "link_prediction", "metric": "MAP"},
    }

    loaded_task_names = []
    for task_name, config in task_configs.items():
        full_task_name = f"rel-avito/{task_name}"
        logger.info(f"  Loading task: {full_task_name}")
        try:
            task = get_task("rel-avito", task_name, download=True)
        except Exception:
            logger.exception(f"  Failed to load task {full_task_name}")
            continue

        loaded_task_names.append(task_name)

        # Get task metadata
        try:
            task_type_str = str(task.task_type).split(".")[-1].lower()
        except Exception:
            task_type_str = config["task_type"]

        task_meta = {
            "task_name": full_task_name,
            "task_type": task_type_str,
        }

        # Entity-level attributes
        for attr in ["entity_table", "entity_col", "target_col", "time_col",
                      "timedelta", "num_eval_timestamps"]:
            if hasattr(task, attr):
                val = getattr(task, attr)
                if attr == "timedelta":
                    task_meta["timedelta_days"] = val.days if hasattr(val, 'days') else int(val)
                else:
                    task_meta[attr] = val

        logger.info(f"  Task meta: {task_meta}")

        # Extract splits
        for split_name in ["train", "val", "test"]:
            try:
                split_table = task.get_table(split_name)
                split_df = split_table.df
                full_split_size = len(split_df)
                logger.info(f"    Split '{split_name}': {full_split_size:,} rows x {len(split_df.columns)} cols")

                target_col = task_meta.get("target_col")

                split_info: Dict[str, Any] = {
                    "type": "task_split",
                    "metadata_fold": split_name,
                    "task_name": full_task_name,
                    "task_type": task_type_str,
                    "num_samples": full_split_size,
                    "columns": list(split_df.columns),
                }

                # Add task-specific info
                for attr in ["entity_table", "entity_col", "target_col", "time_col"]:
                    if attr in task_meta:
                        split_info[attr] = task_meta[attr]

                if "timedelta_days" in task_meta:
                    split_info["timedelta_days"] = task_meta["timedelta_days"]
                if "num_eval_timestamps" in task_meta:
                    split_info["num_eval_timestamps"] = task_meta["num_eval_timestamps"]

                # Compute label statistics on FULL split
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

                # Subsample large splits
                if full_split_size > MAX_TASK_SPLIT_ROWS:
                    logger.info(f"    Subsampling split from {full_split_size:,} to {MAX_TASK_SPLIT_ROWS:,}")
                    if task_type_str == "binary_classification" and target_col and target_col in split_df.columns:
                        # Stratified subsample
                        from sklearn.model_selection import train_test_split
                        try:
                            _, sub_df = train_test_split(
                                split_df,
                                test_size=MAX_TASK_SPLIT_ROWS,
                                stratify=split_df[target_col],
                                random_state=42,
                            )
                            split_df = sub_df
                        except ValueError:
                            split_df = split_df.sample(n=MAX_TASK_SPLIT_ROWS, random_state=42)
                    else:
                        split_df = split_df.sample(n=MAX_TASK_SPLIT_ROWS, random_state=42)
                    split_info["subsampled"] = True
                    split_info["subsample_size"] = len(split_df)
                    split_info["full_size"] = full_split_size

                # Store actual data rows
                split_info["data"] = df_to_records_fast(split_df)

                output_rows.append(split_info)
                logger.info(f"    → Extracted {len(split_df):,} task split records")

                del split_df
                gc.collect()

            except Exception:
                logger.exception(f"    Failed to get split '{split_name}' for {full_task_name}")
                continue

    # ── Step 7: Add dataset-level metadata (insert at beginning) ─────────
    logger.info("=" * 60)
    logger.info("Building dataset metadata...")
    dataset_meta = {
        "type": "dataset_metadata",
        "metadata_fold": "full",
        "dataset_name": "rel-avito",
        "source": "RelBench (Stanford, NeurIPS 2024)",
        "license": "CC-BY-4.0",
        "paper": "arXiv:2407.20060",
        "url": "https://relbench.stanford.edu/databases/avito/",
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
        "task_names": loaded_task_names,
        "total_rows_across_tables": sum(info["num_rows"] for info in table_summary.values()),
    }
    output_rows.insert(0, dataset_meta)

    # ── Step 8: Write output (split into <100MB parts) ─────────────────
    logger.info("=" * 60)
    mid = len(output_rows) // 2
    parts = [output_rows[:mid], output_rows[mid:]]
    for idx, part in enumerate(parts, 1):
        part_path = Path(f"data_out_{idx}.json")
        logger.info(f"Writing {len(part)} records to {part_path}...")
        part_path.write_text(json.dumps(part, indent=2, default=str))
        size_mb = part_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Part {idx}: {size_mb:.1f} MB")

    logger.info(f"Total records in output: {len(output_rows)}")

    elapsed = time.time() - tic_total
    logger.info(f"Total extraction time: {elapsed:.1f}s")

    # Summary
    type_counts = {}
    for row in output_rows:
        t = row.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"Output record types: {type_counts}")

    # Validate output
    assert type_counts.get("dataset_metadata", 0) == 1, "Missing dataset_metadata"
    assert type_counts.get("table_data", 0) >= 1, "Missing table_data"
    assert type_counts.get("fk_join_metadata", 0) >= 1, "Missing fk_join_metadata"
    assert type_counts.get("task_split", 0) >= 1, "Missing task_split"
    logger.info("✓ All required record types present")

    if file_size_mb > 300:
        logger.warning(f"Output file is {file_size_mb:.0f}MB — exceeds 300MB limit!")
    else:
        logger.info(f"✓ Output file size OK ({file_size_mb:.0f}MB < 300MB)")


if __name__ == "__main__":
    main()
