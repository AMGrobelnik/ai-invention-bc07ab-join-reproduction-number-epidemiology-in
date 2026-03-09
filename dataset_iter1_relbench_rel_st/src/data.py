#!/usr/bin/env python3
"""Extract RelBench rel-stack dataset for JRN estimation experiments.

Loads the RelBench rel-stack dataset (Stack Exchange Q&A, 7 tables),
computes comprehensive per-join statistics (fan-out distributions, coverage,
cardinality ratios) on the FULL dataset, documents multi-hop join chains
with cumulative cardinality, extracts task definitions, and outputs a
subsampled version in exp_sel_data_out.json schema format.

Also loads rel-f1 (Formula 1) as a secondary candidate for comparison.
"""

import gc
import json
import math
import os
import resource
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

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
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.7 * 1e9)  # 70% of container limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")


# === HELPERS ===
def compute_histogram(values: pd.Series, bins: int = 10) -> list:
    """Compute histogram as list of [bin_edge, count] pairs."""
    try:
        arr = values.values
        counts, edges = np.histogram(arr, bins=bins)
        return [[float(edges[i]), int(counts[i])] for i in range(len(counts))]
    except Exception:
        return []


def make_json_safe(val):
    """Convert a value to JSON-serializable type."""
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
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, (str, int, float, bool)):
        return val
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val)


def compute_join_stats(db) -> dict:
    """Compute per-FK-join statistics on the full dataset."""
    join_stats = {}
    for table_name, table in db.table_dict.items():
        fk_map = table.fkey_col_to_pkey_table
        if not fk_map:
            continue
        for fk_col, pk_table_name in fk_map.items():
            pk_table = db.table_dict[pk_table_name]
            pk_col = pk_table.pkey_col
            child_df = table.df
            parent_df = pk_table.df

            join_key = f"{table_name}.{fk_col} -> {pk_table_name}.{pk_col}"
            logger.info(f"  Computing join stats: {join_key}")

            # Fan-out: how many child rows reference each parent PK
            fk_values = child_df[fk_col].dropna()
            fan_out = fk_values.value_counts()

            # Coverage
            all_parent_pks = set(parent_df[pk_col].dropna())
            referenced_pks = set(fk_values.unique())
            parent_coverage = len(referenced_pks & all_parent_pks) / max(len(all_parent_pks), 1)

            # Fan-out stats (only for referenced parents)
            if len(fan_out) > 0:
                fan_out_stats = {
                    "mean": float(fan_out.mean()),
                    "median": float(fan_out.median()),
                    "std": float(fan_out.std()) if len(fan_out) > 1 else 0.0,
                    "min": int(fan_out.min()),
                    "max": int(fan_out.max()),
                    "p25": float(fan_out.quantile(0.25)),
                    "p75": float(fan_out.quantile(0.75)),
                    "p95": float(fan_out.quantile(0.95)),
                    "p99": float(fan_out.quantile(0.99)),
                }
            else:
                fan_out_stats = {
                    "mean": 0.0, "median": 0.0, "std": 0.0,
                    "min": 0, "max": 0,
                    "p25": 0.0, "p75": 0.0, "p95": 0.0, "p99": 0.0,
                }

            n_zero_parents = len(all_parent_pks - referenced_pks)
            fan_out_with_zeros_mean = float(fk_values.shape[0] / max(len(all_parent_pks), 1))

            join_stats[join_key] = {
                "child_table": table_name,
                "fk_col": fk_col,
                "parent_table": pk_table_name,
                "pk_col": pk_col,
                "child_rows_total": len(child_df),
                "parent_rows_total": len(parent_df),
                "fk_non_null_count": int(fk_values.shape[0]),
                "fk_null_count": int(child_df[fk_col].isna().sum()),
                "fk_null_ratio": float(child_df[fk_col].isna().mean()),
                "unique_fk_values": int(fk_values.nunique()),
                "unique_pk_values": int(parent_df[pk_col].nunique()),
                "parent_coverage": parent_coverage,
                "cardinality_ratio": float(fk_values.shape[0] / max(len(parent_df), 1)),
                "fan_out_stats": fan_out_stats,
                "fan_out_with_zeros_mean": fan_out_with_zeros_mean,
                "n_parents_with_zero_children": n_zero_parents,
                "fan_out_histogram": compute_histogram(fan_out, bins=10),
            }

            del fk_values, fan_out, all_parent_pks, referenced_pks
            gc.collect()

    return join_stats


def compute_multi_hop_chains(join_stats: dict) -> list:
    """Document multi-hop join chains with cumulative cardinality."""
    fan_out_lookup = {}
    for key, stats in join_stats.items():
        fan_out_lookup[(stats["child_table"], stats["fk_col"])] = stats["fan_out_stats"]["mean"]

    chains = [
        # 2-hop chains
        [("posts", "OwnerUserId", "users"), ("comments", "PostId", "posts")],
        [("posts", "OwnerUserId", "users"), ("votes", "PostId", "posts")],
        [("posts", "OwnerUserId", "users"), ("postLinks", "PostId", "posts")],
        [("posts", "OwnerUserId", "users"), ("postHistory", "PostId", "posts")],
        [("posts", "OwnerUserId", "users"), ("posts", "ParentId", "posts")],
        # 3-hop chains
        [("posts", "OwnerUserId", "users"), ("posts", "ParentId", "posts"), ("comments", "PostId", "posts")],
        [("posts", "OwnerUserId", "users"), ("posts", "ParentId", "posts"), ("votes", "PostId", "posts")],
        # Reverse chain
        [("comments", "PostId", "posts"), ("posts", "OwnerUserId", "users")],
    ]

    multi_hop = []
    for chain in chains:
        individual_fan_outs = []
        valid = True
        for child, fk, parent in chain:
            fo = fan_out_lookup.get((child, fk))
            if fo is None:
                logger.warning(f"  Missing fan-out for ({child}, {fk}), skipping chain")
                valid = False
                break
            individual_fan_outs.append(fo)

        if not valid:
            continue

        cumulative = reduce(lambda a, b: a * b, individual_fan_outs, 1.0)
        multi_hop.append({
            "hops": [f"{c}.{fk} -> {p}" for c, fk, p in chain],
            "depth": len(chain),
            "individual_fan_outs": individual_fan_outs,
            "cumulative_cardinality": cumulative,
        })

    return multi_hop


def load_tasks(dataset_name: str) -> dict:
    """Load task definitions for a RelBench dataset."""
    from relbench.tasks import get_task

    task_names = [
        "user-engagement", "user-badge", "post-votes",
        "user-post-comment", "post-post-related",
    ]

    task_metadata = {}
    for task_name in task_names:
        try:
            logger.info(f"  Loading task: {task_name}")
            task = get_task(dataset_name, task_name, download=True)

            info = {
                "task_type": str(task.task_type),
            }
            # EntityTask attributes
            for attr in ["entity_table", "entity_col", "target_col", "time_col"]:
                if hasattr(task, attr):
                    info[attr] = getattr(task, attr)
            # LinkTask attributes (link prediction tasks)
            for attr in ["src_entity_table", "src_entity_col",
                         "dst_entity_table", "dst_entity_col"]:
                if hasattr(task, attr):
                    info[attr] = getattr(task, attr)

            # timedelta
            if hasattr(task, "timedelta") and task.timedelta is not None:
                info["timedelta_days"] = task.timedelta.days
            # metrics
            if hasattr(task, "metrics"):
                info["metrics"] = [str(m) for m in task.metrics]

            # Get table splits
            for split_name in ["train", "val", "test"]:
                try:
                    t = task.get_table(split_name)
                    info[f"{split_name}_rows"] = len(t.df)
                    if (split_name == "train"
                            and hasattr(task, "target_col")
                            and task.target_col
                            and task.target_col in t.df.columns):
                        desc = t.df[task.target_col].describe()
                        info["train_label_distribution"] = {
                            k: make_json_safe(v) for k, v in desc.to_dict().items()
                        }
                except Exception as e:
                    info[f"{split_name}_rows"] = None
                    logger.warning(f"    Could not load {split_name} for {task_name}: {e}")

            task_metadata[task_name] = info
            logger.info(f"    OK: {info.get('task_type', '?')}")
        except Exception as e:
            logger.warning(f"  Could not load task {task_name}: {e}")
            task_metadata[task_name] = {"error": str(e)}

        gc.collect()

    return task_metadata


def subsample_tables(db, max_rows: int = 30000) -> dict:
    """Subsample tables using stratified temporal sampling."""
    subsampled = {}
    for table_name, table in db.table_dict.items():
        df = table.df
        if len(df) <= max_rows:
            subsampled[table_name] = df.copy()
            logger.info(f"  {table_name}: keeping all {len(df)} rows")
        else:
            if table.time_col and table.time_col in df.columns:
                time_series = pd.to_datetime(df[table.time_col], errors="coerce")
                years = time_series.dt.year
                n_years = years.nunique()
                per_year = max(1, max_rows // max(n_years, 1))

                indices = []
                for year_val, group_idx in df.groupby(years).groups.items():
                    n_sample = min(len(group_idx), per_year)
                    sampled_idx = np.random.RandomState(42).choice(
                        group_idx, size=n_sample, replace=False
                    )
                    indices.extend(sampled_idx)

                subsampled[table_name] = df.loc[indices].copy()
                logger.info(f"  {table_name}: temporal sampled {len(indices)} from {len(df)}")
            else:
                subsampled[table_name] = df.sample(max_rows, random_state=42).copy()
                logger.info(f"  {table_name}: random sampled {max_rows} from {len(df)}")

        gc.collect()
    return subsampled


def construct_rows(
    db, subsampled_tables: dict,
    val_ts_str: str, test_ts_str: str,
    truncate_text_cols: int = 500,
) -> list:
    """Construct example rows from subsampled tables.

    Each row becomes one example with:
    - input: JSON string of column values
    - output: "{}" (empty, relational metadata in metadata fields)
    - metadata_table, metadata_fold, metadata_pk, metadata_fkeys
    """
    val_ts = pd.Timestamp(val_ts_str)
    test_ts = pd.Timestamp(test_ts_str)

    rows = []
    for table_name, df in subsampled_tables.items():
        table = db.table_dict[table_name]
        time_col = table.time_col
        fk_map = table.fkey_col_to_pkey_table or {}

        logger.info(f"  Constructing rows for {table_name} ({len(df)} rows)")

        # Identify text columns to truncate
        text_cols = [c for c in df.columns if df[c].dtype == object]

        for row_idx, (idx, row) in enumerate(df.iterrows()):
            # Determine temporal fold
            fold = 0  # default: train
            if time_col and time_col in row.index:
                try:
                    ts = pd.to_datetime(row[time_col])
                    if pd.notna(ts):
                        if ts >= test_ts:
                            fold = 2
                        elif ts >= val_ts:
                            fold = 1
                except Exception:
                    pass

            # Build input dict
            input_dict = {}
            for col in df.columns:
                val = make_json_safe(row[col])
                # Truncate long text strings
                if isinstance(val, str) and len(val) > truncate_text_cols and col in text_cols:
                    val = val[:truncate_text_cols] + "..."
                input_dict[col] = val

            # Build FK dict
            fkeys = {}
            for fk_col in fk_map:
                fkeys[fk_col] = make_json_safe(row.get(fk_col))

            pk_val = make_json_safe(row.get(table.pkey_col)) if table.pkey_col else None

            rows.append({
                "input": json.dumps(input_dict, default=str),
                "output": "{}",
                "metadata_fold": fold,
                "metadata_table": table_name,
                "metadata_pk": pk_val,
                "metadata_fkeys": json.dumps(fkeys, default=str),
                "metadata_row_index": row_idx,
            })

    return rows


def process_relbench_dataset(
    dataset_name: str, max_rows_per_table: int = 30000,
    val_ts: str = "2019-01-01", test_ts: str = "2021-01-01",
    compute_stats: bool = True,
) -> tuple[list, dict]:
    """Process a relbench dataset and return (rows, metadata)."""
    from relbench.datasets import get_dataset

    logger.info(f"Loading dataset: {dataset_name}")
    dataset = get_dataset(dataset_name, download=True)
    db = dataset.get_db()

    total_rows = sum(len(t.df) for t in db.table_dict.values())
    total_cols = sum(len(t.df.columns) for t in db.table_dict.values())
    logger.info(f"  {len(db.table_dict)} tables, {total_rows} rows, {total_cols} cols")

    # Schema
    schema_info = {}
    for tname, table in db.table_dict.items():
        schema_info[tname] = {
            "pkey_col": table.pkey_col,
            "time_col": table.time_col,
            "fkey_col_to_pkey_table": dict(table.fkey_col_to_pkey_table) if table.fkey_col_to_pkey_table else {},
            "columns": list(table.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in table.df.dtypes.items()},
            "num_rows": len(table.df),
            "num_cols": len(table.df.columns),
        }
        logger.info(f"    {tname}: {len(table.df)} rows, {len(table.df.columns)} cols, PK={table.pkey_col}")

    metadata = {
        "total_rows_full": total_rows,
        "total_tables": len(db.table_dict),
        "total_columns": total_cols,
        "schema": schema_info,
    }

    # Join statistics
    if compute_stats:
        logger.info("  Computing join statistics on FULL data...")
        join_stats = compute_join_stats(db)
        logger.info(f"  Computed stats for {len(join_stats)} FK relationships")
        metadata["join_statistics"] = join_stats

        logger.info("  Computing multi-hop join chains...")
        multi_hop_chains = compute_multi_hop_chains(join_stats)
        logger.info(f"  Documented {len(multi_hop_chains)} multi-hop chains")
        metadata["multi_hop_chains"] = multi_hop_chains

    # Tasks (only for rel-stack)
    if dataset_name == "rel-stack":
        logger.info("  Loading task definitions...")
        task_metadata = load_tasks(dataset_name)
        logger.info(f"  Loaded {len(task_metadata)} tasks")
        metadata["tasks"] = task_metadata
        metadata["temporal_splits"] = {"val_timestamp": val_ts, "test_timestamp": test_ts}

    # Subsample
    logger.info(f"  Subsampling (max {max_rows_per_table} rows/table)...")
    subsampled = subsample_tables(db, max_rows=max_rows_per_table)
    total_sampled = sum(len(df) for df in subsampled.values())
    logger.info(f"  Subsampled: {total_sampled} rows")

    # Construct rows
    logger.info("  Constructing example rows...")
    rows = construct_rows(db, subsampled, val_ts, test_ts)
    logger.info(f"  Constructed {len(rows)} examples")

    del subsampled
    gc.collect()

    return rows, metadata


@logger.catch
def main():
    workspace = Path(__file__).parent
    os.chdir(workspace)

    # Check memory
    try:
        current_mem = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        logger.info(f"Current memory usage: {current_mem / 1e9:.2f} GB / {TOTAL_RAM_GB:.1f} GB")
    except (FileNotFoundError, ValueError):
        pass

    # ============================================================
    # DATASET 1: rel-stack (PRIMARY)
    # ============================================================
    logger.info("=" * 60)
    logger.info("DATASET 1: RelBench rel-stack (Stack Exchange)")
    logger.info("=" * 60)

    rows_stack, meta_stack = process_relbench_dataset(
        "rel-stack",
        max_rows_per_table=30000,
        val_ts="2019-01-01",
        test_ts="2021-01-01",
        compute_stats=True,
    )

    gc.collect()

    # Check memory after loading
    try:
        current_mem = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        logger.info(f"Memory after rel-stack: {current_mem / 1e9:.2f} GB / {TOTAL_RAM_GB:.1f} GB")
    except (FileNotFoundError, ValueError):
        pass

    gc.collect()

    # ============================================================
    # BUILD OUTPUT (rel-stack only)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Building output JSON...")
    logger.info("=" * 60)

    datasets_list = [
        {"dataset": "rel-stack", "examples": rows_stack},
    ]

    output = {
        "metadata": {
            "description": "RelBench rel-stack dataset for JRN estimation experiments",
            "source": "RelBench (Stanford) - Stack Exchange stats-exchange",
            "license": "CC BY-SA 4.0",
            "rel_stack": meta_stack,
        },
        "datasets": datasets_list,
    }

    # Save
    out_path = workspace / "full_data_out.json"
    logger.info(f"Saving to {out_path}...")
    out_path.write_text(json.dumps(output, default=str, ensure_ascii=False))

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")

    if file_size_mb > 300:
        logger.warning("File exceeds 300MB! Need to reduce sample size or truncate more.")

    # Validation summary
    logger.info("--- Validation Summary ---")
    for ds in datasets_list:
        tables = set()
        for ex in ds["examples"]:
            tables.add(ex.get("metadata_table", "unknown"))
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples, tables: {sorted(tables)}")

    if "join_statistics" in meta_stack:
        logger.info(f"  FK relationships with stats: {len(meta_stack['join_statistics'])}")
    if "multi_hop_chains" in meta_stack:
        logger.info(f"  Multi-hop chains: {len(meta_stack['multi_hop_chains'])}")
    if "tasks" in meta_stack:
        logger.info(f"  Tasks: {list(meta_stack['tasks'].keys())}")

    logger.info("DONE!")


if __name__ == "__main__":
    main()
