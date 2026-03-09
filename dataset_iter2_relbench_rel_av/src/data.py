#!/usr/bin/env python3
"""Transform RelBench rel-avito data_out.json into exp_sel_data_out.json schema.

Creates a single unified dataset 'rel-avito' containing all table rows,
FK join metadata, and prediction task samples. Each row/sample is an
individual example with input/output strings and metadata_* fields.
"""

import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
INPUT_FILES = sorted(WORKSPACE.glob("data_out_*.json"))  # split parts
OUTPUT_FILE = WORKSPACE / "full_data_out.json"
SPLIT_DIR = WORKSPACE / "data_out"


def safe_str(val: Any) -> str:
    """Convert value to string, handling None/NaN."""
    if val is None:
        return "null"
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "null"
    return str(val)


@logger.catch
def main():
    raw_data = []
    for f in INPUT_FILES:
        logger.info(f"Loading data from {f}")
        raw_data.extend(json.loads(f.read_text()))
    logger.info(f"Loaded {len(raw_data)} top-level items")

    # Separate by type
    metadata_item = None
    table_items: List[Dict] = []
    fk_items: List[Dict] = []
    task_items: List[Dict] = []

    for item in raw_data:
        t = item.get("type")
        if t == "dataset_metadata":
            metadata_item = item
        elif t == "table_data":
            table_items.append(item)
        elif t == "fk_join_metadata":
            fk_items.append(item)
        elif t == "task_split":
            task_items.append(item)

    logger.info(f"Found: {len(table_items)} tables, {len(fk_items)} FK relations, "
                f"{len(task_items)} task splits")

    # ── Single unified dataset: rel-avito ─────────────────────────────────
    all_examples: List[Dict[str, Any]] = []

    # 1) Table rows → examples
    for tbl in table_items:
        table_name = tbl["table_name"]
        pkey_col = tbl["primary_key_col"]
        time_col = tbl.get("time_col")
        foreign_keys = tbl.get("foreign_keys", {})
        columns = tbl.get("columns", [])
        data_rows = tbl.get("data", [])

        logger.info(f"  Table '{table_name}': {len(data_rows)} rows")

        for row_idx, row in enumerate(data_rows):
            input_features = {k: v for k, v in row.items() if k != pkey_col}
            pkey_val = row.get(pkey_col, row_idx)

            example: Dict[str, Any] = {
                "input": json.dumps(input_features, default=str),
                "output": f"{table_name}/{safe_str(pkey_val)}",
                "metadata_row_type": "table_row",
                "metadata_table_name": table_name,
                "metadata_primary_key_col": pkey_col,
                "metadata_primary_key_value": safe_str(pkey_val),
                "metadata_row_index": row_idx,
                "metadata_num_cols": len(columns),
                "metadata_feature_names": [c for c in columns if c != pkey_col],
            }
            if time_col:
                example["metadata_time_col"] = time_col
            if foreign_keys:
                example["metadata_foreign_keys_json"] = json.dumps(foreign_keys)

            all_examples.append(example)

    logger.info(f"  → {len(all_examples)} table-row examples so far")

    # 2) FK join metadata → examples
    for fk_idx, fk in enumerate(fk_items):
        input_data = {
            "source_table": fk["source_table"],
            "source_fk_col": fk["source_fk_col"],
            "target_table": fk["target_table"],
            "target_pk_col": fk["target_pk_col"],
        }
        output_data = {
            "num_edges": fk["num_edges"],
            "join_coverage": fk["join_coverage"],
            "cardinality_ratio": fk["cardinality_ratio"],
            "fanout_min": fk["fanout_min"],
            "fanout_mean": fk["fanout_mean"],
            "fanout_median": fk["fanout_median"],
            "fanout_max": fk["fanout_max"],
            "fanout_std": fk["fanout_std"],
            "fanout_p25": fk["fanout_p25"],
            "fanout_p75": fk["fanout_p75"],
            "fanout_p95": fk["fanout_p95"],
        }
        all_examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "metadata_row_type": "fk_join_metadata",
            "metadata_row_index": fk_idx,
            "metadata_source_table": fk["source_table"],
            "metadata_target_table": fk["target_table"],
            "metadata_source_fk_col": fk["source_fk_col"],
            "metadata_num_edges": fk["num_edges"],
            "metadata_join_coverage": fk["join_coverage"],
        })

    logger.info(f"  → {len(all_examples)} examples after FK metadata")

    # 3) Task split samples → examples
    for task_item in task_items:
        task_name = task_item["task_name"]
        fold = task_item["metadata_fold"]
        task_type = task_item["task_type"]
        target_col = task_item.get("target_col")
        entity_col = task_item.get("entity_col")
        entity_table = task_item.get("entity_table")
        data_rows = task_item.get("data", [])

        logger.info(f"  Task '{task_name}/{fold}': {len(data_rows)} samples")

        fold_int = {"train": 0, "val": 1, "test": 2}.get(fold, -1)

        for row_idx, row in enumerate(data_rows):
            if target_col and target_col in row:
                input_features = {k: v for k, v in row.items() if k != target_col}
                output_val = safe_str(row[target_col])
            else:
                input_features = dict(row)
                output_val = "masked"

            example: Dict[str, Any] = {
                "input": json.dumps(input_features, default=str),
                "output": output_val,
                "metadata_row_type": "task_sample",
                "metadata_fold": fold_int,
                "metadata_fold_name": fold,
                "metadata_task_name": task_name,
                "metadata_task_type": task_type,
                "metadata_row_index": row_idx,
            }
            if target_col:
                example["metadata_target_col"] = target_col
            if entity_col:
                example["metadata_entity_col"] = entity_col
            if entity_table:
                example["metadata_entity_table"] = entity_table
            if task_type == "binary_classification":
                example["metadata_n_classes"] = 2
                if "positive_rate" in task_item and task_item["positive_rate"] is not None:
                    example["metadata_positive_rate"] = task_item["positive_rate"]

            all_examples.append(example)

    logger.info(f"  → {len(all_examples)} total examples")

    # ── Build output ─────────────────────────────────────────────────────
    output = {
        "metadata": {
            "source": "RelBench rel-avito (Avito ad marketplace) relational database",
            "paper": "arXiv:2407.20060",
            "license": "CC-BY-4.0",
            "relbench_version": "1.0.0",
            "val_timestamp": metadata_item["val_timestamp"] if metadata_item else "",
            "test_timestamp": metadata_item["test_timestamp"] if metadata_item else "",
            "num_tables": len(table_items),
            "num_fk_relationships": len(fk_items),
            "num_tasks": len(set(t["task_name"] for t in task_items)),
            "table_names": [t["table_name"] for t in table_items],
            "task_names": sorted(set(t["task_name"] for t in task_items)),
        },
        "datasets": [
            {
                "dataset": "rel-avito",
                "examples": all_examples,
            },
        ],
    }

    # ── Write output (split into <100MB parts) ─────────────────────────
    CHUNK_SIZE = 112_000  # ~85MB per part

    logger.info(f"Total examples: {len(all_examples)}")

    SPLIT_DIR.mkdir(exist_ok=True)

    parts = []
    for i in range(0, len(all_examples), CHUNK_SIZE):
        parts.append(all_examples[i:i + CHUNK_SIZE])

    logger.info(f"Splitting into {len(parts)} parts in {SPLIT_DIR}")

    for idx, chunk in enumerate(parts, 1):
        part_data = {
            "metadata": output["metadata"],
            "datasets": [{"dataset": "rel-avito", "examples": chunk}],
        }
        part_path = SPLIT_DIR / f"full_data_out_{idx}.json"
        part_path.write_text(json.dumps(part_data, indent=2, default=str))
        size_mb = part_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Part {idx}: {len(chunk)} examples, {size_mb:.1f} MB")

    # Row-type breakdown
    type_counts: Dict[str, int] = {}
    for ex in all_examples:
        rt = ex.get("metadata_row_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1
    logger.info(f"Row-type breakdown: {type_counts}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
