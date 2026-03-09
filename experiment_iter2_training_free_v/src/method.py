#!/usr/bin/env python3
"""Training-Free vs Probe-Based JRN: Diagnostic Comparison on rel-f1.

Compares 5 training-free statistical proxies against probe-based JRN
for predicting join utility across all 13 FK joins in the rel-f1 schema.
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

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

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


def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 16.0
RAM_BUDGET_GB = min(0.7 * TOTAL_RAM_GB, 20)
RAM_BUDGET_BYTES = int(RAM_BUDGET_GB * 1e9)
NUM_WORKERS = max(1, NUM_CPUS)

# Set memory limits
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError):
    pass
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_GB:.1f} GB")
logger.info(f"Workers for parallel training: {NUM_WORKERS}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DATA_FILE = DATA_DIR / "full_data_out.json"

SEEDS = [42, 123, 456]
PROBE_CONFIG = {"hidden_layer_sizes": (32, 16), "max_iter": 50, "learning_rate_init": 0.001}
GT_CONFIG = {"hidden_layer_sizes": (128, 64, 32), "max_iter": 200, "learning_rate_init": 0.001}

# Driver tasks: joins point TO drivers
DRIVER_TASKS = ["rel-f1/driver-dnf", "rel-f1/driver-top3", "rel-f1/driver-position"]

# Maps: target_table -> list of join indices (filled during parsing)
TARGET_TABLE_JOINS: dict[str, list[int]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Step 1: Parse full_data_out.json
# ---------------------------------------------------------------------------
def parse_data(data_path: Path) -> tuple[dict, list, dict]:
    """Parse the full dataset into tables, FK joins, and task samples."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples")

    table_rows: dict[str, list[dict]] = defaultdict(list)
    table_pk_cols: dict[str, str] = {}
    table_fk_info: dict[str, dict] = {}
    fk_joins: list[dict] = []
    task_samples: dict[str, list[dict]] = defaultdict(list)
    task_meta: dict[str, dict] = {}

    for ex in examples:
        row_type = ex.get("metadata_row_type")

        if row_type == "table_row":
            tname = ex["metadata_table_name"]
            pk_col = ex["metadata_primary_key_col"]
            pk_val = ex["metadata_primary_key_value"]
            features = json.loads(ex["input"])
            features[pk_col] = pk_val
            table_rows[tname].append(features)
            table_pk_cols[tname] = pk_col
            if "metadata_foreign_keys_json" in ex:
                table_fk_info[tname] = json.loads(ex["metadata_foreign_keys_json"])

        elif row_type == "fk_join_metadata":
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])
            fk_joins.append({
                "source_table": inp["source_table"],
                "source_fk_col": inp["source_fk_col"],
                "target_table": inp["target_table"],
                "target_pk_col": inp["target_pk_col"],
                "stats": out,
                "join_index": ex["metadata_row_index"],
            })

        elif row_type == "task_sample":
            tname = ex["metadata_task_name"]
            sample = json.loads(ex["input"])
            sample["__label__"] = ex["output"]
            sample["__fold__"] = ex["metadata_fold_name"]
            task_samples[tname].append(sample)
            if tname not in task_meta:
                task_meta[tname] = {
                    "task_type": ex.get("metadata_task_type", "regression"),
                    "target_col": ex.get("metadata_target_col", "label"),
                    "entity_col": ex.get("metadata_entity_col"),
                    "entity_table": ex.get("metadata_entity_table"),
                }

    # Convert table rows to DataFrames
    tables: dict[str, pd.DataFrame] = {}
    for tname, rows in table_rows.items():
        df = pd.DataFrame(rows)
        pk_col = table_pk_cols[tname]
        # Convert PK to int if possible
        try:
            df[pk_col] = pd.to_numeric(df[pk_col], errors="coerce").astype("Int64")
        except (ValueError, TypeError):
            pass
        # Try to convert numeric columns (but preserve date strings)
        for col in df.columns:
            if col == pk_col:
                continue
            # Check if column looks like dates before trying numeric conversion
            sample_vals = df[col].dropna().head(5).astype(str).tolist()
            looks_like_date = any("T" in v and "-" in v for v in sample_vals)
            if looks_like_date:
                continue  # Preserve date strings for datetime parsing in engineer_features
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Only apply if at least 50% of values converted successfully
                if converted.notna().sum() >= len(converted) * 0.5:
                    df[col] = converted
            except (ValueError, TypeError):
                pass
        tables[tname] = df
        logger.info(f"  Table {tname}: {len(df)} rows, {len(df.columns)} cols, "
                     f"numeric: {list(df.select_dtypes(include=[np.number]).columns)[:8]}")

    # Convert task samples to DataFrames
    task_dfs: dict[str, pd.DataFrame] = {}
    for tname, samples in task_samples.items():
        df = pd.DataFrame(samples)
        meta = task_meta[tname]
        # Convert label
        if meta["task_type"] == "binary_classification":
            df["__label__"] = pd.to_numeric(df["__label__"], errors="coerce").astype(float)
        elif meta["task_type"] == "regression":
            df["__label__"] = pd.to_numeric(df["__label__"], errors="coerce").astype(float)
        # Convert entity col
        entity_col = meta.get("entity_col")
        if entity_col and entity_col in df.columns:
            try:
                df[entity_col] = pd.to_numeric(df[entity_col], errors="coerce").astype("Int64")
            except (ValueError, TypeError):
                pass
        task_dfs[tname] = df
        logger.info(f"  Task {tname}: {len(df)} samples, type={meta['task_type']}, "
                     f"entity={meta.get('entity_col')}, folds={df['__fold__'].value_counts().to_dict()}")

    # Build target_table -> join index map
    for i, fk in enumerate(fk_joins):
        TARGET_TABLE_JOINS[fk["target_table"]].append(i)
        logger.info(f"  FK Join {i}: {fk['source_table']}→{fk['target_table']} via "
                     f"{fk['source_fk_col']} (fanout={fk['stats']['fanout_mean']:.1f})")

    return tables, fk_joins, task_dfs, task_meta, table_pk_cols


# ---------------------------------------------------------------------------
# Step 2: Feature engineering for tables
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, exclude_cols: set[str] | None = None) -> pd.DataFrame:
    """Extract numeric features from a DataFrame, handling dates and categoricals."""
    if exclude_cols is None:
        exclude_cols = set()

    result = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col in exclude_cols:
            continue

        series = df[col]

        # Already numeric
        if pd.api.types.is_numeric_dtype(series):
            result[col] = series.astype(float)
            continue

        # Try to parse as datetime
        try:
            dt = pd.to_datetime(series, errors="coerce")
            if dt.notna().sum() > len(dt) * 0.5:
                result[f"{col}_year"] = dt.dt.year.astype(float)
                result[f"{col}_month"] = dt.dt.month.astype(float)
                result[f"{col}_dayofyear"] = dt.dt.dayofyear.astype(float)
                continue
        except Exception:
            pass

        # Try ordinal encoding for categoricals (up to 200 unique)
        try:
            nunique = series.nunique()
        except Exception:
            nunique = 0
        if 2 <= nunique <= 200:
            try:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                vals = enc.fit_transform(series.astype(str).values.reshape(-1, 1))
                result[f"{col}_ord"] = vals.ravel().astype(float)
            except Exception:
                pass
        elif nunique > 200:
            # Hash-encode high-cardinality strings into a single numeric feature
            try:
                hashed = series.astype(str).apply(lambda x: hash(x) % 10000).astype(float)
                result[f"{col}_hash"] = hashed
            except Exception:
                pass

    # Fill NaN with 0
    result = result.fillna(0)

    # Drop constant columns
    for col in list(result.columns):
        if result[col].std() < 1e-10:
            result.drop(columns=[col], inplace=True)

    return result


# ---------------------------------------------------------------------------
# Step 3: Compute aggregated child features per join
# ---------------------------------------------------------------------------
def compute_agg_features(child_df: pd.DataFrame, fk_col: str,
                         pk_col: str | None = None) -> pd.DataFrame:
    """Aggregate child table numeric features by FK column."""
    exclude = {fk_col}
    if pk_col:
        exclude.add(pk_col)

    feat_df = engineer_features(child_df, exclude_cols=exclude)

    if feat_df.shape[1] == 0:
        logger.warning(f"No numeric features in child table after engineering")
        return pd.DataFrame()

    # Add FK col back for groupby
    feat_df[fk_col] = child_df[fk_col].values

    # Aggregate
    try:
        agg = feat_df.groupby(fk_col).agg(["mean", "std", "count"])
    except Exception:
        logger.exception("Aggregation failed")
        return pd.DataFrame()

    # Flatten column names
    agg.columns = ["_".join(str(c) for c in col) for col in agg.columns]
    agg = agg.fillna(0)

    # Drop constant columns
    for col in list(agg.columns):
        if agg[col].std() < 1e-10:
            agg.drop(columns=[col], inplace=True)

    return agg


# ---------------------------------------------------------------------------
# Step 4: Build prediction tasks for all parent tables
# ---------------------------------------------------------------------------
def build_self_supervised_task(tables: dict, table_pk_cols: dict,
                               target_table: str) -> tuple[pd.DataFrame, dict] | None:
    """Create a self-supervised regression task for a parent table."""
    df = tables[target_table].copy()
    pk_col = table_pk_cols[target_table]

    if target_table == "races":
        # Predict "round" from other features
        if "round" not in df.columns:
            logger.warning("races table has no 'round' column for self-supervised task")
            return None
        target_col = "round"
        feat_df = engineer_features(df, exclude_cols={pk_col, target_col})
        if feat_df.shape[1] == 0:
            return None
        feat_df[pk_col] = df[pk_col].values
        feat_df["__label__"] = df[target_col].astype(float).values
        # Temporal split based on year if available
        if "year" in df.columns:
            years = df["year"].astype(float)
            feat_df["__fold__"] = "train"
            feat_df.loc[years >= 2005, "__fold__"] = "val"
            feat_df.loc[years >= 2010, "__fold__"] = "test"
        else:
            # Random split
            rng = np.random.RandomState(42)
            r = rng.rand(len(feat_df))
            feat_df["__fold__"] = np.where(r < 0.7, "train", np.where(r < 0.85, "val", "test"))
        meta = {"task_type": "regression", "target_col": target_col,
                "entity_col": pk_col, "entity_table": target_table}
        return feat_df, meta

    elif target_table == "constructors":
        # Predict number of constructor_standings entries (popularity/longevity proxy)
        if "constructor_standings" in tables:
            cs = tables["constructor_standings"]
            ck = "constructorId"
            if ck in cs.columns:
                counts = cs.groupby(ck).size().reset_index(name="__label__")
                counts["__label__"] = counts["__label__"].astype(float)
                feat_df = engineer_features(df, exclude_cols={pk_col})
                if feat_df.shape[1] == 0:
                    logger.warning("Constructors: no features after engineering, creating minimal")
                    # Create minimal features from nationality ordinal encoding
                    feat_df = pd.DataFrame(index=df.index)
                    if "nationality" in df.columns:
                        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                        feat_df["nationality_ord"] = enc.fit_transform(
                            df["nationality"].astype(str).values.reshape(-1, 1)
                        ).ravel().astype(float)
                    if feat_df.shape[1] == 0:
                        return None
                feat_df[pk_col] = df[pk_col].values
                feat_df = feat_df.merge(counts, left_on=pk_col, right_on=ck, how="inner")
                if len(feat_df) < 10:
                    return None
                # Random split
                rng = np.random.RandomState(42)
                r = rng.rand(len(feat_df))
                feat_df["__fold__"] = np.where(r < 0.7, "train", np.where(r < 0.85, "val", "test"))
                meta = {"task_type": "regression", "target_col": "standing_count",
                        "entity_col": pk_col, "entity_table": target_table}
                return feat_df, meta
        return None

    elif target_table == "circuits":
        # Predict number of races held at each circuit
        if "races" in tables:
            races = tables["races"]
            ck = "circuitId"
            if ck in races.columns:
                counts = races.groupby(ck).size().reset_index(name="__label__")
                counts["__label__"] = counts["__label__"].astype(float)
                feat_df = engineer_features(df, exclude_cols={pk_col})
                if feat_df.shape[1] == 0:
                    return None
                feat_df[pk_col] = df[pk_col].values
                feat_df = feat_df.merge(counts, left_on=pk_col, right_on=ck, how="inner")
                # Random split
                rng = np.random.RandomState(42)
                r = rng.rand(len(feat_df))
                feat_df["__fold__"] = np.where(r < 0.7, "train", np.where(r < 0.85, "val", "test"))
                meta = {"task_type": "regression", "target_col": "race_count",
                        "entity_col": pk_col, "entity_table": target_table}
                return feat_df, meta
        return None

    return None


# ---------------------------------------------------------------------------
# Step 5: Training-free proxies
# ---------------------------------------------------------------------------
def compute_training_free_proxies(
    agg_features: pd.DataFrame,
    parent_features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    fanout_mean: float,
    parent_feature_cols: list[str],
    agg_feature_cols: list[str],
    merged_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute 5 training-free proxies for a (join, task) pair."""

    proxies = {}

    # Proxy 1: Mean Fan-Out (inverted: 1/fanout so higher = better join)
    proxies["fanout"] = fanout_mean

    # Proxy 2: Max Pearson correlation between agg features and label
    if len(agg_feature_cols) > 0 and len(labels) > 5:
        correlations = []
        for col in agg_feature_cols:
            vals = merged_df[col].values.astype(float)
            mask = np.isfinite(vals) & np.isfinite(labels)
            if mask.sum() > 5:
                try:
                    r, _ = pearsonr(vals[mask], labels[mask])
                    if np.isfinite(r):
                        correlations.append(abs(r))
                except Exception:
                    pass
        proxies["correlation"] = max(correlations) if correlations else 0.0
    else:
        proxies["correlation"] = 0.0

    # Proxy 3: Mutual Information
    if len(agg_feature_cols) > 0 and len(labels) > 5:
        try:
            X_agg = merged_df[agg_feature_cols].values.astype(float)
            X_agg = np.nan_to_num(X_agg, 0)
            y = labels.copy()
            if task_type == "binary_classification":
                mi = mutual_info_classif(X_agg, y, random_state=42, n_neighbors=3)
            else:
                mi = mutual_info_regression(X_agg, y, random_state=42, n_neighbors=3)
            proxies["MI"] = float(np.mean(mi))
        except Exception as e:
            logger.debug(f"MI computation failed: {e}")
            proxies["MI"] = 0.0
    else:
        proxies["MI"] = 0.0

    # Proxy 4: Conditional Entropy Reduction
    if len(agg_feature_cols) > 0 and parent_features.shape[1] > 0 and len(labels) > 5:
        try:
            X_parent = np.nan_to_num(parent_features, 0)
            X_agg = merged_df[agg_feature_cols].values.astype(float)
            X_agg = np.nan_to_num(X_agg, 0)
            X_combined = np.hstack([X_parent, X_agg])
            y = labels.copy()
            if task_type == "binary_classification":
                mi_parent = float(np.mean(mutual_info_classif(X_parent, y, random_state=42, n_neighbors=3)))
                mi_combined = float(np.mean(mutual_info_classif(X_combined, y, random_state=42, n_neighbors=3)))
            else:
                mi_parent = float(np.mean(mutual_info_regression(X_parent, y, random_state=42, n_neighbors=3)))
                mi_combined = float(np.mean(mutual_info_regression(X_combined, y, random_state=42, n_neighbors=3)))
            proxies["entropy_reduction"] = mi_combined - mi_parent
        except Exception as e:
            logger.debug(f"Entropy reduction failed: {e}")
            proxies["entropy_reduction"] = 0.0
    else:
        proxies["entropy_reduction"] = 0.0

    # Proxy 5: Homophily (label-feature correlation for mean-aggregated features)
    if len(agg_feature_cols) > 0 and len(labels) > 5:
        mean_cols = [c for c in agg_feature_cols if c.endswith("_mean")]
        if not mean_cols:
            mean_cols = agg_feature_cols[:3]
        homophily_vals = []
        for col in mean_cols:
            vals = merged_df[col].values.astype(float)
            mask = np.isfinite(vals) & np.isfinite(labels)
            if mask.sum() > 5:
                try:
                    r, _ = pearsonr(vals[mask], labels[mask])
                    if np.isfinite(r):
                        homophily_vals.append(abs(r))
                except Exception:
                    pass
        proxies["homophily"] = float(np.mean(homophily_vals)) if homophily_vals else 0.0
    else:
        proxies["homophily"] = 0.0

    return proxies


# ---------------------------------------------------------------------------
# Step 6: Model training and evaluation (single job)
# ---------------------------------------------------------------------------
def train_and_eval_single(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    task_type: str, model_config: dict, seed: int,
    use_gbm: bool = False,
) -> float:
    """Train a model and return evaluation score."""
    try:
        if task_type == "binary_classification":
            if use_gbm:
                model = GradientBoostingClassifier(
                    n_estimators=model_config.get("n_estimators", 50),
                    max_depth=model_config.get("max_depth", 3),
                    random_state=seed,
                    learning_rate=0.1,
                )
            else:
                model = MLPClassifier(
                    hidden_layer_sizes=model_config["hidden_layer_sizes"],
                    max_iter=model_config["max_iter"],
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.15,
                    learning_rate_init=model_config["learning_rate_init"],
                )
            model.fit(X_train, y_train)
            # AUROC
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)
                if y_proba.shape[1] == 2:
                    score = roc_auc_score(y_val, y_proba[:, 1])
                else:
                    score = roc_auc_score(y_val, y_proba, multi_class="ovr")
            else:
                y_pred = model.predict(X_val)
                score = roc_auc_score(y_val, y_pred)
            return float(score)
        else:
            if use_gbm:
                model = GradientBoostingRegressor(
                    n_estimators=model_config.get("n_estimators", 50),
                    max_depth=model_config.get("max_depth", 3),
                    random_state=seed,
                    learning_rate=0.1,
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=model_config["hidden_layer_sizes"],
                    max_iter=model_config["max_iter"],
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.15,
                    learning_rate_init=model_config["learning_rate_init"],
                )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            # Use R² (higher is better)
            score = r2_score(y_val, y_pred)
            return float(score)
    except Exception as e:
        logger.debug(f"Model training failed (seed={seed}): {e}")
        return float("nan")


def compute_jrn(
    X_train_parent: np.ndarray, X_val_parent: np.ndarray,
    X_train_combined: np.ndarray, X_val_combined: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray,
    task_type: str, model_config: dict, seeds: list[int],
    use_gbm: bool = False,
) -> dict:
    """Compute JRN across multiple seeds."""
    base_scores = []
    join_scores = []

    for seed in seeds:
        s_base = train_and_eval_single(
            X_train_parent, y_train, X_val_parent, y_val,
            task_type, model_config, seed, use_gbm
        )
        s_join = train_and_eval_single(
            X_train_combined, y_train, X_val_combined, y_val,
            task_type, model_config, seed, use_gbm
        )
        if np.isfinite(s_base) and np.isfinite(s_join):
            base_scores.append(s_base)
            join_scores.append(s_join)

    if not base_scores:
        return {"jrn_mean": float("nan"), "jrn_std": float("nan"),
                "M_base": float("nan"), "M_join": float("nan"),
                "base_scores": [], "join_scores": []}

    M_base = float(np.mean(base_scores))
    M_join = float(np.mean(join_scores))

    # Compute JRN
    # For classification (AUROC): M_base >= 0.5 usually, ratio is meaningful
    # For regression (R²): M_base can be negative; ratio becomes misleading
    # Use ratio only when M_base > 0.05 (positive and meaningfully above zero)
    # Otherwise use additive: JRN = 1 + (M_join - M_base) so >1 means improvement
    use_ratio = M_base > 0.05
    if use_ratio:
        jrn_values = [j / M_base for j in join_scores]
    else:
        # Additive fallback
        jrn_values = [1.0 + (j - M_base) for j in join_scores]

    return {
        "jrn_mean": float(np.mean(jrn_values)),
        "jrn_std": float(np.std(jrn_values)) if len(jrn_values) > 1 else 0.0,
        "M_base": M_base,
        "M_join": M_join,
        "base_scores": base_scores,
        "join_scores": join_scores,
        "used_additive": not use_ratio,
    }


# ---------------------------------------------------------------------------
# Step 7: Prepare data for a (join, task) pair
# ---------------------------------------------------------------------------
def prepare_join_task_data(
    tables: dict, table_pk_cols: dict,
    fk_join: dict, agg_features: pd.DataFrame,
    task_df: pd.DataFrame, task_meta_entry: dict,
) -> dict | None:
    """Prepare train/val features and labels for a (join, task) pair."""
    target_table = fk_join["target_table"]
    target_pk = table_pk_cols.get(target_table)
    entity_col = task_meta_entry.get("entity_col")
    task_type = task_meta_entry["task_type"]

    if entity_col is None:
        return None

    parent_df = tables[target_table].copy()

    # Engineer parent features
    parent_feat = engineer_features(parent_df, exclude_cols={target_pk})
    if parent_feat.shape[1] == 0:
        logger.warning(f"No parent features for {target_table} from engineer_features, "
                        f"creating constant feature as fallback")
        # Create a constant feature so the baseline model has something to fit
        # This ensures JRN measures the value of the join features alone
        parent_feat = pd.DataFrame({"__const__": np.ones(len(parent_df))}, index=parent_df.index)

    parent_feat[target_pk] = parent_df[target_pk].values

    # Make sure FK col in agg_features index matches parent PK type
    if agg_features.empty:
        logger.warning(f"No aggregated features for join {fk_join['source_table']}→{target_table}")
        return None

    # Merge task labels with parent features
    task_copy = task_df.copy()
    # Convert entity col to match parent PK
    try:
        task_copy[entity_col] = pd.to_numeric(task_copy[entity_col], errors="coerce").astype("Int64")
    except (ValueError, TypeError):
        pass

    # Build merged DataFrame
    merged = parent_feat.merge(task_copy[[entity_col, "__label__", "__fold__"]],
                                left_on=target_pk, right_on=entity_col, how="inner")

    if len(merged) < 20:
        logger.warning(f"Too few samples after parent merge: {len(merged)}")
        return None

    # Merge with agg features
    agg_reset = agg_features.reset_index()
    fk_col_name = fk_join["source_fk_col"]
    # The agg_features index is the FK column values = parent PK values
    # Rename index to match parent PK
    agg_cols = [c for c in agg_reset.columns if c != fk_col_name]
    if fk_col_name in agg_reset.columns:
        try:
            agg_reset[fk_col_name] = pd.to_numeric(agg_reset[fk_col_name], errors="coerce").astype("Int64")
        except (ValueError, TypeError):
            pass
        merged_full = merged.merge(agg_reset, left_on=target_pk, right_on=fk_col_name, how="left")
    else:
        # Index was already the FK column
        agg_reset2 = agg_features.copy()
        agg_reset2.index.name = target_pk
        agg_reset2 = agg_reset2.reset_index()
        try:
            agg_reset2[target_pk] = pd.to_numeric(agg_reset2[target_pk], errors="coerce").astype("Int64")
        except (ValueError, TypeError):
            pass
        merged_full = merged.merge(agg_reset2, on=target_pk, how="left")

    # Fill NaN agg features with 0 (parent rows with no children)
    parent_feature_cols = [c for c in parent_feat.columns if c != target_pk]
    agg_feature_cols = [c for c in agg_features.columns]

    # Ensure all agg columns exist
    for col in agg_feature_cols:
        if col not in merged_full.columns:
            merged_full[col] = 0.0
    merged_full[agg_feature_cols] = merged_full[agg_feature_cols].fillna(0)

    # Labels
    labels = merged_full["__label__"].values.astype(float)
    folds = merged_full["__fold__"].values

    # Check for valid labels
    valid_mask = np.isfinite(labels)
    if valid_mask.sum() < 20:
        logger.warning(f"Too few valid labels: {valid_mask.sum()}")
        return None

    # Train/val split
    train_mask = (folds == "train") & valid_mask
    val_mask = (folds == "val") & valid_mask

    if train_mask.sum() < 10 or val_mask.sum() < 5:
        # Fall back to random split
        rng = np.random.RandomState(42)
        indices = np.where(valid_mask)[0]
        rng.shuffle(indices)
        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)
        train_mask = np.zeros(len(labels), dtype=bool)
        val_mask = np.zeros(len(labels), dtype=bool)
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train + n_val]] = True

    if train_mask.sum() < 5 or val_mask.sum() < 3:
        logger.warning(f"Still too few samples: train={train_mask.sum()}, val={val_mask.sum()}")
        return None

    # Build feature matrices
    X_parent_all = merged_full[parent_feature_cols].values.astype(float)
    X_agg_all = merged_full[agg_feature_cols].values.astype(float)

    X_parent_all = np.nan_to_num(X_parent_all, 0)
    X_agg_all = np.nan_to_num(X_agg_all, 0)

    # Scale
    scaler_parent = StandardScaler()
    scaler_combined = StandardScaler()

    X_parent_train = scaler_parent.fit_transform(X_parent_all[train_mask])
    X_parent_val = scaler_parent.transform(X_parent_all[val_mask])

    X_combined_all = np.hstack([X_parent_all, X_agg_all])
    X_combined_train = scaler_combined.fit_transform(X_combined_all[train_mask])
    X_combined_val = scaler_combined.transform(X_combined_all[val_mask])

    y_train = labels[train_mask]
    y_val = labels[val_mask]

    # For binary classification, check we have both classes
    if task_type == "binary_classification":
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            logger.warning("Binary task but only one class in train or val")
            return None

    return {
        "X_train_parent": X_parent_train,
        "X_val_parent": X_parent_val,
        "X_train_combined": X_combined_train,
        "X_val_combined": X_combined_val,
        "y_train": y_train,
        "y_val": y_val,
        "task_type": task_type,
        "parent_features": X_parent_all[valid_mask],
        "labels": labels[valid_mask],
        "merged_df": merged_full[valid_mask].reset_index(drop=True),
        "parent_feature_cols": parent_feature_cols,
        "agg_feature_cols": agg_feature_cols,
    }


# ---------------------------------------------------------------------------
# Step 8: Steiger's Z-test for comparing correlations
# ---------------------------------------------------------------------------
def steiger_z_test(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """Compare two dependent correlations (r12 vs r13 sharing variable 1).

    Tests H0: r12 = r13  (where variable 1 = ground truth).
    Returns (z_statistic, p_value).
    """
    if n < 4:
        return 0.0, 1.0

    # Fisher Z-transform
    def fisher_z(r):
        r = np.clip(r, -0.999, 0.999)
        return 0.5 * np.log((1 + r) / (1 - r))

    z12 = fisher_z(r12)
    z13 = fisher_z(r13)

    # Steiger's formula for the SE of the difference
    rm = 0.5 * (r12 + r13)
    f = (1 - r23) / (2 * (1 - rm ** 2))
    h = (1 - f * (n - 1)) / (n - 3)

    if h <= 0:
        return 0.0, 1.0

    z = (z12 - z13) / np.sqrt(2 * (1 - r23) / ((n - 3) * (1 + rm)))

    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))

    return float(z), float(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Training-Free vs Probe-Based JRN: Diagnostic Comparison")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Parse data
    # -----------------------------------------------------------------------
    tables, fk_joins, task_dfs, task_meta, table_pk_cols = parse_data(DATA_FILE)

    logger.info(f"Parsed {len(tables)} tables, {len(fk_joins)} FK joins, {len(task_dfs)} tasks")

    # Validation
    assert len(tables) == 9, f"Expected 9 tables, got {len(tables)}"
    assert len(fk_joins) == 13, f"Expected 13 FK joins, got {len(fk_joins)}"
    total_rows = sum(len(t) for t in tables.values())
    logger.info(f"Total table rows: {total_rows}")

    # -----------------------------------------------------------------------
    # Compute aggregated features for each join
    # -----------------------------------------------------------------------
    logger.info("Computing aggregated features for each FK join...")
    join_agg_features: dict[int, pd.DataFrame] = {}

    for i, fk in enumerate(fk_joins):
        child_df = tables[fk["source_table"]]
        fk_col = fk["source_fk_col"]

        # Find child table PK
        child_pk = table_pk_cols.get(fk["source_table"])

        agg = compute_agg_features(child_df, fk_col, child_pk)
        join_agg_features[i] = agg
        logger.info(f"  Join {i} ({fk['source_table']}→{fk['target_table']}): "
                     f"{agg.shape[0]} groups, {agg.shape[1]} agg features")

    # -----------------------------------------------------------------------
    # Build all (join, task) pairs
    # -----------------------------------------------------------------------
    logger.info("Building (join, task) pairs...")

    # Identify which tasks map to which target tables
    # Driver tasks: entity_table=drivers, target of joins 4,8,11
    driver_join_indices = [i for i, fk in enumerate(fk_joins) if fk["target_table"] == "drivers"]
    logger.info(f"Driver joins: {driver_join_indices} (expect [4, 8, 11])")

    # Build self-supervised tasks for non-driver parent tables
    ss_tasks: dict[str, tuple[pd.DataFrame, dict]] = {}
    for target_table in ["races", "constructors", "circuits"]:
        result = build_self_supervised_task(tables, table_pk_cols, target_table)
        if result is not None:
            task_name = f"ss_{target_table}"
            ss_tasks[task_name] = result
            logger.info(f"  Self-supervised task for {target_table}: "
                         f"{len(result[0])} samples, {result[0].shape[1]-2} features")

    # Enumerate all (join_index, task_name) pairs
    all_pairs: list[tuple[int, str]] = []

    # Driver task × driver joins
    for task_name in DRIVER_TASKS:
        if task_name in task_dfs:
            for ji in driver_join_indices:
                all_pairs.append((ji, task_name))

    # Self-supervised tasks × relevant joins
    for target_table in ["races", "constructors", "circuits"]:
        task_name = f"ss_{target_table}"
        if task_name not in ss_tasks:
            continue
        join_indices = [i for i, fk in enumerate(fk_joins) if fk["target_table"] == target_table]
        for ji in join_indices:
            all_pairs.append((ji, task_name))

    logger.info(f"Total (join, task) pairs to evaluate: {len(all_pairs)}")

    # -----------------------------------------------------------------------
    # Evaluate all pairs
    # -----------------------------------------------------------------------
    all_results: list[dict] = []
    per_join_data: dict[int, list[dict]] = defaultdict(list)

    for pair_idx, (join_idx, task_name) in enumerate(all_pairs):
        fk = fk_joins[join_idx]
        logger.info(f"[{pair_idx+1}/{len(all_pairs)}] Join {join_idx} "
                     f"({fk['source_table']}→{fk['target_table']}) × {task_name}")

        # Get task data
        if task_name.startswith("ss_"):
            if task_name not in ss_tasks:
                logger.warning(f"  Skipping: self-supervised task {task_name} not available")
                continue
            task_df_full, tm = ss_tasks[task_name]
            # For self-supervised tasks, the task_df already has features + labels
            # We need to restructure for prepare_join_task_data
            entity_col = tm["entity_col"]
            # Create a simplified task_df with just entity_col, label, fold
            task_df = task_df_full[[entity_col, "__label__", "__fold__"]].copy()
        else:
            if task_name not in task_dfs:
                logger.warning(f"  Skipping: task {task_name} not found")
                continue
            task_df = task_dfs[task_name]
            tm = task_meta[task_name]

        agg = join_agg_features[join_idx]
        if agg.empty:
            logger.warning(f"  Skipping: no aggregated features")
            continue

        # Prepare data
        data = prepare_join_task_data(
            tables, table_pk_cols, fk, agg, task_df, tm
        )
        if data is None:
            logger.warning(f"  Skipping: data preparation failed")
            continue

        # Compute training-free proxies
        proxies = compute_training_free_proxies(
            agg_features=agg,
            parent_features=data["parent_features"],
            labels=data["labels"],
            task_type=data["task_type"],
            fanout_mean=fk["stats"]["fanout_mean"],
            parent_feature_cols=data["parent_feature_cols"],
            agg_feature_cols=data["agg_feature_cols"],
            merged_df=data["merged_df"],
        )
        logger.info(f"  Proxies: fanout={proxies['fanout']:.2f}, corr={proxies['correlation']:.3f}, "
                     f"MI={proxies['MI']:.3f}, ent_red={proxies['entropy_reduction']:.3f}, "
                     f"homophily={proxies['homophily']:.3f}")

        # Compute probe JRN with BOTH MLP and GBM, keep the one that works better
        # MLP probe
        mlp_probe_result = compute_jrn(
            data["X_train_parent"], data["X_val_parent"],
            data["X_train_combined"], data["X_val_combined"],
            data["y_train"], data["y_val"],
            data["task_type"], PROBE_CONFIG, SEEDS, use_gbm=False
        )

        # GBM probe (more robust for small tabular data)
        gbm_probe_config = {"n_estimators": 50, "max_depth": 3}
        gbm_probe_result = compute_jrn(
            data["X_train_parent"], data["X_val_parent"],
            data["X_train_combined"], data["X_val_combined"],
            data["y_train"], data["y_val"],
            data["task_type"], gbm_probe_config, SEEDS, use_gbm=True
        )

        # Use GBM as primary (better on small tabular), MLP as secondary
        probe_result = gbm_probe_result
        use_gbm = True
        if np.isnan(probe_result["jrn_mean"]) and not np.isnan(mlp_probe_result["jrn_mean"]):
            probe_result = mlp_probe_result
            use_gbm = False

        logger.info(f"  Probe JRN (GBM): mean={gbm_probe_result['jrn_mean']:.4f}, "
                     f"M_base={gbm_probe_result['M_base']:.4f}, M_join={gbm_probe_result['M_join']:.4f}")
        logger.info(f"  Probe JRN (MLP): mean={mlp_probe_result['jrn_mean']:.4f}, "
                     f"M_base={mlp_probe_result['M_base']:.4f}, M_join={mlp_probe_result['M_join']:.4f}")

        # Compute ground-truth JRN (larger GBM model)
        gt_config_gbm = {"n_estimators": 200, "max_depth": 5}
        gt_result = compute_jrn(
            data["X_train_parent"], data["X_val_parent"],
            data["X_train_combined"], data["X_val_combined"],
            data["y_train"], data["y_val"],
            data["task_type"], gt_config_gbm, SEEDS, use_gbm=True
        )

        # Also compute MLP GT for comparison
        mlp_gt_result = compute_jrn(
            data["X_train_parent"], data["X_val_parent"],
            data["X_train_combined"], data["X_val_combined"],
            data["y_train"], data["y_val"],
            data["task_type"], GT_CONFIG, SEEDS, use_gbm=False
        )

        if np.isnan(gt_result["jrn_mean"]) and not np.isnan(mlp_gt_result["jrn_mean"]):
            gt_result = mlp_gt_result

        logger.info(f"  GT JRN (GBM): mean={gt_result['jrn_mean']:.4f}, "
                     f"M_base={gt_result['M_base']:.4f}, M_join={gt_result['M_join']:.4f}")

        result_entry = {
            "join_idx": join_idx,
            "task_name": task_name,
            "source_table": fk["source_table"],
            "target_table": fk["target_table"],
            "source_fk_col": fk["source_fk_col"],
            "task_type": data["task_type"],
            "n_train": int(len(data["y_train"])),
            "n_val": int(len(data["y_val"])),
            **{f"proxy_{k}": v for k, v in proxies.items()},
            # GBM probe (primary)
            "jrn_probe_mean": gbm_probe_result["jrn_mean"],
            "jrn_probe_std": gbm_probe_result["jrn_std"],
            "M_base_probe": gbm_probe_result["M_base"],
            "M_join_probe": gbm_probe_result["M_join"],
            # MLP probe (secondary)
            "jrn_probe_mlp_mean": mlp_probe_result["jrn_mean"],
            "jrn_probe_mlp_std": mlp_probe_result["jrn_std"],
            "M_base_probe_mlp": mlp_probe_result["M_base"],
            "M_join_probe_mlp": mlp_probe_result["M_join"],
            # GT (GBM)
            "jrn_gt_mean": gt_result["jrn_mean"],
            "jrn_gt_std": gt_result["jrn_std"],
            "M_base_gt": gt_result["M_base"],
            "M_join_gt": gt_result["M_join"],
            "model_type": "GBM+MLP",
        }
        all_results.append(result_entry)
        per_join_data[join_idx].append(result_entry)

    logger.info(f"Successfully evaluated {len(all_results)} (join, task) pairs")

    if len(all_results) < 3:
        logger.error("Too few results for meaningful analysis")
        # Still output what we have
        _write_output(all_results, per_join_data, fk_joins, {}, {}, [])
        return

    # -----------------------------------------------------------------------
    # Compute Spearman correlation matrix
    # -----------------------------------------------------------------------
    logger.info("Computing Spearman correlation matrix...")

    metric_names = ["fanout", "correlation", "MI", "entropy_reduction",
                    "homophily", "JRN_probe_GBM", "JRN_probe_MLP", "JRN_ground_truth"]

    # Filter out degenerate pairs
    valid_results = []
    for r in all_results:
        if not np.isfinite(r["jrn_gt_mean"]):
            continue
        # Skip extremely degenerate JRN values (>10x)
        if abs(r["jrn_gt_mean"]) > 10:
            logger.info(f"  Excluding degenerate pair: {r['source_table']}→{r['target_table']} × "
                         f"{r['task_name']} (JRN_gt={r['jrn_gt_mean']:.2f})")
            r["_excluded"] = True
            continue
        valid_results.append(r)

    logger.info(f"Valid pairs after filtering: {len(valid_results)} / {len(all_results)}")

    # Build matrix
    matrix_data = []
    for r in valid_results:
        gbm_jrn = r["jrn_probe_mean"] if np.isfinite(r["jrn_probe_mean"]) else 0.0
        mlp_jrn = r["jrn_probe_mlp_mean"] if np.isfinite(r["jrn_probe_mlp_mean"]) else 0.0
        matrix_data.append([
            r["proxy_fanout"],
            r["proxy_correlation"],
            r["proxy_MI"],
            r["proxy_entropy_reduction"],
            r["proxy_homophily"],
            gbm_jrn,
            mlp_jrn,
            r["jrn_gt_mean"],
        ])

    n_pairs = len(matrix_data)
    logger.info(f"Valid pairs for Spearman: {n_pairs}")

    n_metrics = len(metric_names)
    gt_idx = n_metrics - 1  # Last column = ground truth
    gbm_probe_idx = 5
    mlp_probe_idx = 6
    spearman_rho = np.full((n_metrics, n_metrics), np.nan)
    spearman_pval = np.full((n_metrics, n_metrics), np.nan)

    if n_pairs >= 4:
        matrix_arr = np.array(matrix_data)
        for i in range(n_metrics):
            for j in range(n_metrics):
                if i == j:
                    spearman_rho[i, j] = 1.0
                    spearman_pval[i, j] = 0.0
                else:
                    try:
                        rho, pval = spearmanr(matrix_arr[:, i], matrix_arr[:, j])
                        spearman_rho[i, j] = float(rho) if np.isfinite(rho) else 0.0
                        spearman_pval[i, j] = float(pval) if np.isfinite(pval) else 1.0
                    except Exception:
                        spearman_rho[i, j] = 0.0
                        spearman_pval[i, j] = 1.0

        # Log key comparisons
        logger.info(f"\nSpearman correlations with ground truth (JRN_gt):")
        for i, name in enumerate(metric_names):
            logger.info(f"  {name}: ρ={spearman_rho[i, gt_idx]:.3f}, p={spearman_pval[i, gt_idx]:.4f}")

    # Key comparisons
    key_comparisons = {}
    if n_pairs >= 4:
        # GBM Probe vs GT
        key_comparisons["probe_gbm_vs_gt_rho"] = float(spearman_rho[gbm_probe_idx, gt_idx])
        key_comparisons["probe_gbm_vs_gt_pval"] = float(spearman_pval[gbm_probe_idx, gt_idx])
        # MLP Probe vs GT
        key_comparisons["probe_mlp_vs_gt_rho"] = float(spearman_rho[mlp_probe_idx, gt_idx])
        key_comparisons["probe_mlp_vs_gt_pval"] = float(spearman_pval[mlp_probe_idx, gt_idx])
        # Best probe overall
        best_probe_rho = max(
            abs(float(spearman_rho[gbm_probe_idx, gt_idx])),
            abs(float(spearman_rho[mlp_probe_idx, gt_idx]))
        )
        if abs(float(spearman_rho[gbm_probe_idx, gt_idx])) >= abs(float(spearman_rho[mlp_probe_idx, gt_idx])):
            key_comparisons["best_probe_type"] = "GBM"
            key_comparisons["probe_vs_gt_rho"] = float(spearman_rho[gbm_probe_idx, gt_idx])
            key_comparisons["probe_vs_gt_pval"] = float(spearman_pval[gbm_probe_idx, gt_idx])
        else:
            key_comparisons["best_probe_type"] = "MLP"
            key_comparisons["probe_vs_gt_rho"] = float(spearman_rho[mlp_probe_idx, gt_idx])
            key_comparisons["probe_vs_gt_pval"] = float(spearman_pval[mlp_probe_idx, gt_idx])

        # Best training-free proxy
        free_indices = list(range(5))
        free_rhos = [(i, float(spearman_rho[i, gt_idx])) for i in free_indices]
        free_rhos_valid = [(i, r) for i, r in free_rhos if np.isfinite(r)]
        if free_rhos_valid:
            best_free_idx, best_free_rho = max(free_rhos_valid, key=lambda x: abs(x[1]))
            key_comparisons["best_training_free_proxy"] = metric_names[best_free_idx]
            key_comparisons["best_training_free_rho"] = best_free_rho
            key_comparisons["best_training_free_pval"] = float(spearman_pval[best_free_idx, gt_idx])
            best_probe_idx = gbm_probe_idx if key_comparisons.get("best_probe_type") == "GBM" else mlp_probe_idx
            best_probe_rho_val = float(spearman_rho[best_probe_idx, gt_idx])
            key_comparisons["probe_outperforms_all_free"] = (
                abs(best_probe_rho_val) > abs(best_free_rho)
            )

            # Steiger's Z-test
            r_probe_gt = best_probe_rho_val
            r_best_gt = best_free_rho
            r_probe_best = float(spearman_rho[best_probe_idx, best_free_idx])
            z, p = steiger_z_test(r_probe_gt, r_best_gt, r_probe_best, n_pairs)
            key_comparisons["steiger_z_probe_vs_best_free"] = z
            key_comparisons["steiger_z_pval"] = p

    # -----------------------------------------------------------------------
    # Per-task analysis
    # -----------------------------------------------------------------------
    logger.info("Per-task analysis...")
    per_task_analysis = []
    task_names_seen = set(r["task_name"] for r in valid_results)

    for task_name in sorted(task_names_seen):
        task_results = [r for r in valid_results
                        if r["task_name"] == task_name]
        if len(task_results) < 3:
            per_task_analysis.append({
                "task_name": task_name,
                "num_joins_evaluated": len(task_results),
                "note": "too few pairs for Spearman"
            })
            continue

        task_matrix = []
        for r in task_results:
            gbm_jrn = r["jrn_probe_mean"] if np.isfinite(r["jrn_probe_mean"]) else 0.0
            mlp_jrn = r["jrn_probe_mlp_mean"] if np.isfinite(r["jrn_probe_mlp_mean"]) else 0.0
            task_matrix.append([
                r["proxy_fanout"], r["proxy_correlation"], r["proxy_MI"],
                r["proxy_entropy_reduction"], r["proxy_homophily"],
                gbm_jrn, mlp_jrn, r["jrn_gt_mean"],
            ])
        task_arr = np.array(task_matrix)

        # Compute correlations with GT for this task
        best_free_name = "fanout"
        best_free_rho_val = 0.0
        probe_gbm_gt_rho = 0.0
        probe_mlp_gt_rho = 0.0

        if len(task_results) >= 3:
            try:
                rho, _ = spearmanr(task_arr[:, 5], task_arr[:, 7])
                probe_gbm_gt_rho = float(rho) if np.isfinite(rho) else 0.0
            except Exception:
                pass
            try:
                rho, _ = spearmanr(task_arr[:, 6], task_arr[:, 7])
                probe_mlp_gt_rho = float(rho) if np.isfinite(rho) else 0.0
            except Exception:
                pass

            for fi, fname in enumerate(metric_names[:5]):
                try:
                    rho, _ = spearmanr(task_arr[:, fi], task_arr[:, 7])
                    if np.isfinite(rho) and abs(rho) > abs(best_free_rho_val):
                        best_free_rho_val = float(rho)
                        best_free_name = fname
                except Exception:
                    pass

        best_probe_rho = max(probe_gbm_gt_rho, probe_mlp_gt_rho, key=abs)
        per_task_analysis.append({
            "task_name": task_name,
            "num_joins_evaluated": len(task_results),
            "probe_gbm_vs_gt_rho": probe_gbm_gt_rho,
            "probe_mlp_vs_gt_rho": probe_mlp_gt_rho,
            "probe_vs_gt_rho": best_probe_rho,
            "best_free_proxy": best_free_name,
            "best_free_rho": best_free_rho_val,
        })
        logger.info(f"  {task_name}: {len(task_results)} joins, "
                     f"probe_gbm-gt ρ={probe_gbm_gt_rho:.3f}, probe_mlp-gt ρ={probe_mlp_gt_rho:.3f}, "
                     f"best_free={best_free_name} (ρ={best_free_rho_val:.3f})")

    # -----------------------------------------------------------------------
    # Conclusion
    # -----------------------------------------------------------------------
    probe_justified = key_comparisons.get("probe_outperforms_all_free", False)
    best_alt = key_comparisons.get("best_training_free_proxy", "MI")
    best_probe_type = key_comparisons.get("best_probe_type", "GBM")
    probe_rho = key_comparisons.get("probe_vs_gt_rho", 0.0)
    best_free_rho_val = key_comparisons.get("best_training_free_rho", 0.0)

    if probe_justified:
        recommendation = (
            f"Probe-based JRN ({best_probe_type}, ρ={probe_rho:.3f}) correlates more strongly "
            f"with ground truth than all training-free proxies. The best free alternative is "
            f"'{best_alt}' (ρ={best_free_rho_val:.3f}) but probe provides better ranking fidelity."
        )
    else:
        recommendation = (
            f"Training-free proxy '{best_alt}' (ρ={best_free_rho_val:.3f}) achieves comparable "
            f"or better correlation with ground truth than probe-based JRN "
            f"({best_probe_type}, ρ={probe_rho:.3f}), suggesting probes may not be necessary "
            f"for join utility ranking on this dataset."
        )

    conclusion = {
        "probe_justified": probe_justified,
        "best_training_free_alternative": best_alt,
        "recommendation": recommendation,
    }

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    _write_output(all_results, per_join_data, fk_joins,
                  key_comparisons, conclusion, per_task_analysis,
                  spearman_rho, spearman_pval, metric_names, n_pairs)


def _write_output(
    all_results, per_join_data, fk_joins,
    key_comparisons, conclusion, per_task_analysis,
    spearman_rho=None, spearman_pval=None, metric_names=None, n_pairs=0
):
    """Write the method_out.json in exp_gen_sol_out.json schema format."""

    # Build per-join results structure
    per_join_results = []
    for i, fk in enumerate(fk_joins):
        join_entry = {
            "join_id": i,
            "source_table": fk["source_table"],
            "target_table": fk["target_table"],
            "source_fk_col": fk["source_fk_col"],
            "fanout_mean": fk["stats"]["fanout_mean"],
            "tasks": per_join_data.get(i, []),
        }
        per_join_results.append(join_entry)

    # Build internal results structure
    internal_results = {
        "metadata": {
            "dataset": "rel-f1",
            "num_joins": 13,
            "num_tasks_evaluated": len(set(r["task_name"] for r in all_results)),
            "num_join_task_pairs": len(all_results),
            "model_probe_config": {"hidden": list(PROBE_CONFIG["hidden_layer_sizes"]),
                                    "epochs": PROBE_CONFIG["max_iter"], "seeds": len(SEEDS)},
            "model_gt_config": {"hidden": list(GT_CONFIG["hidden_layer_sizes"]),
                                 "epochs": GT_CONFIG["max_iter"], "seeds": len(SEEDS)},
        },
        "per_join_results": per_join_results,
        "spearman_correlation_matrix": {
            "metrics": metric_names or [],
            "rho_matrix": spearman_rho.tolist() if spearman_rho is not None else [],
            "pvalue_matrix": spearman_pval.tolist() if spearman_pval is not None else [],
            "n_pairs": n_pairs,
        },
        "key_comparisons": key_comparisons,
        "per_task_analysis": per_task_analysis,
        "conclusion": conclusion,
    }

    # Now build the exp_gen_sol_out.json schema format
    # Each (join, task) pair becomes an example
    examples = []
    for r in all_results:
        # Input: description of the join+task pair
        input_desc = json.dumps({
            "join": f"{r['source_table']}→{r['target_table']} via {r['source_fk_col']}",
            "task": r["task_name"],
            "task_type": r["task_type"],
            "fanout_mean": r["proxy_fanout"],
        })

        # Output: ground truth JRN
        output_val = json.dumps({
            "jrn_gt_mean": r["jrn_gt_mean"],
            "jrn_gt_std": r.get("jrn_gt_std", 0),
            "M_base_gt": r["M_base_gt"],
            "M_join_gt": r["M_join_gt"],
        })

        example = {
            "input": input_desc,
            "output": output_val,
            "metadata_join_idx": r["join_idx"],
            "metadata_task_name": r["task_name"],
            "metadata_source_table": r["source_table"],
            "metadata_target_table": r["target_table"],
            "metadata_task_type": r["task_type"],
            "metadata_n_train": r["n_train"],
            "metadata_n_val": r["n_val"],
            "metadata_model_type": r.get("model_type", "MLP"),
            # Predictions from training-free proxies (baseline)
            "predict_proxy_fanout": str(r["proxy_fanout"]),
            "predict_proxy_correlation": str(r["proxy_correlation"]),
            "predict_proxy_MI": str(r["proxy_MI"]),
            "predict_proxy_entropy_reduction": str(r["proxy_entropy_reduction"]),
            "predict_proxy_homophily": str(r["proxy_homophily"]),
            # Prediction from probe-based method (our method)
            "predict_jrn_probe_gbm": str(r["jrn_probe_mean"]),
            "predict_jrn_probe_mlp": str(r["jrn_probe_mlp_mean"]),
        }
        examples.append(example)

    # Add summary examples for the Spearman analysis
    if key_comparisons:
        summary_input = json.dumps({
            "type": "spearman_correlation_summary",
            "n_pairs": n_pairs,
        })
        summary_output = json.dumps(key_comparisons)
        examples.append({
            "input": summary_input,
            "output": summary_output,
            "metadata_type": "summary",
            "predict_probe_justified": str(conclusion.get("probe_justified", False)),
            "predict_best_alternative": str(conclusion.get("best_training_free_alternative", "")),
        })

    output = {
        "metadata": internal_results["metadata"],
        "datasets": [{
            "dataset": "rel-f1",
            "examples": examples,
        }],
    }

    # Ensure all values are JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    output = make_serializable(output)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output written to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Also write detailed internal results
    internal_path = WORKSPACE / "detailed_results.json"
    internal_results = make_serializable(internal_results)
    internal_path.write_text(json.dumps(internal_results, indent=2))
    logger.info(f"Detailed results written to {internal_path}")

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
