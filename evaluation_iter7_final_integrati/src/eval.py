#!/usr/bin/env python3
"""Final Integrative Evaluation: 9 Paper Tables, Hypothesis Scorecard v5, Statistical Summary.

Loads ALL 5 dependency experiments and produces eval_out.json conforming to
exp_eval_sol_out schema with metrics_agg and datasets structure.
"""

import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware / memory limits
# ---------------------------------------------------------------------------
def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or 57.0
RAM_BUDGET = int(min(4, TOTAL_RAM_GB * 0.3) * 1e9)  # 4 GB max for this script
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/ai-inventor/aii_pipeline/runs/run__20260309_024817/3_invention_loop")
DEP_PATHS = {
    "exp_id1_it6": BASE / "iter_6/gen_art/exp_id1_it6__opus/full_method_out.json",
    "exp_id2_it6": BASE / "iter_6/gen_art/exp_id2_it6__opus/full_method_out.json",
    "exp_id4_it5": BASE / "iter_5/gen_art/exp_id4_it5__opus/full_method_out.json",
    "exp_id4_it4": BASE / "iter_4/gen_art/exp_id4_it4__opus/full_method_out.json",
    "exp_id1_it5": BASE / "iter_5/gen_art/exp_id1_it5__opus/full_method_out.json",
}
WORKSPACE = Path(__file__).parent


def load_all_deps() -> dict:
    """Load all 5 dependency JSON files."""
    deps = {}
    for name, path in DEP_PATHS.items():
        logger.info(f"Loading {name} from {path}")
        data = json.loads(path.read_text())
        deps[name] = data
        n_ex = sum(len(ds["examples"]) for ds in data["datasets"])
        logger.info(f"  -> {n_ex} examples loaded")
    return deps


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def fisher_z(r: float) -> float:
    """Fisher z-transform of correlation."""
    r = max(-0.9999, min(0.9999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def inv_fisher_z(z: float) -> float:
    """Inverse Fisher z-transform."""
    return math.tanh(z)


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Clopper-Pearson exact binomial CI."""
    if n == 0:
        return 0.0, 1.0
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def safe_float(v, default=0.0) -> float:
    """Convert value to float safely."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ===================================================================
# TABLE 1: JRN MATRIX
# ===================================================================

def build_table_1(deps: dict) -> tuple[list[dict], dict]:
    """Build unified JRN matrix across all datasets, joins, tasks."""
    logger.info("Building Table 1: JRN Matrix")
    examples = []
    all_jrn_values = []

    # From exp_id1_it6: cross-task JRN measurements (rel-f1 + rel-stack)
    d1 = deps["exp_id1_it6"]
    for ds in d1["datasets"]:
        ds_name = ds["dataset"]
        for ex in ds["examples"]:
            if ex.get("metadata_type") != "jrn_measurement":
                continue
            out = json.loads(ex["output"])
            jrn_mean = out.get("jrn_mean", safe_float(ex.get("predict_jrn_probe")))
            jrn_std = out.get("jrn_std", 0.0)
            ci_lo = jrn_mean - 1.96 * jrn_std / math.sqrt(3)
            ci_hi = jrn_mean + 1.96 * jrn_std / math.sqrt(3)
            jrn_val = safe_float(jrn_mean)
            all_jrn_values.append(jrn_val)

            if jrn_val > 1.1:
                cat = "high"
            elif jrn_val < 0.9:
                cat = "low"
            else:
                cat = "critical"

            inp = json.loads(ex["input"])
            examples.append({
                "input": f"JRN measurement: dataset={ds_name}, join={inp.get('join', ex.get('metadata_join', ''))}, task={inp.get('task', ex.get('metadata_task', ''))}",
                "output": f"JRN={jrn_val:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], category={cat}",
                "metadata_dataset": ds_name,
                "metadata_join": str(inp.get("join", ex.get("metadata_join", ""))),
                "metadata_task": str(inp.get("task", ex.get("metadata_task", ""))),
                "metadata_hop_distance": safe_float(ex.get("metadata_hop_distance", inp.get("hop_distance", 1))),
                "metadata_source": "exp_id1_it6",
                "predict_jrn_probe": f"{jrn_val:.4f}",
                "predict_baseline_uniform": "1.0",
                "eval_jrn_mean": round(jrn_val, 6),
                "eval_jrn_std": round(safe_float(jrn_std), 6),
                "eval_jrn_ci_lo": round(ci_lo, 6),
                "eval_jrn_ci_hi": round(ci_hi, 6),
                "eval_jrn_above_threshold": 1 if jrn_val > 1.0 else 0,
                "eval_jrn_category_high": 1 if cat == "high" else 0,
                "eval_jrn_category_critical": 1 if cat == "critical" else 0,
                "eval_jrn_category_low": 1 if cat == "low" else 0,
            })

    # From exp_id4_it4: rel-f1 FK-shuffling (13 joins x 3 tasks = 39 normal JRN)
    d4 = deps["exp_id4_it4"]
    seen_pairs = {(e["metadata_join"], e["metadata_task"]) for e in examples if e["metadata_dataset"] == "rel-f1"}
    for ds in d4["datasets"]:
        for ex in ds["examples"]:
            inp_data = json.loads(ex["input"])
            join_name = f"{inp_data.get('source_table', '')}.{inp_data.get('source_fk_col', '')}->{inp_data.get('target_table', '')}"
            task_name = inp_data.get("task", ex.get("metadata_task_name", "").split("/")[-1])
            if (join_name, task_name) in seen_pairs:
                continue
            out_data = json.loads(ex["output"])
            jrn_val = safe_float(out_data.get("normal_jrn", ex.get("predict_normal_jrn")))
            if jrn_val == 0.0:
                continue
            all_jrn_values.append(jrn_val)
            if jrn_val > 1.1:
                cat = "high"
            elif jrn_val < 0.9:
                cat = "low"
            else:
                cat = "critical"
            examples.append({
                "input": f"JRN measurement: dataset=rel-f1, join={join_name}, task={task_name}",
                "output": f"JRN={jrn_val:.4f}, category={cat} (from FK-shuffling experiment)",
                "metadata_dataset": "rel-f1",
                "metadata_join": join_name,
                "metadata_task": task_name,
                "metadata_hop_distance": 1.0,
                "metadata_source": "exp_id4_it4",
                "predict_jrn_probe": f"{jrn_val:.4f}",
                "predict_baseline_uniform": "1.0",
                "eval_jrn_mean": round(jrn_val, 6),
                "eval_jrn_std": 0.0,
                "eval_jrn_ci_lo": round(jrn_val, 6),
                "eval_jrn_ci_hi": round(jrn_val, 6),
                "eval_jrn_above_threshold": 1 if jrn_val > 1.0 else 0,
                "eval_jrn_category_high": 1 if cat == "high" else 0,
                "eval_jrn_category_critical": 1 if cat == "critical" else 0,
                "eval_jrn_category_low": 1 if cat == "low" else 0,
            })

    # From exp_id2_it6: MLP/GBM dual measurements on rel-f1
    d2 = deps["exp_id2_it6"]
    for ds in d2["datasets"]:
        for ex in ds["examples"]:
            if ex.get("metadata_measurement_type") != "individual_jrn":
                continue
            jrn_val = safe_float(ex.get("metadata_jrn_mean"))
            probe = ex.get("metadata_probe", "")
            join_label = ex.get("metadata_join_label", "")
            task = ex.get("metadata_task", "")
            jrn_std = safe_float(ex.get("metadata_jrn_std", 0))
            ci_lo = jrn_val - 1.96 * jrn_std / math.sqrt(3)
            ci_hi = jrn_val + 1.96 * jrn_std / math.sqrt(3)
            all_jrn_values.append(jrn_val)
            if jrn_val > 1.1:
                cat = "high"
            elif jrn_val < 0.9:
                cat = "low"
            else:
                cat = "critical"
            examples.append({
                "input": f"JRN measurement: dataset=rel-f1, join={join_label}, task={task}, probe={probe}",
                "output": f"JRN={jrn_val:.4f} +/- {jrn_std:.4f}, probe={probe}, category={cat}",
                "metadata_dataset": "rel-f1",
                "metadata_join": join_label,
                "metadata_task": task,
                "metadata_hop_distance": 1.0,
                "metadata_source": "exp_id2_it6",
                "predict_jrn_probe": f"{jrn_val:.4f}",
                "predict_baseline_uniform": "1.0",
                "eval_jrn_mean": round(jrn_val, 6),
                "eval_jrn_std": round(jrn_std, 6),
                "eval_jrn_ci_lo": round(ci_lo, 6),
                "eval_jrn_ci_hi": round(ci_hi, 6),
                "eval_jrn_above_threshold": 1 if jrn_val > 1.0 else 0,
                "eval_jrn_category_high": 1 if cat == "high" else 0,
                "eval_jrn_category_critical": 1 if cat == "critical" else 0,
                "eval_jrn_category_low": 1 if cat == "low" else 0,
            })

    # From exp_id1_it5: rel-stack JRN measurements
    d5 = deps["exp_id1_it5"]
    seen_stack = {(e["metadata_join"], e["metadata_task"]) for e in examples if e["metadata_dataset"] == "rel-stack"}
    for ds in d5["datasets"]:
        for ex in ds["examples"]:
            join_name = ex.get("metadata_join", "")
            task_name = ex.get("metadata_task", "")
            if (join_name, task_name) in seen_stack:
                continue
            jrn_val = safe_float(ex.get("metadata_jrn_mean"))
            if jrn_val == 0.0:
                continue
            all_jrn_values.append(jrn_val)
            if jrn_val > 1.1:
                cat = "high"
            elif jrn_val < 0.9:
                cat = "low"
            else:
                cat = "critical"
            examples.append({
                "input": f"JRN measurement: dataset=rel-stack, join={join_name}, task={task_name}",
                "output": f"JRN={jrn_val:.4f}, category={cat} (GBM probe, rel-stack)",
                "metadata_dataset": "rel-stack",
                "metadata_join": join_name,
                "metadata_task": task_name,
                "metadata_hop_distance": 1.0,
                "metadata_source": "exp_id1_it5",
                "predict_jrn_probe": f"{jrn_val:.4f}",
                "predict_baseline_uniform": "1.0",
                "eval_jrn_mean": round(jrn_val, 6),
                "eval_jrn_std": 0.0,
                "eval_jrn_ci_lo": round(jrn_val, 6),
                "eval_jrn_ci_hi": round(jrn_val, 6),
                "eval_jrn_above_threshold": 1 if jrn_val > 1.0 else 0,
                "eval_jrn_category_high": 1 if cat == "high" else 0,
                "eval_jrn_category_critical": 1 if cat == "critical" else 0,
                "eval_jrn_category_low": 1 if cat == "low" else 0,
            })

    arr = np.array(all_jrn_values) if all_jrn_values else np.array([1.0])
    metrics = {
        "t1_n_measurements": len(examples),
        "t1_n_high": sum(1 for e in examples if e["eval_jrn_category_high"] == 1),
        "t1_n_critical": sum(1 for e in examples if e["eval_jrn_category_critical"] == 1),
        "t1_n_low": sum(1 for e in examples if e["eval_jrn_category_low"] == 1),
        "t1_mean_jrn": round(float(arr.mean()), 6),
        "t1_std_jrn": round(float(arr.std()), 6),
        "t1_min_jrn": round(float(arr.min()), 6),
        "t1_max_jrn": round(float(arr.max()), 6),
    }
    logger.info(f"Table 1: {len(examples)} JRN measurements, mean={metrics['t1_mean_jrn']:.4f}")
    return examples, metrics


# ===================================================================
# TABLE 2: PROBE VALIDITY (Fisher z meta-analysis)
# ===================================================================

def build_table_2(deps: dict) -> tuple[list[dict], dict]:
    """Probe validity via Fisher z random-effects meta-analysis."""
    logger.info("Building Table 2: Probe Validity")
    examples = []
    fisher_zs = []
    fisher_vars = []

    # From exp_id4_it5: convergence analysis rho values
    d3 = deps["exp_id4_it5"]
    meta = d3["metadata"]
    min_cost = meta["part_a_convergence"]["minimum_cost_config"]
    for task, vals in min_cost["spearman_rho_per_task"].items():
        rho = safe_float(vals["rho"])
        n = 13  # 13 joins
        fz = fisher_z(rho)
        fz_var = 1.0 / max(n - 3, 1)
        fisher_zs.append(fz)
        fisher_vars.append(fz_var)
        examples.append({
            "input": f"Probe validity: dataset=rel-f1, task={task}, source=convergence_cheapest",
            "output": f"Spearman rho={rho:.4f}, Fisher_z={fz:.4f}, var={fz_var:.4f}",
            "metadata_dataset": "rel-f1",
            "metadata_task": task,
            "metadata_source": "exp_id4_it5_convergence",
            "predict_jrn_probe": f"{rho:.4f}",
            "predict_baseline_uniform": "0.0",
            "eval_spearman_rho": round(rho, 6),
            "eval_fisher_z": round(fz, 6),
            "eval_fisher_z_var": round(fz_var, 6),
            "eval_probe_valid": 1 if rho > 0.7 else 0,
            "eval_n_joins": n,
        })

    # From exp_id1_it6: cross-task transfer rho values
    d1 = deps["exp_id1_it6"]
    for ds in d1["datasets"]:
        ds_name = ds["dataset"]
        for ex in ds["examples"]:
            if ex.get("metadata_type") == "transfer_analysis":
                out = json.loads(ex["output"])
                for task, vals in out.get("per_task_transfer_rho", {}).items():
                    rho = safe_float(vals.get("rho", 0))
                    n_joins = out.get("num_common_joins", 11)
                    fz = fisher_z(rho)
                    fz_var = 1.0 / max(n_joins - 3, 1)
                    fisher_zs.append(fz)
                    fisher_vars.append(fz_var)
                    examples.append({
                        "input": f"Probe validity: dataset={ds_name}, task={task}, source=transfer_rho",
                        "output": f"Leave-one-out transfer rho={rho:.4f}, Fisher_z={fz:.4f}",
                        "metadata_dataset": ds_name,
                        "metadata_task": task,
                        "metadata_source": "exp_id1_it6_transfer",
                        "predict_jrn_probe": f"{rho:.4f}",
                        "predict_baseline_uniform": "0.0",
                        "eval_spearman_rho": round(rho, 6),
                        "eval_fisher_z": round(fz, 6),
                        "eval_fisher_z_var": round(fz_var, 6),
                        "eval_probe_valid": 1 if rho > 0.7 else 0,
                        "eval_n_joins": n_joins,
                    })

    # From exp_id1_it5: compounding Spearman r
    d5 = deps["exp_id1_it5"]
    comp = d5["metadata"].get("part_a_compounding", {})
    spearman_r = safe_float(comp.get("spearman_r", 0))
    n_chains = comp.get("n_chains_tested", 14)
    fz = fisher_z(spearman_r)
    fz_var = 1.0 / max(n_chains - 3, 1)
    fisher_zs.append(fz)
    fisher_vars.append(fz_var)
    examples.append({
        "input": "Probe validity: dataset=rel-stack, source=compounding_spearman",
        "output": f"Compounding Spearman r={spearman_r:.4f}, Fisher_z={fz:.4f}",
        "metadata_dataset": "rel-stack",
        "metadata_task": "compounding",
        "metadata_source": "exp_id1_it5_compounding",
        "predict_jrn_probe": f"{spearman_r:.4f}",
        "predict_baseline_uniform": "0.0",
        "eval_spearman_rho": round(spearman_r, 6),
        "eval_fisher_z": round(fz, 6),
        "eval_fisher_z_var": round(fz_var, 6),
        "eval_probe_valid": 1 if spearman_r > 0.7 else 0,
        "eval_n_joins": n_chains,
    })

    # Random-effects meta-analysis (DerSimonian-Laird)
    fz_arr = np.array(fisher_zs)
    var_arr = np.array(fisher_vars)
    w_fe = 1.0 / var_arr  # fixed-effect weights

    # Q statistic
    pooled_fe = np.sum(w_fe * fz_arr) / np.sum(w_fe)
    Q = np.sum(w_fe * (fz_arr - pooled_fe) ** 2)
    k = len(fz_arr)
    df = max(k - 1, 1)

    # tau^2 (DerSimonian-Laird)
    c = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0, (Q - df) / c)

    # Random-effects weights
    w_re = 1.0 / (var_arr + tau2)
    pooled_z = np.sum(w_re * fz_arr) / np.sum(w_re)
    pooled_var = 1.0 / np.sum(w_re)
    pooled_rho = inv_fisher_z(float(pooled_z))

    # I^2
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # Prediction interval
    pred_lo = inv_fisher_z(float(pooled_z - 1.96 * math.sqrt(pooled_var + tau2)))
    pred_hi = inv_fisher_z(float(pooled_z + 1.96 * math.sqrt(pooled_var + tau2)))

    metrics = {
        "t2_pooled_rho": round(pooled_rho, 6),
        "t2_I_squared": round(I2, 4),
        "t2_Q_stat": round(float(Q), 4),
        "t2_tau_squared": round(float(tau2), 6),
        "t2_prediction_interval_lo": round(pred_lo, 6),
        "t2_prediction_interval_hi": round(pred_hi, 6),
        "t2_n_estimates": k,
    }
    logger.info(f"Table 2: pooled_rho={pooled_rho:.4f}, I2={I2:.1f}%, k={k}")
    return examples, metrics


# ===================================================================
# TABLE 3: COST EFFICIENCY
# ===================================================================

def build_table_3(deps: dict) -> tuple[list[dict], dict]:
    """Cost efficiency from exp_id4_it5."""
    logger.info("Building Table 3: Cost Efficiency")
    examples = []
    d = deps["exp_id4_it5"]
    meta = d["metadata"]

    # Convergence examples (64 configs)
    for ds in d["datasets"]:
        for ex in ds["examples"]:
            if ex.get("metadata_experiment_part") == "A_convergence":
                rho = safe_float(ex.get("metadata_mean_rho"))
                tau = safe_float(ex.get("metadata_mean_tau"))
                wc = safe_float(json.loads(ex["output"]).get("wall_clock_seconds", 0))
                examples.append({
                    "input": ex["input"],
                    "output": ex["output"],
                    "metadata_task": "convergence",
                    "metadata_method": "jrn_probe",
                    "metadata_source": "exp_id4_it5",
                    "predict_jrn_probe": f"{rho:.4f}",
                    "predict_baseline_uniform": "1.0",
                    "eval_mean_spearman_rho": round(rho, 6),
                    "eval_mean_kendall_tau": round(tau, 6),
                    "eval_wall_clock_seconds": round(wc, 4),
                    "eval_converged": 1 if rho > 0.9 else 0,
                })

    # Cost comparison examples (3 tasks x 4 methods)
    cost_comp = meta.get("part_b_cost_comparison", {})
    cost_ratios = []
    perf_ratios = []
    for task, vals in cost_comp.items():
        for method in ["jrn", "greedy", "exhaustive", "random"]:
            if method == "jrn":
                wc = safe_float(vals.get("jrn_time"))
                perf = safe_float(vals.get("jrn_perf"))
                n_models = safe_float(vals.get("n_models_jrn"))
                cost_ratio = safe_float(vals.get("cost_ratio_jrn_vs_greedy"))
                pps = safe_float(vals.get("perf_per_second_jrn"))
                cost_ratios.append(cost_ratio)
                pr = safe_float(vals.get("jrn_perf")) / max(safe_float(vals.get("greedy_perf")), 1e-10)
                perf_ratios.append(pr)
            elif method == "greedy":
                wc = safe_float(vals.get("greedy_time"))
                perf = safe_float(vals.get("greedy_perf"))
                n_models = safe_float(vals.get("n_models_greedy"))
                cost_ratio = 1.0
                pps = safe_float(vals.get("perf_per_second_greedy"))
            elif method == "exhaustive":
                wc = safe_float(vals.get("exhaust_time_extrapolated"))
                perf = safe_float(vals.get("exhaust_perf"))
                n_models = safe_float(vals.get("n_models_exhaust_full"))
                cost_ratio = safe_float(vals.get("cost_ratio_jrn_vs_exhaust"))
                pps = perf / max(wc, 0.001)
            else:  # random
                wc = safe_float(vals.get("random_time"))
                perf = safe_float(vals.get("random_perf"))
                n_models = safe_float(vals.get("n_models_random"))
                cost_ratio = wc / max(safe_float(vals.get("greedy_time")), 0.001)
                pps = perf / max(wc, 0.001)

            examples.append({
                "input": f"Cost comparison: task={task}, method={method}",
                "output": f"wall_clock={wc:.2f}s, perf={perf:.6f}, n_models={int(n_models)}, perf_per_sec={pps:.6f}",
                "metadata_task": task,
                "metadata_method": method,
                "metadata_source": "exp_id4_it5",
                "predict_jrn_probe": f"{perf:.6f}",
                "predict_baseline_uniform": "1.0",
                "eval_wall_clock_seconds": round(wc, 4),
                "eval_n_models_trained": round(n_models, 0),
                "eval_performance": round(perf, 6),
                "eval_perf_per_second": round(pps, 6),
                "eval_cost_ratio_vs_greedy": round(cost_ratio, 6),
            })

    # Breakeven examples
    be = meta.get("breakeven_analysis", {})
    for row in be.get("scaling_table", []):
        examples.append({
            "input": f"Breakeven analysis: J={row['J']} joins",
            "output": f"JRN models={row['jrn_models']}, greedy={row['greedy_models']}, speedup={row['speedup_jrn_vs_greedy']:.1f}x",
            "metadata_task": "breakeven",
            "metadata_method": "scaling",
            "metadata_source": "exp_id4_it5",
            "predict_jrn_probe": f"{row['jrn_models']}",
            "predict_baseline_uniform": f"{row['greedy_models']}",
            "eval_n_joins": safe_float(row["J"]),
            "eval_jrn_models": safe_float(row["jrn_models"]),
            "eval_greedy_models": safe_float(row["greedy_models"]),
            "eval_speedup_vs_greedy": safe_float(row["speedup_jrn_vs_greedy"]),
        })

    conv_examples = [e for e in examples if e.get("metadata_task") == "convergence"]
    n_converged = sum(1 for e in conv_examples if e.get("eval_converged", 0) == 1)
    n_conv_total = len(conv_examples)

    metrics = {
        "t3_mean_cost_ratio_jrn_vs_greedy": round(float(np.mean(cost_ratios)) if cost_ratios else 0, 6),
        "t3_mean_perf_ratio": round(float(np.mean(perf_ratios)) if perf_ratios else 0, 6),
        "t3_n_converged": n_converged,
        "t3_n_convergence_configs": n_conv_total,
        "t3_convergence_rate": round(n_converged / max(n_conv_total, 1), 4),
        "t3_n_cost_examples": len(examples),
    }
    logger.info(f"Table 3: {len(examples)} examples, mean cost ratio={metrics['t3_mean_cost_ratio_jrn_vs_greedy']:.4f}")
    return examples, metrics


# ===================================================================
# TABLE 4: FK-SHUFFLING DECOMPOSITION
# ===================================================================

def build_table_4(deps: dict) -> tuple[list[dict], dict]:
    """FK-shuffling structural vs feature decomposition."""
    logger.info("Building Table 4: FK-Shuffling Decomposition")
    examples = []
    all_structural = []
    all_feature = []
    all_normal = []
    all_shuffled = []

    # From exp_id4_it4: rel-f1 (39 pairs)
    d4 = deps["exp_id4_it4"]
    for ds in d4["datasets"]:
        for ex in ds["examples"]:
            out = json.loads(ex["output"])
            normal_jrn = safe_float(out.get("normal_jrn"))
            shuffled_jrn = safe_float(out.get("shuffled_jrn_mean", out.get("shuffled_jrn")))
            structural = safe_float(out.get("jrn_structural"))
            feature = safe_float(out.get("jrn_feature"))
            struct_frac = safe_float(out.get("structural_fraction"))

            all_normal.append(normal_jrn)
            all_shuffled.append(shuffled_jrn)
            all_structural.append(structural)
            all_feature.append(feature)

            task_name = ex.get("metadata_task_name", "").split("/")[-1]
            join_label = f"{ex.get('metadata_source_table', '')}->{ex.get('metadata_target_table', '')}"

            examples.append({
                "input": f"FK-shuffling: dataset=rel-f1, join={join_label}, task={task_name}",
                "output": f"normal_JRN={normal_jrn:.4f}, shuffled_JRN={shuffled_jrn:.4f}, structural={structural:.4f}, feature={feature:.4f}",
                "metadata_dataset": "rel-f1",
                "metadata_join": join_label,
                "metadata_task": task_name,
                "metadata_source": "exp_id4_it4",
                "predict_jrn_probe": f"{normal_jrn:.4f}",
                "predict_baseline_uniform": f"{shuffled_jrn:.4f}",
                "eval_normal_jrn": round(normal_jrn, 6),
                "eval_shuffled_jrn": round(shuffled_jrn, 6),
                "eval_structural_component": round(structural, 6),
                "eval_feature_component": round(feature, 6),
                "eval_structural_fraction": round(struct_frac, 6),
                "eval_structural_dominant": 1 if abs(structural) > abs(feature) else 0,
            })

    # From exp_id1_it5: rel-stack FK-shuffling (16 pairs)
    d5 = deps["exp_id1_it5"]
    decomp = d5["metadata"].get("part_b_fk_shuffling", {}).get("decomposition", [])
    for item in decomp:
        normal_jrn = safe_float(item.get("normal_jrn"))
        shuffled_jrn = safe_float(item.get("shuffled_jrn"))
        structural = safe_float(item.get("jrn_structural"))
        feature = safe_float(item.get("jrn_feature"))
        struct_frac = safe_float(item.get("structural_fraction"))

        all_normal.append(normal_jrn)
        all_shuffled.append(shuffled_jrn)
        all_structural.append(structural)
        all_feature.append(feature)

        examples.append({
            "input": f"FK-shuffling: dataset=rel-stack, join={item.get('join', '')}, task={item.get('task', '')}",
            "output": f"normal_JRN={normal_jrn:.4f}, shuffled_JRN={shuffled_jrn:.4f}, structural={structural:.4f}, feature={feature:.4f}",
            "metadata_dataset": "rel-stack",
            "metadata_join": str(item.get("join", "")),
            "metadata_task": str(item.get("task", "")),
            "metadata_source": "exp_id1_it5",
            "predict_jrn_probe": f"{normal_jrn:.4f}",
            "predict_baseline_uniform": f"{shuffled_jrn:.4f}",
            "eval_normal_jrn": round(normal_jrn, 6),
            "eval_shuffled_jrn": round(shuffled_jrn, 6),
            "eval_structural_component": round(structural, 6),
            "eval_feature_component": round(feature, 6),
            "eval_structural_fraction": round(struct_frac, 6),
            "eval_structural_dominant": 1 if abs(structural) > abs(feature) else 0,
        })

    # Pooled statistics
    normal_arr = np.array(all_normal)
    shuffled_arr = np.array(all_shuffled)
    struct_arr = np.array(all_structural)
    feat_arr = np.array(all_feature)

    # Paired t-test (pooled)
    if len(normal_arr) > 1:
        t_stat, t_pval = stats.ttest_rel(normal_arr, shuffled_arr)
        # Cohen's d for paired data
        diffs = normal_arr - shuffled_arr
        d_val = float(np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-10))
    else:
        t_stat, t_pval, d_val = 0.0, 1.0, 0.0

    # Wilcoxon
    try:
        w_stat, w_pval = stats.wilcoxon(normal_arr - shuffled_arr)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    struct_dom_pct = sum(1 for e in examples if e["eval_structural_dominant"] == 1) / max(len(examples), 1) * 100

    metrics = {
        "t4_pooled_cohens_d": round(d_val, 6),
        "t4_pooled_t_stat": round(float(t_stat), 6),
        "t4_pooled_t_p_value": round(float(t_pval), 8),
        "t4_wilcoxon_w": round(float(w_stat), 4),
        "t4_wilcoxon_p": round(float(w_pval), 8),
        "t4_structural_dominant_pct": round(struct_dom_pct, 4),
        "t4_mean_structural": round(float(struct_arr.mean()), 6),
        "t4_mean_feature": round(float(feat_arr.mean()), 6),
        "t4_n_pairs": len(examples),
    }
    logger.info(f"Table 4: {len(examples)} pairs, Cohen's d={d_val:.4f}, p={t_pval:.6f}")
    return examples, metrics


# ===================================================================
# TABLE 5: TRAINING-FREE PROXIES
# ===================================================================

def build_table_5(deps: dict) -> tuple[list[dict], dict]:
    """Rank proxy approaches by correlation with full JRN."""
    logger.info("Building Table 5: Training-Free Proxies")
    examples = []

    # GBM reference (from exp_id4_it5 convergence)
    d3 = deps["exp_id4_it5"]
    meta = d3["metadata"]
    conv = meta["part_a_convergence"]
    ref_time = conv.get("convergence_summary", {}).get("full_budget_time_per_task", 14.3)

    # Cheapest config proxy
    min_cfg = conv.get("minimum_cost_config", {})
    mean_rho_cheap = safe_float(min_cfg.get("mean_rho", 0.954))
    mean_tau_cheap = safe_float(min_cfg.get("mean_tau", 0.877))
    wc_cheap = safe_float(min_cfg.get("wall_clock_seconds", 0.953))

    examples.append({
        "input": "Proxy: LightGBM cheapest config (25 trees, depth 3, 10% subsample), dataset=rel-f1",
        "output": f"rho={mean_rho_cheap:.4f}, tau={mean_tau_cheap:.4f}, time={wc_cheap:.3f}s",
        "metadata_dataset": "rel-f1",
        "metadata_proxy_method": "lgbm_cheapest",
        "metadata_source": "exp_id4_it5",
        "predict_jrn_probe": f"{mean_rho_cheap:.4f}",
        "predict_baseline_uniform": "0.0",
        "eval_spearman_rho_vs_reference": round(mean_rho_cheap, 6),
        "eval_kendall_tau_vs_reference": round(mean_tau_cheap, 6),
        "eval_wall_clock_seconds": round(wc_cheap, 4),
        "eval_cost_normalized": round(wc_cheap / max(ref_time, 0.001), 6),
    })

    # Full budget LightGBM reference
    examples.append({
        "input": "Proxy: LightGBM full budget (100 trees, depth 5, 100% data), dataset=rel-f1",
        "output": f"rho=1.0 (reference), time={ref_time:.3f}s",
        "metadata_dataset": "rel-f1",
        "metadata_proxy_method": "lgbm_reference",
        "metadata_source": "exp_id4_it5",
        "predict_jrn_probe": "1.0000",
        "predict_baseline_uniform": "0.0",
        "eval_spearman_rho_vs_reference": 1.0,
        "eval_kendall_tau_vs_reference": 1.0,
        "eval_wall_clock_seconds": round(ref_time, 4),
        "eval_cost_normalized": 1.0,
    })

    # MLP probe (from exp_id2_it6)
    d2 = deps["exp_id2_it6"]
    # Compounding summaries have R2/spearman for MLP
    for ex in d2["datasets"][0]["examples"]:
        if ex.get("metadata_measurement_type") == "compounding_summary" and ex.get("metadata_probe") == "MLP":
            if ex.get("metadata_model") == "log_linear":
                mlp_spearman = safe_float(ex.get("metadata_spearman_r"))
                examples.append({
                    "input": "Proxy: MLP probe (log-linear compounding), dataset=rel-f1",
                    "output": f"Compounding Spearman r={mlp_spearman:.4f}",
                    "metadata_dataset": "rel-f1",
                    "metadata_proxy_method": "mlp_probe",
                    "metadata_source": "exp_id2_it6",
                    "predict_jrn_probe": f"{mlp_spearman:.4f}",
                    "predict_baseline_uniform": "0.0",
                    "eval_spearman_rho_vs_reference": round(mlp_spearman, 6),
                    "eval_kendall_tau_vs_reference": 0.0,
                    "eval_wall_clock_seconds": 0.0,
                    "eval_cost_normalized": 0.0,
                })
                break

    # GBM probe (from exp_id2_it6)
    for ex in d2["datasets"][0]["examples"]:
        if ex.get("metadata_measurement_type") == "compounding_summary" and ex.get("metadata_probe") == "GBM":
            if ex.get("metadata_model") == "log_linear":
                gbm_spearman = safe_float(ex.get("metadata_spearman_r"))
                examples.append({
                    "input": "Proxy: GBM probe (log-linear compounding), dataset=rel-f1",
                    "output": f"Compounding Spearman r={gbm_spearman:.4f}",
                    "metadata_dataset": "rel-f1",
                    "metadata_proxy_method": "gbm_probe",
                    "metadata_source": "exp_id2_it6",
                    "predict_jrn_probe": f"{gbm_spearman:.4f}",
                    "predict_baseline_uniform": "0.0",
                    "eval_spearman_rho_vs_reference": round(gbm_spearman, 6),
                    "eval_kendall_tau_vs_reference": 0.0,
                    "eval_wall_clock_seconds": 0.0,
                    "eval_cost_normalized": 0.0,
                })
                break

    # Fanout statistics proxy (from exp_id4_it4 correlation_analysis)
    d4 = deps["exp_id4_it4"]
    corr = d4["metadata"]["results"].get("correlation_analysis", {})
    fanout_rho = safe_float(corr.get("structural_vs_fanout_mean", {}).get("rho", 0))
    examples.append({
        "input": "Proxy: Fanout statistics (training-free), dataset=rel-f1",
        "output": f"Correlation structural JRN vs fanout: rho={fanout_rho:.4f}",
        "metadata_dataset": "rel-f1",
        "metadata_proxy_method": "fanout_stats",
        "metadata_source": "exp_id4_it4",
        "predict_jrn_probe": f"{fanout_rho:.4f}",
        "predict_baseline_uniform": "0.0",
        "eval_spearman_rho_vs_reference": round(fanout_rho, 6),
        "eval_kendall_tau_vs_reference": 0.0,
        "eval_wall_clock_seconds": 0.0,
        "eval_cost_normalized": 0.0,
    })

    # Add per-task convergence configs as proxy examples
    for ds in d3["datasets"]:
        for ex in ds["examples"]:
            if ex.get("metadata_experiment_part") == "A_convergence":
                rho = safe_float(ex.get("metadata_mean_rho"))
                wc = safe_float(json.loads(ex["output"]).get("wall_clock_seconds", 0))
                n_est = ex.get("metadata_n_estimators", 0)
                depth = ex.get("metadata_max_depth", 0)
                sub = ex.get("metadata_subsample_frac", 0)
                examples.append({
                    "input": f"Proxy config: n_est={n_est}, depth={depth}, subsample={sub}, dataset=rel-f1",
                    "output": f"rho={rho:.4f}, time={wc:.3f}s",
                    "metadata_dataset": "rel-f1",
                    "metadata_proxy_method": f"lgbm_{n_est}_{depth}_{sub}",
                    "metadata_source": "exp_id4_it5",
                    "predict_jrn_probe": f"{rho:.4f}",
                    "predict_baseline_uniform": "0.0",
                    "eval_spearman_rho_vs_reference": round(rho, 6),
                    "eval_kendall_tau_vs_reference": 0.0,
                    "eval_wall_clock_seconds": round(wc, 4),
                    "eval_cost_normalized": round(wc / max(ref_time, 0.001), 6),
                })

    rhos = [e["eval_spearman_rho_vs_reference"] for e in examples if e["eval_spearman_rho_vs_reference"] != 0]
    metrics = {
        "t5_n_proxy_configs": len(examples),
        "t5_max_rho": round(max(rhos) if rhos else 0, 6),
        "t5_min_rho": round(min(rhos) if rhos else 0, 6),
        "t5_mean_rho": round(float(np.mean(rhos)) if rhos else 0, 6),
    }
    logger.info(f"Table 5: {len(examples)} proxy configs")
    return examples, metrics


# ===================================================================
# TABLE 6: ARCHITECTURE COMPARISON
# ===================================================================

def build_table_6(deps: dict) -> tuple[list[dict], dict]:
    """JRN-guided vs greedy vs random selection."""
    logger.info("Building Table 6: Architecture Comparison")
    examples = []

    d = deps["exp_id4_it5"]
    cost_comp = d["metadata"].get("part_b_cost_comparison", {})
    be = d["metadata"].get("breakeven_analysis", {}).get("perf_comparison", {})

    jrn_wins_greedy = 0
    jrn_wins_random = 0
    total_comparisons = 0
    perf_ratios = []

    for task, vals in cost_comp.items():
        jrn_perf = safe_float(vals.get("jrn_perf"))
        greedy_perf = safe_float(vals.get("greedy_perf"))
        random_perf = safe_float(vals.get("random_perf"))
        exhaust_perf = safe_float(vals.get("exhaust_perf"))

        perf_ratio = jrn_perf / max(greedy_perf, 1e-10)
        perf_ratios.append(perf_ratio)
        total_comparisons += 1

        be_task = be.get(task, {})

        # JRN vs Greedy
        jrn_wins = 1 if jrn_perf >= greedy_perf else 0
        jrn_wins_greedy += jrn_wins
        examples.append({
            "input": f"Architecture: dataset=rel-f1, task={task}, comparison=jrn_vs_greedy",
            "output": f"JRN perf={jrn_perf:.6f}, greedy perf={greedy_perf:.6f}, ratio={perf_ratio:.4f}",
            "metadata_dataset": "rel-f1",
            "metadata_task": task,
            "metadata_method_pair": "jrn_vs_greedy",
            "metadata_source": "exp_id4_it5",
            "predict_jrn_probe": f"{jrn_perf:.6f}",
            "predict_baseline_uniform": f"{greedy_perf:.6f}",
            "eval_jrn_perf": round(jrn_perf, 6),
            "eval_baseline_perf": round(greedy_perf, 6),
            "eval_jrn_wins": jrn_wins,
            "eval_perf_difference": round(jrn_perf - greedy_perf, 6),
            "eval_perf_ratio": round(perf_ratio, 6),
        })

        # JRN vs Random
        jrn_wins_r = 1 if jrn_perf >= random_perf else 0
        jrn_wins_random += jrn_wins_r
        examples.append({
            "input": f"Architecture: dataset=rel-f1, task={task}, comparison=jrn_vs_random",
            "output": f"JRN perf={jrn_perf:.6f}, random perf={random_perf:.6f}",
            "metadata_dataset": "rel-f1",
            "metadata_task": task,
            "metadata_method_pair": "jrn_vs_random",
            "metadata_source": "exp_id4_it5",
            "predict_jrn_probe": f"{jrn_perf:.6f}",
            "predict_baseline_uniform": f"{random_perf:.6f}",
            "eval_jrn_perf": round(jrn_perf, 6),
            "eval_baseline_perf": round(random_perf, 6),
            "eval_jrn_wins": jrn_wins_r,
            "eval_perf_difference": round(jrn_perf - random_perf, 6),
            "eval_perf_ratio": round(jrn_perf / max(random_perf, 1e-10), 6),
        })

        # JRN vs Exhaustive
        examples.append({
            "input": f"Architecture: dataset=rel-f1, task={task}, comparison=jrn_vs_exhaustive",
            "output": f"JRN perf={jrn_perf:.6f}, exhaustive perf={exhaust_perf:.6f}",
            "metadata_dataset": "rel-f1",
            "metadata_task": task,
            "metadata_method_pair": "jrn_vs_exhaustive",
            "metadata_source": "exp_id4_it5",
            "predict_jrn_probe": f"{jrn_perf:.6f}",
            "predict_baseline_uniform": f"{exhaust_perf:.6f}",
            "eval_jrn_perf": round(jrn_perf, 6),
            "eval_baseline_perf": round(exhaust_perf, 6),
            "eval_jrn_wins": 1 if jrn_perf >= exhaust_perf else 0,
            "eval_perf_difference": round(jrn_perf - exhaust_perf, 6),
            "eval_perf_ratio": round(jrn_perf / max(exhaust_perf, 1e-10), 6),
        })

    # Binomial CIs
    wr_greedy = jrn_wins_greedy / max(total_comparisons, 1)
    wr_random = jrn_wins_random / max(total_comparisons, 1)
    ci_lo_g, ci_hi_g = clopper_pearson(jrn_wins_greedy, total_comparisons)
    ci_lo_r, ci_hi_r = clopper_pearson(jrn_wins_random, total_comparisons)

    metrics = {
        "t6_win_rate_jrn_vs_greedy": round(wr_greedy, 6),
        "t6_win_rate_jrn_vs_random": round(wr_random, 6),
        "t6_ci_lo_vs_greedy": round(ci_lo_g, 6),
        "t6_ci_hi_vs_greedy": round(ci_hi_g, 6),
        "t6_ci_lo_vs_random": round(ci_lo_r, 6),
        "t6_ci_hi_vs_random": round(ci_hi_r, 6),
        "t6_mean_perf_ratio": round(float(np.mean(perf_ratios)) if perf_ratios else 0, 6),
        "t6_n_comparisons": len(examples),
    }
    logger.info(f"Table 6: {len(examples)} comparisons, win_rate_vs_greedy={wr_greedy:.2f}")
    return examples, metrics


# ===================================================================
# TABLE 7: COMPOUNDING UPDATED
# ===================================================================

def build_table_7(deps: dict) -> tuple[list[dict], dict]:
    """Compounding models comparison across probes and datasets."""
    logger.info("Building Table 7: Compounding Updated")
    examples = []

    # From exp_id2_it6: rel-f1, MLP + GBM, 4 models
    d2 = deps["exp_id2_it6"]
    for ex in d2["datasets"][0]["examples"]:
        if ex.get("metadata_measurement_type") != "compounding_summary":
            continue
        model = ex.get("metadata_model", "")
        probe = ex.get("metadata_probe", "")
        r2 = safe_float(ex.get("metadata_r2"))
        spearman = safe_float(ex.get("metadata_spearman_r"))
        mae = safe_float(ex.get("metadata_mae"))
        rmse = safe_float(ex.get("metadata_rmse"))

        examples.append({
            "input": f"Compounding: dataset=rel-f1, probe={probe}, model={model}",
            "output": f"R2={r2:.4f}, Spearman={spearman:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}",
            "metadata_dataset": "rel-f1",
            "metadata_probe": probe,
            "metadata_compounding_model": model,
            "metadata_source": "exp_id2_it6",
            "predict_jrn_probe": f"{r2:.4f}",
            "predict_baseline_uniform": "0.0",
            "eval_r_squared": round(r2, 6),
            "eval_spearman_r": round(spearman, 6),
            "eval_mae": round(mae, 6),
            "eval_rmse": round(rmse, 6),
            "eval_n_chains": 12,
        })

    # From exp_id1_it5: rel-stack compounding (multiplicative only, with Spearman)
    d5 = deps["exp_id1_it5"]
    comp = d5["metadata"].get("part_a_compounding", {})
    r2_mult = safe_float(comp.get("compounding_r2"))
    r2_add = safe_float(comp.get("additive_log_r2"))
    spearman_r = safe_float(comp.get("spearman_r"))
    n_chains = comp.get("n_chains_tested", 14)

    examples.append({
        "input": "Compounding: dataset=rel-stack, probe=GBM, model=multiplicative",
        "output": f"R2={r2_mult:.4f}, Spearman={spearman_r:.4f}, n_chains={n_chains}",
        "metadata_dataset": "rel-stack",
        "metadata_probe": "GBM",
        "metadata_compounding_model": "multiplicative",
        "metadata_source": "exp_id1_it5",
        "predict_jrn_probe": f"{r2_mult:.4f}",
        "predict_baseline_uniform": "0.0",
        "eval_r_squared": round(r2_mult, 6),
        "eval_spearman_r": round(spearman_r, 6),
        "eval_mae": 0.0,
        "eval_rmse": 0.0,
        "eval_n_chains": n_chains,
    })

    examples.append({
        "input": "Compounding: dataset=rel-stack, probe=GBM, model=additive_log",
        "output": f"R2={r2_add:.4f}, Spearman={spearman_r:.4f}, n_chains={n_chains}",
        "metadata_dataset": "rel-stack",
        "metadata_probe": "GBM",
        "metadata_compounding_model": "additive_log",
        "metadata_source": "exp_id1_it5",
        "predict_jrn_probe": f"{r2_add:.4f}",
        "predict_baseline_uniform": "0.0",
        "eval_r_squared": round(r2_add, 6),
        "eval_spearman_r": round(spearman_r, 6),
        "eval_mae": 0.0,
        "eval_rmse": 0.0,
        "eval_n_chains": n_chains,
    })

    # Find best model per dataset-probe combo
    best_r2 = max((e["eval_r_squared"] for e in examples), default=0)
    log_linear_r2_vals = [e["eval_r_squared"] for e in examples if e.get("metadata_compounding_model") == "log_linear"]
    best_log_linear = max(log_linear_r2_vals) if log_linear_r2_vals else 0

    metrics = {
        "t7_best_r2": round(best_r2, 6),
        "t7_best_log_linear_r2": round(best_log_linear, 6),
        "t7_n_compounding_entries": len(examples),
    }
    logger.info(f"Table 7: {len(examples)} compounding entries, best R2={best_r2:.4f}")
    return examples, metrics


# ===================================================================
# TABLE 8: CROSS-TASK TRANSFER
# ===================================================================

def build_table_8(deps: dict) -> tuple[list[dict], dict]:
    """Cross-task transfer analysis."""
    logger.info("Building Table 8: Cross-Task Transfer")
    examples = []

    d1 = deps["exp_id1_it6"]
    summary = d1["metadata"]["summary"]
    all_W = []
    all_same_entity = []
    all_cross_entity = []

    for ds in d1["datasets"]:
        ds_name = ds["dataset"]
        ds_summary = summary.get(ds_name, {})
        W = safe_float(ds_summary.get("kendalls_W"))
        all_W.append(W)

        same_rho = safe_float(ds_summary.get("same_entity_mean_rho"))
        cross_rho = safe_float(ds_summary.get("cross_entity_mean_rho"))
        all_same_entity.append(same_rho)
        all_cross_entity.append(cross_rho)

        mean_transfer = safe_float(ds_summary.get("mean_transfer_rho"))
        gap_pct = safe_float(ds_summary.get("mean_gap_pct"))

        # Dataset-level summary
        examples.append({
            "input": f"Transfer summary: dataset={ds_name}",
            "output": f"Kendall's W={W:.4f}, same_entity_rho={same_rho:.4f}, cross_entity_rho={cross_rho:.4f}, gap={gap_pct:.1f}%",
            "metadata_dataset": ds_name,
            "metadata_type": "dataset_summary",
            "metadata_source": "exp_id1_it6",
            "predict_jrn_probe": f"{W:.4f}",
            "predict_baseline_uniform": "0.0",
            "eval_kendalls_W": round(W, 6),
            "eval_mean_transfer_rho": round(mean_transfer, 6),
            "eval_same_entity_mean_rho": round(same_rho, 6),
            "eval_cross_entity_mean_rho": round(cross_rho, 6),
            "eval_universal_vs_pertask_gap_pct": round(gap_pct, 4),
        })

        # Pairwise transfer examples
        for ex in ds["examples"]:
            if ex.get("metadata_type") == "concordance_analysis":
                out = json.loads(ex["output"])
                pairwise = out.get("pairwise_spearman_rho", {})
                for pair_name, pair_vals in pairwise.items():
                    rho = safe_float(pair_vals.get("rho"))
                    tasks = pair_name.split("_vs_")
                    # Determine if same entity
                    same_entity = 0
                    if ds_name == "rel-f1":
                        # Same entity if both are driver-* tasks
                        if len(tasks) == 2 and tasks[0].startswith("driver") and tasks[1].startswith("driver"):
                            same_entity = 1
                    elif ds_name == "rel-stack":
                        if len(tasks) == 2 and tasks[0].startswith("user") and tasks[1].startswith("user"):
                            same_entity = 1

                    examples.append({
                        "input": f"Pairwise transfer: dataset={ds_name}, pair={pair_name}",
                        "output": f"Spearman rho={rho:.4f}, same_entity={same_entity}",
                        "metadata_dataset": ds_name,
                        "metadata_type": "pairwise_transfer",
                        "metadata_source": "exp_id1_it6",
                        "predict_jrn_probe": f"{rho:.4f}",
                        "predict_baseline_uniform": "0.0",
                        "eval_spearman_rho": round(rho, 6),
                        "eval_same_entity": same_entity,
                    })

        # Leave-one-task-out transfer
        for ex in ds["examples"]:
            if ex.get("metadata_type") == "transfer_analysis":
                out = json.loads(ex["output"])
                for task, vals in out.get("per_task_transfer_rho", {}).items():
                    rho = safe_float(vals.get("rho"))
                    examples.append({
                        "input": f"Leave-one-out transfer: dataset={ds_name}, held_out_task={task}",
                        "output": f"Transfer rho={rho:.4f}",
                        "metadata_dataset": ds_name,
                        "metadata_type": "leave_one_out",
                        "metadata_source": "exp_id1_it6",
                        "predict_jrn_probe": f"{rho:.4f}",
                        "predict_baseline_uniform": "0.0",
                        "eval_spearman_rho": round(rho, 6),
                        "eval_same_entity": 0,
                    })

    metrics = {
        "t8_mean_W": round(float(np.mean(all_W)), 6),
        "t8_same_entity_rho_pooled": round(float(np.mean(all_same_entity)), 6),
        "t8_cross_entity_rho_pooled": round(float(np.mean(all_cross_entity)), 6),
        "t8_n_transfer_examples": len(examples),
    }
    logger.info(f"Table 8: {len(examples)} transfer examples, mean W={metrics['t8_mean_W']:.4f}")
    return examples, metrics


# ===================================================================
# TABLE 9: PRACTITIONER GUIDELINES
# ===================================================================

def build_table_9(deps: dict, all_metrics: dict) -> tuple[list[dict], dict]:
    """Synthesize practitioner guidelines from all experiments."""
    logger.info("Building Table 9: Practitioner Guidelines")

    guidelines = [
        {
            "condition": "Schema with >10 FK joins",
            "recommendation": "Use JRN probe to rank joins before feature engineering (6.5x speedup vs greedy on rel-f1 with 13 joins)",
            "grade": "B",
            "n_datasets": 1,
            "effect_size": round(all_metrics.get("t3_mean_cost_ratio_jrn_vs_greedy", 0.07), 4),
        },
        {
            "condition": "Same-entity prediction tasks",
            "recommendation": "Universal JRN ranking is viable (same-entity rho=0.82-0.87); compute JRN once and reuse across tasks",
            "grade": "B",
            "n_datasets": 2,
            "effect_size": round(all_metrics.get("t8_same_entity_rho_pooled", 0.85), 4),
        },
        {
            "condition": "Cross-entity prediction tasks",
            "recommendation": "Compute per-task JRN rankings (cross-entity rho=0.24 to -0.70); universal rankings fail across entity boundaries",
            "grade": "B",
            "n_datasets": 2,
            "effect_size": round(all_metrics.get("t8_cross_entity_rho_pooled", -0.23), 4),
        },
        {
            "condition": "Multi-hop join chains",
            "recommendation": "Use log-linear compounding model to predict chain JRN from individual JRN values (R2=0.83 on GBM, Spearman=0.91)",
            "grade": "C",
            "n_datasets": 1,
            "effect_size": round(all_metrics.get("t7_best_log_linear_r2", 0.83), 4),
        },
        {
            "condition": "Need to understand JRN signal source",
            "recommendation": "Feature quality dominates structural signal; FK-shuffling shows structural component is significant but secondary (Cohen's d=0.63-0.69)",
            "grade": "B",
            "n_datasets": 2,
            "effect_size": round(all_metrics.get("t4_pooled_cohens_d", 0.65), 4),
        },
        {
            "condition": "Tight compute budget for JRN estimation",
            "recommendation": "Use cheapest LightGBM config (25 trees, depth 3, 10% subsample): achieves rho>0.95 with 15x speedup",
            "grade": "B",
            "n_datasets": 1,
            "effect_size": 0.954,
        },
        {
            "condition": "Choosing between MLP and GBM probes",
            "recommendation": "Prefer GBM probes for compounding analysis (R2=0.83 vs 0.12 for MLP on log-linear model)",
            "grade": "C",
            "n_datasets": 1,
            "effect_size": 0.83,
        },
        {
            "condition": "Cost-constrained join selection",
            "recommendation": "JRN probe achieves 96-99% of greedy performance at 5-11% of the cost; always preferable for J>1 joins",
            "grade": "B",
            "n_datasets": 1,
            "effect_size": round(all_metrics.get("t6_mean_perf_ratio", 0.978), 4),
        },
        {
            "condition": "Interpreting JRN values near 1.0",
            "recommendation": "No direct threshold experiment was run; treat JRN near 1.0 as uncertain signal. JRN>1.1 likely informative, JRN<0.9 likely uninformative",
            "grade": "D",
            "n_datasets": 0,
            "effect_size": 0.0,
        },
        {
            "condition": "New relational dataset",
            "recommendation": "Start with cheapest GBM config (25 trees, 10% data) for initial JRN rankings, validate with full budget if rankings seem unexpected",
            "grade": "B",
            "n_datasets": 1,
            "effect_size": 0.954,
        },
    ]

    examples = []
    for g in guidelines:
        grade_num = {"A": 4, "B": 3, "C": 2, "D": 1}.get(g["grade"], 0)
        examples.append({
            "input": f"Scenario: {g['condition']}",
            "output": f"Recommendation: {g['recommendation']}",
            "metadata_evidence_grade": g["grade"],
            "metadata_source": "synthesis",
            "predict_jrn_probe": f"{g['effect_size']:.4f}",
            "predict_baseline_uniform": "0.0",
            "eval_evidence_grade": grade_num,
            "eval_supporting_datasets": g["n_datasets"],
            "eval_effect_size": g["effect_size"],
        })

    metrics = {
        "t9_n_guidelines": len(examples),
        "t9_n_grade_A": sum(1 for g in guidelines if g["grade"] == "A"),
        "t9_n_grade_B": sum(1 for g in guidelines if g["grade"] == "B"),
        "t9_n_grade_C": sum(1 for g in guidelines if g["grade"] == "C"),
        "t9_n_grade_D": sum(1 for g in guidelines if g["grade"] == "D"),
    }
    logger.info(f"Table 9: {len(examples)} guidelines")
    return examples, metrics


# ===================================================================
# SCORECARD v5
# ===================================================================

def build_scorecard(deps: dict, all_metrics: dict) -> tuple[list[dict], dict]:
    """Hypothesis scorecard v5 with 7 claims."""
    logger.info("Building Hypothesis Scorecard v5")

    claims = [
        {
            "claim_id": 1,
            "claim": "JRN as valid per-join metric",
            "grade": "A",
            "strength": "strong",
            "effect_size": all_metrics.get("t2_pooled_rho", 0.9),
            "ci_lo": all_metrics.get("t2_prediction_interval_lo", 0.5),
            "ci_hi": all_metrics.get("t2_prediction_interval_hi", 1.0),
            "n_datasets": 2,
            "supported": 1,
            "evidence": "Probe rho>0.9 (exp_id4_it5), structural signal significant p=0.0001 d=0.69 (rel-f1), p=0.024 d=0.63 (rel-stack)",
        },
        {
            "claim_id": 2,
            "claim": "Threshold behavior (non-monotonic variance near JRN~1)",
            "grade": "D",
            "strength": "absent",
            "effect_size": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "n_datasets": 0,
            "supported": 0,
            "evidence": "No direct aggregation-strategy variance experiment was run. Cannot evaluate peak near JRN=1.",
        },
        {
            "claim_id": 3,
            "claim": "Cost-efficiency of JRN probes",
            "grade": "B",
            "strength": "strong",
            "effect_size": all_metrics.get("t3_mean_cost_ratio_jrn_vs_greedy", 0.07),
            "ci_lo": 0.05,
            "ci_hi": 0.11,
            "n_datasets": 1,
            "supported": 1,
            "evidence": "5.5-18x speedup vs greedy, 96-99% performance. Single dataset (rel-f1).",
        },
        {
            "claim_id": 4,
            "claim": "Multiplicative compounding along chains",
            "grade": "C",
            "strength": "moderate",
            "effect_size": all_metrics.get("t7_best_log_linear_r2", 0.83),
            "ci_lo": 0.12,
            "ci_hi": 0.83,
            "n_datasets": 2,
            "supported": 0,
            "evidence": "Log-linear model R2=0.83 on GBM (rel-f1), Spearman=0.68 on rel-stack. Naive multiplicative fails (negative R2). Partially supported: signal compounds but not multiplicatively.",
        },
        {
            "claim_id": 5,
            "claim": "JRN-guided architecture outperforms uniform",
            "grade": "C",
            "strength": "moderate",
            "effect_size": all_metrics.get("t6_mean_perf_ratio", 0.978),
            "ci_lo": all_metrics.get("t6_ci_lo_vs_greedy", 0.0),
            "ci_hi": all_metrics.get("t6_ci_hi_vs_greedy", 0.3),
            "n_datasets": 1,
            "supported": 0,
            "evidence": "JRN achieves 96-99% of greedy at 5-11% cost. Does NOT outperform on raw performance. Cost-adjusted win.",
        },
        {
            "claim_id": 6,
            "claim": "FK-shuffling structural signal",
            "grade": "B",
            "strength": "moderate",
            "effect_size": all_metrics.get("t4_pooled_cohens_d", 0.65),
            "ci_lo": 0.0,
            "ci_hi": 1.0,
            "n_datasets": 2,
            "supported": 1,
            "evidence": f"Structural component significant but secondary (dominant in {all_metrics.get('t4_structural_dominant_pct', 3):.1f}% of pairs). Feature component larger.",
        },
        {
            "claim_id": 7,
            "claim": "Cross-task transferability of JRN rankings",
            "grade": "B",
            "strength": "moderate",
            "effect_size": all_metrics.get("t8_same_entity_rho_pooled", 0.85),
            "ci_lo": all_metrics.get("t8_cross_entity_rho_pooled", -0.23),
            "ci_hi": all_metrics.get("t8_same_entity_rho_pooled", 0.85),
            "n_datasets": 2,
            "supported": 1,
            "evidence": "Same-entity rho=0.82-0.87 (strong), cross-entity rho=0.24 to -0.70 (fails). Per-task recommended.",
        },
    ]

    examples = []
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for c in claims:
        grade_num = {"A": 4, "B": 3, "C": 2, "D": 1}.get(c["grade"], 0)
        grade_counts[c["grade"]] = grade_counts.get(c["grade"], 0) + 1
        examples.append({
            "input": f"Claim {c['claim_id']}: {c['claim']}",
            "output": f"Grade: {c['grade']} ({c['strength']}). {c['evidence']}",
            "metadata_claim_id": c["claim_id"],
            "metadata_claim": c["claim"],
            "metadata_source": "scorecard_v5",
            "predict_jrn_probe": f"{c['effect_size']:.4f}",
            "predict_baseline_uniform": "0.0",
            "eval_evidence_grade": grade_num,
            "eval_effect_size": round(c["effect_size"], 6),
            "eval_ci_lo": round(c["ci_lo"], 6),
            "eval_ci_hi": round(c["ci_hi"], 6),
            "eval_n_datasets": c["n_datasets"],
            "eval_claim_supported": c["supported"],
        })

    metrics = {
        "sc_n_claims": len(claims),
        "sc_n_grade_A": grade_counts["A"],
        "sc_n_grade_B": grade_counts["B"],
        "sc_n_grade_C": grade_counts["C"],
        "sc_n_grade_D": grade_counts["D"],
        "sc_n_supported": sum(c["supported"] for c in claims),
    }
    logger.info(f"Scorecard: {len(claims)} claims, A={grade_counts['A']}, B={grade_counts['B']}, C={grade_counts['C']}, D={grade_counts['D']}")
    return examples, metrics


# ===================================================================
# STATISTICAL SUMMARY
# ===================================================================

def build_statistical_summary(deps: dict, all_metrics: dict) -> tuple[list[dict], dict]:
    """Overall statistical summary."""
    logger.info("Building Statistical Summary")
    examples = []

    total_examples = sum(
        sum(len(ds["examples"]) for ds in d["datasets"])
        for d in deps.values()
    )

    stats_items = [
        ("total_dependency_examples", total_examples),
        ("n_datasets", 2),
        ("n_tasks", 8),
        ("n_fk_joins_rel_f1", 13),
        ("n_fk_joins_rel_stack", 11),
        ("n_fk_joins_total", 24),
        ("n_hypothesis_claims", 7),
        ("n_claims_supported", all_metrics.get("sc_n_supported", 4)),
        ("pooled_probe_validity_rho", all_metrics.get("t2_pooled_rho", 0)),
        ("pooled_structural_cohens_d", all_metrics.get("t4_pooled_cohens_d", 0)),
        ("best_compounding_R2", all_metrics.get("t7_best_log_linear_r2", 0)),
        ("mean_cost_ratio_jrn_vs_greedy", all_metrics.get("t3_mean_cost_ratio_jrn_vs_greedy", 0)),
        ("same_entity_transfer_rho", all_metrics.get("t8_same_entity_rho_pooled", 0)),
        ("cross_entity_transfer_rho", all_metrics.get("t8_cross_entity_rho_pooled", 0)),
        ("mean_jrn_across_all", all_metrics.get("t1_mean_jrn", 0)),
        ("structural_dominant_pct", all_metrics.get("t4_structural_dominant_pct", 0)),
        ("convergence_rate_cheapest", all_metrics.get("t3_convergence_rate", 0)),
    ]

    for name, value in stats_items:
        examples.append({
            "input": f"Statistic: {name}",
            "output": f"{name} = {value}",
            "metadata_statistic_name": name,
            "metadata_source": "statistical_summary",
            "predict_jrn_probe": f"{safe_float(value):.6f}",
            "predict_baseline_uniform": "0.0",
            "eval_statistic_value": round(safe_float(value), 6),
        })

    metrics = {
        "ss_n_statistics": len(examples),
        "ss_total_dep_examples": total_examples,
    }
    logger.info(f"Statistical summary: {len(examples)} statistics")
    return examples, metrics


# ===================================================================
# MAIN
# ===================================================================

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Final Integrative Evaluation: Starting")
    logger.info("=" * 60)

    # Load all dependencies
    deps = load_all_deps()

    # Build all tables and collect metrics
    all_metrics = {}

    t1_ex, t1_m = build_table_1(deps)
    all_metrics.update(t1_m)

    t2_ex, t2_m = build_table_2(deps)
    all_metrics.update(t2_m)

    t3_ex, t3_m = build_table_3(deps)
    all_metrics.update(t3_m)

    t4_ex, t4_m = build_table_4(deps)
    all_metrics.update(t4_m)

    t5_ex, t5_m = build_table_5(deps)
    all_metrics.update(t5_m)

    t6_ex, t6_m = build_table_6(deps)
    all_metrics.update(t6_m)

    t7_ex, t7_m = build_table_7(deps)
    all_metrics.update(t7_m)

    t8_ex, t8_m = build_table_8(deps)
    all_metrics.update(t8_m)

    t9_ex, t9_m = build_table_9(deps, all_metrics)
    all_metrics.update(t9_m)

    sc_ex, sc_m = build_scorecard(deps, all_metrics)
    all_metrics.update(sc_m)

    ss_ex, ss_m = build_statistical_summary(deps, all_metrics)
    all_metrics.update(ss_m)

    # Build datasets list
    datasets = [
        {"dataset": "paper_table_1_jrn_matrix", "examples": t1_ex},
        {"dataset": "paper_table_2_probe_validity", "examples": t2_ex},
        {"dataset": "paper_table_3_cost_efficiency", "examples": t3_ex},
        {"dataset": "paper_table_4_fk_shuffling", "examples": t4_ex},
        {"dataset": "paper_table_5_proxies", "examples": t5_ex},
        {"dataset": "paper_table_6_architecture", "examples": t6_ex},
        {"dataset": "paper_table_7_compounding", "examples": t7_ex},
        {"dataset": "paper_table_8_transfer", "examples": t8_ex},
        {"dataset": "paper_table_9_guidelines", "examples": t9_ex},
        {"dataset": "scorecard_v5", "examples": sc_ex},
        {"dataset": "statistical_summary", "examples": ss_ex},
    ]

    # Top-level metrics_agg (numbers only)
    total_examples = sum(len(ds["examples"]) for ds in datasets)

    metrics_agg = {
        "n_paper_tables": 9,
        "n_total_measurements": total_examples,
        "n_datasets": 2,
        "n_tasks": 8,
        "n_fk_joins": 24,
        "n_hypothesis_claims": 7,
        "n_claims_grade_A": all_metrics.get("sc_n_grade_A", 1),
        "n_claims_grade_B": all_metrics.get("sc_n_grade_B", 3),
        "n_claims_grade_C": all_metrics.get("sc_n_grade_C", 2),
        "n_claims_grade_D": all_metrics.get("sc_n_grade_D", 1),
        "pooled_probe_validity_rho": all_metrics.get("t2_pooled_rho", 0),
        "pooled_structural_cohens_d": all_metrics.get("t4_pooled_cohens_d", 0),
        "best_compounding_R2": all_metrics.get("t7_best_log_linear_r2", 0),
        "mean_cost_ratio_jrn_vs_greedy": all_metrics.get("t3_mean_cost_ratio_jrn_vs_greedy", 0),
        "same_entity_transfer_rho": all_metrics.get("t8_same_entity_rho_pooled", 0),
        "t1_n_measurements": all_metrics.get("t1_n_measurements", 0),
        "t1_mean_jrn": all_metrics.get("t1_mean_jrn", 0),
        "t2_I_squared": all_metrics.get("t2_I_squared", 0),
        "t3_convergence_rate": all_metrics.get("t3_convergence_rate", 0),
        "t4_pooled_t_p_value": all_metrics.get("t4_pooled_t_p_value", 0),
        "t4_structural_dominant_pct": all_metrics.get("t4_structural_dominant_pct", 0),
        "t6_mean_perf_ratio": all_metrics.get("t6_mean_perf_ratio", 0),
        "t8_mean_W": all_metrics.get("t8_mean_W", 0),
    }

    # Ensure all metrics_agg values are numbers
    for k, v in list(metrics_agg.items()):
        metrics_agg[k] = round(safe_float(v), 6)

    output = {
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }

    # Write output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote eval_out.json ({out_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Datasets: {[ds['dataset'] for ds in datasets]}")

    # Verify all datasets have at least 1 example
    for ds in datasets:
        assert len(ds["examples"]) >= 1, f"Dataset {ds['dataset']} has no examples!"
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
