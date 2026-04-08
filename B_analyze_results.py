#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fractional_optimizer_results.py

Load saved experiment JSON files produced by
run_fractional_optimizer_activation_experiments.py, aggregate results across
random seed runs, rank configurations by average accuracy, and write a
human-readable text report.

Expected input structure
------------------------
results_fractional_activation_optimizer_study/
    breast_cancer/
        *.json
    wine/
        *.json
    digits/
        *.json

Expected JSON schema
--------------------
Each configuration file is expected to contain fields such as:
    - dataset
    - optimizer
    - activation
    - history_size
    - adaptation_mode
    - schedule_type
    - n_runs
    - vderivs: [
          {
              "vderiv": ...,
              "avg accuracy": ...,
              "std accuracy": ...,
              "results": [
                  {
                      "run_seed": ...,
                      "accuracy": ...,
                      "f1_macro": ...,
                      "precision_macro": ...,
                      "recall_macro": ...,
                      "training_time_seconds": ...,
                      "test_time_seconds": ...,
                      "epochs_completed": ...,
                      "best_val_loss": ...
                  },
                  ...
              ]
          }
      ]

Main outputs
------------
1. Console summary:
   - top configurations per dataset by average accuracy
   - best optimizer family summaries
   - best activation summaries

2. Text report:
   - detailed dataset-wise ranking
   - optimizer-level aggregation
   - activation-level aggregation
   - family-level aggregation
   - global ranking across all datasets

Notes
-----
- This script ranks by mean test accuracy across the repeated random-seed runs.
- Ties are broken by:
    1) lower std accuracy,
    2) higher mean F1,
    3) lower mean training time.
- The script is robust to partially missing keys and reports skipped files.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# 0) USER SETTINGS
# =============================================================================

RESULTS_ROOT = Path("results_fractional_activation_optimizer_study")
REPORTS_DIR = RESULTS_ROOT / "analysis_reports"

TOP_K_PER_DATASET = 20
TOP_K_GLOBAL = 30
PRINT_TO_CONSOLE = True

# If True, include per-seed details for every configuration in the text report.
INCLUDE_PER_SEED_DETAILS = True

# If True, include epoch-history-derived summary when available.
INCLUDE_HISTORY_SUMMARY = True


# =============================================================================
# 1) DATA CONTAINERS
# =============================================================================

@dataclass
class RunMetric:
    run_seed: Optional[int]
    accuracy: float
    f1_macro: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    training_time_seconds: Optional[float]
    test_time_seconds: Optional[float]
    epochs_completed: Optional[int]
    best_val_loss: Optional[float]


@dataclass
class ConfigSummary:
    dataset: str
    source_file: str
    optimizer: str
    activation: str
    history_size: Optional[int]
    adaptation_mode: Optional[str]
    schedule_type: Optional[str]
    vderiv: Optional[float]
    n_runs_declared: Optional[int]
    n_runs_found: int

    mean_accuracy: float
    std_accuracy: float
    max_accuracy: float
    min_accuracy: float

    mean_f1_macro: Optional[float]
    mean_precision_macro: Optional[float]
    mean_recall_macro: Optional[float]
    mean_training_time_seconds: Optional[float]
    mean_test_time_seconds: Optional[float]
    mean_epochs_completed: Optional[float]
    mean_best_val_loss: Optional[float]

    optimizer_family: str
    config_label: str
    run_metrics: List[RunMetric]
    history_summary: Optional[Dict[str, float]]


# =============================================================================
# 2) SMALL UTILITIES
# =============================================================================

def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def std_population_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(statistics.pstdev(vals))


def family_from_optimizer(optimizer_name: str) -> str:
    if optimizer_name in {"SGD", "Adam", "RMSprop", "Adagrad", "Adadelta", "AdamW"}:
        return "baseline"
    if optimizer_name.startswith("Memory"):
        return "memory_fractional"
    if optimizer_name.startswith("AdaptiveModesVariableOrder"):
        return "variable_order_fractional"
    if optimizer_name.startswith("F"):
        return "herrera_fractional"
    return "other"


def fmt(x: Optional[float], digits: int = 5) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "NA"
    return str(x)


def build_config_label(
    optimizer: str,
    activation: str,
    history_size: Optional[int],
    adaptation_mode: Optional[str],
    schedule_type: Optional[str],
    vderiv: Optional[float],
) -> str:
    parts = [f"optimizer={optimizer}", f"activation={activation}"]
    if history_size is not None:
        parts.append(f"history={history_size}")
    if adaptation_mode is not None:
        parts.append(f"mode={adaptation_mode}")
    if schedule_type is not None:
        parts.append(f"schedule={schedule_type}")
    if vderiv is not None:
        parts.append(f"vderiv={vderiv}")
    return ", ".join(parts)


def sort_key_config(summary: ConfigSummary) -> Tuple[float, float, float, float]:
    """
    Sort descending by mean accuracy, then ascending std accuracy,
    then descending mean F1, then ascending mean training time.
    """
    std_acc = summary.std_accuracy if summary.std_accuracy is not None else float("inf")
    mean_f1 = summary.mean_f1_macro if summary.mean_f1_macro is not None else -float("inf")
    mean_train = (
        summary.mean_training_time_seconds
        if summary.mean_training_time_seconds is not None
        else float("inf")
    )
    return (
        summary.mean_accuracy,
        -std_acc,
        mean_f1,
        -mean_train,
    )


# =============================================================================
# 3) FILE PARSING
# =============================================================================

def summarize_history_block(run_histories: Any) -> Optional[Dict[str, float]]:
    """
    Aggregate epoch-history statistics if present.
    """
    if not isinstance(run_histories, list) or len(run_histories) == 0:
        return None

    final_val_accs = []
    final_val_losses = []
    best_epoch_val_accs = []
    min_val_losses = []

    for entry in run_histories:
        epoch_history = entry.get("epoch_history")
        if not isinstance(epoch_history, list) or len(epoch_history) == 0:
            continue

        val_accs = []
        val_losses = []
        for ep in epoch_history:
            va = safe_float(ep.get("val_accuracy"))
            vl = safe_float(ep.get("val_loss"))
            if va is not None:
                val_accs.append(va)
            if vl is not None:
                val_losses.append(vl)

        if val_accs:
            final_val_accs.append(val_accs[-1])
            best_epoch_val_accs.append(max(val_accs))
        if val_losses:
            final_val_losses.append(val_losses[-1])
            min_val_losses.append(min(val_losses))

    if not (final_val_accs or final_val_losses or best_epoch_val_accs or min_val_losses):
        return None

    return {
        "mean_final_val_accuracy": mean_or_none(final_val_accs) or float("nan"),
        "mean_best_epoch_val_accuracy": mean_or_none(best_epoch_val_accs) or float("nan"),
        "mean_final_val_loss": mean_or_none(final_val_losses) or float("nan"),
        "mean_min_val_loss": mean_or_none(min_val_losses) or float("nan"),
    }


def parse_config_json(json_path: Path) -> ConfigSummary:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    dataset = str(payload.get("dataset", json_path.parent.name))
    optimizer = str(payload.get("optimizer", "UNKNOWN"))
    activation = str(payload.get("activation", "UNKNOWN"))
    history_size = payload.get("history_size", None)
    adaptation_mode = payload.get("adaptation_mode", None)
    schedule_type = payload.get("schedule_type", None)
    n_runs_declared = safe_int(payload.get("n_runs"))

    vderivs = payload.get("vderivs", [])
    if not isinstance(vderivs, list) or len(vderivs) == 0:
        raise ValueError(f"No valid 'vderivs' section in file: {json_path}")

    # The runner currently writes one vderiv entry per file.
    first_v = vderivs[0]
    vderiv = safe_float(first_v.get("vderiv"))
    results = first_v.get("results", [])
    if not isinstance(results, list) or len(results) == 0:
        raise ValueError(f"No valid results list in file: {json_path}")

    run_metrics: List[RunMetric] = []
    for run in results:
        run_metrics.append(
            RunMetric(
                run_seed=safe_int(run.get("run_seed")),
                accuracy=safe_float(run.get("accuracy")) or 0.0,
                f1_macro=safe_float(run.get("f1_macro")),
                precision_macro=safe_float(run.get("precision_macro")),
                recall_macro=safe_float(run.get("recall_macro")),
                training_time_seconds=safe_float(run.get("training_time_seconds")),
                test_time_seconds=safe_float(run.get("test_time_seconds")),
                epochs_completed=safe_int(run.get("epochs_completed")),
                best_val_loss=safe_float(run.get("best_val_loss")),
            )
        )

    accuracies = [r.accuracy for r in run_metrics]
    mean_accuracy = float(sum(accuracies) / len(accuracies))
    std_accuracy = float(statistics.pstdev(accuracies)) if len(accuracies) > 1 else 0.0
    max_accuracy = float(max(accuracies))
    min_accuracy = float(min(accuracies))

    history_summary = None
    if INCLUDE_HISTORY_SUMMARY:
        history_summary = summarize_history_block(payload.get("run_histories"))

    config_label = build_config_label(
        optimizer=optimizer,
        activation=activation,
        history_size=history_size,
        adaptation_mode=adaptation_mode,
        schedule_type=schedule_type,
        vderiv=vderiv,
    )

    return ConfigSummary(
        dataset=dataset,
        source_file=str(json_path),
        optimizer=optimizer,
        activation=activation,
        history_size=history_size,
        adaptation_mode=adaptation_mode,
        schedule_type=schedule_type,
        vderiv=vderiv,
        n_runs_declared=n_runs_declared,
        n_runs_found=len(run_metrics),

        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
        max_accuracy=max_accuracy,
        min_accuracy=min_accuracy,

        mean_f1_macro=mean_or_none(r.f1_macro for r in run_metrics),
        mean_precision_macro=mean_or_none(r.precision_macro for r in run_metrics),
        mean_recall_macro=mean_or_none(r.recall_macro for r in run_metrics),
        mean_training_time_seconds=mean_or_none(r.training_time_seconds for r in run_metrics),
        mean_test_time_seconds=mean_or_none(r.test_time_seconds for r in run_metrics),
        mean_epochs_completed=mean_or_none(
            float(r.epochs_completed) if r.epochs_completed is not None else None
            for r in run_metrics
        ),
        mean_best_val_loss=mean_or_none(r.best_val_loss for r in run_metrics),

        optimizer_family=family_from_optimizer(optimizer),
        config_label=config_label,
        run_metrics=run_metrics,
        history_summary=history_summary,
    )


def load_all_results(results_root: Path) -> Tuple[List[ConfigSummary], List[str]]:
    all_summaries: List[ConfigSummary] = []
    skipped: List[str] = []

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    for dataset_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if dataset_dir.name == "analysis_reports":
            continue

        for json_file in sorted(dataset_dir.glob("*.json")):
            try:
                summary = parse_config_json(json_file)
                all_summaries.append(summary)
            except Exception as e:
                skipped.append(f"{json_file}: {e}")

    return all_summaries, skipped


# =============================================================================
# 4) AGGREGATION
# =============================================================================

def aggregate_group(
    summaries: List[ConfigSummary],
    key_fn,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[ConfigSummary]] = defaultdict(list)
    for s in summaries:
        grouped[key_fn(s)].append(s)

    rows: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        rows.append(
            {
                "group": key,
                "n_configs": len(items),
                "mean_accuracy": float(sum(i.mean_accuracy for i in items) / len(items)),
                "std_of_config_mean_accuracy": (
                    float(statistics.pstdev(i.mean_accuracy for i in items))
                    if len(items) > 1 else 0.0
                ),
                "best_accuracy": float(max(i.mean_accuracy for i in items)),
                "best_config": max(items, key=lambda x: x.mean_accuracy).config_label,
                "mean_f1_macro": mean_or_none(i.mean_f1_macro for i in items),
                "mean_training_time_seconds": mean_or_none(i.mean_training_time_seconds for i in items),
                "mean_epochs_completed": mean_or_none(i.mean_epochs_completed for i in items),
            }
        )

    rows.sort(
        key=lambda r: (
            r["mean_accuracy"],
            -(r["std_of_config_mean_accuracy"] if r["std_of_config_mean_accuracy"] is not None else float("inf")),
            r["mean_f1_macro"] if r["mean_f1_macro"] is not None else -float("inf"),
        ),
        reverse=True,
    )
    return rows


# =============================================================================
# 5) REPORT WRITING
# =============================================================================

def report_header_lines(title: str) -> List[str]:
    return [
        "=" * 100,
        title,
        "=" * 100,
    ]


def config_block_lines(rank: int, s: ConfigSummary) -> List[str]:
    lines = [
        f"[{rank}] {s.config_label}",
        f"    dataset={s.dataset}",
        f"    mean_accuracy={fmt(s.mean_accuracy)}  std_accuracy={fmt(s.std_accuracy)}"
        f"  max_accuracy={fmt(s.max_accuracy)}  min_accuracy={fmt(s.min_accuracy)}",
        f"    mean_f1_macro={fmt(s.mean_f1_macro)}"
        f"  mean_precision_macro={fmt(s.mean_precision_macro)}"
        f"  mean_recall_macro={fmt(s.mean_recall_macro)}",
        f"    mean_training_time_seconds={fmt(s.mean_training_time_seconds, 4)}"
        f"  mean_test_time_seconds={fmt(s.mean_test_time_seconds, 4)}"
        f"  mean_epochs_completed={fmt(s.mean_epochs_completed, 2)}"
        f"  mean_best_val_loss={fmt(s.mean_best_val_loss)}",
        f"    optimizer_family={s.optimizer_family}",
        f"    source_file={s.source_file}",
    ]

    if s.history_summary is not None:
        lines.append(
            "    history_summary: "
            f"mean_final_val_accuracy={fmt(s.history_summary.get('mean_final_val_accuracy'))}, "
            f"mean_best_epoch_val_accuracy={fmt(s.history_summary.get('mean_best_epoch_val_accuracy'))}, "
            f"mean_final_val_loss={fmt(s.history_summary.get('mean_final_val_loss'))}, "
            f"mean_min_val_loss={fmt(s.history_summary.get('mean_min_val_loss'))}"
        )

    if INCLUDE_PER_SEED_DETAILS and s.run_metrics:
        lines.append("    per_seed_runs:")
        for rm in s.run_metrics:
            lines.append(
                f"      seed={fmt_int(rm.run_seed)}"
                f"  accuracy={fmt(rm.accuracy)}"
                f"  f1={fmt(rm.f1_macro)}"
                f"  precision={fmt(rm.precision_macro)}"
                f"  recall={fmt(rm.recall_macro)}"
                f"  train_time={fmt(rm.training_time_seconds, 4)}"
                f"  test_time={fmt(rm.test_time_seconds, 4)}"
                f"  epochs={fmt_int(rm.epochs_completed)}"
                f"  best_val_loss={fmt(rm.best_val_loss)}"
            )

    return lines


def grouped_table_lines(title: str, rows: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[str]:
    lines = report_header_lines(title)
    use_rows = rows if top_k is None else rows[:top_k]

    if not use_rows:
        lines.append("No data.")
        return lines

    for rank, row in enumerate(use_rows, start=1):
        lines.append(
            f"[{rank}] {row['group']} | "
            f"n_configs={row['n_configs']} | "
            f"mean_accuracy={fmt(row['mean_accuracy'])} | "
            f"std_of_config_mean_accuracy={fmt(row['std_of_config_mean_accuracy'])} | "
            f"best_accuracy={fmt(row['best_accuracy'])} | "
            f"mean_f1={fmt(row['mean_f1_macro'])} | "
            f"mean_train_time={fmt(row['mean_training_time_seconds'], 4)} | "
            f"mean_epochs={fmt(row['mean_epochs_completed'], 2)}"
        )
        lines.append(f"    best_config: {row['best_config']}")

    return lines


def write_text_report(
    all_summaries: List[ConfigSummary],
    skipped: List[str],
    output_file: Path,
) -> None:
    lines: List[str] = []
    lines.extend(report_header_lines("Fractional Optimizer Experiment Analysis Report"))

    lines.append(f"Results root: {RESULTS_ROOT}")
    lines.append(f"Total parsed configuration files: {len(all_summaries)}")
    lines.append(f"Skipped files: {len(skipped)}")
    lines.append("")

    if skipped:
        lines.extend(report_header_lines("Skipped Files"))
        lines.extend(skipped)
        lines.append("")

    datasets = sorted({s.dataset for s in all_summaries})

    # Dataset-wise rankings
    for dataset in datasets:
        ds_items = [s for s in all_summaries if s.dataset == dataset]
        ds_items_sorted = sorted(ds_items, key=sort_key_config, reverse=True)

        lines.extend(report_header_lines(f"Dataset Ranking: {dataset}"))
        lines.append(f"Number of configurations: {len(ds_items_sorted)}")
        if ds_items_sorted:
            best = ds_items_sorted[0]
            lines.append(
                "Best configuration by mean accuracy: "
                f"{best.config_label} | mean_accuracy={fmt(best.mean_accuracy)} | std_accuracy={fmt(best.std_accuracy)}"
            )
        lines.append("")

        for rank, summary in enumerate(ds_items_sorted[:TOP_K_PER_DATASET], start=1):
            lines.extend(config_block_lines(rank, summary))
            lines.append("")

        # Aggregations within dataset
        lines.extend(
            grouped_table_lines(
                f"Dataset Aggregation by Optimizer: {dataset}",
                aggregate_group(ds_items, key_fn=lambda s: s.optimizer),
            )
        )
        lines.append("")

        lines.extend(
            grouped_table_lines(
                f"Dataset Aggregation by Activation: {dataset}",
                aggregate_group(ds_items, key_fn=lambda s: s.activation),
            )
        )
        lines.append("")

        lines.extend(
            grouped_table_lines(
                f"Dataset Aggregation by Optimizer Family: {dataset}",
                aggregate_group(ds_items, key_fn=lambda s: s.optimizer_family),
            )
        )
        lines.append("")

    # Global ranking
    all_sorted = sorted(all_summaries, key=sort_key_config, reverse=True)
    lines.extend(report_header_lines("Global Ranking Across All Datasets"))
    for rank, summary in enumerate(all_sorted[:TOP_K_GLOBAL], start=1):
        lines.extend(config_block_lines(rank, summary))
        lines.append("")

    # Global aggregations
    lines.extend(
        grouped_table_lines(
            "Global Aggregation by Optimizer",
            aggregate_group(all_summaries, key_fn=lambda s: s.optimizer),
        )
    )
    lines.append("")

    lines.extend(
        grouped_table_lines(
            "Global Aggregation by Activation",
            aggregate_group(all_summaries, key_fn=lambda s: s.activation),
        )
    )
    lines.append("")

    lines.extend(
        grouped_table_lines(
            "Global Aggregation by Optimizer Family",
            aggregate_group(all_summaries, key_fn=lambda s: s.optimizer_family),
        )
    )
    lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 6) CONSOLE SUMMARY
# =============================================================================

def print_console_summary(all_summaries: List[ConfigSummary], skipped: List[str]) -> None:
    print("=" * 100)
    print("Saved-result analysis")
    print("=" * 100)
    print(f"Parsed configuration files: {len(all_summaries)}")
    print(f"Skipped files: {len(skipped)}")

    datasets = sorted({s.dataset for s in all_summaries})
    for dataset in datasets:
        ds_items = [s for s in all_summaries if s.dataset == dataset]
        ds_items_sorted = sorted(ds_items, key=sort_key_config, reverse=True)

        print("\n" + "-" * 100)
        print(f"DATASET: {dataset}")
        print("-" * 100)

        if ds_items_sorted:
            best = ds_items_sorted[0]
            print(
                f"Best config: {best.config_label} | "
                f"mean_accuracy={best.mean_accuracy:.5f} | "
                f"std_accuracy={best.std_accuracy:.5f} | "
                f"mean_f1={fmt(best.mean_f1_macro)}"
            )

        print("\nTop configurations:")
        for rank, s in enumerate(ds_items_sorted[:min(TOP_K_PER_DATASET, len(ds_items_sorted))], start=1):
            print(
                f"[{rank:02d}] "
                f"{s.optimizer:40s} "
                f"act={s.activation:35s} "
                f"acc={s.mean_accuracy:.5f} "
                f"std={s.std_accuracy:.5f} "
                f"f1={fmt(s.mean_f1_macro)} "
                f"hist={s.history_size} "
                f"mode={s.adaptation_mode} "
                f"sched={s.schedule_type} "
                f"v={s.vderiv}"
            )

        optimizer_rows = aggregate_group(ds_items, key_fn=lambda s: s.optimizer)
        print("\nBest optimizers by dataset-level average over configurations:")
        for rank, row in enumerate(optimizer_rows[:10], start=1):
            print(
                f"[{rank:02d}] {row['group']:40s} "
                f"mean_acc={row['mean_accuracy']:.5f} "
                f"best_acc={row['best_accuracy']:.5f} "
                f"n_configs={row['n_configs']}"
            )

    global_sorted = sorted(all_summaries, key=sort_key_config, reverse=True)
    print("\n" + "=" * 100)
    print("GLOBAL TOP CONFIGURATIONS")
    print("=" * 100)
    for rank, s in enumerate(global_sorted[:min(TOP_K_GLOBAL, len(global_sorted))], start=1):
        print(
            f"[{rank:02d}] {s.dataset:15s} "
            f"{s.optimizer:40s} "
            f"act={s.activation:30s} "
            f"acc={s.mean_accuracy:.5f} "
            f"std={s.std_accuracy:.5f} "
            f"v={s.vderiv}"
        )


# =============================================================================
# 7) MAIN
# =============================================================================

def main() -> None:
    all_summaries, skipped = load_all_results(RESULTS_ROOT)

    if not all_summaries:
        raise RuntimeError(
            f"No valid result JSON files found under: {RESULTS_ROOT}"
        )

    report_name = "fractional_optimizer_analysis_report.txt"
    output_file = REPORTS_DIR / report_name

    write_text_report(
        all_summaries=all_summaries,
        skipped=skipped,
        output_file=output_file,
    )

    if PRINT_TO_CONSOLE:
        print_console_summary(all_summaries, skipped)

    print("\nReport written to:")
    print(output_file)


if __name__ == "__main__":
    main()