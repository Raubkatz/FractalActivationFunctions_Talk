#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fractional_optimizer_times.py

Read saved JSON results from run_fractional_optimizer_activation_experiments.py
and generate a human-readable timing report in TXT format.

What this script analyzes
-------------------------
The training script saves:
    - per-run training_time_seconds
    - per-run test_time_seconds
    - per-run epochs_completed
    - per-run best_val_loss
    - configuration-level averages for training and test time

This script aggregates those timing fields and reports:
    1. Fastest and slowest configurations by dataset
    2. Timing summaries by optimizer
    3. Timing summaries by activation
    4. Timing summaries by optimizer family
    5. Time per epoch estimates
    6. Global ranking tables
    7. Per-run timing details for every configuration

Important limitation
--------------------
The original run script does NOT save:
    - per-epoch wall-clock time
    - validation-only time
    - dataset preparation time
    - model construction time
    - optimizer step time

Therefore this report can only analyze the timing fields that were actually saved.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# 0) SETTINGS
# =============================================================================

RESULTS_ROOT = Path("results_fractional_activation_optimizer_study")
REPORTS_DIR = RESULTS_ROOT / "analysis_reports"

OUTPUT_REPORT_NAME = "fractional_optimizer_timing_report.txt"

TOP_K_FASTEST = 20
TOP_K_SLOWEST = 20

PRINT_TO_CONSOLE = True
INCLUDE_PER_RUN_DETAILS = True


# =============================================================================
# 1) DATA STRUCTURES
# =============================================================================

@dataclass
class RunTimeRecord:
    run_seed: Optional[int]
    training_time_seconds: Optional[float]
    test_time_seconds: Optional[float]
    epochs_completed: Optional[int]
    best_val_loss: Optional[float]
    accuracy: Optional[float]
    f1_macro: Optional[float]


@dataclass
class ConfigTimeSummary:
    dataset: str
    source_file: str
    optimizer: str
    activation: str
    history_size: Optional[int]
    adaptation_mode: Optional[str]
    schedule_type: Optional[str]
    vderiv: Optional[float]
    optimizer_family: str

    n_runs: int

    mean_training_time_seconds: Optional[float]
    std_training_time_seconds: Optional[float]
    min_training_time_seconds: Optional[float]
    max_training_time_seconds: Optional[float]

    mean_test_time_seconds: Optional[float]
    std_test_time_seconds: Optional[float]
    min_test_time_seconds: Optional[float]
    max_test_time_seconds: Optional[float]

    mean_total_time_seconds: Optional[float]
    std_total_time_seconds: Optional[float]
    min_total_time_seconds: Optional[float]
    max_total_time_seconds: Optional[float]

    mean_epochs_completed: Optional[float]
    mean_training_time_per_epoch_seconds: Optional[float]

    mean_accuracy: Optional[float]
    mean_f1_macro: Optional[float]
    mean_best_val_loss: Optional[float]

    run_records: List[RunTimeRecord]
    config_label: str


# =============================================================================
# 2) HELPERS
# =============================================================================

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def pstdev_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(statistics.pstdev(vals))


def min_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(min(vals))


def max_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(max(vals))


def fmt(x: Optional[float], digits: int = 5) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "NA"
    return str(x)


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


def sort_by_fastest_training(summary: ConfigTimeSummary) -> Tuple[float, float, float]:
    return (
        -(summary.mean_training_time_seconds if summary.mean_training_time_seconds is not None else float("inf")),
        -(summary.mean_total_time_seconds if summary.mean_total_time_seconds is not None else float("inf")),
        (summary.mean_accuracy if summary.mean_accuracy is not None else -float("inf")),
    )


def sort_by_slowest_training(summary: ConfigTimeSummary) -> Tuple[float, float, float]:
    return (
        summary.mean_training_time_seconds if summary.mean_training_time_seconds is not None else -float("inf"),
        summary.mean_total_time_seconds if summary.mean_total_time_seconds is not None else -float("inf"),
        -(summary.mean_accuracy if summary.mean_accuracy is not None else -float("inf")),
    )


# =============================================================================
# 3) PARSING
# =============================================================================

def parse_result_file(json_file: Path) -> ConfigTimeSummary:
    with open(json_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    dataset = str(payload.get("dataset", json_file.parent.name))
    optimizer = str(payload.get("optimizer", "UNKNOWN"))
    activation = str(payload.get("activation", "UNKNOWN"))
    history_size = payload.get("history_size")
    adaptation_mode = payload.get("adaptation_mode")
    schedule_type = payload.get("schedule_type")

    vderivs = payload.get("vderivs", [])
    if not isinstance(vderivs, list) or not vderivs:
        raise ValueError("Missing or invalid 'vderivs' list.")

    v_info = vderivs[0]
    vderiv = safe_float(v_info.get("vderiv"))

    results = v_info.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError("Missing or invalid 'results' list.")

    run_records: List[RunTimeRecord] = []
    total_times: List[Optional[float]] = []

    for r in results:
        train_t = safe_float(r.get("training_time_seconds"))
        test_t = safe_float(r.get("test_time_seconds"))
        total_t = None
        if train_t is not None and test_t is not None:
            total_t = train_t + test_t
        total_times.append(total_t)

        run_records.append(
            RunTimeRecord(
                run_seed=safe_int(r.get("run_seed")),
                training_time_seconds=train_t,
                test_time_seconds=test_t,
                epochs_completed=safe_int(r.get("epochs_completed")),
                best_val_loss=safe_float(r.get("best_val_loss")),
                accuracy=safe_float(r.get("accuracy")),
                f1_macro=safe_float(r.get("f1_macro")),
            )
        )

    mean_training = mean_or_none(rr.training_time_seconds for rr in run_records)
    mean_test = mean_or_none(rr.test_time_seconds for rr in run_records)
    mean_epochs = mean_or_none(
        float(rr.epochs_completed) if rr.epochs_completed is not None else None
        for rr in run_records
    )

    if mean_training is not None and mean_epochs is not None and mean_epochs > 0:
        mean_train_per_epoch = mean_training / mean_epochs
    else:
        mean_train_per_epoch = None

    return ConfigTimeSummary(
        dataset=dataset,
        source_file=str(json_file),
        optimizer=optimizer,
        activation=activation,
        history_size=history_size,
        adaptation_mode=adaptation_mode,
        schedule_type=schedule_type,
        vderiv=vderiv,
        optimizer_family=family_from_optimizer(optimizer),

        n_runs=len(run_records),

        mean_training_time_seconds=mean_training,
        std_training_time_seconds=pstdev_or_none(rr.training_time_seconds for rr in run_records),
        min_training_time_seconds=min_or_none(rr.training_time_seconds for rr in run_records),
        max_training_time_seconds=max_or_none(rr.training_time_seconds for rr in run_records),

        mean_test_time_seconds=mean_test,
        std_test_time_seconds=pstdev_or_none(rr.test_time_seconds for rr in run_records),
        min_test_time_seconds=min_or_none(rr.test_time_seconds for rr in run_records),
        max_test_time_seconds=max_or_none(rr.test_time_seconds for rr in run_records),

        mean_total_time_seconds=mean_or_none(total_times),
        std_total_time_seconds=pstdev_or_none(total_times),
        min_total_time_seconds=min_or_none(total_times),
        max_total_time_seconds=max_or_none(total_times),

        mean_epochs_completed=mean_epochs,
        mean_training_time_per_epoch_seconds=mean_train_per_epoch,

        mean_accuracy=mean_or_none(rr.accuracy for rr in run_records),
        mean_f1_macro=mean_or_none(rr.f1_macro for rr in run_records),
        mean_best_val_loss=mean_or_none(rr.best_val_loss for rr in run_records),

        run_records=run_records,
        config_label=build_config_label(
            optimizer=optimizer,
            activation=activation,
            history_size=history_size,
            adaptation_mode=adaptation_mode,
            schedule_type=schedule_type,
            vderiv=vderiv,
        ),
    )


def load_all_time_summaries(results_root: Path) -> Tuple[List[ConfigTimeSummary], List[str]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    summaries: List[ConfigTimeSummary] = []
    skipped: List[str] = []

    for dataset_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if dataset_dir.name == "analysis_reports":
            continue

        for json_file in sorted(dataset_dir.glob("*.json")):
            try:
                summaries.append(parse_result_file(json_file))
            except Exception as e:
                skipped.append(f"{json_file}: {e}")

    return summaries, skipped


# =============================================================================
# 4) GROUP AGGREGATION
# =============================================================================

def aggregate_group(summaries: List[ConfigTimeSummary], key_fn) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[ConfigTimeSummary]] = defaultdict(list)
    for s in summaries:
        grouped[key_fn(s)].append(s)

    rows: List[Dict[str, Any]] = []
    for group_name, items in grouped.items():
        rows.append(
            {
                "group": group_name,
                "n_configs": len(items),
                "mean_training_time_seconds": mean_or_none(i.mean_training_time_seconds for i in items),
                "mean_test_time_seconds": mean_or_none(i.mean_test_time_seconds for i in items),
                "mean_total_time_seconds": mean_or_none(i.mean_total_time_seconds for i in items),
                "mean_training_time_per_epoch_seconds": mean_or_none(
                    i.mean_training_time_per_epoch_seconds for i in items
                ),
                "mean_accuracy": mean_or_none(i.mean_accuracy for i in items),
                "fastest_config_training_time_seconds": min_or_none(
                    i.mean_training_time_seconds for i in items
                ),
                "slowest_config_training_time_seconds": max_or_none(
                    i.mean_training_time_seconds for i in items
                ),
                "best_accuracy_in_group": max_or_none(i.mean_accuracy for i in items),
                "fastest_config_label": min(
                    items,
                    key=lambda x: (
                        x.mean_training_time_seconds if x.mean_training_time_seconds is not None else float("inf")
                    ),
                ).config_label,
                "slowest_config_label": max(
                    items,
                    key=lambda x: (
                        x.mean_training_time_seconds if x.mean_training_time_seconds is not None else -float("inf")
                    ),
                ).config_label,
            }
        )

    rows.sort(
        key=lambda r: (
            r["mean_training_time_seconds"] if r["mean_training_time_seconds"] is not None else float("inf"),
            r["mean_total_time_seconds"] if r["mean_total_time_seconds"] is not None else float("inf"),
        )
    )
    return rows


# =============================================================================
# 5) REPORT BUILDING
# =============================================================================

def block_header(title: str) -> List[str]:
    return [
        "=" * 100,
        title,
        "=" * 100,
    ]


def config_time_block(rank: int, s: ConfigTimeSummary) -> List[str]:
    lines = [
        f"[{rank}] {s.config_label}",
        f"    dataset={s.dataset}",
        f"    mean_training_time_seconds={fmt(s.mean_training_time_seconds, 4)}"
        f"  std_training_time_seconds={fmt(s.std_training_time_seconds, 4)}"
        f"  min_training_time_seconds={fmt(s.min_training_time_seconds, 4)}"
        f"  max_training_time_seconds={fmt(s.max_training_time_seconds, 4)}",
        f"    mean_test_time_seconds={fmt(s.mean_test_time_seconds, 4)}"
        f"  std_test_time_seconds={fmt(s.std_test_time_seconds, 4)}"
        f"  min_test_time_seconds={fmt(s.min_test_time_seconds, 4)}"
        f"  max_test_time_seconds={fmt(s.max_test_time_seconds, 4)}",
        f"    mean_total_time_seconds={fmt(s.mean_total_time_seconds, 4)}"
        f"  std_total_time_seconds={fmt(s.std_total_time_seconds, 4)}"
        f"  min_total_time_seconds={fmt(s.min_total_time_seconds, 4)}"
        f"  max_total_time_seconds={fmt(s.max_total_time_seconds, 4)}",
        f"    mean_epochs_completed={fmt(s.mean_epochs_completed, 2)}"
        f"  mean_training_time_per_epoch_seconds={fmt(s.mean_training_time_per_epoch_seconds, 4)}",
        f"    mean_accuracy={fmt(s.mean_accuracy)}"
        f"  mean_f1_macro={fmt(s.mean_f1_macro)}"
        f"  mean_best_val_loss={fmt(s.mean_best_val_loss)}",
        f"    optimizer_family={s.optimizer_family}",
        f"    source_file={s.source_file}",
    ]

    if INCLUDE_PER_RUN_DETAILS and s.run_records:
        lines.append("    per_run_times:")
        for rr in s.run_records:
            total = None
            if rr.training_time_seconds is not None and rr.test_time_seconds is not None:
                total = rr.training_time_seconds + rr.test_time_seconds

            if rr.training_time_seconds is not None and rr.epochs_completed not in (None, 0):
                train_per_epoch = rr.training_time_seconds / rr.epochs_completed
            else:
                train_per_epoch = None

            lines.append(
                f"      seed={fmt_int(rr.run_seed)}"
                f"  train={fmt(rr.training_time_seconds, 4)}s"
                f"  test={fmt(rr.test_time_seconds, 4)}s"
                f"  total={fmt(total, 4)}s"
                f"  epochs={fmt_int(rr.epochs_completed)}"
                f"  train_per_epoch={fmt(train_per_epoch, 4)}s"
                f"  accuracy={fmt(rr.accuracy)}"
                f"  f1={fmt(rr.f1_macro)}"
                f"  best_val_loss={fmt(rr.best_val_loss)}"
            )

    return lines


def grouped_rows_block(title: str, rows: List[Dict[str, Any]]) -> List[str]:
    lines = block_header(title)
    if not rows:
        lines.append("No data.")
        return lines

    for i, row in enumerate(rows, start=1):
        lines.append(
            f"[{i}] {row['group']} | "
            f"n_configs={row['n_configs']} | "
            f"mean_train={fmt(row['mean_training_time_seconds'], 4)}s | "
            f"mean_test={fmt(row['mean_test_time_seconds'], 4)}s | "
            f"mean_total={fmt(row['mean_total_time_seconds'], 4)}s | "
            f"mean_train_per_epoch={fmt(row['mean_training_time_per_epoch_seconds'], 4)}s | "
            f"mean_accuracy={fmt(row['mean_accuracy'])} | "
            f"best_accuracy={fmt(row['best_accuracy_in_group'])}"
        )
        lines.append(
            f"    fastest_config_train={fmt(row['fastest_config_training_time_seconds'], 4)}s | "
            f"{row['fastest_config_label']}"
        )
        lines.append(
            f"    slowest_config_train={fmt(row['slowest_config_training_time_seconds'], 4)}s | "
            f"{row['slowest_config_label']}"
        )
    return lines


def write_timing_report(
    summaries: List[ConfigTimeSummary],
    skipped: List[str],
    output_file: Path,
) -> None:
    lines: List[str] = []
    lines.extend(block_header("Fractional Optimizer Timing Report"))

    lines.append(f"Results root: {RESULTS_ROOT}")
    lines.append(f"Parsed configuration files: {len(summaries)}")
    lines.append(f"Skipped files: {len(skipped)}")
    lines.append("")
    lines.append("Saved timing fields available from the original run script:")
    lines.append("  - training_time_seconds per run")
    lines.append("  - test_time_seconds per run")
    lines.append("  - epochs_completed per run")
    lines.append("  - configuration-level average training and test times")
    lines.append("")
    lines.append("Timing fields NOT available in the saved JSON:")
    lines.append("  - per-epoch wall-clock time")
    lines.append("  - validation-only time")
    lines.append("  - data preparation time")
    lines.append("  - model build time")
    lines.append("  - optimizer step time")
    lines.append("")

    if skipped:
        lines.extend(block_header("Skipped Files"))
        lines.extend(skipped)
        lines.append("")

    datasets = sorted({s.dataset for s in summaries})

    for dataset in datasets:
        ds = [s for s in summaries if s.dataset == dataset]
        fastest = sorted(
            ds,
            key=lambda x: (
                x.mean_training_time_seconds if x.mean_training_time_seconds is not None else float("inf"),
                x.mean_total_time_seconds if x.mean_total_time_seconds is not None else float("inf"),
            )
        )
        slowest = sorted(
            ds,
            key=lambda x: (
                x.mean_training_time_seconds if x.mean_training_time_seconds is not None else -float("inf"),
                x.mean_total_time_seconds if x.mean_total_time_seconds is not None else -float("inf"),
            ),
            reverse=True,
        )

        lines.extend(block_header(f"Dataset Timing Summary: {dataset}"))
        lines.append(f"Number of configurations: {len(ds)}")
        lines.append("")

        if fastest:
            lines.append(
                "Fastest configuration by mean training time: "
                f"{fastest[0].config_label} | "
                f"mean_train={fmt(fastest[0].mean_training_time_seconds, 4)}s | "
                f"mean_total={fmt(fastest[0].mean_total_time_seconds, 4)}s | "
                f"mean_accuracy={fmt(fastest[0].mean_accuracy)}"
            )
        if slowest:
            lines.append(
                "Slowest configuration by mean training time: "
                f"{slowest[0].config_label} | "
                f"mean_train={fmt(slowest[0].mean_training_time_seconds, 4)}s | "
                f"mean_total={fmt(slowest[0].mean_total_time_seconds, 4)}s | "
                f"mean_accuracy={fmt(slowest[0].mean_accuracy)}"
            )
        lines.append("")

        lines.extend(block_header(f"Fastest Configurations by Training Time: {dataset}"))
        for rank, s in enumerate(fastest[:TOP_K_FASTEST], start=1):
            lines.extend(config_time_block(rank, s))
            lines.append("")

        lines.extend(block_header(f"Slowest Configurations by Training Time: {dataset}"))
        for rank, s in enumerate(slowest[:TOP_K_SLOWEST], start=1):
            lines.extend(config_time_block(rank, s))
            lines.append("")

        lines.extend(
            grouped_rows_block(
                f"Timing Aggregation by Optimizer: {dataset}",
                aggregate_group(ds, key_fn=lambda s: s.optimizer),
            )
        )
        lines.append("")

        lines.extend(
            grouped_rows_block(
                f"Timing Aggregation by Activation: {dataset}",
                aggregate_group(ds, key_fn=lambda s: s.activation),
            )
        )
        lines.append("")

        lines.extend(
            grouped_rows_block(
                f"Timing Aggregation by Optimizer Family: {dataset}",
                aggregate_group(ds, key_fn=lambda s: s.optimizer_family),
            )
        )
        lines.append("")

    global_fastest = sorted(
        summaries,
        key=lambda x: (
            x.mean_training_time_seconds if x.mean_training_time_seconds is not None else float("inf"),
            x.mean_total_time_seconds if x.mean_total_time_seconds is not None else float("inf"),
        )
    )
    global_slowest = sorted(
        summaries,
        key=lambda x: (
            x.mean_training_time_seconds if x.mean_training_time_seconds is not None else -float("inf"),
            x.mean_total_time_seconds if x.mean_total_time_seconds is not None else -float("inf"),
        ),
        reverse=True,
    )

    lines.extend(block_header("Global Fastest Configurations by Mean Training Time"))
    for rank, s in enumerate(global_fastest[:TOP_K_FASTEST], start=1):
        lines.extend(config_time_block(rank, s))
        lines.append("")

    lines.extend(block_header("Global Slowest Configurations by Mean Training Time"))
    for rank, s in enumerate(global_slowest[:TOP_K_SLOWEST], start=1):
        lines.extend(config_time_block(rank, s))
        lines.append("")

    lines.extend(
        grouped_rows_block(
            "Global Timing Aggregation by Optimizer",
            aggregate_group(summaries, key_fn=lambda s: s.optimizer),
        )
    )
    lines.append("")

    lines.extend(
        grouped_rows_block(
            "Global Timing Aggregation by Activation",
            aggregate_group(summaries, key_fn=lambda s: s.activation),
        )
    )
    lines.append("")

    lines.extend(
        grouped_rows_block(
            "Global Timing Aggregation by Optimizer Family",
            aggregate_group(summaries, key_fn=lambda s: s.optimizer_family),
        )
    )
    lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 6) CONSOLE SUMMARY
# =============================================================================

def print_console_summary(summaries: List[ConfigTimeSummary], skipped: List[str]) -> None:
    print("=" * 100)
    print("Timing analysis of saved neural-network experiment results")
    print("=" * 100)
    print(f"Parsed configuration files: {len(summaries)}")
    print(f"Skipped files: {len(skipped)}")

    datasets = sorted({s.dataset for s in summaries})
    for dataset in datasets:
        ds = [s for s in summaries if s.dataset == dataset]
        fastest = sorted(
            ds,
            key=lambda x: (
                x.mean_training_time_seconds if x.mean_training_time_seconds is not None else float("inf"),
                x.mean_total_time_seconds if x.mean_total_time_seconds is not None else float("inf"),
            )
        )

        print("\n" + "-" * 100)
        print(f"DATASET: {dataset}")
        print("-" * 100)

        if fastest:
            best = fastest[0]
            print(
                f"Fastest config: {best.config_label} | "
                f"mean_train={fmt(best.mean_training_time_seconds, 4)}s | "
                f"mean_test={fmt(best.mean_test_time_seconds, 4)}s | "
                f"mean_total={fmt(best.mean_total_time_seconds, 4)}s | "
                f"mean_acc={fmt(best.mean_accuracy)}"
            )

        print("\nTop fastest configurations:")
        for rank, s in enumerate(fastest[:min(TOP_K_FASTEST, len(fastest))], start=1):
            print(
                f"[{rank:02d}] "
                f"{s.optimizer:40s} "
                f"act={s.activation:35s} "
                f"train={fmt(s.mean_training_time_seconds, 4)}s "
                f"test={fmt(s.mean_test_time_seconds, 4)}s "
                f"total={fmt(s.mean_total_time_seconds, 4)}s "
                f"train/epoch={fmt(s.mean_training_time_per_epoch_seconds, 4)}s "
                f"acc={fmt(s.mean_accuracy)}"
            )

    print("\nReport will include detailed per-run timing information.")


# =============================================================================
# 7) MAIN
# =============================================================================

def main() -> None:
    summaries, skipped = load_all_time_summaries(RESULTS_ROOT)

    if not summaries:
        raise RuntimeError(f"No valid result JSON files found under: {RESULTS_ROOT}")

    output_file = REPORTS_DIR / OUTPUT_REPORT_NAME
    write_timing_report(summaries, skipped, output_file)

    if PRINT_TO_CONSOLE:
        print_console_summary(summaries, skipped)

    print("\nTiming report written to:")
    print(output_file)


if __name__ == "__main__":
    main()