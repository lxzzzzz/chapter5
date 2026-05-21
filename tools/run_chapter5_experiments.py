#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


METRIC_COLUMNS = ["sAMOTA", "AMOTA", "AMOTP", "MOTA", "MOTP", "IDS", "FRAG", "FP", "FN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan, run, and collect Chapter 5 tracking experiments.")
    parser.add_argument("--matrix", default="configs/chapter5_experiments.yaml")
    parser.add_argument("--tables", nargs="*", default=None, help="Table ids to process. Defaults to all tables.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset ids to process.")
    parser.add_argument("--methods", nargs="*", default=None, help="Method ids to process.")
    parser.add_argument("--output_root", default="outputs/chapter5")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--run_cache", action="store_true", help="Generate unified detector caches.")
    parser.add_argument("--run_tracking", action="store_true", help="Run implemented tracking rows.")
    parser.add_argument("--collect", action="store_true", help="Collect metrics into CSV and Markdown tables.")
    parser.add_argument("--fail_on_external", action="store_true", help="Fail if a selected method has no local runner.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    matrix = _load_yaml(repo_root / args.matrix)
    output_root = _resolve_path(repo_root, args.output_root)

    selected = _selected_jobs(matrix, args.tables, args.datasets, args.methods)
    if not selected:
        raise SystemExit("No jobs selected.")

    if args.run_cache:
        for dataset_id in _selected_dataset_ids(matrix, selected):
            command = _cache_command(repo_root, output_root, matrix, dataset_id, dry_run=args.dry_run)
            _run_or_print(command, cwd=repo_root, dry_run=args.dry_run)

    if args.run_tracking or (not args.run_cache and not args.collect):
        for job in selected:
            command = _tracking_command(repo_root, output_root, matrix, job)
            if command is None:
                message = (
                    f"external row skipped: table={job['table_id']} "
                    f"dataset={job['dataset_id']} method={job['method']['id']}"
                )
                note = job["method"].get("note")
                if note:
                    message += f" ({note})"
                if args.fail_on_external:
                    raise SystemExit(message)
                print(message)
                continue
            _run_or_print(command, cwd=repo_root, dry_run=args.dry_run)

    if args.collect:
        _collect_tables(repo_root, output_root, matrix, selected)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _selected_jobs(
    matrix: dict[str, Any],
    table_filter: list[str] | None,
    dataset_filter: list[str] | None,
    method_filter: list[str] | None,
) -> list[dict[str, Any]]:
    tables = matrix.get("tables", {})
    selected: list[dict[str, Any]] = []
    for table_id, table in tables.items():
        if table_filter and table_id not in table_filter:
            continue
        for dataset_id in table.get("datasets", []):
            if dataset_filter and dataset_id not in dataset_filter:
                continue
            for method in table.get("methods", []):
                if method_filter and method.get("id") not in method_filter:
                    continue
                selected.append({"table_id": table_id, "table": table, "dataset_id": dataset_id, "method": method})
    return selected


def _selected_dataset_ids(matrix: dict[str, Any], selected: list[dict[str, Any]]) -> list[str]:
    known = set(matrix.get("datasets", {}).keys())
    out = sorted({str(job["dataset_id"]) for job in selected if str(job["dataset_id"]) in known})
    return out


def _cache_command(repo_root: Path, output_root: Path, matrix: dict[str, Any], dataset_id: str, *, dry_run: bool) -> list[str]:
    datasets = matrix["datasets"]
    detector = matrix["unified_detector"]
    dataset = datasets[dataset_id]
    cfg_file = _materialized_cfg(repo_root, output_root, matrix, dataset_id, require_detector_ckpt=not dry_run)
    cache_root = _resolve_path(repo_root, dataset["cache_root"])
    command = [
        str(matrix.get("python", "python")),
        "tools/cache_voxelnext_detections.py",
        "--cfg_file",
        str(cfg_file),
        "--split",
        str(dataset.get("split", "val")),
        "--score_thresh",
        str(detector.get("cache_score_thresh", 0.05)),
        "--max_dets_per_frame",
        str(detector.get("max_dets_per_frame", 100)),
        "--output_dir",
        str(cache_root),
    ]
    return command


def _tracking_command(
    repo_root: Path,
    output_root: Path,
    matrix: dict[str, Any],
    job: dict[str, Any],
) -> list[str] | None:
    method = job["method"]
    runner = str(method.get("runner", "external"))
    if runner == "external":
        return None

    dataset_id = str(job["dataset_id"])
    dataset = matrix["datasets"][dataset_id]
    defaults = _tracking_params(matrix, dataset_id, method)
    metrics = matrix.get("metrics", {})
    cfg_file = _materialized_cfg(repo_root, output_root, matrix, dataset_id, require_detector_ckpt=False)
    cache_root = _resolve_path(repo_root, dataset["cache_root"])
    method_id = str(method["id"])
    table_id = str(job["table_id"])
    split = str(dataset.get("split", "val"))
    method_output = output_root / "runs" / table_id / dataset_id / method_id

    common = [
        "--cfg_file",
        str(cfg_file),
        "--split",
        split,
        "--detection_cache_root",
        str(cache_root),
        "--output_dir",
        str(method_output),
        "--score_thresh",
        str(defaults.get("eval_score_thresh", defaults.get("score_thresh", 0.6))),
        "--max_dets_per_frame",
        str(matrix.get("unified_detector", {}).get("max_dets_per_frame", 100)),
        "--max_lost_frames",
        str(defaults.get("max_lost_frames", 2)),
        "--min_hits",
        str(defaults.get("min_hits", 1)),
        "--eval_iou_threshold",
        str(metrics.get("eval_iou_threshold", 0.5)),
        "--ab3dmot_recall_points",
        str(metrics.get("recall_points", 40)),
        "--eval_ab3dmot",
    ]
    _append_dt_hypotheses(common, defaults)
    _append_motion_prior(common, defaults)
    if runner == "ab3dmot":
        return [
            str(matrix.get("python", "python")),
            "tools/run_ab3dmot_tracking.py",
            *common,
            "--iou_threshold",
            str(defaults.get("association_threshold", defaults.get("ab3dmot_iou_threshold", 0.1))),
        ]
    if runner == "hierarchical":
        return [
            str(matrix.get("python", "python")),
            "tools/run_hierarchical_tracking.py",
            *common,
            "--mode",
            str(method.get("mode", "stage1_stage2")),
            "--high_score_thresh",
            str(defaults.get("high_score_thresh", defaults.get("eval_score_thresh", defaults.get("score_thresh", 0.6)))),
            "--low_score_thresh",
            str(defaults.get("low_score_thresh", 0.3)),
            "--stage1_iou_threshold",
            str(defaults.get("association_threshold", defaults.get("stage1_iou_threshold", 0.1))),
            "--stage2_center_distance",
            str(defaults.get("stage2_center_distance", 4.0)),
        ]
    if runner == "chapter5":
        return [
            str(matrix.get("python", "python")),
            "tools/run_chapter5_tracking.py",
            *common,
            "--variant",
            str(method.get("variant", method_id)),
            "--high_score_thresh",
            str(defaults.get("high_score_thresh", defaults.get("eval_score_thresh", defaults.get("score_thresh", 0.6)))),
            "--low_score_thresh",
            str(defaults.get("low_score_thresh", 0.3)),
            "--iou_threshold",
            str(defaults.get("association_threshold", defaults.get("stage1_iou_threshold", defaults.get("ab3dmot_iou_threshold", 0.1)))),
            "--center_distance",
            str(method.get("center_distance", defaults.get("center_distance", defaults.get("stage2_center_distance", 4.0)))),
            "--tlom_threshold",
            str(method.get("tlom_threshold", defaults.get("tlom_threshold", 0.5))),
        ]
    raise ValueError(f"Unknown runner {runner!r} for method {method_id!r}")


def _tracking_params(matrix: dict[str, Any], dataset_id: str, method: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = dict(matrix.get("tracking_defaults", {}))
    params.update(matrix.get("tracking_overrides", {}).get(dataset_id, {}))
    params.update({key: value for key, value in method.items() if key in {
        "eval_score_thresh",
        "score_thresh",
        "association_threshold",
        "max_lost_frames",
        "min_hits",
        "high_score_thresh",
        "low_score_thresh",
        "stage2_center_distance",
        "center_distance",
        "tlom_threshold",
        "dt_hypotheses",
        "init_velocity_mode",
        "init_speed_prior",
    }})
    return params


def _append_dt_hypotheses(command: list[str], params: dict[str, Any]) -> None:
    values = params.get("dt_hypotheses")
    if values is None:
        return
    if not isinstance(values, (list, tuple)):
        values = [values]
    cleaned = [str(float(value)) for value in values if float(value) > 0]
    if cleaned:
        command.extend(["--dt_hypotheses", *cleaned])


def _append_motion_prior(command: list[str], params: dict[str, Any]) -> None:
    mode = str(params.get("init_velocity_mode", "zero"))
    speed = float(params.get("init_speed_prior", 0.0))
    if mode != "zero" or speed > 0:
        command.extend(["--init_velocity_mode", mode, "--init_speed_prior", str(speed)])


def _materialized_cfg(
    repo_root: Path,
    output_root: Path,
    matrix: dict[str, Any],
    dataset_id: str,
    *,
    require_detector_ckpt: bool,
) -> Path:
    dataset = matrix["datasets"][dataset_id]
    detector = matrix["unified_detector"]
    base_cfg_path = _resolve_path(repo_root, dataset["cfg_file"])
    cfg = _load_yaml(base_cfg_path)
    detector_cfg = detector.get("detector_cfg_file", {}).get(dataset_id, "")
    detector_ckpt = detector.get("detector_ckpt", {}).get(dataset_id, "")
    if require_detector_ckpt and not detector_ckpt:
        raise SystemExit(
            f"Missing unified_detector.detector_ckpt.{dataset_id} in {repo_root / 'configs/chapter5_experiments.yaml'}"
        )
    cfg["chapter3_root"] = matrix.get("chapter3_root", cfg.get("chapter3_root", ""))
    if detector_cfg:
        cfg["detector_cfg_file"] = detector_cfg
    cfg["detector_ckpt"] = detector_ckpt or ""
    cfg.setdefault("detection_cache", {})
    cfg["detection_cache"]["root"] = str(_resolve_path(repo_root, dataset["cache_root"]))
    cfg["detection_cache"]["score_thresh"] = float(detector.get("cache_score_thresh", cfg["detection_cache"].get("score_thresh", 0.05)))
    cfg["detection_cache"]["max_dets_per_frame"] = int(detector.get("max_dets_per_frame", cfg["detection_cache"].get("max_dets_per_frame", 100)))
    cfg["output_dir"] = str(output_root / "runs" / "_default" / dataset_id)

    generated_dir = output_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_dir / f"{dataset_id}.yaml"
    with generated_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return generated_path


def _run_or_print(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    printable = " ".join(_quote(part) for part in command)
    if dry_run:
        print(printable)
        return
    print(printable, flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def _collect_tables(repo_root: Path, output_root: Path, matrix: dict[str, Any], selected: list[dict[str, Any]]) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in selected:
        grouped.setdefault(str(job["table_id"]), []).append(job)

    for table_id, jobs in grouped.items():
        if table_id == "table_5_runtime":
            _collect_runtime_table(output_root, matrix, table_id, jobs)
            continue
        metric_columns = list(matrix.get("metrics", {}).get("columns", METRIC_COLUMNS))
        rows = []
        for job in jobs:
            method_id = str(job["method"]["id"])
            dataset_id = str(job["dataset_id"])
            metrics_path = output_root / "runs" / table_id / dataset_id / method_id / f"ab3dmot_{matrix['datasets'][dataset_id].get('split', 'val')}_metrics_ab3dmot.json"
            row = {
                "dataset": dataset_id,
                "method": job["method"].get("paper_name", method_id),
                "status": "missing",
            }
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                row["status"] = "ok"
                for key in metric_columns:
                    row[key] = metrics.get(key, "")
            else:
                for key in metric_columns:
                    row[key] = ""
            rows.append(row)
        csv_path = tables_dir / f"{table_id}.csv"
        md_path = tables_dir / f"{table_id}.md"
        _write_csv(csv_path, rows, ["dataset", "method", *metric_columns, "status"])
        _write_markdown(md_path, rows, ["dataset", "method", *metric_columns, "status"])
        print(f"wrote {csv_path}")
        print(f"wrote {md_path}")


def _collect_runtime_table(output_root: Path, matrix: dict[str, Any], table_id: str, jobs: list[dict[str, Any]]) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "method",
        "detector",
        "tracker",
        "num_frames",
        "tracking_fps",
        "tracking_ms_per_frame",
        "tracking_time_s",
        "status",
    ]
    rows = []
    for job in jobs:
        method_id = str(job["method"]["id"])
        dataset_id = str(job["dataset_id"])
        runtime_path = output_root / "runs" / table_id / dataset_id / method_id / "runtime_summary.json"
        row = {
            "dataset": dataset_id,
            "method": job["method"].get("paper_name", method_id),
            "detector": matrix.get("unified_detector", {}).get("name", "chapter4_best"),
            "tracker": job["method"].get("variant", method_id),
            "num_frames": "",
            "tracking_fps": "",
            "tracking_ms_per_frame": "",
            "tracking_time_s": "",
            "status": "missing",
        }
        if runtime_path.exists():
            with runtime_path.open("r", encoding="utf-8") as f:
                runtime = json.load(f)
            row.update(
                {
                    "num_frames": runtime.get("num_frames", ""),
                    "tracking_fps": runtime.get("tracking_fps", ""),
                    "tracking_ms_per_frame": runtime.get("tracking_ms_per_frame", ""),
                    "tracking_time_s": runtime.get("tracking_time_s", ""),
                    "status": "ok",
                }
            )
        rows.append(row)
    csv_path = tables_dir / f"{table_id}.csv"
    md_path = tables_dir / f"{table_id}.md"
    _write_csv(csv_path, rows, columns)
    _write_markdown(md_path, rows, columns)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in columns})


def _write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = [_format_value(row.get(key, "")) for key in columns]
            f.write("| " + " | ".join(values) + " |\n")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.4f}"
    return str(value)


def _quote(value: str) -> str:
    if not value:
        return "''"
    if any(char.isspace() for char in value):
        return "'" + value.replace("'", "'\\''") + "'"
    return value


if __name__ == "__main__":
    main()
