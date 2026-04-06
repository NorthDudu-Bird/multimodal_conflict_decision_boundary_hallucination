#!/usr/bin/env python
"""Run the Stanford Cars clean car-color pipeline end to end."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from metadata_paths import (
    METADATA_DIR,
    STANFORD_CARS_AUTOLABEL_SUMMARY_MD,
    STANFORD_CARS_FINAL_LABELED_CSV,
    STANFORD_CARS_MANUAL_REVIEW_CSV,
    STANFORD_CARS_PRELABELED_CSV,
    STANFORD_CARS_PROMPTS_CSV,
    STANFORD_CARS_RAW_CSV,
    STANFORD_CARS_REVIEW_CSV,
    STANFORD_CARS_RUN_SUMMARY_MD,
    STANFORD_CARS_RUNTIME_CSV,
    STANFORD_CARS_SAMPLE_CSV,
    STANFORD_CARS_SANITY_JSON,
    STANFORD_CARS_SMOKE_RUNTIME_CSV,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
LOGS_DIR = ROOT / "logs"
OUTPUT_DIR = METADATA_DIR

SAMPLES_CSV = STANFORD_CARS_SAMPLE_CSV
REVIEW_CSV = STANFORD_CARS_REVIEW_CSV
PROMPTS_CSV = STANFORD_CARS_PROMPTS_CSV
MANIFEST_CSV = ROOT / "data" / "processed" / "stanford_cars" / "clean_subset_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stanford Cars clean car-color experiment pipeline.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--results-dir", dest="output_dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--clean-subset-size", type=int, default=320)
    parser.add_argument("--experiment-sample-size", type=int, default=30)
    parser.add_argument("--candidate-pool-size", type=int, default=1600)
    parser.add_argument("--target-short-edge", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--smoke-limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-autolabel", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_car_color_stanford_clean_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_command(command: list[str], logger: logging.Logger) -> None:
    logger.info("Running: %s", " ".join(command))
    env = dict(**subprocess.os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "output_dir": output_dir,
        "pipeline_log": LOGS_DIR / "run_car_color_stanford_clean_pipeline.log",
        "smoke_runtime": STANFORD_CARS_SMOKE_RUNTIME_CSV,
        "runtime_csv": STANFORD_CARS_RUNTIME_CSV,
        "raw_csv": STANFORD_CARS_RAW_CSV,
        "prelabeled_csv": STANFORD_CARS_PRELABELED_CSV,
        "manual_review_csv": STANFORD_CARS_MANUAL_REVIEW_CSV,
        "final_labeled_csv": STANFORD_CARS_FINAL_LABELED_CSV,
        "autolabel_summary_md": STANFORD_CARS_AUTOLABEL_SUMMARY_MD,
        "sanity_json": STANFORD_CARS_SANITY_JSON,
        "run_summary_md": STANFORD_CARS_RUN_SUMMARY_MD,
    }


def run_sanity_check(paths: dict[str, Path], logger: logging.Logger) -> dict[str, object]:
    prompt_rows = read_csv_rows(PROMPTS_CSV)
    sample_rows = read_csv_rows(SAMPLES_CSV)
    runtime_rows = read_csv_rows(paths["runtime_csv"]) if paths["runtime_csv"].exists() else []
    raw_rows = read_csv_rows(paths["raw_csv"]) if paths["raw_csv"].exists() else []
    final_rows = read_csv_rows(paths["final_labeled_csv"]) if paths["final_labeled_csv"].exists() else []

    expected_levels = {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    missing_images = [row["image_path"] for row in prompt_rows if not (ROOT / row["image_path"]).exists()]
    runtime_status_counts = Counter((row.get("status", "") or "").strip().lower() or "blank" for row in runtime_rows)
    empty_outputs = [row.get("sample_id", "") for row in runtime_rows if not (row.get("raw_output", "") or "").strip()]

    prompt_levels_by_image: dict[str, set[str]] = defaultdict(set)
    runtime_levels_by_image: dict[str, set[str]] = defaultdict(set)
    for row in prompt_rows:
        prompt_levels_by_image[row["image_id"]].add(row["prompt_code"])
    for row in raw_rows:
        runtime_levels_by_image[row["image_id"]].add(row.get("prompt_code", ""))

    incomplete_prompt_images = sorted(image_id for image_id, levels in prompt_levels_by_image.items() if levels != expected_levels)
    incomplete_runtime_images = sorted(image_id for image_id, levels in runtime_levels_by_image.items() if levels != expected_levels)
    abnormal_output_rows = [
        row.get("sample_id", "")
        for row in runtime_rows
        if (row.get("raw_output", "") or "").strip() and len((row.get("raw_output", "") or "").strip()) < 2
    ]

    label_counts = Counter(row.get("label", "") for row in final_rows)
    sanity = {
        "sample_count": len(sample_rows),
        "prompt_count": len(prompt_rows),
        "runtime_count": len(runtime_rows),
        "raw_count": len(raw_rows),
        "final_labeled_count": len(final_rows),
        "missing_image_count": len(missing_images),
        "missing_images": missing_images[:20],
        "runtime_status_counts": dict(runtime_status_counts),
        "empty_output_count": len(empty_outputs),
        "empty_output_sample_ids": empty_outputs[:20],
        "abnormal_output_count": len(abnormal_output_rows),
        "abnormal_output_sample_ids": abnormal_output_rows[:20],
        "incomplete_prompt_images": incomplete_prompt_images[:20],
        "incomplete_runtime_images": incomplete_runtime_images[:20],
        "label_counts": dict(label_counts),
        "levels_complete_in_prompts": len(incomplete_prompt_images) == 0,
        "levels_complete_in_runtime": len(incomplete_runtime_images) == 0,
    }

    paths["sanity_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["sanity_json"].write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "# Stanford Cars Clean Run Summary",
        "",
        "## Counts",
        f"- clean subset manifest: {relative_str(MANIFEST_CSV)}",
        f"- experiment sample images: {len(sample_rows)}",
        f"- prompts: {len(prompt_rows)}",
        f"- runtime rows: {len(runtime_rows)}",
        f"- raw merged rows: {len(raw_rows)}",
        f"- final labeled rows: {len(final_rows)}",
        "",
        "## Sanity Check",
        f"- missing images: {len(missing_images)}",
        f"- empty outputs: {len(empty_outputs)}",
        f"- abnormal short outputs: {len(abnormal_output_rows)}",
        f"- prompt S0-S7 coverage complete: {len(incomplete_prompt_images) == 0}",
        f"- runtime S0-S7 coverage complete: {len(incomplete_runtime_images) == 0}",
        "",
        "## Runtime Status",
    ]
    for status, count in sorted(runtime_status_counts.items()):
        summary_lines.append(f"- {status}: {count}")
    summary_lines.extend(["", "## Final Labels"])
    for label, count in sorted(label_counts.items()):
        summary_lines.append(f"- {label or 'blank'}: {count}")
    paths["run_summary_md"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logger.info("Wrote sanity report: %s", relative_str(paths["run_summary_md"]))
    return sanity


def main() -> int:
    args = parse_args()
    paths = build_paths(args.output_dir)
    logger = build_logger(paths["pipeline_log"])
    logger.info("Starting Stanford Cars clean pipeline.")
    logger.info("Output dir: %s", paths["output_dir"])

    try:
        for key, path in paths.items():
            if key.endswith("_csv") or key.endswith("_md") or key.endswith("_json") or key.endswith("_runtime"):
                path.parent.mkdir(parents=True, exist_ok=True)

        if not args.skip_prepare:
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "generate_car_color_stanford_clean_table.py"),
                "--clean-subset-size",
                str(args.clean_subset_size),
                "--experiment-sample-size",
                str(args.experiment_sample_size),
                "--candidate-pool-size",
                str(args.candidate_pool_size),
                "--target-short-edge",
                str(args.target_short_edge),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.skip_download:
                command.append("--skip-download")
            run_command(command, logger)

        if not args.skip_smoke:
            smoke_command = [
                sys.executable,
                str(SCRIPTS_DIR / "run_qwen2vl_smoke_test.py"),
                "--input-csv",
                str(PROMPTS_CSV),
                "--output-csv",
                str(paths["smoke_runtime"]),
                "--log-path",
                str(LOGS_DIR / "qwen2vl_car_color_stanford_clean_smoke.log"),
                "--limit",
                str(args.smoke_limit),
                "--batch-size",
                "1",
                "--max-new-tokens",
                "96",
                "--temperature",
                "0.0",
            ]
            if args.use_4bit:
                smoke_command.append("--use-4bit")
            run_command(smoke_command, logger)

        if not args.skip_batch:
            batch_command = [
                sys.executable,
                str(SCRIPTS_DIR / "run_qwen2vl_batch.py"),
                "--input-csv",
                str(PROMPTS_CSV),
                "--output-csv",
                str(paths["runtime_csv"]),
                "--log-path",
                str(LOGS_DIR / "qwen2vl_car_color_stanford_clean_batch.log"),
                "--batch-size",
                str(args.batch_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
            ]
            if args.use_4bit:
                batch_command.append("--use-4bit")
            run_command(batch_command, logger)

        if not args.skip_export:
            export_command = [
                sys.executable,
                str(SCRIPTS_DIR / "export_qwen2vl_raw_results.py"),
                "--source-csv",
                str(PROMPTS_CSV),
                "--runtime-csv",
                str(paths["runtime_csv"]),
                "--output-csv",
                str(paths["raw_csv"]),
            ]
            run_command(export_command, logger)

        if not args.skip_autolabel:
            autolabel_command = [
                sys.executable,
                str(SCRIPTS_DIR / "auto_label_car_color_attribute_conflict_outputs.py"),
                "--prompt-csv",
                str(PROMPTS_CSV),
                "--runtime-csv",
                str(paths["runtime_csv"]),
                "--prelabeled-csv",
                str(paths["prelabeled_csv"]),
                "--manual-review-csv",
                str(paths["manual_review_csv"]),
                "--final-labeled-csv",
                str(paths["final_labeled_csv"]),
                "--summary-md",
                str(paths["autolabel_summary_md"]),
            ]
            run_command(autolabel_command, logger)

        sanity = run_sanity_check(paths, logger)
        print(json.dumps({"output_dir": relative_str(paths["output_dir"]), "sanity": sanity}, ensure_ascii=False, indent=2))
        logger.info("Stanford Cars clean pipeline completed successfully.")
        return 0
    except subprocess.CalledProcessError as exc:
        logger.error("Pipeline command failed with exit code %s: %s", exc.returncode, exc.cmd)
        return exc.returncode or 1
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
