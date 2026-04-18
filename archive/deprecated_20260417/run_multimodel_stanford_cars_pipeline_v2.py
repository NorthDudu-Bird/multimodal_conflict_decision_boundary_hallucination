#!/usr/bin/env python
"""Run the full current strict-colors Stanford Cars multimodel pipeline end to end."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import build_logger, ensure_dirs, load_config, relative_str, repo_path


ROOT = REPO_ROOT
DATA_PREP_DIR = ROOT / "scripts" / "data_prep"
INFERENCE_DIR = ROOT / "scripts" / "inference"
PARSING_DIR = ROOT / "scripts" / "parsing"
ANALYSIS_DIR = ROOT / "scripts" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full current strict-colors Stanford Cars multimodel pipeline.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--truth-source", choices=["auto", "provisional", "reviewed", "adjudicated"], default="reviewed")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model_key values from the config.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-primary", action="store_true")
    parser.add_argument("--skip-auxiliary", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], logger) -> None:
    logger.info("Running command: %s", " ".join(str(item) for item in command))
    env = dict(subprocess.os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def selected_model_keys(config: dict, requested: list[str] | None) -> list[str]:
    configured = [model["model_key"] for model in config.get("models", [])]
    if not requested:
        return configured
    requested_set = set(requested)
    return [model_key for model_key in configured if model_key in requested_set]


def ensure_previous_run_note(config: dict) -> Path:
    legacy_dir = repo_path(config["legacy"]["archive_dir"]) / "previous_run"
    ensure_dirs([legacy_dir])
    note_path = legacy_dir / "README.md"
    lines = [
        "# Previous Run Preservation Note",
        "",
        "The pre-strict restructured outputs remain in place and were not overwritten:",
        "",
        f"- {relative_str(repo_path(config['outputs']['main_dir']))}",
        f"- {relative_str(repo_path(config['outputs']['auxiliary_dir']))}",
        f"- {relative_str(repo_path(config['outputs']['smoke_dir']))}",
        "",
        "The current strict-colors multimodel outputs are written to model-specific directories under `outputs/current/`.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def copy_prompt_table(source_csv: Path, destination_dir: Path, filename: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    shutil.copy2(source_csv, destination)
    return destination


def model_output_dir(config: dict, outputs_root: Path, model_key: str, family: str) -> Path:
    templates = config["outputs"].get("model_dir_templates", {}) or {}
    default_templates = {
        "smoke": "{model_key}_strict_colors_smoke",
        "primary": "{model_key}_strict_colors_primary",
        "auxiliary": "{model_key}_strict_colors_auxiliary",
    }
    template = str(templates.get(family, default_templates[family]))
    return outputs_root / template.format(model_key=model_key)


def run_family_for_model(config: dict, logger, model_key: str, family: str, prompt_csv: Path, output_dir: Path) -> Path:
    prompt_copy = copy_prompt_table(prompt_csv, output_dir, f"{family}_prompts.csv")
    runtime_csv = output_dir / f"{family}_runtime.csv"
    raw_csv = output_dir / f"{family}_raw_results.csv"
    parsed_csv = output_dir / f"{family}_parsed_results.csv"
    review_csv = output_dir / f"{family}_parse_review.csv"
    summary_md = output_dir / f"{family}_parse_summary.md"

    run_command(
        [
            sys.executable,
            str(INFERENCE_DIR / "run_multimodel_batch.py"),
            "--config",
            str(config["_config_path"]),
            "--model-key",
            model_key,
            "--input-csv",
            str(prompt_copy),
            "--output-csv",
            str(runtime_csv),
            "--log-path",
            str(output_dir / f"{family}.log"),
        ],
        logger,
    )
    run_command(
        [
            sys.executable,
            str(INFERENCE_DIR / "export_qwen2vl_raw_results.py"),
            "--source-csv",
            str(prompt_copy),
            "--runtime-csv",
            str(runtime_csv),
            "--output-csv",
            str(raw_csv),
        ],
        logger,
    )
    run_command(
        [
            sys.executable,
            str(PARSING_DIR / "parse_restructured_car_color_outputs.py"),
            "--config",
            str(config["_config_path"]),
            "--input-csv",
            str(raw_csv),
            "--output-csv",
            str(parsed_csv),
            "--review-csv",
            str(review_csv),
            "--summary-md",
            str(summary_md),
            "--log-path",
            str(output_dir / "parse.log"),
        ],
        logger,
    )
    return parsed_csv


def create_cross_model_index(primary_dir: Path, auxiliary_dir: Path, cross_model_dir: Path) -> None:
    ensure_dirs([cross_model_dir])
    files_to_copy = {
        "primary_model_condition_metrics.csv": primary_dir / "model_condition_metrics.csv",
        "primary_cross_model_pairwise.csv": primary_dir / "cross_model_pairwise.csv",
        "primary_summary_metrics.csv": primary_dir / "summary_metrics.csv",
        "auxiliary_model_condition_metrics.csv": auxiliary_dir / "model_condition_metrics.csv",
        "auxiliary_cross_model_pairwise.csv": auxiliary_dir / "cross_model_pairwise.csv",
        "auxiliary_summary_metrics.csv": auxiliary_dir / "summary_metrics.csv",
        "auxiliary_answer_space_compliance_metrics.csv": auxiliary_dir / "answer_space_compliance_metrics.csv",
        "auxiliary_answer_space_behavior_breakdown.csv": auxiliary_dir / "answer_space_behavior_breakdown.csv",
        "primary_cross_model_comparison.png": primary_dir / "plots" / "primary_cross_model_comparison.png",
        "auxiliary_cross_model_comparison.png": auxiliary_dir / "plots" / "auxiliary_cross_model_comparison.png",
        "primary_hr_heatmap.png": primary_dir / "plots" / "primary_hr_heatmap.png",
        "auxiliary_hr_heatmap.png": auxiliary_dir / "plots" / "auxiliary_hr_heatmap.png",
        "auxiliary_answer_space_compliance.png": auxiliary_dir / "plots" / "auxiliary_answer_space_compliance.png",
        "auxiliary_answer_space_breakdown.png": auxiliary_dir / "plots" / "auxiliary_answer_space_breakdown.png",
    }
    copied = {}
    for target_name, source_path in files_to_copy.items():
        if source_path.exists():
            destination = cross_model_dir / target_name
            shutil.copy2(source_path, destination)
            copied[target_name] = relative_str(destination)

    lines = [
        "# Cross-Model Strict-Colors Index",
        "",
        "This directory copies the core cross-model tables and figures from the family-level analyses.",
        "",
        "## Copied Files",
    ]
    for target_name in sorted(copied):
        lines.append(f"- {target_name}: {copied[target_name]}")
    (cross_model_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs([ROOT / "logs"])
    logger = build_logger("run_multimodel_stanford_cars_pipeline_v2", ROOT / "logs" / "run_multimodel_stanford_cars_pipeline_v2.log")
    model_keys = selected_model_keys(config, args.models)
    logger.info("Selected model_keys=%s", model_keys)

    if not model_keys:
        raise ValueError("No models were selected for the current strict-colors pipeline.")

    previous_run_note = ensure_previous_run_note(config)
    logger.info("Previous run preservation note: %s", relative_str(previous_run_note))

    if not args.skip_prepare:
        run_command(
            [
                sys.executable,
            str(DATA_PREP_DIR / "prepare_stanford_cars_multimodel_v2.py"),
                "--config",
                str(config["_config_path"]),
                "--truth-source",
                args.truth_source,
            ],
            logger,
        )

    prompts_cfg = config["prompts"]
    primary_prompt_csv = repo_path(prompts_cfg["v2_primary_csv"])
    auxiliary_prompt_csv = repo_path(prompts_cfg["v2_auxiliary_csv"])
    smoke_prompt_csv = repo_path(prompts_cfg["v2_smoke_csv"])

    primary_parsed_paths: list[Path] = []
    auxiliary_parsed_paths: list[Path] = []
    smoke_parsed_paths: list[Path] = []
    outputs_root = repo_path(config["outputs"]["v2_root_dir"])
    for model_key in model_keys:
        smoke_dir = model_output_dir(config=config, outputs_root=outputs_root, model_key=model_key, family="smoke")
        primary_dir = model_output_dir(config=config, outputs_root=outputs_root, model_key=model_key, family="primary")
        auxiliary_dir = model_output_dir(config=config, outputs_root=outputs_root, model_key=model_key, family="auxiliary")
        ensure_dirs([smoke_dir, primary_dir, auxiliary_dir])

        if not args.skip_smoke:
            smoke_parsed_paths.append(
                run_family_for_model(config=config, logger=logger, model_key=model_key, family="smoke", prompt_csv=smoke_prompt_csv, output_dir=smoke_dir)
            )
        if not args.skip_primary:
            primary_parsed_paths.append(
                run_family_for_model(config=config, logger=logger, model_key=model_key, family="primary", prompt_csv=primary_prompt_csv, output_dir=primary_dir)
            )
        if not args.skip_auxiliary:
            auxiliary_parsed_paths.append(
                run_family_for_model(config=config, logger=logger, model_key=model_key, family="auxiliary", prompt_csv=auxiliary_prompt_csv, output_dir=auxiliary_dir)
            )

    analysis_cfg = config["analysis"]
    analysis_primary_dir = repo_path(analysis_cfg["v2_primary_dir"])
    analysis_auxiliary_dir = repo_path(analysis_cfg["v2_auxiliary_dir"])
    analysis_cross_model_dir = repo_path(analysis_cfg["v2_cross_model_dir"])

    if not args.skip_analysis:
        if primary_parsed_paths:
            run_command(
                [
                    sys.executable,
                    str(ANALYSIS_DIR / "analyze_multimodel_car_color_results.py"),
                    "--config",
                    str(config["_config_path"]),
                    "--family",
                    "primary",
                    "--output-dir",
                    str(analysis_primary_dir),
                    "--input-csvs",
                    *[str(path) for path in primary_parsed_paths],
                ],
                logger,
            )
        if auxiliary_parsed_paths:
            run_command(
                [
                    sys.executable,
                    str(ANALYSIS_DIR / "analyze_multimodel_car_color_results.py"),
                    "--config",
                    str(config["_config_path"]),
                    "--family",
                    "auxiliary",
                    "--output-dir",
                    str(analysis_auxiliary_dir),
                    "--input-csvs",
                    *[str(path) for path in auxiliary_parsed_paths],
                    "--reference-primary-csvs",
                    *[str(path) for path in primary_parsed_paths],
                ],
                logger,
            )
        if analysis_primary_dir.exists() and analysis_auxiliary_dir.exists():
            create_cross_model_index(analysis_primary_dir, analysis_auxiliary_dir, analysis_cross_model_dir)

    result = {
        "truth_source": args.truth_source,
        "model_keys": model_keys,
        "smoke_parsed_paths": [relative_str(path) for path in smoke_parsed_paths],
        "primary_parsed_paths": [relative_str(path) for path in primary_parsed_paths],
        "auxiliary_parsed_paths": [relative_str(path) for path in auxiliary_parsed_paths],
        "analysis_primary_dir": relative_str(analysis_primary_dir),
        "analysis_auxiliary_dir": relative_str(analysis_auxiliary_dir),
        "analysis_cross_model_dir": relative_str(analysis_cross_model_dir),
    }
    logger.info("V2 pipeline completed: %s", json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
