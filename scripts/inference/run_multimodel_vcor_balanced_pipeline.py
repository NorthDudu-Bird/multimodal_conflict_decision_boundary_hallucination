#!/usr/bin/env python
"""Run the VCoR-balanced primary/auxiliary pipeline without overwriting current strict-colors results."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Run the current VCoR-balanced multimodel pipeline.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--core-only", action="store_true")
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
    wanted = set(requested)
    return [model_key for model_key in configured if model_key in wanted]


def model_output_dir(config: dict, outputs_root: Path, model_key: str, family: str) -> Path:
    template = str((config["outputs"].get("model_dir_templates", {}) or {})[family])
    return outputs_root / template.format(model_key=model_key)


def run_family_for_model(config: dict, logger, model_key: str, family: str, prompt_csv: Path, output_dir: Path) -> Path:
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
            str(prompt_csv),
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
            str(prompt_csv),
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


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs([ROOT / "logs"])
    log_path = repo_path((config.get("dataset_builder") or {}).get("pipeline_log_path", "logs/run_multimodel_vcor_balanced_pipeline.log"))
    logger = build_logger("run_multimodel_vcor_balanced_pipeline", log_path)
    model_keys = selected_model_keys(config=config, requested=args.models)
    if not model_keys:
        raise ValueError("No models selected for the VCoR-balanced pipeline.")

    if not args.skip_prepare:
        command = [
            sys.executable,
            str(DATA_PREP_DIR / "build_primary_vcor_balanced_manifests.py"),
            "--config",
            str(config["_config_path"]),
        ]
        if args.core_only:
            command.append("--core-only")
        run_command(command, logger)

    prompts_cfg = config["prompts"]
    primary_prompt_csv = repo_path(prompts_cfg["v2_primary_csv"])
    auxiliary_prompt_csv = repo_path(prompts_cfg["v2_auxiliary_csv"])
    smoke_prompt_csv = repo_path(prompts_cfg["v2_smoke_csv"])
    outputs_root = repo_path(config["outputs"]["v2_root_dir"])

    primary_parsed_paths: list[Path] = []
    auxiliary_parsed_paths: list[Path] = []
    smoke_parsed_paths: list[Path] = []
    for model_key in model_keys:
        smoke_dir = model_output_dir(config=config, outputs_root=outputs_root, model_key=model_key, family="smoke")
        primary_dir = model_output_dir(config=config, outputs_root=outputs_root, model_key=model_key, family="primary")
        auxiliary_dir = model_output_dir(config=config, outputs_root=outputs_root, model_key=model_key, family="auxiliary")
        ensure_dirs([smoke_dir, primary_dir, auxiliary_dir])

        if not args.skip_smoke:
            smoke_parsed_paths.append(run_family_for_model(config, logger, model_key, "smoke", smoke_prompt_csv, smoke_dir))
        if not args.skip_primary:
            primary_parsed_paths.append(run_family_for_model(config, logger, model_key, "primary", primary_prompt_csv, primary_dir))
        if not args.skip_auxiliary:
            auxiliary_parsed_paths.append(run_family_for_model(config, logger, model_key, "auxiliary", auxiliary_prompt_csv, auxiliary_dir))

    analysis_cfg = config["analysis"]
    primary_analysis_dir = repo_path(analysis_cfg["v2_primary_dir"])
    auxiliary_analysis_dir = repo_path(analysis_cfg["v2_auxiliary_dir"])

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
                    str(primary_analysis_dir),
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
                    str(auxiliary_analysis_dir),
                    "--input-csvs",
                    *[str(path) for path in auxiliary_parsed_paths],
                    "--reference-primary-csvs",
                    *[str(path) for path in primary_parsed_paths],
                ],
                logger,
            )

    payload = {
        "model_keys": model_keys,
        "core_only": args.core_only,
        "smoke_parsed_paths": [relative_str(path) for path in smoke_parsed_paths],
        "primary_parsed_paths": [relative_str(path) for path in primary_parsed_paths],
        "auxiliary_parsed_paths": [relative_str(path) for path in auxiliary_parsed_paths],
        "analysis_primary_dir": relative_str(primary_analysis_dir),
        "analysis_auxiliary_dir": relative_str(auxiliary_analysis_dir),
    }
    logger.info("VCoR-balanced pipeline complete: %s", json.dumps(payload, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
