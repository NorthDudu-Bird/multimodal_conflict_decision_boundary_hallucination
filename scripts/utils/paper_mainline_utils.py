#!/usr/bin/env python
"""Helpers for the balanced-eval-set paper mainline workflow."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.utils.restructured_experiment_utils import ensure_dirs, load_config, repo_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "paper_mainline.yaml"
PRIMARY_CONDITION_ORDER = [
    "C0_neutral",
    "C1_weak_suggestion",
    "C2_false_assertion_open",
    "C3_presupposition_correction_allowed",
    "C4_stronger_open_conflict",
]
PRIMARY_CONDITION_SHORT_LABELS = {
    "C0_neutral": "C0",
    "C1_weak_suggestion": "C1",
    "C2_false_assertion_open": "C2",
    "C3_presupposition_correction_allowed": "C3",
    "C4_stronger_open_conflict": "C4",
}
AUXILIARY_CONDITION_ORDER = [
    "A1_forced_choice_red_family",
    "A2_counterfactual_assumption",
]
MODEL_ORDER = ["qwen2vl7b", "llava15_7b", "internvl2_8b"]
ROBUSTNESS_VARIANT_ORDER = ["C3_original", "C3_v2", "C3_v3"]


def load_paper_config(config_path: Path | None = None) -> dict:
    return load_config(config_path or DEFAULT_CONFIG_PATH)


def paper_paths(config: dict) -> dict[str, Path]:
    paper_cfg = config.get("paper", {}) or {}
    results_cfg = paper_cfg.get("results", {}) or {}
    return {
        "manifest_csv": repo_path(config["dataset"]["final_manifest_csv"]),
        "core_manifest_csv": repo_path(config["dataset"]["core_manifest_csv"]),
        "core_excluded_csv": repo_path(config["dataset"]["core_excluded_csv"]),
        "selected_manifest_csv": repo_path(config["vcor"]["selected_manifest_csv"]),
        "rejected_manifest_csv": repo_path(config["vcor"]["rejected_manifest_csv"]),
        "metadata_dir": repo_path(paper_cfg["metadata_dir"]),
        "baseline_prompt_csv": repo_path(paper_cfg["baseline_prompt_csv"]),
        "main_prompt_csv": repo_path(paper_cfg["main_prompt_csv"]),
        "main_nonbaseline_prompt_csv": repo_path(paper_cfg["main_nonbaseline_prompt_csv"]),
        "aux_prompt_csv": repo_path(paper_cfg["aux_prompt_csv"]),
        "robustness_prompt_csv": repo_path(paper_cfg["robustness_prompt_csv"]),
        "baseline_dir": repo_path(results_cfg["baseline_dir"]),
        "main_dir": repo_path(results_cfg["main_dir"]),
        "aux_dir": repo_path(results_cfg["aux_dir"]),
        "robustness_dir": repo_path(results_cfg["robustness_dir"]),
        "appendix_dir": repo_path(results_cfg["appendix_dir"]),
    }


def selected_model_keys(config: dict, requested: list[str] | None) -> list[str]:
    configured = [model["model_key"] for model in config.get("models", [])]
    if not requested:
        return configured
    requested_set = set(requested)
    return [model_key for model_key in configured if model_key in requested_set]


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=str(cwd or ROOT), check=True, env=env)


def filter_prompt_csv(source_csv: Path, target_csv: Path, condition_names: list[str]) -> int:
    ensure_dirs([target_csv.parent])
    keep = set(condition_names)
    with source_csv.open("r", encoding="utf-8-sig", newline="") as source_fh:
        reader = csv.DictReader(source_fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("condition_name") in keep]
    with target_csv.open("w", encoding="utf-8-sig", newline="") as target_fh:
        writer = csv.DictWriter(target_fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def limit_prompt_csv(source_csv: Path, target_csv: Path, limit: int) -> int:
    ensure_dirs([target_csv.parent])
    with source_csv.open("r", encoding="utf-8-sig", newline="") as source_fh:
        reader = csv.DictReader(source_fh)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    with target_csv.open("w", encoding="utf-8-sig", newline="") as target_fh:
        writer = csv.DictWriter(target_fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def model_raw_dir(results_dir: Path, model_key: str) -> Path:
    return results_dir / "raw" / model_key


def run_and_parse_prompt_set(
    *,
    config: dict,
    model_key: str,
    prompt_csv: Path,
    output_dir: Path,
    prefix: str,
    limit: int | None = None,
) -> Path:
    ensure_dirs([output_dir])
    active_prompt_csv = prompt_csv
    if limit is not None:
        active_prompt_csv = output_dir / f"{prefix}_active_prompts.csv"
        limit_prompt_csv(prompt_csv, active_prompt_csv, limit)
    runtime_csv = output_dir / f"{prefix}_runtime.csv"
    raw_csv = output_dir / f"{prefix}_raw_results.csv"
    parsed_csv = output_dir / f"{prefix}_parsed_results.csv"
    review_csv = output_dir / f"{prefix}_parse_review.csv"
    summary_md = output_dir / f"{prefix}_parse_summary.md"
    metadata_json = output_dir / f"{prefix}_run_metadata.json"

    inference_command = [
        sys.executable,
        str(ROOT / "scripts" / "inference" / "run_multimodel_batch.py"),
        "--config",
        str(config["_config_path"]),
        "--model-key",
        model_key,
        "--input-csv",
        str(active_prompt_csv),
        "--output-csv",
        str(runtime_csv),
        "--metadata-json",
        str(metadata_json),
        "--log-path",
        str(output_dir / f"{prefix}.log"),
    ]
    run_command(inference_command)

    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "inference" / "export_qwen2vl_raw_results.py"),
            "--source-csv",
            str(active_prompt_csv),
            "--runtime-csv",
            str(runtime_csv),
            "--output-csv",
            str(raw_csv),
        ]
    )
    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "parsing" / "parse_restructured_car_color_outputs.py"),
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
            str(output_dir / f"{prefix}_parse.log"),
        ]
    )
    return parsed_csv


def load_bool_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    for field in [
        "parse_success",
        "is_conflict_aligned",
        "is_faithful",
        "is_other_wrong",
        "is_refusal_or_correction",
        "is_parse_error",
        "in_allowed_answer_space",
    ]:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip().isin(["1", "true", "True"])
    return df


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_ci(low: float, high: float) -> str:
    return f"[{low * 100:.2f}%, {high * 100:.2f}%]"


def write_markdown(path: Path, text: str) -> None:
    ensure_dirs([path.parent])
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def dump_json(path: Path, payload: dict) -> None:
    ensure_dirs([path.parent])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
