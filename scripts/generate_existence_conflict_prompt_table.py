#!/usr/bin/env python
"""Generate a baseline existence-conflict prompt table with the shared schema."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from existence_conflict_schema import (
    BASELINE_EXPERIMENT_TYPE,
    COMMON_PROMPT_FIELDS,
    build_common_prompt_row,
    get_baseline_prompt_templates,
    get_expected_answer,
)
from metadata_paths import (
    BASELINE_PROMPTS_CSV,
    BASELINE_PROMPTS_EN_CSV,
    LEGACY_NO_DOG_SAMPLE_50_CSV,
    NO_DOG_SAMPLE_50_CSV,
    ensure_metadata_dirs,
    resolve_existing_path,
)


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = resolve_existing_path(NO_DOG_SAMPLE_50_CSV, LEGACY_NO_DOG_SAMPLE_50_CSV)
LOG_FILE = ROOT / "logs" / "generate_prompt_table.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline existence-conflict prompt tables with zh/en prompt templates.")
    parser.add_argument("--prompt-language", choices=["zh", "en"], default="zh")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_output_csv(prompt_language: str, output_csv: Path | None) -> Path:
    if output_csv is not None:
        return output_csv
    if prompt_language == "en":
        return BASELINE_PROMPTS_EN_CSV
    return BASELINE_PROMPTS_CSV


def setup_logging() -> logging.Logger:
    ensure_metadata_dirs()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generate_existence_conflict_prompt_table")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def load_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_rows(base_rows: list[dict[str, str]], prompt_language: str) -> list[dict[str, str]]:
    prompt_templates = get_baseline_prompt_templates(prompt_language)
    expected_answer = get_expected_answer(prompt_language)
    output_rows: list[dict[str, str]] = []
    for row in base_rows:
        for prompt_level, prompt_code, conflict_strength, prompt_text in prompt_templates:
            output_rows.append(
                build_common_prompt_row(
                    row,
                    experiment_type=BASELINE_EXPERIMENT_TYPE,
                    prompt_level=prompt_level,
                    prompt_code=prompt_code,
                    conflict_strength=conflict_strength,
                    prompt_text=prompt_text,
                    expected_answer=expected_answer,
                )
            )
    return output_rows


def write_rows(output_csv: Path, rows: list[dict[str, str]], logger: logging.Logger) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMMON_PROMPT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s rows to %s", len(rows), output_csv.relative_to(ROOT).as_posix())


def main() -> int:
    args = parse_args()
    output_csv = resolve_output_csv(args.prompt_language, args.output_csv)
    logger = setup_logging()
    logger.info("Generating baseline existence-conflict prompt table from %s", INPUT_CSV.relative_to(ROOT).as_posix())
    logger.info("Prompt language: %s | Output CSV: %s", args.prompt_language, output_csv.relative_to(ROOT).as_posix())
    if not INPUT_CSV.exists():
        logger.error("Input CSV not found: %s", INPUT_CSV)
        return 1

    base_rows = load_rows()
    rows = build_rows(base_rows, args.prompt_language)
    write_rows(output_csv, rows, logger)
    logger.info("Baseline prompt table generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
