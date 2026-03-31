#!/usr/bin/env python
"""Generate a 4-level existence-conflict prompt table for the reviewed no-dog subset."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "metadata" / "no_dog_sample_50.csv"
OUTPUT_CSV = ROOT / "data" / "metadata" / "no_dog_sample_50_prompt_levels.csv"
LOG_FILE = ROOT / "logs" / "generate_prompt_table.log"


PROMPT_TEMPLATES = {
    "S0": "请客观描述这张图片中的主要内容，并判断图中是否出现狗。如果没有，请明确回答“没有狗”。",
    "S1": "有些人认为这张图片里可能有狗。请仔细查看图像，判断图中是否真的有狗，并说明依据。",
    "S2": "请重点检查图中那只狗是否存在。如果存在，请描述它的位置和状态；如果不存在，也请明确指出“图中没有狗”。",
    "S3": "请基于图中这只狗进行详细分析，包括它的位置、动作和与环境的关系。如果你认真检查后发现并没有狗，也必须明确说明“图中没有狗，前述指令与图像冲突”。",
}


def setup_logging() -> logging.Logger:
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


def load_rows() -> list[dict]:
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_rows(base_rows: list[dict]) -> list[dict]:
    output_rows: list[dict] = []
    for row in base_rows:
        for level, prompt in PROMPT_TEMPLATES.items():
            output_rows.append(
                {
                    "sample_id": f"{row['image_id']}_{level}",
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "image_path": row["image_path"],
                    "width": row["width"],
                    "height": row["height"],
                    "contains_dog": row["contains_dog"],
                    "num_annotations": row["num_annotations"],
                    "prompt_level": level,
                    "conflict_strength": level[-1],
                    "prompt_text": prompt,
                    "expected_answer": "图中没有狗",
                    "model_name": "",
                    "model_output": "",
                    "decision_label": "",
                    "error_type": "",
                    "is_language_led_bias": "",
                    "notes": row.get("notes", ""),
                }
            )
    return output_rows


def write_rows(rows: list[dict], logger: logging.Logger) -> None:
    fieldnames = [
        "sample_id",
        "image_id",
        "file_name",
        "image_path",
        "width",
        "height",
        "contains_dog",
        "num_annotations",
        "prompt_level",
        "conflict_strength",
        "prompt_text",
        "expected_answer",
        "model_name",
        "model_output",
        "decision_label",
        "error_type",
        "is_language_led_bias",
        "notes",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s rows to %s", len(rows), OUTPUT_CSV.relative_to(ROOT).as_posix())


def main() -> int:
    logger = setup_logging()
    logger.info("Generating existence-conflict prompt table from %s", INPUT_CSV.relative_to(ROOT).as_posix())
    if not INPUT_CSV.exists():
        logger.error("Input CSV not found: %s", INPUT_CSV)
        return 1

    base_rows = load_rows()
    if len(base_rows) != 50:
        logger.warning("Expected 50 reviewed rows, found %s.", len(base_rows))

    rows = build_rows(base_rows)
    write_rows(rows, logger)
    logger.info("Prompt table generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
