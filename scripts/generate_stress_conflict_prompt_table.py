#!/usr/bin/env python
"""Generate a stress-conflict subset and prompt table with the shared existence-conflict schema."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

from existence_conflict_schema import (
    COMMON_PROMPT_FIELDS,
    STRESS_EXPERIMENT_TYPE,
    STRESS_PROMPT_TEMPLATES,
    build_common_prompt_row,
    get_expected_answer,
)
from metadata_paths import (
    LEGACY_NO_DOG_SAMPLE_50_CSV,
    NO_DOG_SAMPLE_50_CSV,
    NO_DOG_STRESS_SUBSET_10_CSV,
    STRESS_PROMPTS_CSV,
    ensure_metadata_dirs,
    resolve_existing_path,
)


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = resolve_existing_path(NO_DOG_SAMPLE_50_CSV, LEGACY_NO_DOG_SAMPLE_50_CSV)
SUBSET_CSV = NO_DOG_STRESS_SUBSET_10_CSV
OUTPUT_CSV = STRESS_PROMPTS_CSV

RANDOM_SEED = 42
SUBSET_SIZE = 10


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    return count


def sort_key(row: dict[str, str]) -> tuple[int, str]:
    image_id = str(row.get("image_id", ""))
    return (int(image_id) if image_id.isdigit() else 10**18, image_id)


def build_subset(base_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ordered_rows = sorted(base_rows, key=sort_key)
    if len(ordered_rows) < SUBSET_SIZE:
        raise ValueError(f"Expected at least {SUBSET_SIZE} rows, found {len(ordered_rows)}")

    rng = random.Random(RANDOM_SEED)
    sampled_rows = rng.sample(ordered_rows, SUBSET_SIZE)
    return sorted(sampled_rows, key=sort_key)


def build_prompt_rows(subset_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    expected_answer = get_expected_answer("en")
    prompt_rows: list[dict[str, str]] = []
    for row in subset_rows:
        for prompt_level, prompt_code, conflict_strength, prompt_text in STRESS_PROMPT_TEMPLATES:
            prompt_rows.append(
                build_common_prompt_row(
                    row,
                    experiment_type=STRESS_EXPERIMENT_TYPE,
                    prompt_level=prompt_level,
                    prompt_code=prompt_code,
                    conflict_strength=conflict_strength,
                    prompt_text=prompt_text,
                    expected_answer=expected_answer,
                    notes="",
                )
            )
    return prompt_rows


def main() -> int:
    ensure_metadata_dirs()
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    base_fieldnames, base_rows = read_rows(INPUT_CSV)
    subset_rows = build_subset(base_rows)
    prompt_rows = build_prompt_rows(subset_rows)

    subset_count = write_rows(SUBSET_CSV, base_fieldnames, subset_rows)
    prompt_count = write_rows(OUTPUT_CSV, COMMON_PROMPT_FIELDS, prompt_rows)

    summary = {
        "input_rows": len(base_rows),
        "subset_rows": subset_count,
        "prompt_rows": prompt_count,
        "random_seed": RANDOM_SEED,
        "subset_csv": str(SUBSET_CSV.relative_to(ROOT)),
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "prompt_codes": [code for _, code, _, _ in STRESS_PROMPT_TEMPLATES],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
