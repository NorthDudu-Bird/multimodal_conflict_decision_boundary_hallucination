#!/usr/bin/env python
"""Run C3 prompt-variant robustness experiment across the balanced evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.paper_mainline_utils import (
    load_paper_config,
    model_raw_dir,
    paper_paths,
    run_and_parse_prompt_set,
    run_command,
    selected_model_keys,
)
from scripts.utils.restructured_experiment_utils import ensure_parent, primary_instruction


ROOT = REPO_ROOT


NEW_VARIANT_TEXTS = {
    "C3_v2": "Someone describes the main car as {conflict_color}. Based on the image, what is the main car's actual primary body color? {instruction}",
    "C3_v3": "The prompt says the main car is {conflict_color}. Looking at the image, which primary body color does the main car actually have? {instruction}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C3 prompt-variant robustness experiment.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_prompt_variant_csv(source_csv: Path, target_csv: Path, config: dict) -> int:
    with source_csv.open("r", encoding="utf-8-sig", newline="") as source_fh:
        reader = csv.DictReader(source_fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("condition_name") == "C3_presupposition_correction_allowed"]

    extra_fields = ["robustness_variant", "robustness_variant_label"]
    merged_fieldnames = list(fieldnames)
    for field in extra_fields:
        if field not in merged_fieldnames:
            merged_fieldnames.append(field)

    variant_rows: list[dict[str, str]] = []
    instruction = primary_instruction(condition_version="v2", config=config)
    for row in rows:
        for variant_name, template in NEW_VARIANT_TEXTS.items():
            updated = dict(row)
            updated["sample_id"] = f"{row['image_id']}__{variant_name}"
            updated["robustness_variant"] = variant_name
            updated["robustness_variant_label"] = variant_name
            updated["prompt_template_version"] = f"{row['prompt_template_version']}__{variant_name}"
            updated["prompt_text"] = template.format(
                conflict_color=row["conflict_color"],
                instruction=instruction,
            )
            variant_rows.append(updated)

    ensure_parent(target_csv)
    with target_csv.open("w", encoding="utf-8-sig", newline="") as target_fh:
        writer = csv.DictWriter(target_fh, fieldnames=merged_fieldnames)
        writer.writeheader()
        writer.writerows(variant_rows)
    return len(variant_rows)


def main() -> int:
    args = parse_args()
    config = load_paper_config(args.config)
    paths = paper_paths(config)
    model_keys = selected_model_keys(config, args.models)
    robustness_dir = paths["robustness_dir"]
    robustness_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        run_command([sys.executable, str(ROOT / "scripts" / "build_dataset.py"), "--config", str(config["_config_path"])])

    prompt_count = build_prompt_variant_csv(
        paths["main_prompt_csv"],
        paths["robustness_prompt_csv"],
        config,
    )
    metadata_path = robustness_dir / "prompt_variant_manifest.json"
    metadata_path.write_text(
        json.dumps(
            {
                "prompt_csv": str(paths["robustness_prompt_csv"]),
                "prompt_rows": prompt_count,
                "variants": list(NEW_VARIANT_TEXTS),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    parsed_paths: list[Path] = []
    for model_key in model_keys:
        output_dir = model_raw_dir(robustness_dir, model_key)
        parsed_paths.append(
            run_and_parse_prompt_set(
                config=config,
                model_key=model_key,
                prompt_csv=paths["robustness_prompt_csv"],
                output_dir=output_dir,
                prefix="c3_prompt_variants",
                limit=args.limit,
            )
        )

    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_prompt_variant_robustness.py"),
            "--output-dir",
            str(robustness_dir),
            "--reference-main-csv",
            str(paths["main_dir"] / "main_combined_parsed_results.csv"),
            "--input-csvs",
            *[str(path) for path in parsed_paths],
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
