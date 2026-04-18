#!/usr/bin/env python
"""Build the paper mainline balanced evaluation set and paper-facing metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.utils.paper_mainline_utils import (
    AUXILIARY_CONDITION_ORDER,
    PRIMARY_CONDITION_ORDER,
    dump_json,
    filter_prompt_csv,
    load_paper_config,
    paper_paths,
    run_command,
    write_markdown,
)

ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the balanced evaluation set used by the current paper.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def build_summary_tables(config: dict) -> dict[str, object]:
    paths = paper_paths(config)
    manifest_df = pd.read_csv(paths["manifest_csv"], encoding="utf-8-sig")
    core_df = pd.read_csv(paths["core_manifest_csv"], encoding="utf-8-sig")
    excluded_df = pd.read_csv(paths["core_excluded_csv"], encoding="utf-8-sig")
    selected_df = pd.read_csv(paths["selected_manifest_csv"], encoding="utf-8-sig")
    rejected_df = pd.read_csv(paths["rejected_manifest_csv"], encoding="utf-8-sig")

    metadata_dir = paths["metadata_dir"]
    metadata_dir.mkdir(parents=True, exist_ok=True)

    counts_by_color = (
        manifest_df.groupby(["true_color", "source_dataset"]).size().rename("count").reset_index()
    )
    counts_by_color.to_csv(metadata_dir / "balanced_eval_set_color_by_source.csv", index=False, encoding="utf-8-sig")

    final_counts = manifest_df["true_color"].value_counts().sort_index()
    source_counts = manifest_df["source_dataset"].value_counts().sort_index()
    core_counts = core_df["true_color"].value_counts().sort_index()
    rejected_counts = rejected_df["assigned_true_color"].value_counts().sort_index()

    summary_rows = [
        {"metric": "final_total", "value": int(len(manifest_df))},
        {"metric": "core_stanford_total", "value": int(len(core_df))},
        {"metric": "vcor_selected_total", "value": int(len(selected_df))},
        {"metric": "vcor_rejected_total", "value": int(len(rejected_df))},
        {"metric": "core_excluded_total", "value": int(len(excluded_df))},
        {"metric": "target_per_color", "value": int(config["dataset_builder"]["target_per_color"])},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(metadata_dir / "balanced_eval_set_summary.csv", index=False, encoding="utf-8-sig")

    distribution_rows = []
    color_order = ["black", "blue", "green", "red", "white", "yellow"]
    for color in color_order:
        distribution_rows.append(
            {
                "true_color": color,
                "stanford_core_n": int(core_counts.get(color, 0)),
                "final_balanced_total_n": int(final_counts.get(color, 0)),
                "final_from_stanford_n": int(
                    len(manifest_df[(manifest_df["true_color"] == color) & (manifest_df["source_dataset"] == "StanfordCars")])
                ),
                "final_from_vcor_n": int(
                    len(manifest_df[(manifest_df["true_color"] == color) & (manifest_df["source_dataset"] == "VCoR")])
                ),
                "vcor_rejected_n": int(rejected_counts.get(color, 0)),
            }
        )
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(metadata_dir / "dataset_distribution.csv", index=False, encoding="utf-8-sig")

    source_df = pd.DataFrame(
        [{"source_dataset": source_name, "count": int(count)} for source_name, count in source_counts.items()]
    )
    source_df.to_csv(metadata_dir / "source_dataset_breakdown.csv", index=False, encoding="utf-8-sig")

    cleaning_rules = """# Balanced Eval Set Cleaning Rules

1. The official paper dataset is the final balanced evaluation set in `data/balanced_eval_set/final_manifest.csv`.
2. Only the six paper colors are retained for the mainline: `red`, `blue`, `green`, `yellow`, `black`, `white`.
3. The Stanford Cars strict-clean subset is treated as the seed source, not the final paper benchmark.
4. Ten Stanford seed examples flagged by the latest manual review are excluded before balancing.
5. VCoR supplementation is used only to fill per-color shortages until each color reaches 50 examples.
6. The benchmark keeps the strict reviewed truth labels and does not relax the faithful-match rule.
7. `gray`, `silver`, and `other` remain excluded from the paper main analysis.
"""
    write_markdown(metadata_dir / "cleaning_rules.md", cleaning_rules)

    summary_payload = {
        "final_total": int(len(manifest_df)),
        "core_stanford_total": int(len(core_df)),
        "vcor_selected_total": int(len(selected_df)),
        "vcor_rejected_total": int(len(rejected_df)),
        "core_excluded_total": int(len(excluded_df)),
        "colors": distribution_rows,
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
    }
    dump_json(metadata_dir / "balanced_eval_set_summary.json", summary_payload)
    return summary_payload


def main() -> int:
    args = parse_args()
    config = load_paper_config(args.config)
    paths = paper_paths(config)

    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "data_prep" / "build_primary_vcor_balanced_manifests.py"),
            "--config",
            str(config["_config_path"]),
        ]
    )

    baseline_rows = filter_prompt_csv(
        paths["main_prompt_csv"],
        paths["baseline_prompt_csv"],
        ["C0_neutral"],
    )
    main_nonbaseline_rows = filter_prompt_csv(
        paths["main_prompt_csv"],
        paths["main_nonbaseline_prompt_csv"],
        PRIMARY_CONDITION_ORDER[1:],
    )
    aux_rows = filter_prompt_csv(
        paths["aux_prompt_csv"],
        paths["aux_prompt_csv"],
        AUXILIARY_CONDITION_ORDER,
    )

    summary_payload = build_summary_tables(config)
    summary_payload.update(
        {
            "baseline_prompt_rows": baseline_rows,
            "main_nonbaseline_prompt_rows": main_nonbaseline_rows,
            "aux_prompt_rows": aux_rows,
            "manifest_csv": str(paths["manifest_csv"]),
            "baseline_prompt_csv": str(paths["baseline_prompt_csv"]),
            "main_prompt_csv": str(paths["main_prompt_csv"]),
            "aux_prompt_csv": str(paths["aux_prompt_csv"]),
        }
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
