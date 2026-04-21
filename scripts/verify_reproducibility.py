#!/usr/bin/env python
"""Compare rerun paper artifacts against a locked snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = ROOT / "logs" / "reproducibility_snapshot" / "latest"

CANONICAL_FILES = [
    "data/balanced_eval_set/appendix_stanford_core_excluded.csv",
    "data/balanced_eval_set/appendix_stanford_core_manifest.csv",
    "data/balanced_eval_set/final_manifest.csv",
    "data/metadata/balanced_eval_set/balanced_eval_set_color_by_source.csv",
    "data/metadata/balanced_eval_set/balanced_eval_set_summary.csv",
    "data/metadata/balanced_eval_set/balanced_eval_set_summary.json",
    "data/metadata/balanced_eval_set/cleaning_rules.md",
    "data/metadata/balanced_eval_set/dataset_distribution.csv",
    "data/metadata/balanced_eval_set/source_dataset_breakdown.csv",
    "prompts/a1_a2/a1_a2_prompts.csv",
    "prompts/c0_c4/c0_baseline_prompts.csv",
    "prompts/c0_c4/main_c0_c4_prompts.csv",
    "prompts/c0_c4/main_c1_c4_prompts.csv",
    "prompts/c0_c4/smoke_prompts.csv",
    "prompts/robustness/c3_prompt_variants.csv",
    "results/appendix/dataset_distribution.md",
    "results/appendix/dataset_distribution.png",
    "results/appendix/dataset_distribution_table.csv",
    "results/appendix/figure_manifest.json",
    "results/appendix/stanford_core_sanity_check.csv",
    "results/appendix/stanford_core_sanity_check.md",
    "results/appendix/stanford_core_to_balanced_counts.csv",
    "results/appendix/stanford_core_to_balanced_counts.md",
    "results/auxiliary/aux_combined_parsed_results.csv",
    "results/auxiliary/aux_condition_metrics.csv",
    "results/auxiliary/aux_exact_tests.csv",
    "results/auxiliary/aux_summary.md",
    "results/auxiliary/table3_aux_metrics.csv",
    "results/auxiliary/table3_aux_metrics.md",
    "results/baseline/baseline_combined_parsed_results.csv",
    "results/baseline/baseline_condition_metrics.csv",
    "results/baseline/baseline_exact_tests.csv",
    "results/baseline/baseline_summary.md",
    "results/final_result_summary.md",
    "results/main/figure2_conflict_aligned_rates.png",
    "results/main/main_combined_parsed_results.csv",
    "results/main/main_condition_metrics.csv",
    "results/main/main_exact_tests.csv",
    "results/main/main_key_tests.csv",
    "results/main/main_stats_summary.md",
    "results/main/main_summary.md",
    "results/main/table1_main_metrics.csv",
    "results/main/table1_main_metrics.md",
    "results/parser/ambiguous_outputs_sample.csv",
    "results/parser/label_mapping_audit.md",
    "results/parser/label_mapping_audit_table.csv",
    "results/robustness/prompt_variant_combined_parsed_results.csv",
    "results/robustness/prompt_variant_exact_tests.csv",
    "results/robustness/prompt_variant_metrics.csv",
    "results/robustness/prompt_variant_summary.json",
    "results/robustness/prompt_variant_summary.md",
]

TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
ALLOWED_VOLATILE_PATTERNS = [
    "*.log",
    "*_runtime.csv",
    "*_run_metadata.json",
    "*_raw_results.csv",
    "*_parse_review.csv",
    "results/**/raw/**",
]
COMBINED_RESULT_VOLATILE_COLUMNS = {
    "elapsed_seconds",
    "raw_output",
    "normalized_output",
    "device",
    "device_map",
    "parse_notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify reproducibility of rerun paper artifacts.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    return parser.parse_args()


def normalize_text(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8-sig")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized = "\n".join(lines).strip() + "\n"
    return normalized.encode("utf-8")


def normalize_combined_results_csv(path: Path) -> bytes:
    df = pd.read_csv(path, encoding="utf-8-sig")
    drop_cols = [column for column in COMBINED_RESULT_VOLATILE_COLUMNS if column in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    sort_cols = [column for column in ["sample_id", "model_key", "robustness_variant"] if column in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df.to_csv(index=False).replace("\r\n", "\n").encode("utf-8")


def file_digest(path: Path, relative_path: str) -> str:
    if relative_path.endswith("combined_parsed_results.csv"):
        data = normalize_combined_results_csv(path)
    else:
        data = normalize_text(path) if path.suffix.lower() in TEXT_SUFFIXES else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def compare_file(snapshot_root: Path, relative_path: str) -> dict[str, object]:
    snapshot_path = snapshot_root / relative_path
    current_path = ROOT / relative_path
    snapshot_exists = snapshot_path.exists()
    current_exists = current_path.exists()

    row: dict[str, object] = {
        "relative_path": relative_path,
        "snapshot_exists": snapshot_exists,
        "current_exists": current_exists,
        "snapshot_sha256": "",
        "current_sha256": "",
        "status": "",
        "blocking": True,
        "note": "",
    }

    if not snapshot_exists and not current_exists:
        row["status"] = "missing_both"
        row["note"] = "Neither snapshot nor current file exists."
        return row
    if not snapshot_exists:
        row["status"] = "missing_snapshot"
        row["note"] = "File exists after rerun but was absent in the locked snapshot."
        return row
    if not current_exists:
        row["status"] = "missing_current"
        row["note"] = "File existed in the locked snapshot but is missing after rerun."
        return row

    row["snapshot_sha256"] = file_digest(snapshot_path, relative_path)
    row["current_sha256"] = file_digest(current_path, relative_path)
    if row["snapshot_sha256"] == row["current_sha256"]:
        row["status"] = "match"
        row["note"] = "Canonical artifact matched."
    else:
        row["status"] = "different"
        row["note"] = "Canonical artifact differs from the locked snapshot."
    return row


def write_summary(rows: list[dict[str, object]], output_md: Path, snapshot_root: Path) -> None:
    total = len(rows)
    matched = sum(1 for row in rows if row["status"] == "match")
    blocking_failures = [row for row in rows if row["status"] != "match"]
    status_line = (
        "The rerun reproduced all locked canonical artifacts."
        if not blocking_failures
        else "The rerun did not reproduce all locked canonical artifacts."
    )
    lines = [
        "# Reproducibility Audit",
        "",
        f"- Snapshot root: `{snapshot_root}`",
        f"- Canonical files checked: {total}",
        f"- Exact/normalized matches: {matched}",
        f"- Blocking mismatches or missing files: {len(blocking_failures)}",
        f"- Verdict: {status_line}",
        "",
        "## Allowed Non-Canonical Differences",
        "",
    ]
    for pattern in ALLOWED_VOLATILE_PATTERNS:
        lines.append(f"- `{pattern}`")

    if blocking_failures:
        lines.extend(["", "## Blocking Items", ""])
        for row in blocking_failures:
            lines.append(f"- `{row['relative_path']}`: {row['status']}. {row['note']}")
    else:
        lines.extend(
            [
                "",
                "## Result",
                "",
                "- All tracked canonical manifests, prompts, parsed outputs, condition metrics, key tests, summary files, parser audit files, and appendix sanity files matched the locked snapshot.",
                "- Any log/runtime/raw-output differences are outside the reproducibility gate.",
            ]
        )

    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    snapshot_root = args.snapshot_dir
    if not snapshot_root.exists():
        print(
            json.dumps(
                {
                    "error": "snapshot_not_found",
                    "snapshot_dir": str(snapshot_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    rows = [compare_file(snapshot_root, relative_path) for relative_path in CANONICAL_FILES]
    comparison_df = pd.DataFrame(rows)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = results_dir / "reproducibility_comparison.csv"
    audit_md = results_dir / "reproducibility_audit.md"
    comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    write_summary(rows, audit_md, snapshot_root)

    failures = [row for row in rows if row["status"] != "match"]
    print(
        json.dumps(
            {
                "snapshot_dir": str(snapshot_root),
                "comparison_csv": str(comparison_csv),
                "audit_md": str(audit_md),
                "checked_files": len(rows),
                "blocking_failures": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
