#!/usr/bin/env python
"""Build a controlled <=25-file paper writing pack."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


PACK_FILES = [
    "README.md",
    "GPT_PROMPT_TEMPLATE.md",
    "docs/experiment_plan.md",
    "docs/reproduction.md",
    "docs/strengthening_master_plan.md",
    "docs/writing_pack_upgrade_note.md",
    "data/balanced_eval_set/final_manifest.csv",
    "data/metadata/balanced_eval_set/balanced_eval_set_summary.json",
    "results/final_result_summary.md",
    "results/results_discussion_summary.md",
    "results/main/table1_main_metrics.csv",
    "results/main/main_stats_summary.md",
    "results/main/main_key_tests.csv",
    "results/main/paired_flip_summary.md",
    "results/main/main_results_paper_ready.md",
    "results/main/figure2_conflict_aligned_rates.png",
    "results/robustness/prompt_boundary_summary.md",
    "results/robustness/prompt_boundary_metrics.csv",
    "results/auxiliary/table3_aux_metrics.csv",
    "results/auxiliary/aux_role_note.md",
    "results/parser/label_mapping_audit.md",
    "results/appendix/stanford_core_sanity_check.md",
    "results/audit/visual_clarity_audit_readme.md",
    "results/threats_to_validity_summary.md",
    "results/reproducibility_audit.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paper writing pack.")
    parser.add_argument("--pack-date", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "deliverables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack_dir = args.output_root / f"gpt_paper_writing_pack_25files_{args.pack_date}"
    zip_path = args.output_root / f"{pack_dir.name}.zip"
    if len(PACK_FILES) > 25:
        raise ValueError(f"Writing pack contains {len(PACK_FILES)} files, exceeding the 25-file cap.")
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for rel in PACK_FILES:
        source = REPO_ROOT / rel
        if not source.exists():
            missing.append(rel)
            continue
        target = pack_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if missing:
        raise FileNotFoundError("Missing files for writing pack: " + ", ".join(missing))

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in PACK_FILES:
            zf.write(pack_dir / rel, arcname=f"{pack_dir.name}/{rel}")

    print(f"pack_dir={pack_dir}")
    print(f"zip_path={zip_path}")
    print(f"file_count={len(PACK_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
