"""Build the final 20-file writing pack for GPT paper drafting."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables"
ZIP_PATH = OUT_DIR / "gpt_paper_writing_pack_20files_final.zip"
MANIFEST_PATH = OUT_DIR / "gpt_paper_writing_pack_20files_manifest.md"


FILES = [
    (
        "README.md",
        "Top-level project scope, frozen empirical-paper boundary, and result map.",
    ),
    (
        "GPT_PROMPT_TEMPLATE.md",
        "Drafting instructions and explicit overclaim boundaries for downstream GPT writing.",
    ),
    (
        "docs/final_writing_interface_note.md",
        "Final writing interface: evidence hierarchy, terminology, caveats, and upload order.",
    ),
    (
        "docs/experiment_plan.md",
        "Original C0-C4/A1-A2 experimental design and main protocol framing.",
    ),
    (
        "docs/reproduction.md",
        "Reproduction and rerun instructions, including Phase 2 diagnostic scripts.",
    ),
    (
        "data/metadata/balanced_eval_set/balanced_eval_set_summary.json",
        "Dataset balance summary: 300 images, six colors, source composition.",
    ),
    (
        "results/final_result_summary.md",
        "Writing-facing final result summary with Phase 2 addendum.",
    ),
    (
        "results/results_discussion_summary.md",
        "Discussion-ready synthesis and limitation language.",
    ),
    (
        "results/main/table1_main_metrics.csv",
        "Canonical main C0-C4 metrics across all three models.",
    ),
    (
        "results/main/main_key_tests.csv",
        "Canonical statistical tests for the main experiment.",
    ),
    (
        "results/main/main_results_paper_ready.md",
        "Paper-ready main experiment narrative and table/figure references.",
    ),
    (
        "results/main/paired_flip_summary.md",
        "Same-image C0-to-conflict paired flip interpretation.",
    ),
    (
        "results/main/figure2_conflict_aligned_rates.png",
        "Main conflict-aligned rate figure for paper drafting.",
    ),
    (
        "results/auxiliary/table3_aux_metrics.csv",
        "A1/A2 auxiliary diagnostic metrics.",
    ),
    (
        "results/auxiliary/aux_role_note.md",
        "Boundary note keeping A1/A2 as auxiliary diagnostics, not main evidence.",
    ),
    (
        "results/robustness/prompt_boundary_summary.md",
        "C3 wording robustness and template-sensitivity summary.",
    ),
    (
        "results/parser/label_mapping_audit.md",
        "Parser reliability audit and label-mapping notes.",
    ),
    (
        "results/appendix/stanford_core_sanity_check.md",
        "Source-stratified sanity check for StanfordCars core images.",
    ),
    (
        "results/reproducibility_audit.md",
        "Canonical reproducibility audit and Phase 2 summary-drift interpretation.",
    ),
    (
        "results/phase2_final_summary.md",
        "Compact A-G Phase 2 synthesis covering all new diagnostics.",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(FILES) != 20:
        raise RuntimeError(f"Expected exactly 20 files, got {len(FILES)}")

    missing = [rel for rel, _ in FILES if not (ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError("Missing pack inputs: " + ", ".join(missing))

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as zf:
        for rel, _purpose in FILES:
            zf.write(ROOT / rel, rel)

    lines = [
        "# GPT Paper Writing Pack - 20 Files Final",
        "",
        f"- Zip: `{ZIP_PATH.relative_to(ROOT).as_posix()}`",
        "- Count: 20 selected project files, excluding this manifest.",
        "- Purpose: compact, writing-facing handoff that covers the final empirical study without dragging GPT into raw-result sprawl.",
        "",
        "## Files And Purposes",
        "",
        "| Upload order | File | Purpose |",
        "| --- | --- | --- |",
    ]
    for idx, (rel, purpose) in enumerate(FILES, start=1):
        lines.append(f"| {idx} | `{rel}` | {purpose} |")

    lines.extend(
        [
            "",
            "## Selection Logic",
            "",
            "The pack prioritizes project boundary, reproduction, dataset balance, canonical main results, A1/A2 auxiliary diagnostics, wording robustness, parser/source/reproducibility checks, and one compact Phase 2 synthesis. It intentionally does not include every raw Phase 2 CSV because the downstream writing task needs the final interpretation and boundaries, not a new benchmark-style data dump.",
            "",
            "## Recommended Upload Order",
            "",
        ]
    )
    for idx, (rel, _purpose) in enumerate(FILES, start=1):
        lines.append(f"{idx}. `{rel}`")

    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {ZIP_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print("Pack files: 20")


if __name__ == "__main__":
    main()
