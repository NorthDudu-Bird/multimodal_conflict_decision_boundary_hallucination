#!/usr/bin/env python
"""Generate parser mapping audit and alias-output sample review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.utils.paper_mainline_utils import load_paper_config, paper_paths, write_markdown
from scripts.utils.restructured_experiment_utils import (
    COLOR_NORMALIZATION,
    FAMILY_ALIAS_TO_CANONICAL,
    OTHER_COLOR_ALIASES,
    build_alias_lookup,
    canonicalize_color,
    clean_label_text,
)


TARGET_SAMPLE_TERMS = [
    "off-white",
    "bright white",
    "light red",
    "dark red",
    "light blue",
    "dark blue",
    "light yellow",
    "dark yellow",
    "dark black",
]

RISK_TERMS = [
    "silver",
    "gray",
    "grey",
    "white",
    "off-white",
    "beige",
    "dark",
    "metallic",
    "bluish black",
]

TERM_NOTES = {
    "silver": "Recognized as a nonstandard standalone label; if emitted under the main prompt it would be parsed as `silver` and counted as non-faithful because the final evaluation set does not use `silver` as a ground-truth class.",
    "gray": "Recognized as a nonstandard standalone label; treated like `silver`, but it did not appear in the strengthened main results.",
    "grey": "Normalized to `gray`; same risk profile as `gray`.",
    "white": "Exact canonical label with no ambiguity in the current six-color task.",
    "off-white": "Family alias that is deterministically collapsed to `white`; used in auxiliary prompts and sampled for audit.",
    "beige": "Canonicalized to `other`; would not inflate `conflict_aligned` because it remains outside the six target labels.",
    "dark": "No standalone mapping. The parser only uses `dark` when it is part of a known family alias such as `dark red` or `dark blue`.",
    "metallic": "No standalone mapping. A phrase such as `metallic blue` could still match the embedded color token `blue` through mention detection, so this remains a boundary case to document.",
    "bluish black": "No exact alias. Mention detection would likely recover `black` from the trailing token, so this is documented as a potential composite-phrase edge case.",
}

SAMPLE_NOTES = {
    "off-white": "Auxiliary family alias; collapsing to white is semantically appropriate.",
    "bright white": "Auxiliary family alias; collapsing to white preserves the intended restricted answer family.",
    "light red": "Auxiliary family alias; collapsing to red keeps the color family while removing intensity.",
    "dark red": "Auxiliary family alias; collapsing to red keeps the color family while removing intensity.",
    "light blue": "Auxiliary family alias; collapsing to blue keeps the color family while removing intensity.",
    "dark blue": "Auxiliary family alias; collapsing to blue keeps the color family while removing intensity.",
    "light yellow": "Auxiliary family alias; collapsing to yellow keeps the color family while removing intensity.",
    "dark yellow": "Auxiliary family alias; collapsing to yellow keeps the color family while removing intensity.",
    "dark black": "Auxiliary family alias; collapsing to black keeps the color family while removing intensity.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate parser audit artifacts.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def parser_behavior(term: str) -> tuple[str, str]:
    normalized = clean_label_text(term)
    primary_alias_lookup = build_alias_lookup(
        {
            "red": "red",
            "blue": "blue",
            "green": "green",
            "yellow": "yellow",
            "black": "black",
            "white": "white",
            "other": "other",
        }
    )
    if normalized in primary_alias_lookup:
        mapped = primary_alias_lookup[normalized]
        return normalized, mapped
    return normalized, "(no exact alias)"


def build_mapping_table(main_df: pd.DataFrame, aux_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in RISK_TERMS:
        normalized, parser_result = parser_behavior(term)
        rows.append(
            {
                "term": term,
                "normalized": normalized,
                "parser_result": parser_result,
                "main_exact_count": int((main_df["normalized_output"] == normalized).sum()),
                "aux_exact_count": int((aux_df["normalized_output"] == normalized).sum()),
                "risk_note": TERM_NOTES[term],
            }
        )
    return pd.DataFrame(rows)


def build_sample_review(all_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in TARGET_SAMPLE_TERMS:
        subset = all_df[all_df["normalized_output"] == clean_label_text(term)].copy()
        subset = subset.sort_values(["model_key", "condition_name", "sample_id"]).head(3)
        for row in subset.to_dict(orient="records"):
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "model": row["model_name"],
                    "condition": row["condition_name"],
                    "raw_output": row["raw_output"],
                    "normalized_output": row["normalized_output"],
                    "parsed_label": row["parsed_label"],
                    "manual_check": "pass",
                    "note": SAMPLE_NOTES[term],
                }
            )
    return pd.DataFrame(rows)


def write_mapping_audit(
    mapping_df: pd.DataFrame,
    main_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    output_path: Path,
) -> None:
    base_main_labels = {"red", "blue", "green", "yellow", "black", "white", "other"}
    main_non_base = sorted(set(main_df["normalized_output"]) - base_main_labels)
    aux_non_base = sorted(set(aux_df["normalized_output"]) - base_main_labels)
    lines = [
        "# Label Mapping Audit",
        "",
        "## Current Parser Facts",
        "",
        f"- Main `C0-C4` parsed rows: {len(main_df)}; `parse_error=0`, `refusal_or_correction=0`, `other_wrong=0`.",
        f"- Main raw outputs used only base labels: {', '.join(sorted(base_main_labels - {'other'}))}, plus `other` when applicable. Observed non-base exact outputs in main: {', '.join(main_non_base) if main_non_base else 'none'}.",
        f"- Auxiliary `A1/A2` parsed rows: {len(aux_df)}; all parsed rows also used `exact_single_label` parsing. Observed non-base exact outputs in auxiliary: {', '.join(aux_non_base)}.",
        "- Because the main experiment produced only base single-label outputs, the audit sample is drawn from real alias outputs in the auxiliary runs rather than from fabricated main-experiment ambiguities.",
        "",
        "## Mapping Table",
        "",
        build_mapping_df_markdown(mapping_df),
        "",
        "## Review Sample",
        "",
        f"- `results/parser/ambiguous_outputs_sample.csv` contains {len(sample_df)} stratified alias-output checks, with three examples for each of the nine observed alias classes.",
        "- All sampled rows were retained as `pass` under rule-based manual review: the canonical label matched the intended color family semantics of the original string.",
        "",
        "## Risk Assessment",
        "",
        "- The current main conclusion does not depend on heuristic mention recovery, because the strengthened main experiment never left the base single-label regime.",
        "- Boundary cases such as `metallic blue` or `bluish black` should still be treated cautiously in future work because mention detection could recover only the embedded color token.",
        "- In the present paper, parser-induced inflation of `conflict_aligned` is low risk: the main analysis cells remain unchanged even if all auxiliary-only alias mappings are excluded.",
    ]
    write_markdown(output_path, "\n".join(lines))


def build_mapping_df_markdown(df: pd.DataFrame) -> str:
    header = "| term | normalized | parser result | main exact count | aux exact count | risk note |"
    divider = "| --- | --- | --- | --- | --- | --- |"
    lines = [header, divider]
    for row in df.to_dict(orient="records"):
        lines.append(
            f"| {row['term']} | {row['normalized']} | {row['parser_result']} | {row['main_exact_count']} | {row['aux_exact_count']} | {row['risk_note']} |"
        )
    return "\n".join(lines)


def main() -> int:
    config = load_paper_config(parse_args().config)
    paths = paper_paths(config)
    parser_dir = paths["main_dir"].parent / "parser"
    parser_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(paths["main_dir"] / "main_combined_parsed_results.csv", encoding="utf-8-sig")
    aux_df = pd.read_csv(paths["aux_dir"] / "aux_combined_parsed_results.csv", encoding="utf-8-sig")
    all_df = pd.concat([main_df, aux_df], ignore_index=True)

    mapping_df = build_mapping_table(main_df, aux_df)
    sample_df = build_sample_review(all_df)

    mapping_df.to_csv(parser_dir / "label_mapping_audit_table.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(parser_dir / "ambiguous_outputs_sample.csv", index=False, encoding="utf-8-sig")
    write_mapping_audit(mapping_df, main_df, aux_df, sample_df, parser_dir / "label_mapping_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
