#!/usr/bin/env python
"""Rule-based prelabeling for car color attribute-conflict outputs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from metadata_paths import ensure_metadata_dirs


ROOT = Path(__file__).resolve().parent.parent
PROMPTS_CSV = ROOT / "data" / "metadata" / "prompts" / "car_color_attribute_conflict_s0_s7_30x8.csv"
RUNTIME_CSV = ROOT / "data" / "metadata" / "outputs_raw" / "qwen2vl7b_car_color_attribute_conflict_runtime.csv"
PRELABELED_CSV = ROOT / "data" / "metadata" / "outputs_labeled" / "qwen2vl7b_car_color_attribute_conflict_prelabeled.csv"
MANUAL_REVIEW_CSV = ROOT / "data" / "metadata" / "outputs_labeled" / "qwen2vl7b_car_color_attribute_conflict_manual_review.csv"
FINAL_LABELED_CSV = ROOT / "data" / "metadata" / "outputs_labeled" / "qwen2vl7b_car_color_attribute_conflict_final_labeled.csv"
SUMMARY_MD = ROOT / "reports" / "car_color_attribute_conflict_autolabel_summary.md"

OK_STATUSES = {"", "ok", "success"}
COLOR_TOKENS = ["red", "blue", "green", "yellow", "white", "black", "gray", "silver", "brown", "orange"]
COLOR_ALIAS_MAP = {"grey": "gray"}
TRUE_COLOR_COMPATIBILITY = {
    "gray": {"gray", "silver"},
    "silver": {"gray", "silver"},
}
UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:appears?\s+to\s+be|appears?|seems?|possibly|maybe|perhaps|likely|unclear|hard\s+to\s+tell|"
    r"difficult\s+to\s+tell|not\s+sure|unsure)\b",
    re.IGNORECASE,
)
CAR_ABSENCE_PATTERN = re.compile(
    r"\b(?:there\s+(?:is|are)\s+no|there\s+is\s+no\s+existence\s+of|no\s+existence\s+of)\s+(?:a\s+)?car\b",
    re.IGNORECASE,
)

LABEL_FIELDS = [
    "label",
    "language_consistent",
    "vision_consistent",
    "ambiguous",
    "notes",
    "auto_label_confidence",
    "auto_label_reason",
    "compatible_true_colors",
    "detected_positive_colors",
    "detected_negated_colors",
]


def build_negation_patterns(color: str) -> list[re.Pattern[str]]:
    return [
        re.compile(
            rf"\bnot\s+(?:a\s+|the\s+)?(?:light\s+|dark\s+|metallic\s+|glossy\s+|matte\s+)?{color}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:is\s+not|isn't|was\s+not|wasn't|does\s+not\s+look|doesn't\s+look|looks\s+not)\s+"
            rf"(?:a\s+|the\s+)?(?:light\s+|dark\s+|metallic\s+|glossy\s+|matte\s+)?{color}\b",
            re.IGNORECASE,
        ),
        re.compile(rf"\bthere\s+(?:is|are)\s+no\s+(?:\w+\s+){{0,3}}{color}\s+car\b", re.IGNORECASE),
        re.compile(rf"\bno\s+(?:\w+\s+){{0,3}}{color}\s+car\b", re.IGNORECASE),
    ]


NEGATION_PATTERNS = {color: build_negation_patterns(color) for color in COLOR_TOKENS}
POSITIVE_PATTERNS = {color: re.compile(rf"\b{color}\b", re.IGNORECASE) for color in COLOR_TOKENS}


def merge_fieldnames(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for field in group:
            if not field or field in seen:
                continue
            merged.append(field)
            seen.add(field)
    return merged


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_text(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    for src, dst in COLOR_ALIAS_MAP.items():
        normalized = re.sub(rf"\b{src}\b", dst, normalized)
    return normalized


def join_prompt_and_runtime_rows(
    prompt_rows: list[dict[str, str]],
    runtime_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    prompt_by_id = {row.get("sample_id", ""): row for row in prompt_rows}
    merged_rows: list[dict[str, str]] = []
    for runtime_row in runtime_rows:
        sample_id = runtime_row.get("sample_id", "")
        prompt_row = prompt_by_id.get(sample_id)
        if prompt_row is None:
            merged_rows.append(dict(runtime_row))
            continue

        merged = dict(prompt_row)
        merged.update(runtime_row)
        merged["model_output"] = runtime_row.get("raw_output", "")
        merged_rows.append(merged)
    return merged_rows


def detect_negated_colors(text: str) -> set[str]:
    hits: set[str] = set()
    for color, patterns in NEGATION_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            hits.add(color)
    return hits


def remove_negated_segments(text: str) -> str:
    scrubbed = text
    for patterns in NEGATION_PATTERNS.values():
        for pattern in patterns:
            scrubbed = pattern.sub(" ", scrubbed)
    return scrubbed


def detect_positive_colors(text: str) -> set[str]:
    scrubbed = remove_negated_segments(text)
    hits: set[str] = set()
    for color, pattern in POSITIVE_PATTERNS.items():
        if pattern.search(scrubbed):
            hits.add(color)
    return hits


def format_color_set(values: set[str]) -> str:
    return ",".join(sorted(values))


def classify_output(row: dict[str, str]) -> dict[str, str]:
    status = (row.get("status", "") or "").strip().lower()
    raw_output = row.get("raw_output", "") or ""
    normalized = normalize_text(raw_output)
    true_color = (row.get("true_color", "") or "").strip().lower()
    conflict_color = (row.get("conflict_color", "") or "").strip().lower()
    compatible_true_colors = TRUE_COLOR_COMPATIBILITY.get(true_color, {true_color}) if true_color else set()

    base = {
        "label": "needs_manual_review",
        "language_consistent": "",
        "vision_consistent": "",
        "ambiguous": "",
        "notes": "",
        "auto_label_confidence": "low",
        "auto_label_reason": "",
        "compatible_true_colors": format_color_set(compatible_true_colors),
        "detected_positive_colors": "",
        "detected_negated_colors": "",
    }

    if status not in OK_STATUSES:
        base["notes"] = f"Runtime status is not ok: {row.get('status', '')}; {row.get('error', '')}".strip("; ")
        base["auto_label_reason"] = "status_not_ok"
        return base

    if not normalized:
        base["notes"] = "raw_output is empty, so the response cannot be auto-labeled."
        base["auto_label_reason"] = "empty_output"
        return base

    negated_colors = detect_negated_colors(normalized)
    positive_colors = detect_positive_colors(normalized)
    base["detected_positive_colors"] = format_color_set(positive_colors)
    base["detected_negated_colors"] = format_color_set(negated_colors)

    positive_true = bool(positive_colors & compatible_true_colors)
    positive_conflict = conflict_color in positive_colors if conflict_color else False
    negated_conflict = conflict_color in negated_colors if conflict_color else False
    other_positive_colors = positive_colors - compatible_true_colors - ({conflict_color} if conflict_color else set())
    has_uncertainty = bool(UNCERTAINTY_PATTERN.search(normalized) or " or " in normalized)
    car_absence = bool(CAR_ABSENCE_PATTERN.search(normalized))

    if car_absence:
        return {
            **base,
            "label": "needs_manual_review",
            "auto_label_reason": "car_absence_error",
            "notes": "The model denied that a car exists in the image, so this case needs manual review.",
            "auto_label_confidence": "low",
        }

    if positive_conflict and not negated_conflict and not positive_true:
        return {
            **base,
            "label": "hallucination",
            "language_consistent": "1",
            "vision_consistent": "0",
            "ambiguous": "0",
            "notes": f"The output positively asserted conflict_color={conflict_color}.",
            "auto_label_confidence": "high" if not has_uncertainty else "medium",
            "auto_label_reason": "positive_conflict_color",
        }

    if positive_true and negated_conflict:
        return {
            **base,
            "label": "faithful",
            "language_consistent": "0",
            "vision_consistent": "1",
            "ambiguous": "0",
            "notes": (
                "The output gave the true color and explicitly rejected the conflict color; "
                f"true={true_color}; conflict={conflict_color}."
            ),
            "auto_label_confidence": "high",
            "auto_label_reason": "positive_true_and_negated_conflict",
        }

    if positive_true and not positive_conflict:
        return {
            **base,
            "label": "faithful",
            "language_consistent": "0",
            "vision_consistent": "1",
            "ambiguous": "0",
            "notes": f"The output positively matched the true-color set: {format_color_set(compatible_true_colors)}.",
            "auto_label_confidence": "high" if not has_uncertainty else "medium",
            "auto_label_reason": "positive_true_color",
        }

    if negated_conflict and not positive_conflict and not other_positive_colors:
        return {
            **base,
            "label": "faithful",
            "language_consistent": "0",
            "vision_consistent": "1",
            "ambiguous": "0",
            "notes": f"The output explicitly rejected conflict_color={conflict_color}.",
            "auto_label_confidence": "high" if not has_uncertainty else "medium",
            "auto_label_reason": "negated_conflict_color",
        }

    if positive_true and positive_conflict:
        return {
            **base,
            "label": "needs_manual_review",
            "auto_label_reason": "true_and_conflict_both_positive",
            "notes": (
                "The output positively mentioned both true_color and conflict_color; "
                f"positive={format_color_set(positive_colors)}."
            ),
            "auto_label_confidence": "low",
        }

    if other_positive_colors:
        return {
            **base,
            "label": "needs_manual_review",
            "auto_label_reason": "other_positive_color",
            "notes": (
                "The output mentioned another positive color, so the label is not stable; "
                f"other_positive={format_color_set(other_positive_colors)}."
            ),
            "auto_label_confidence": "low",
        }

    if has_uncertainty:
        return {
            **base,
            "label": "ambiguous",
            "language_consistent": "0",
            "vision_consistent": "0",
            "ambiguous": "1",
            "notes": "The output used uncertainty language and did not hit a stable color rule.",
            "auto_label_confidence": "medium",
            "auto_label_reason": "uncertain_without_stable_color",
        }

    return {
        **base,
        "label": "needs_manual_review",
        "auto_label_reason": "no_stable_rule_hit",
        "notes": "No stable auto-label rule was matched, so manual review is needed.",
        "auto_label_confidence": "low",
    }


def build_outputs(
    prompt_csv: Path,
    runtime_csv: Path,
    prelabeled_csv: Path,
    manual_review_csv: Path,
    final_labeled_csv: Path,
    summary_md: Path,
) -> dict[str, object]:
    prompt_fieldnames, prompt_rows = read_rows(prompt_csv)
    runtime_fieldnames, runtime_rows = read_rows(runtime_csv)
    merged_rows = join_prompt_and_runtime_rows(prompt_rows, runtime_rows)
    output_fieldnames = merge_fieldnames(prompt_fieldnames, runtime_fieldnames, LABEL_FIELDS)

    prelabeled_rows: list[dict[str, str]] = []
    manual_review_rows: list[dict[str, str]] = []
    label_counter: Counter[str] = Counter()
    review_counter: Counter[str] = Counter()
    by_level: dict[str, Counter[str]] = defaultdict(Counter)

    for row in merged_rows:
        labeled = dict(row)
        labeled.update(classify_output(row))
        prelabeled_rows.append(labeled)
        label_counter[labeled.get("label", "")] += 1
        by_level[str(labeled.get("prompt_code", ""))][labeled.get("label", "")] += 1

        if (
            labeled.get("label") in {"ambiguous", "needs_manual_review"}
            or (labeled.get("auto_label_confidence") or "").lower() != "high"
        ):
            manual_review_rows.append(labeled)
            review_counter[labeled.get("auto_label_reason", "")] += 1

    write_rows(prelabeled_csv, output_fieldnames, prelabeled_rows)
    write_rows(manual_review_csv, output_fieldnames, manual_review_rows)
    write_rows(final_labeled_csv, output_fieldnames, prelabeled_rows)

    summary_lines = [
        "# Car Color Attribute Conflict Auto-Label Summary",
        "",
        "## Files",
        f"- input prompt csv: {relative_str(prompt_csv)}",
        f"- input runtime csv: {relative_str(runtime_csv)}",
        f"- prelabeled output: {relative_str(prelabeled_csv)}",
        f"- manual review list: {relative_str(manual_review_csv)}",
        f"- final labeled template: {relative_str(final_labeled_csv)}",
        "",
        "## Overall Counts",
        f"- input rows: {len(merged_rows)}",
    ]
    for label, count in sorted(label_counter.items()):
        summary_lines.append(f"- {label}: {count}")
    summary_lines.append(f"- manual review rows: {len(manual_review_rows)}")
    summary_lines.append("")
    summary_lines.append("## By Prompt Level")
    for prompt_code in sorted(by_level):
        parts = ", ".join(f"{label}={count}" for label, count in sorted(by_level[prompt_code].items()))
        summary_lines.append(f"- {prompt_code}: {parts}")
    summary_lines.append("")
    summary_lines.append("## Main Manual-Review Reasons")
    for reason, count in review_counter.most_common():
        summary_lines.append(f"- {reason or 'unknown'}: {count}")
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "input_rows": len(merged_rows),
        "prelabeled_rows": len(prelabeled_rows),
        "manual_review_rows": len(manual_review_rows),
        "label_counts": dict(label_counter),
        "review_reason_counts": dict(review_counter),
        "prelabeled_csv": relative_str(prelabeled_csv),
        "manual_review_csv": relative_str(manual_review_csv),
        "final_labeled_csv": relative_str(final_labeled_csv),
        "summary_md": relative_str(summary_md),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-prelabel Qwen2-VL car-color attribute-conflict outputs.")
    parser.add_argument("--prompt-csv", type=Path, default=PROMPTS_CSV)
    parser.add_argument("--runtime-csv", type=Path, default=RUNTIME_CSV)
    parser.add_argument("--prelabeled-csv", type=Path, default=PRELABELED_CSV)
    parser.add_argument("--manual-review-csv", type=Path, default=MANUAL_REVIEW_CSV)
    parser.add_argument("--final-labeled-csv", type=Path, default=FINAL_LABELED_CSV)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    return parser.parse_args()


def main() -> int:
    ensure_metadata_dirs()
    args = parse_args()
    summary = build_outputs(
        prompt_csv=args.prompt_csv,
        runtime_csv=args.runtime_csv,
        prelabeled_csv=args.prelabeled_csv,
        manual_review_csv=args.manual_review_csv,
        final_labeled_csv=args.final_labeled_csv,
        summary_md=args.summary_md,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
