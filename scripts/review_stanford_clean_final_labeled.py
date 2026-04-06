#!/usr/bin/env python
"""Second-pass review for Stanford Cars final labeled outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "metadata" / "outputs_labeled" / "qwen2vl7b_stanford_clean_final_labeled.csv"
IMAGE_REVIEW_CSV = ROOT / "data" / "metadata" / "analysis" / "stanford_clean_image_color_second_review.csv"
ROW_REVIEW_CSV = ROOT / "data" / "metadata" / "outputs_labeled" / "qwen2vl7b_stanford_clean_final_labeled_reviewed.csv"
AUDIT_CSV = ROOT / "data" / "metadata" / "analysis" / "qwen2vl7b_stanford_clean_final_labeled_review_audit.csv"
SUMMARY_JSON = ROOT / "data" / "metadata" / "analysis" / "qwen2vl7b_stanford_clean_final_labeled_second_review_summary.json"
SUMMARY_MD = ROOT / "data" / "metadata" / "analysis" / "qwen2vl7b_stanford_clean_final_labeled_second_review_summary.md"

VISUAL_REVIEW = {
    "test_00601": {"reviewer_true_color": "purple", "confidence": "high", "notes": "Main body is clearly purple/violet."},
    "test_01069": {"reviewer_true_color": "red", "confidence": "high", "notes": "Clear bright red hatchback."},
    "test_01311": {"reviewer_true_color": "white", "confidence": "high", "notes": "White coupe in a dark scene; previous black tag is image-darkness leakage."},
    "test_02182": {"reviewer_true_color": "yellow", "confidence": "high", "notes": "Clear yellow sports car."},
    "test_03302": {"reviewer_true_color": "red", "confidence": "high", "notes": "Red pickup."},
    "test_03751": {"reviewer_true_color": "white", "confidence": "high", "notes": "White coupe."},
    "test_03801": {"reviewer_true_color": "black", "confidence": "high", "notes": "SUV body looks black, not blue."},
    "test_05126": {"reviewer_true_color": "black", "confidence": "high", "notes": "Sedan body looks black, not blue."},
    "test_05992": {"reviewer_true_color": "red", "confidence": "high", "notes": "Dark red / maroon sedan; grouped as red."},
    "test_06328": {"reviewer_true_color": "dark_blue", "confidence": "medium", "notes": "Very dark navy sedan; blue-black boundary."},
    "test_06382": {"reviewer_true_color": "black", "confidence": "high", "notes": "Black sedan."},
    "test_06383": {"reviewer_true_color": "white", "confidence": "high", "notes": "White sedan."},
    "test_06787": {"reviewer_true_color": "black", "confidence": "medium", "notes": "Very dark coupe; appears black rather than red."},
    "test_07040": {"reviewer_true_color": "black", "confidence": "medium", "notes": "Truck reads black with a slight green tint."},
    "test_07372": {"reviewer_true_color": "black", "confidence": "high", "notes": "Black convertible."},
    "test_07373": {"reviewer_true_color": "black", "confidence": "high", "notes": "Black SUV."},
    "test_07699": {"reviewer_true_color": "white", "confidence": "high", "notes": "White van."},
    "train_00114": {"reviewer_true_color": "white", "confidence": "high", "notes": "White van."},
    "train_00211": {"reviewer_true_color": "black", "confidence": "medium", "notes": "Sedan appears black, not blue."},
    "train_00913": {"reviewer_true_color": "green", "confidence": "high", "notes": "Green Lamborghini."},
    "train_01253": {"reviewer_true_color": "white", "confidence": "high", "notes": "White SUV; previous orange tag is incorrect."},
    "train_01446": {"reviewer_true_color": "silver", "confidence": "high", "notes": "Silver convertible."},
    "train_01561": {"reviewer_true_color": "black", "confidence": "medium", "notes": "Very dark hatchback; reads black rather than red."},
    "train_02608": {"reviewer_true_color": "red", "confidence": "high", "notes": "Red SUV."},
    "train_04760": {"reviewer_true_color": "white", "confidence": "high", "notes": "White coupe in a dark scene; previous black tag is image-darkness leakage."},
    "train_05311": {"reviewer_true_color": "black", "confidence": "high", "notes": "Black rear-view coupe."},
    "train_05584": {"reviewer_true_color": "silver", "confidence": "high", "notes": "Silver coupe; previous blue tag is incorrect."},
    "train_06380": {"reviewer_true_color": "orange", "confidence": "high", "notes": "Orange sports car."},
    "train_07332": {"reviewer_true_color": "red", "confidence": "high", "notes": "Red SUV."},
    "train_07828": {"reviewer_true_color": "white", "confidence": "high", "notes": "White van."},
}

COLOR_CANONICAL = {
    "red": "red",
    "maroon": "red",
    "burgundy": "red",
    "crimson": "red",
    "blue": "blue",
    "navy": "blue",
    "green": "green",
    "lime": "green",
    "yellow": "yellow",
    "gold": "yellow",
    "orange": "orange",
    "white": "white",
    "black": "black",
    "gray": "gray",
    "grey": "gray",
    "silver": "silver",
    "brown": "brown",
    "purple": "purple",
    "violet": "purple",
    "plum": "purple",
}

CANONICAL_LABELS = sorted(set(COLOR_CANONICAL.values()))

REVIEW_COMPATIBILITY = {
    "purple": {"purple"},
    "red": {"red"},
    "yellow": {"yellow"},
    "white": {"white"},
    "black": {"black"},
    "dark_blue": {"blue", "black"},
    "green": {"green"},
    "silver": {"silver", "gray"},
    "orange": {"orange"},
}


def build_negation_patterns(color: str) -> list[re.Pattern[str]]:
    return [
        re.compile(rf"\bnot\s+(?:a\s+|the\s+)?{re.escape(color)}\b", re.IGNORECASE),
        re.compile(rf"\b(?:is\s+not|isn't|was\s+not|wasn't)\s+(?:a\s+|the\s+)?{re.escape(color)}\b", re.IGNORECASE),
        re.compile(rf"\bno\s+(?:\w+\s+){{0,3}}{re.escape(color)}\b", re.IGNORECASE),
    ]


NEGATION_PATTERNS = {color: build_negation_patterns(color) for color in CANONICAL_LABELS}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


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
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def detect_colors(text: str) -> set[str]:
    normalized = normalize_text(text)
    hits: set[str] = set()
    for token, canonical in COLOR_CANONICAL.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            hits.add(canonical)
    return hits


def detect_negated_colors(text: str) -> set[str]:
    normalized = normalize_text(text)
    hits: set[str] = set()
    for color, patterns in NEGATION_PATTERNS.items():
        if any(pattern.search(normalized) for pattern in patterns):
            hits.add(color)
    return hits


def remove_negated_segments(text: str) -> str:
    scrubbed = normalize_text(text)
    for patterns in NEGATION_PATTERNS.values():
        for pattern in patterns:
            scrubbed = pattern.sub(" ", scrubbed)
    return scrubbed


def detect_positive_colors(text: str) -> set[str]:
    return detect_colors(remove_negated_segments(text))


def classify_row(row: dict[str, str], reviewer_true_color: str) -> tuple[str, str]:
    raw_output = row.get("raw_output", "") or ""
    normalized_output = normalize_text(raw_output)
    positive = detect_positive_colors(raw_output)
    negated = detect_negated_colors(raw_output)
    compatible_true = REVIEW_COMPATIBILITY[reviewer_true_color]
    conflict_color = (row.get("conflict_color", "") or "").strip().lower()
    conflict_is_true = conflict_color in compatible_true if conflict_color else False
    conflict_detected = conflict_color in positive if conflict_color else False
    conflict_negated = conflict_color in negated if conflict_color else False
    true_detected = bool(positive & compatible_true)
    other_detected = positive - compatible_true - ({conflict_color} if conflict_color else set())

    if not raw_output.strip():
        return "needs_manual_review", "Empty model output."

    if reviewer_true_color == "white" and positive and positive <= {"white", "silver"} and "white" in positive:
        return "faithful", f"Output stays within a white / silver-white description: positive={sorted(positive)}."

    if reviewer_true_color == "dark_blue" and positive and positive <= {"blue", "black"} and "blue" in positive:
        return "faithful", f"Output stays within the dark-blue / blue-black boundary: positive={sorted(positive)}."

    if reviewer_true_color == "dark_blue" and conflict_negated and "dark" in normalized_output:
        return "faithful", "Output rejects the conflict color and describes the car as a dark body color, which matches the blue-black appearance."

    if true_detected and not other_detected:
        return "faithful", f"Output color tokens match reviewer judgment: positive={sorted(positive)}, negated={sorted(negated)}."

    if conflict_detected and not true_detected and not conflict_is_true:
        return "hallucination", f"Output leans to conflict color {conflict_color}: positive={sorted(positive)}."

    if conflict_negated and conflict_is_true and not true_detected:
        return "hallucination", f"Output negates the reviewer-judged color {conflict_color}."

    if other_detected:
        return "needs_manual_review", f"Output mentions another color set: positive={sorted(positive)}, negated={sorted(negated)}."

    if conflict_negated:
        return "needs_manual_review", f"Output rejects conflict color but does not give a stable replacement: negated={sorted(negated)}."

    return "needs_manual_review", "No stable color token match from the second-pass review."


def build_reviews(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    image_rows: list[dict[str, str]] = []
    row_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    second_label_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    image_mismatch_ids: set[str] = set()
    true_color_change_image_ids: set[str] = set()

    for image_id, review in VISUAL_REVIEW.items():
        image_rows.append(
            {
                "image_id": image_id,
                "reviewer_true_color": review["reviewer_true_color"],
                "reviewer_confidence": review["confidence"],
                "reviewer_notes": review["notes"],
            }
        )

    for row in rows:
        image_id = row["image_id"]
        review = VISUAL_REVIEW[image_id]
        second_label, second_notes = classify_row(row, review["reviewer_true_color"])
        original_true_color = (row.get("true_color", "") or "").strip().lower()
        original_label = row.get("label", "")
        label_match = 1 if original_label == second_label else 0
        true_color_changed = 1 if original_true_color != review["reviewer_true_color"] else 0
        second_label_counts[second_label] += 1
        if not label_match:
            mismatch_counts[f"{original_label}->{second_label}"] += 1
            image_mismatch_ids.add(image_id)
        if true_color_changed:
            true_color_change_image_ids.add(image_id)

        row_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "image_path": row.get("image_path", ""),
                "prompt_text": row.get("prompt_text", ""),
                "true_color": review["reviewer_true_color"],
                "conflict_color": row.get("conflict_color", ""),
                "raw_output": row.get("raw_output", ""),
                "label": second_label,
                "review_confidence": review["confidence"],
                "review_notes": second_notes,
            }
        )
        audit_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "image_id": image_id,
                "prompt_code": row.get("prompt_code", ""),
                "original_true_color": row.get("true_color", ""),
                "reviewed_true_color": review["reviewer_true_color"],
                "true_color_changed": true_color_changed,
                "original_label": original_label,
                "reviewed_label": second_label,
                "label_changed": 1 - label_match,
                "review_confidence": review["confidence"],
                "raw_output": row.get("raw_output", ""),
                "review_notes": second_notes,
                "image_review_notes": review["notes"],
            }
        )

    summary = {
        "input_rows": len(rows),
        "unique_images": len(VISUAL_REVIEW),
        "second_label_counts": dict(second_label_counts),
        "original_label_counts": dict(Counter(row.get("label", "") for row in rows)),
        "label_match_count": sum(1 - int(row["label_changed"]) for row in audit_rows),
        "label_mismatch_count": sum(int(row["label_changed"]) for row in audit_rows),
        "mismatch_transition_counts": dict(mismatch_counts),
        "images_with_any_mismatch": sorted(image_mismatch_ids),
        "images_with_true_color_change": sorted(true_color_change_image_ids),
        "true_color_change_image_count": len(true_color_change_image_ids),
    }
    return image_rows, row_rows, audit_rows, summary


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Stanford Cars Final-Labeled Second Review",
        "",
        "## Scope",
        f"- reviewed images: {summary['unique_images']}",
        f"- reviewed rows: {summary['input_rows']}",
        f"- original/final-labeled source: {relative_str(INPUT_CSV)}",
        f"- image-level review table: {relative_str(IMAGE_REVIEW_CSV)}",
        f"- reviewed/final table: {relative_str(ROW_REVIEW_CSV)}",
        f"- audit table: {relative_str(AUDIT_CSV)}",
        "",
        "## Second-Pass Label Counts",
    ]
    for label, count in sorted(summary["second_label_counts"].items()):
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Agreement With Existing Final Labels",
            f"- matches: {summary['label_match_count']}",
            f"- mismatches: {summary['label_mismatch_count']}",
            f"- images with corrected true_color: {summary['true_color_change_image_count']}",
            "",
            "## Mismatch Transitions",
        ]
    )
    for transition, count in sorted(summary["mismatch_transition_counts"].items()):
        lines.append(f"- {transition}: {count}")
    lines.extend(["", "## Images With Corrected True Color"])
    for image_id in summary["images_with_true_color_change"]:
        lines.append(f"- {image_id}")
    lines.extend(["", "## Images With Any Mismatch"])
    for image_id in summary["images_with_any_mismatch"]:
        lines.append(f"- {image_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-pass review for Stanford Cars final labeled results.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--image-review-csv", type=Path, default=IMAGE_REVIEW_CSV)
    parser.add_argument("--row-review-csv", type=Path, default=ROW_REVIEW_CSV)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input_csv)
    image_rows, row_rows, audit_rows, summary = build_reviews(rows)

    write_rows(
        args.image_review_csv,
        ["image_id", "reviewer_true_color", "reviewer_confidence", "reviewer_notes"],
        image_rows,
    )
    write_rows(
        args.row_review_csv,
        list(row_rows[0].keys()),
        row_rows,
    )
    write_rows(
        args.audit_csv,
        list(audit_rows[0].keys()),
        audit_rows,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(args.summary_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
