#!/usr/bin/env python
"""Rule-based prelabeling for existence-conflict outputs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from metadata_paths import (
    BASELINE_FINAL_LABELED_CSV,
    BASELINE_MANUAL_REVIEW_CSV,
    BASELINE_PRELABELED_CSV,
    BASELINE_RAW_CSV,
    LEGACY_BASELINE_RAW_CSV,
    resolve_existing_path,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = resolve_existing_path(BASELINE_RAW_CSV, LEGACY_BASELINE_RAW_CSV)
DEFAULT_PRELABELED_CSV = BASELINE_PRELABELED_CSV
DEFAULT_PRIORITY_CSV = BASELINE_MANUAL_REVIEW_CSV
DEFAULT_TEMPLATE_CSV = BASELINE_FINAL_LABELED_CSV
LABEL_FIELDS = [
    "label",
    "language_consistent",
    "vision_consistent",
    "ambiguous",
    "notes",
    "auto_label_confidence",
]
OK_STATUSES = {"", "ok", "success"}
TRAILING_SENTENCE_ENDINGS = "。！？.!?;；\"'”’】)]」』"


NEGATIVE_RULES = [
    ("en_there_is_no_dog", r"\bthere\s+(?:is|are)\s+no\s+dog(?:s)?\b"),
    ("en_no_dog_visible", r"\bno\s+(?:visible\s+)?dog(?:s)?\b"),
    ("en_no_existence_of_dog", r"\bthere\s+is\s+no\s+(?:existence|presence|sign)\s+of\s+(?:any\s+|a\s+)?dog(?:s)?\b"),
    ("en_do_not_see_dog", r"\b(?:i\s+)?(?:do not|don't|did not|didn't|cannot|can't)\s+(?:see|find|observe|spot|detect|identify)\s+(?:any\s+|a\s+)?dog(?:s)?\b"),
    ("zh_no_dog", r"(?:图中|图片中|画面中|照片中)?[^。；，,]{0,8}没有(?:出现)?(?:任何)?狗"),
    ("zh_not_see_dog", r"(?:未|没(?:有)?)\s*(?:看到|发现|见到|观察到)\s*狗"),
    ("zh_no_visible_dog", r"看不到狗|未见狗|不存在狗"),
]

POSITIVE_RULES = [
    ("en_yes_dog", r"\byes[,! ]+(?:there\s+is\s+)?(?:a\s+|one\s+|the\s+)?dog\b"),
    ("en_there_is_dog", r"\bthere\s+(?:is|appears to be|seems to be)\s+(?:a\s+|one\s+)?dog\b"),
    ("en_i_see_dog", r"\b(?:i\s+can\s+see|i\s+see)\s+(?:a\s+|one\s+|the\s+)?dog\b"),
    ("en_dog_description", r"\b(?:a|the|one)\s+dog\s+(?:is|was|appears|seems|standing|sitting|running|lying|visible|present|near|next to|on|under|beside)\b"),
    ("zh_image_has_dog", r"(?:图中|图片中|画面中|照片中)有(?:一只|一条)?狗"),
    ("zh_one_dog", r"(?:一只|一条)[^。；，,]{0,8}狗"),
    ("zh_can_see_dog", r"(?:可以|能够|能)?看到(?:一只|一条)?狗"),
    ("zh_showing_dog", r"(?:这张图片|图片|图像|画面|照片)(?:展示|显示|呈现|描绘)[^。；，,]{0,12}狗"),
    ("zh_dog_position", r"狗(?:在|位于|正在|站在|坐在|跑|趴|躺|靠近|旁边|附近)"),
    ("zh_dog_closeup", r"狗的特写"),
    ("zh_dog_attributes", r"狗的(?:毛色|眼睛|鼻子|耳朵|嘴巴|身体|姿态|位置|状态|动作)"),
    ("zh_this_dog", r"(?:这只|那只)狗"),
]

AMBIGUOUS_RULES = [
    ("en_hard_to_tell", r"\b(?:hard|difficult)\s+to\s+tell\b"),
    ("en_unclear", r"\bunclear\b"),
    ("en_maybe", r"\bmaybe\b|\bpossibly\b|\bperhaps\b"),
    ("en_not_sure", r"\bnot\s+sure\b|\bunsure\b"),
    ("en_cannot_determine", r"\bcannot\s+(?:determine|confirm)\b|\bcan't\s+(?:determine|confirm)\b"),
    ("zh_hard_to_tell", r"很难判断|难以判断"),
    ("zh_uncertain", r"无法判断|无法确认|不能确定|不确定"),
    ("zh_unclear", r"看不清|图像不清晰|图片不清晰|画面不清晰"),
    ("zh_maybe_dog", r"可能[^。；，,]{0,8}狗|也许[^。；，,]{0,8}狗|似乎[^。；，,]{0,8}狗"),
]

COMPILED_NEGATIVE_RULES = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in NEGATIVE_RULES]
COMPILED_POSITIVE_RULES = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in POSITIVE_RULES]
COMPILED_AMBIGUOUS_RULES = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in AMBIGUOUS_RULES]


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def find_hits(text: str, rules: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    hits: list[str] = []
    for name, pattern in rules:
        if pattern.search(text):
            hits.append(name)
    return hits


def looks_truncated(raw_output: str) -> bool:
    stripped = (raw_output or "").strip()
    if not stripped:
        return False
    return stripped[-1] not in TRAILING_SENTENCE_ENDINGS


def classify_output(row: dict[str, str]) -> dict[str, str]:
    status = (row.get("status", "") or "").strip().lower()
    raw_output = row.get("raw_output", "") or ""
    normalized = normalize_text(raw_output)
    base = {
        "label": "needs_manual_review",
        "language_consistent": "",
        "vision_consistent": "",
        "ambiguous": "",
        "notes": "",
        "auto_label_confidence": "low",
    }

    if status not in OK_STATUSES:
        base["notes"] = f"推理状态非 ok: {row.get('status', '')}; {row.get('error', '')}".strip("; ")
        return base

    if not normalized:
        base["notes"] = "raw_output 为空，无法命中稳定规则"
        return base

    negative_hits = find_hits(normalized, COMPILED_NEGATIVE_RULES)
    positive_hits = find_hits(normalized, COMPILED_POSITIVE_RULES)
    ambiguous_hits = find_hits(normalized, COMPILED_AMBIGUOUS_RULES)

    if positive_hits and negative_hits:
        base["notes"] = f"同时命中承认与否认规则；positive={','.join(positive_hits)}; negative={','.join(negative_hits)}"
        return base

    if positive_hits:
        return {
            "label": "hallucination",
            "language_consistent": "1",
            "vision_consistent": "0",
            "ambiguous": "0",
            "notes": f"命中承认 dog 存在规则: {','.join(positive_hits)}",
            "auto_label_confidence": "high",
        }

    if negative_hits:
        return {
            "label": "faithful",
            "language_consistent": "0",
            "vision_consistent": "1",
            "ambiguous": "0",
            "notes": f"命中否认 dog 存在规则: {','.join(negative_hits)}",
            "auto_label_confidence": "high",
        }

    if ambiguous_hits:
        return {
            "label": "ambiguous",
            "language_consistent": "0",
            "vision_consistent": "0",
            "ambiguous": "1",
            "notes": f"命中模糊/不确定规则: {','.join(ambiguous_hits)}",
            "auto_label_confidence": "medium",
        }

    if looks_truncated(raw_output):
        base["notes"] = "输出疑似截断，且未命中稳定规则，需人工复核"
    else:
        base["notes"] = "未命中稳定规则，需人工复核"
    return base


def build_outputs(input_csv: Path, prelabeled_csv: Path, priority_csv: Path, template_csv: Path) -> dict[str, object]:
    input_fieldnames, rows = read_rows(input_csv)
    output_fieldnames = merge_fieldnames(input_fieldnames, LABEL_FIELDS)

    prelabeled_rows: list[dict[str, str]] = []
    priority_rows: list[dict[str, str]] = []
    label_counter: Counter[str] = Counter()

    for row in rows:
        labeled = dict(row)
        labeled.update(classify_output(row))
        prelabeled_rows.append(labeled)
        label_counter[labeled.get("label", "")] += 1

        if (
            labeled.get("label") in {"ambiguous", "needs_manual_review"}
            or (labeled.get("auto_label_confidence") or "").lower() == "low"
        ):
            priority_rows.append(labeled)

    write_rows(prelabeled_csv, output_fieldnames, prelabeled_rows)
    write_rows(priority_csv, output_fieldnames, priority_rows)
    write_rows(template_csv, output_fieldnames, prelabeled_rows)

    return {
        "input_rows": len(rows),
        "prelabeled_rows": len(prelabeled_rows),
        "priority_rows": len(priority_rows),
        "label_counts": dict(label_counter),
        "prelabeled_csv": str(prelabeled_csv),
        "priority_csv": str(priority_csv),
        "template_csv": str(template_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-prelabel Qwen2-VL existence-conflict outputs and export review files.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--prelabeled-csv", type=Path, default=DEFAULT_PRELABELED_CSV)
    parser.add_argument("--priority-csv", type=Path, default=DEFAULT_PRIORITY_CSV)
    parser.add_argument("--template-csv", type=Path, default=DEFAULT_TEMPLATE_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(args.input_csv, args.prelabeled_csv, args.priority_csv, args.template_csv)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
