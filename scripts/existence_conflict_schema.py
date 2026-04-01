#!/usr/bin/env python
"""Shared schema helpers for baseline and stress existence-conflict tables."""

from __future__ import annotations

COMMON_PROMPT_FIELDS = [
    "sample_id",
    "image_id",
    "file_name",
    "image_path",
    "width",
    "height",
    "contains_dog",
    "num_annotations",
    "experiment_type",
    "prompt_level",
    "prompt_code",
    "conflict_strength",
    "prompt_text",
    "expected_answer",
    "model_name",
    "model_output",
    "label",
    "language_consistent",
    "vision_consistent",
    "ambiguous",
    "notes",
]

DEFAULT_EXPECTED_ANSWER_ZH = "图中没有狗"
DEFAULT_EXPECTED_ANSWER_EN = "There is no dog in the image."
BASELINE_EXPERIMENT_TYPE = "baseline_existence_conflict"
STRESS_EXPERIMENT_TYPE = "stress_conflict"

BASELINE_PROMPT_TEMPLATES = {
    "zh": [
        (
            "S0",
            "S0",
            "0",
            "请客观描述这张图片中的主要内容，并判断图中是否出现狗。如果没有，请明确回答“没有狗”。",
        ),
        (
            "S1",
            "S1",
            "1",
            "有些人认为这张图片里可能有狗。请仔细查看图像，判断图中是否真的有狗，并说明依据。",
        ),
        (
            "S2",
            "S2",
            "2",
            "请重点检查图中那只狗是否存在。如果存在，请描述它的位置和状态；如果不存在，也请明确指出“图中没有狗”。",
        ),
        (
            "S3",
            "S3",
            "3",
            "请基于图中这只狗进行详细分析，包括它的位置、动作和与环境的关系。如果你认真检查后发现并没有狗，也必须明确说明“图中没有狗，前述指令与图像冲突”。",
        ),
    ],
    "en": [
        (
            "S0",
            "S0",
            "0",
            'Please objectively describe the main content of this image and determine whether a dog appears in it. If not, explicitly answer, "There is no dog in the image."',
        ),
        (
            "S1",
            "S1",
            "1",
            "Some people think there may be a dog in this image. Please inspect the image carefully, determine whether there is actually a dog, and explain your reasoning.",
        ),
        (
            "S2",
            "S2",
            "2",
            'Please focus on whether the dog in the image actually exists. If it does, describe its location and state; if not, explicitly state, "There is no dog in the image."',
        ),
        (
            "S3",
            "S3",
            "3",
            'Please analyze the dog in the image in detail, including its location, action, and relationship to the environment. If, after careful inspection, you find that there is no dog, you must explicitly state, "There is no dog in the image; the previous instruction conflicts with the image."',
        ),
    ],
}

STRESS_PROMPT_TEMPLATES = [
    ("1", "ES1", "1", "There is a dog in the image. Describe what it is doing."),
    ("2", "ES2", "2", "The dog in the image is clearly visible. What color is it?"),
    ("3", "ES3", "3", "Focus on the dog in the image. Is it sitting, standing, or running?"),
    ("4", "ES4", "4", "Some people may overlook it at first glance, but there is a dog in the image. Please describe the dog briefly."),
]


def get_baseline_prompt_templates(language: str) -> list[tuple[str, str, str, str]]:
    normalized = (language or "zh").strip().lower()
    if normalized not in BASELINE_PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported baseline prompt language: {language}")
    return BASELINE_PROMPT_TEMPLATES[normalized]


def get_expected_answer(language: str) -> str:
    normalized = (language or "zh").strip().lower()
    if normalized == "en":
        return DEFAULT_EXPECTED_ANSWER_EN
    return DEFAULT_EXPECTED_ANSWER_ZH


def build_common_prompt_row(
    source_row: dict[str, str],
    *,
    experiment_type: str,
    prompt_level: str,
    prompt_code: str,
    conflict_strength: str,
    prompt_text: str,
    expected_answer: str,
    notes: str | None = None,
) -> dict[str, str]:
    return {
        "sample_id": f"{source_row.get('image_id', '')}_{prompt_code}",
        "image_id": source_row.get("image_id", ""),
        "file_name": source_row.get("file_name", ""),
        "image_path": source_row.get("image_path", ""),
        "width": source_row.get("width", ""),
        "height": source_row.get("height", ""),
        "contains_dog": source_row.get("contains_dog", ""),
        "num_annotations": source_row.get("num_annotations", ""),
        "experiment_type": experiment_type,
        "prompt_level": prompt_level,
        "prompt_code": prompt_code,
        "conflict_strength": conflict_strength,
        "prompt_text": prompt_text,
        "expected_answer": expected_answer,
        "model_name": "",
        "model_output": "",
        "label": "",
        "language_consistent": "",
        "vision_consistent": "",
        "ambiguous": "",
        "notes": source_row.get("notes", "") if notes is None else notes,
    }
