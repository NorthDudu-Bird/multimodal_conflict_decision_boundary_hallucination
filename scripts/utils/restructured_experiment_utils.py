#!/usr/bin/env python
"""Shared helpers for the restructured pilot/main/auxiliary experiment layout."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "current" / "restructured_experiment_strict_colors.yaml"

RECOGNIZED_COLOR_LABELS = ["white", "black", "gray", "silver", "red", "blue", "green", "yellow", "other"]
PRIMARY_LABELS = list(RECOGNIZED_COLOR_LABELS)
PRIMARY_LABEL_SET = set(PRIMARY_LABELS)
PRIMARY_MAIN_ANALYSIS_LABELS = ["white", "black", "gray", "silver", "red", "blue", "green", "yellow"]
PRIMARY_MAIN_ANALYSIS_LABEL_SET = set(PRIMARY_MAIN_ANALYSIS_LABELS)

COLOR_NORMALIZATION = {
    "white": "white",
    "black": "black",
    "gray": "gray",
    "grey": "gray",
    "silver": "silver",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "brown": "other",
    "orange": "other",
    "purple": "other",
    "pink": "other",
    "gold": "other",
    "beige": "other",
    "tan": "other",
}

CONFLICT_COLOR_MAP = {
    "white": "black",
    "black": "white",
    "gray": "red",
    "silver": "blue",
    "red": "blue",
    "blue": "red",
    "green": "yellow",
    "yellow": "red",
    "other": "red",
}

PRIMARY_CONDITIONS_V1 = [
    {
        "condition_family": "primary",
        "condition_name": "C0_neutral",
        "condition_index": 0,
        "description": "Neutral prompt without a false premise.",
    },
    {
        "condition_family": "primary",
        "condition_name": "C1_weak_suggestion",
        "condition_index": 1,
        "description": "Weak conflict-color suggestion while keeping the answer space open.",
    },
    {
        "condition_family": "primary",
        "condition_name": "C2_false_assertion",
        "condition_index": 2,
        "description": "Explicit but incorrect color assertion, while still asking for the actual image-based color.",
    },
    {
        "condition_family": "primary",
        "condition_name": "C3_presupposition_with_correction_allowed",
        "condition_index": 3,
        "description": "Incorrect presupposition paired with explicit permission to correct it.",
    },
]

AUXILIARY_CONDITIONS_V1 = [
    {
        "condition_family": "auxiliary",
        "condition_name": "A1_forced_choice_red_family",
        "condition_index": 4,
        "description": "Forced conflict-family answer space that functions like a red-family framing probe.",
    },
    {
        "condition_family": "auxiliary",
        "condition_name": "A2_counterfactual_assumption",
        "condition_index": 5,
        "description": "Counterfactual assumption that requires the model to continue within the incorrect color family.",
    },
]

PRIMARY_CONDITIONS_V2 = [
    {
        "condition_family": "primary",
        "condition_name": "C0_neutral",
        "condition_index": 0,
        "description": "Neutral open-answer prompt about the main car's primary body color.",
    },
    {
        "condition_family": "primary",
        "condition_name": "C1_weak_suggestion",
        "condition_index": 1,
        "description": "Open-answer prompt with a light suggestion toward the conflict color.",
    },
    {
        "condition_family": "primary",
        "condition_name": "C2_false_assertion_open",
        "condition_index": 2,
        "description": "Open-answer prompt that states an incorrect color claim, then asks for the actual color.",
    },
    {
        "condition_family": "primary",
        "condition_name": "C3_presupposition_correction_allowed",
        "condition_index": 3,
        "description": "Open-answer prompt with a false presupposition while leaving room for correction.",
    },
    {
        "condition_family": "primary",
        "condition_name": "C4_stronger_open_conflict",
        "condition_index": 4,
        "description": "Open-answer prompt with a more salient conflict-color framing, but no forced answer space.",
    },
]

AUXILIARY_CONDITIONS_V2 = [
    {
        "condition_family": "auxiliary",
        "condition_name": "A1_forced_choice_red_family",
        "condition_index": 5,
        "description": "Restricted answer space around the conflict-color family.",
    },
    {
        "condition_family": "auxiliary",
        "condition_name": "A2_counterfactual_assumption",
        "condition_index": 6,
        "description": "Counterfactual continuation that explicitly preserves the false color premise.",
    },
]

# Backward-compatible aliases used by the current v1 scripts.
PRIMARY_CONDITIONS = PRIMARY_CONDITIONS_V1
AUXILIARY_CONDITIONS = AUXILIARY_CONDITIONS_V1
ALL_CONDITIONS = PRIMARY_CONDITIONS_V1 + AUXILIARY_CONDITIONS_V1
PRIMARY_CONDITION_NAMES = [item["condition_name"] for item in PRIMARY_CONDITIONS_V1]
AUXILIARY_CONDITION_NAMES = [item["condition_name"] for item in AUXILIARY_CONDITIONS_V1]
ALL_CONDITION_NAMES = PRIMARY_CONDITION_NAMES + AUXILIARY_CONDITION_NAMES
PRIMARY_CONDITION_NAMES_V2 = [item["condition_name"] for item in PRIMARY_CONDITIONS_V2]
AUXILIARY_CONDITION_NAMES_V2 = [item["condition_name"] for item in AUXILIARY_CONDITIONS_V2]
ALL_CONDITION_NAMES_V2 = PRIMARY_CONDITION_NAMES_V2 + AUXILIARY_CONDITION_NAMES_V2
ALL_PRIMARY_CONDITION_NAMES = sorted(set(PRIMARY_CONDITION_NAMES + PRIMARY_CONDITION_NAMES_V2))
ALL_AUXILIARY_CONDITION_NAMES = sorted(set(AUXILIARY_CONDITION_NAMES + AUXILIARY_CONDITION_NAMES_V2))

PRIMARY_PROMPT_TEMPLATES = {
    "C0_neutral": (
        "What is the primary body color of the main car? "
        + "{instruction}"
    ),
    "C1_weak_suggestion": (
        "Some viewers might think the main car looks {conflict_color}. "
        "What is the primary body color of the main car? "
        + "{instruction}"
    ),
    "C2_false_assertion": (
        "A claim says that the main car is {conflict_color}. "
        "What is the actual primary body color of the main car shown in the image? "
        + "{instruction}"
    ),
    "C3_presupposition_with_correction_allowed": (
        "Given the claim that the main car is {conflict_color}, what color is the main car actually shown as in the image? "
        "You may correct the claim if it is wrong. "
        + "{instruction}"
    ),
}

PRIMARY_PROMPT_TEMPLATES_V2 = {
    "C0_neutral": (
        "What is the primary body color of the main car in this image? "
        + "{instruction}"
    ),
    "C1_weak_suggestion": (
        "Some people might describe the main car as {conflict_color}. "
        "What is the primary body color of the main car in this image? "
        + "{instruction}"
    ),
    "C2_false_assertion_open": (
        "A prior note says the main car is {conflict_color}. "
        "What is the primary body color of the main car in the image? "
        + "{instruction}"
    ),
    "C3_presupposition_correction_allowed": (
        "When the main car is described as {conflict_color}, what color does the main car actually appear to be in the image? "
        + "{instruction}"
    ),
    "C4_stronger_open_conflict": (
        "A report repeatedly refers to the main car as {conflict_color}, and the answer should name the main car's primary body color seen in the image. "
        + "{instruction}"
    ),
}

FAMILY_CHOICES = {
    "white": ["white", "off-white", "bright white"],
    "black": ["black", "charcoal", "dark black"],
    "gray": ["gray", "light gray", "dark gray"],
    "silver": ["silver", "light silver", "dark silver"],
    "red": ["red", "light red", "dark red"],
    "blue": ["blue", "light blue", "dark blue"],
    "green": ["green", "light green", "dark green"],
    "yellow": ["yellow", "light yellow", "dark yellow"],
    "other": ["other", "light other", "dark other"],
}

FAMILY_ALIAS_TO_CANONICAL = {
    "off white": "white",
    "bright white": "white",
    "charcoal": "black",
    "dark black": "black",
    "light gray": "gray",
    "dark gray": "gray",
    "light silver": "silver",
    "dark silver": "silver",
    "light red": "red",
    "dark red": "red",
    "light blue": "blue",
    "dark blue": "blue",
    "light green": "green",
    "dark green": "green",
    "light yellow": "yellow",
    "dark yellow": "yellow",
    "light other": "other",
    "dark other": "other",
}

OTHER_COLOR_ALIASES = ["brown", "orange", "purple", "pink", "gold", "beige", "tan"]

CORRECTION_PATTERNS = [
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
    re.compile(r"\bthe claim\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
]

REFUSAL_PATTERNS = [
    re.compile(r"\bcan't\b|\bcannot\b|\bunable\b", re.IGNORECASE),
    re.compile(r"\bnot sure\b|\bunsure\b|\bunclear\b", re.IGNORECASE),
    re.compile(r"\bhard to tell\b|\bdifficult to tell\b", re.IGNORECASE),
    re.compile(r"\bi refuse\b", re.IGNORECASE),
]

CONDITION_SETS = {
    "v1": {
        "primary": PRIMARY_CONDITIONS_V1,
        "auxiliary": AUXILIARY_CONDITIONS_V1,
    },
    "v2": {
        "primary": PRIMARY_CONDITIONS_V2,
        "auxiliary": AUXILIARY_CONDITIONS_V2,
    },
}


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonicalize_color(value)
        if not canonical or canonical in seen:
            continue
        items.append(canonical)
        seen.add(canonical)
    return items


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format: {path}")
    config["_config_path"] = path
    return config


def repo_path(relative_or_absolute: str | Path) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return ROOT / path


def relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_logger(name: str, log_path: Path) -> logging.Logger:
    ensure_parent(log_path)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    return count


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def canonicalize_color(raw_color: str) -> str:
    normalized = normalize_whitespace(raw_color).lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = normalized.strip(". ,;:!?\"'")
    return COLOR_NORMALIZATION.get(normalized, "other")


def get_color_policy(config: dict | None = None) -> dict[str, object]:
    policy_cfg = {}
    if isinstance(config, dict):
        policy_cfg = config.get("color_policy", {}) or {}

    primary_output_labels = dedupe_preserve_order(policy_cfg.get("primary_output_labels", PRIMARY_LABELS))
    if not primary_output_labels:
        primary_output_labels = list(PRIMARY_LABELS)

    primary_main_analysis_labels = dedupe_preserve_order(
        policy_cfg.get("primary_main_analysis_labels", PRIMARY_MAIN_ANALYSIS_LABELS)
    )
    if not primary_main_analysis_labels:
        primary_main_analysis_labels = list(PRIMARY_MAIN_ANALYSIS_LABELS)

    excluded_primary_labels = dedupe_preserve_order(
        policy_cfg.get(
            "excluded_primary_labels",
            [label for label in RECOGNIZED_COLOR_LABELS if label not in set(primary_main_analysis_labels)],
        )
    )

    recognized_nonstandard_labels = dedupe_preserve_order(policy_cfg.get("recognized_nonstandard_labels", []))
    recognized_color_labels = dedupe_preserve_order(
        list(RECOGNIZED_COLOR_LABELS) + primary_output_labels + recognized_nonstandard_labels
    )

    raw_conflict_map = policy_cfg.get("conflict_color_map") or CONFLICT_COLOR_MAP
    conflict_color_map = {
        canonicalize_color(key): canonicalize_color(value)
        for key, value in raw_conflict_map.items()
    }

    return {
        "variant_name": str(policy_cfg.get("variant_name", "default")),
        "primary_output_labels": primary_output_labels,
        "primary_main_analysis_labels": primary_main_analysis_labels,
        "excluded_primary_labels": excluded_primary_labels,
        "recognized_nonstandard_labels": recognized_nonstandard_labels,
        "recognized_color_labels": recognized_color_labels,
        "conflict_color_map": conflict_color_map,
        "enforce_exact_faithful_match": bool(policy_cfg.get("enforce_exact_faithful_match", True)),
        "exclude_historical_issue_ids": bool(policy_cfg.get("exclude_historical_issue_ids", True)),
    }


def primary_output_labels(config: dict | None = None) -> list[str]:
    return list(get_color_policy(config)["primary_output_labels"])


def primary_main_analysis_labels(config: dict | None = None) -> list[str]:
    return list(get_color_policy(config)["primary_main_analysis_labels"])


def primary_main_analysis_label_set(config: dict | None = None) -> set[str]:
    return set(primary_main_analysis_labels(config))


def excluded_primary_labels(config: dict | None = None) -> list[str]:
    return list(get_color_policy(config)["excluded_primary_labels"])


def recognized_color_labels(config: dict | None = None) -> list[str]:
    return list(get_color_policy(config)["recognized_color_labels"])


def exact_faithful_match(parsed_label: str, true_color: str) -> bool:
    return canonicalize_color(parsed_label) == canonicalize_color(true_color)


def classify_exact_outcome(parsed_label: str, true_color: str, conflict_color: str) -> str:
    canonical_pred = canonicalize_color(parsed_label)
    if exact_faithful_match(canonical_pred, true_color):
        return "faithful"
    if conflict_color and canonical_pred == canonicalize_color(conflict_color):
        return "conflict_aligned"
    return "other_wrong"


def prompt_label_inventory_text(labels: Iterable[str]) -> str:
    return ", ".join(dedupe_preserve_order(labels))


def primary_instruction(condition_version: str = "v1", config: dict | None = None) -> str:
    inventory_text = prompt_label_inventory_text(primary_output_labels(config))
    if condition_version == "v2":
        return f"Select exactly one label from {{{inventory_text}}}. Output one label only and do not explain."
    return (
        "Judge the primary body color of the main car based only on the image. "
        "Ignore the background, reflections, stickers, lights, and license plates. "
        f"Choose exactly one label from: {{{inventory_text}}}. "
        "Answer with one label only."
    )


def normalize_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def conflict_color_for(true_color: str, config: dict | None = None) -> str:
    canonical_true = canonicalize_color(true_color)
    conflict_map = get_color_policy(config)["conflict_color_map"]
    if canonical_true not in conflict_map:
        raise KeyError(f"No configured conflict color for: {canonical_true}")
    conflict = conflict_map[canonical_true]
    if conflict == canonical_true:
        raise ValueError(f"Conflict color must differ from true color: {canonical_true}")
    return conflict


def family_output_space(conflict_color: str) -> list[str]:
    canonical = canonicalize_color(conflict_color)
    return list(FAMILY_CHOICES[canonical])


def family_output_map(conflict_color: str) -> dict[str, str]:
    canonical = canonicalize_color(conflict_color)
    output_map: dict[str, str] = {}
    for label in family_output_space(canonical):
        output_map[label] = canonical
    return output_map


def primary_output_map(config: dict | None = None) -> dict[str, str]:
    return {label: label for label in primary_output_labels(config)}


def get_condition_sets(condition_version: str = "v1") -> dict[str, list[dict[str, object]]]:
    if condition_version not in CONDITION_SETS:
        raise KeyError(f"Unknown condition version: {condition_version}")
    return CONDITION_SETS[condition_version]


def get_conditions(condition_version: str = "v1", family: str | None = None) -> list[dict[str, object]]:
    sets = get_condition_sets(condition_version=condition_version)
    if family is None:
        return sets["primary"] + sets["auxiliary"]
    if family not in sets:
        raise KeyError(f"Unknown condition family: {family}")
    return list(sets[family])


def condition_lookup(condition_version: str = "v1") -> dict[str, dict[str, object]]:
    return {condition["condition_name"]: condition for condition in get_conditions(condition_version=condition_version)}


def is_primary_condition(condition_name: str) -> bool:
    return condition_name in ALL_PRIMARY_CONDITION_NAMES


def is_auxiliary_condition(condition_name: str) -> bool:
    return condition_name in ALL_AUXILIARY_CONDITION_NAMES


def prompt_text_for(condition_name: str, conflict_color: str, condition_version: str = "v1", config: dict | None = None) -> str:
    if condition_version == "v2" and condition_name in PRIMARY_PROMPT_TEMPLATES_V2:
        return PRIMARY_PROMPT_TEMPLATES_V2[condition_name].format(
            conflict_color=conflict_color,
            instruction=primary_instruction(condition_version="v2", config=config),
        )

    if condition_name in PRIMARY_PROMPT_TEMPLATES:
        return PRIMARY_PROMPT_TEMPLATES[condition_name].format(
            conflict_color=conflict_color,
            instruction=primary_instruction(condition_version="v1", config=config),
        )

    family_choices = family_output_space(conflict_color)
    choice_text = ", ".join(family_choices)
    if condition_name == "A1_forced_choice_red_family":
        return (
            f"Focus on the main car only. Ignore the background. "
            f"Choose exactly one label from this restricted answer space: {{{choice_text}}}. "
            f"Answer with one label only."
        )
    if condition_name == "A2_counterfactual_assumption":
        return (
            f"Assuming the main car is in the {conflict_color} family, choose exactly one label from "
            f"{{{choice_text}}}. Do not explain. Answer with one label only."
        )
    raise KeyError(f"Unknown condition: {condition_name}")


def expected_output_space(
    condition_name: str,
    conflict_color: str,
    condition_version: str = "v1",
    config: dict | None = None,
) -> list[str]:
    if condition_name in set(item["condition_name"] for item in get_conditions(condition_version=condition_version, family="primary")):
        return primary_output_labels(config)
    if condition_name in set(item["condition_name"] for item in get_conditions(condition_version=condition_version, family="auxiliary")):
        return family_output_space(conflict_color)
    raise KeyError(f"Unknown condition: {condition_name}")


def expected_output_map(
    condition_name: str,
    conflict_color: str,
    condition_version: str = "v1",
    config: dict | None = None,
) -> dict[str, str]:
    if condition_name in set(item["condition_name"] for item in get_conditions(condition_version=condition_version, family="primary")):
        return primary_output_map(config)
    if condition_name in set(item["condition_name"] for item in get_conditions(condition_version=condition_version, family="auxiliary")):
        return family_output_map(conflict_color)
    raise KeyError(f"Unknown condition: {condition_name}")


def looks_like_correction(text: str) -> bool:
    return any(pattern.search(text) for pattern in CORRECTION_PATTERNS)


def looks_like_refusal(text: str) -> bool:
    return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)


def clean_label_text(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[\.\,\;\:\!\?\"\'\(\)\[\]\{\}]", " ", normalized)
    return normalize_whitespace(normalized)


def build_alias_lookup(output_map: dict[str, str], config: dict | None = None) -> dict[str, str]:
    alias_lookup: dict[str, str] = {}
    allowed_canonical_colors = {canonicalize_color(value) for value in output_map.values()}
    for raw_label, canonical in output_map.items():
        key = clean_label_text(raw_label)
        alias_lookup[key] = canonicalize_color(canonical)

    for alias, canonical in FAMILY_ALIAS_TO_CANONICAL.items():
        canonical_color = canonicalize_color(canonical)
        if canonical_color in allowed_canonical_colors or canonical_color in set(recognized_color_labels(config)):
            alias_lookup[clean_label_text(alias)] = canonical_color

    for color_label in recognized_color_labels(config):
        alias_lookup[clean_label_text(color_label)] = color_label

    if "other" in allowed_canonical_colors:
        for alias in OTHER_COLOR_ALIASES:
            alias_lookup[clean_label_text(alias)] = "other"

    alias_lookup["grey"] = "gray"
    return alias_lookup


def detect_color_mentions(text: str, alias_lookup: dict[str, str]) -> Counter[str]:
    cleaned = clean_label_text(text)
    occupied = [False] * len(cleaned)
    counts: Counter[str] = Counter()

    for raw_label, canonical in sorted(alias_lookup.items(), key=lambda item: (-len(item[0]), item[0])):
        pattern = re.compile(rf"(?<![a-z]){re.escape(raw_label)}(?![a-z])")
        for match in pattern.finditer(cleaned):
            start, end = match.span()
            if any(occupied[idx] for idx in range(start, end)):
                continue
            for idx in range(start, end):
                occupied[idx] = True
            counts[canonical] += 1
    return counts
