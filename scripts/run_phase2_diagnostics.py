#!/usr/bin/env python
"""Build, run, and analyze phase-2 diagnostic prompt modules."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from scripts.utils.paper_mainline_utils import (
    MODEL_ORDER,
    dump_json,
    load_bool_results,
    load_paper_config,
    model_raw_dir,
    paper_paths,
    run_and_parse_prompt_set,
    run_command,
    selected_model_keys,
    write_markdown,
)
from scripts.utils.restructured_experiment_utils import json_dumps, primary_instruction


MODEL_DISPLAY = {
    "qwen2vl7b": "Qwen2-VL-7B-Instruct",
    "llava15_7b": "LLaVA-1.5-7B",
    "internvl2_8b": "InternVL2-8B",
}
PRIMARY_OUTPUT_SPACE = ["red", "blue", "green", "yellow", "black", "white", "other"]
PRIMARY_OUTPUT_MAP = {label: label for label in PRIMARY_OUTPUT_SPACE}
YES_NO_SPACE = ["yes", "no"]
YES_NO_MAP = {"yes": "yes", "no": "no"}
EXTRA_FIELDS = [
    "diagnostic_family",
    "factor_id",
    "tone_strength",
    "injection_position",
    "false_text_form",
    "answer_format",
    "response_schema",
    "context_turns_json",
]
FAMILY_PREFIX = {
    "factorization": "factorized_prompt",
    "format_control": "format_control",
    "multiturn": "multiturn",
}
FAMILY_PROMPT_DIR = {
    "factorization": REPO_ROOT / "prompts" / "factorization",
    "format_control": REPO_ROOT / "prompts" / "format_control",
    "multiturn": REPO_ROOT / "prompts" / "multiturn",
}
FAMILY_RESULTS_DIR = {
    "factorization": REPO_ROOT / "results" / "factorization",
    "format_control": REPO_ROOT / "results" / "format_control",
    "multiturn": REPO_ROOT / "results" / "multiturn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-2 diagnostic experiments.")
    parser.add_argument("--family", choices=["factorization", "format_control", "multiturn"], required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-run", action="store_true", help="Only build prompts and analyze existing outputs.")
    parser.add_argument("--skip-build-prompts", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def prompt_csv_for(family: str) -> Path:
    return FAMILY_PROMPT_DIR[family] / f"{FAMILY_PREFIX[family]}_prompts.csv"


def read_base_rows(main_prompt_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    with main_prompt_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("condition_name") == "C0_neutral"]
    return fieldnames, rows


def final_question_instruction(config: dict) -> str:
    return primary_instruction(condition_version="v2", config=config)


def neutral_color_question(config: dict) -> str:
    return f"What is the primary body color of the main car in this image? {final_question_instruction(config)}"


def one_color_word_question() -> str:
    return "Answer with one color word only and do not explain."


def context_json(turns: list[dict[str, str]]) -> str:
    return json.dumps(turns, ensure_ascii=False)


def factorization_specs(config: dict) -> list[dict[str, object]]:
    instruction = final_question_instruction(config)
    return [
        {
            "condition_name": "F1_quoted_claim",
            "factor_id": "quoted_claim_user",
            "tone_strength": "assertive",
            "injection_position": "question_body",
            "false_text_form": "quoted_false_claim",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": "A previous caption says, \"The main car is {conflict_color}.\" Based on the image, what is the main car's actual primary body color? {instruction}",
            "context_turns_json": "",
        },
        {
            "condition_name": "F2_indirect_hint",
            "factor_id": "indirect_hint_user",
            "tone_strength": "suggestive",
            "injection_position": "question_body",
            "false_text_form": "indirect_hint",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": "The main car may look closer to {conflict_color} than other colors. What is its primary body color in the image? {instruction}",
            "context_turns_json": "",
        },
        {
            "condition_name": "F3_title_prefix",
            "factor_id": "title_prefix_framing",
            "tone_strength": "assertive",
            "injection_position": "title_prefix",
            "false_text_form": "title_or_prefix",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": "Title: {conflict_color} main car. Question: what is the primary body color of the main car in the image? {instruction}",
            "context_turns_json": "",
        },
        {
            "condition_name": "F4_presupposition_no_correction",
            "factor_id": "presupposition_no_correction",
            "tone_strength": "presuppositional",
            "injection_position": "question_body",
            "false_text_form": "presupposition_without_correction_cue",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": "When identifying this {conflict_color} main car, which primary body-color label should be selected? {instruction}",
            "context_turns_json": "",
        },
        {
            "condition_name": "F5_previous_turn_context",
            "factor_id": "previous_turn_false_context",
            "tone_strength": "assertive",
            "injection_position": "previous_user_turn",
            "false_text_form": "previous_turn_false_statement",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": neutral_color_question(config),
            "context_turns_json": context_json(
                [
                    {
                        "role": "user",
                        "content": "Before answering, note that the main car in the image is {conflict_color}.",
                    },
                    {"role": "assistant", "content": "Understood."},
                ]
            ),
        },
    ]


def format_control_specs(config: dict) -> list[dict[str, object]]:
    labels = ", ".join(PRIMARY_OUTPUT_SPACE)
    mc = "Choose exactly one color label from this multiple-choice list: red, blue, green, yellow, black, white, other. Output the color label only."
    return [
        {
            "condition_name": "FC1_free_neutral",
            "factor_id": "free_neutral",
            "tone_strength": "neutral",
            "injection_position": "none",
            "false_text_form": "none",
            "answer_format": "free_color_word",
            "response_schema": "one_color_word",
            "prompt_template": "What is the primary body color of the main car in this image? " + one_color_word_question(),
            "context_turns_json": "",
        },
        {
            "condition_name": "FC2_free_c3",
            "factor_id": "free_c3_presupposition",
            "tone_strength": "presuppositional",
            "injection_position": "question_body",
            "false_text_form": "presupposition",
            "answer_format": "free_color_word",
            "response_schema": "one_color_word",
            "prompt_template": "When the main car is described as {conflict_color}, what color does the main car actually appear to be in the image? " + one_color_word_question(),
            "context_turns_json": "",
        },
        {
            "condition_name": "FC3_free_c4",
            "factor_id": "free_c4_stronger_open",
            "tone_strength": "stronger_open_conflict",
            "injection_position": "question_body",
            "false_text_form": "repeated_false_report",
            "answer_format": "free_color_word",
            "response_schema": "one_color_word",
            "prompt_template": "A report repeatedly refers to the main car as {conflict_color}, and the answer should name the main car's primary body color seen in the image. " + one_color_word_question(),
            "context_turns_json": "",
        },
        {
            "condition_name": "FC4_mc_neutral",
            "factor_id": "multiple_choice_neutral",
            "tone_strength": "neutral",
            "injection_position": "none",
            "false_text_form": "none",
            "answer_format": "multiple_choice_label",
            "response_schema": "one_label_from_multiple_choice",
            "prompt_template": f"What is the primary body color of the main car in this image? {mc}",
            "context_turns_json": "",
        },
        {
            "condition_name": "FC5_mc_c3",
            "factor_id": "multiple_choice_c3_presupposition",
            "tone_strength": "presuppositional",
            "injection_position": "question_body",
            "false_text_form": "presupposition",
            "answer_format": "multiple_choice_label",
            "response_schema": "one_label_from_multiple_choice",
            "prompt_template": f"When the main car is described as {{conflict_color}}, what color does the main car actually appear to be in the image? {mc}",
            "context_turns_json": "",
        },
        {
            "condition_name": "FC6_mc_c4",
            "factor_id": "multiple_choice_c4_stronger_open",
            "tone_strength": "stronger_open_conflict",
            "injection_position": "question_body",
            "false_text_form": "repeated_false_report",
            "answer_format": "multiple_choice_label",
            "response_schema": "one_label_from_multiple_choice",
            "prompt_template": f"A report repeatedly refers to the main car as {{conflict_color}}. What is the main car's primary body color seen in the image? {mc}",
            "context_turns_json": "",
        },
        {
            "condition_name": "FC7_yesno_false_claim",
            "factor_id": "yesno_false_claim",
            "tone_strength": "assertive",
            "injection_position": "question_body",
            "false_text_form": "direct_false_claim",
            "answer_format": "yes_no",
            "response_schema": "yes_means_conflict_no_means_reject_false_claim",
            "prompt_template": "Is the main car {conflict_color}? Answer yes or no only.",
            "context_turns_json": "",
        },
        {
            "condition_name": "FC8_yesno_report_correct",
            "factor_id": "yesno_report_correct",
            "tone_strength": "stronger_open_conflict",
            "injection_position": "question_body",
            "false_text_form": "repeated_false_report",
            "answer_format": "yes_no",
            "response_schema": "yes_means_conflict_no_means_reject_false_claim",
            "prompt_template": "A report repeatedly says the main car is {conflict_color}. Is that report correct for the image? Answer yes or no only.",
            "context_turns_json": "",
        },
    ]


def multiturn_specs(config: dict) -> list[dict[str, object]]:
    neutral = neutral_color_question(config)
    return [
        {
            "condition_name": "MT1_single_turn_false_note",
            "factor_id": "single_turn_false_note",
            "tone_strength": "assertive",
            "injection_position": "question_body",
            "false_text_form": "single_turn_false_note",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": "A prior note says the main car is {conflict_color}. " + neutral,
            "context_turns_json": "",
        },
        {
            "condition_name": "MT2_two_turn_persuasion",
            "factor_id": "two_turn_persuasion",
            "tone_strength": "assertive_context",
            "injection_position": "previous_user_turn",
            "false_text_form": "previous_turn_false_statement",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": neutral,
            "context_turns_json": context_json(
                [
                    {"role": "user", "content": "For this image, remember that the main car is {conflict_color}."},
                    {"role": "assistant", "content": "Understood."},
                ]
            ),
        },
        {
            "condition_name": "MT3_three_turn_persuasion",
            "factor_id": "three_turn_persuasion",
            "tone_strength": "repeated_context",
            "injection_position": "previous_user_turn",
            "false_text_form": "repeated_previous_turn_false_statement",
            "answer_format": "constrained_short_label",
            "response_schema": "one_label_from_primary_space",
            "prompt_template": neutral,
            "context_turns_json": context_json(
                [
                    {"role": "user", "content": "For this image, remember that the main car is {conflict_color}."},
                    {"role": "assistant", "content": "Understood."},
                    {"role": "user", "content": "The earlier note is important: treat the main car as {conflict_color}."},
                    {"role": "assistant", "content": "Understood."},
                ]
            ),
        },
    ]


def specs_for_family(family: str, config: dict) -> list[dict[str, object]]:
    if family == "factorization":
        return factorization_specs(config)
    if family == "format_control":
        return format_control_specs(config)
    if family == "multiturn":
        return multiturn_specs(config)
    raise KeyError(family)


def build_prompt_csv(family: str, config: dict, output_csv: Path) -> int:
    paths = paper_paths(config)
    source_fields, base_rows = read_base_rows(paths["main_prompt_csv"])
    fieldnames = list(source_fields)
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    specs = specs_for_family(family, config)
    rows: list[dict[str, str]] = []
    for base in base_rows:
        for condition_index, spec in enumerate(specs, start=100):
            row = dict(base)
            conflict_color = row["conflict_color"]
            row["sample_id"] = f"{row['image_id']}__{spec['condition_name']}"
            row["condition_family"] = f"phase2_{family}"
            row["condition_name"] = str(spec["condition_name"])
            row["condition_index"] = str(condition_index)
            row["prompt_template_version"] = f"phase2_{family}_{spec['condition_name']}"
            prompt_template = str(spec["prompt_template"])
            row["prompt_text"] = (
                prompt_template.replace("{conflict_color}", conflict_color)
                .replace("{labels}", ", ".join(PRIMARY_OUTPUT_SPACE))
                .replace("{instruction}", final_question_instruction(config))
            )
            if spec["answer_format"] == "yes_no":
                row["expected_output_space"] = json_dumps(YES_NO_SPACE)
                row["expected_output_map"] = json_dumps(YES_NO_MAP)
            else:
                row["expected_output_space"] = json_dumps(PRIMARY_OUTPUT_SPACE)
                row["expected_output_map"] = json_dumps(PRIMARY_OUTPUT_MAP)
            row["diagnostic_family"] = family
            for field in EXTRA_FIELDS:
                if field == "diagnostic_family":
                    continue
                value = str(spec.get(field, ""))
                if field == "context_turns_json" and value:
                    turns = json.loads(value)
                    for turn in turns:
                        if isinstance(turn, dict):
                            turn["content"] = str(turn.get("content", "")).format(conflict_color=conflict_color)
                    value = json.dumps(turns, ensure_ascii=False)
                row[field] = value
            rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def classify_yes_no(text: object) -> str:
    cleaned = clean_text(text)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = set(cleaned.split())
    has_yes = "yes" in tokens
    has_no = "no" in tokens
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    return ""


def normalize_phase2_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ["is_conflict_aligned", "is_faithful", "is_other_wrong", "is_refusal_or_correction", "is_parse_error"]:
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        else:
            result[column] = False
    result["phase2_is_conflict_aligned"] = result["is_conflict_aligned"]
    result["phase2_is_faithful"] = result["is_faithful"]
    result["phase2_is_parse_error"] = result["is_parse_error"]
    result["phase2_parsed_label"] = result.get("parsed_label", "").astype(str)
    yesno_mask = result.get("answer_format", "").astype(str).eq("yes_no")
    if yesno_mask.any():
        classifications = result.loc[yesno_mask, "raw_output"].map(classify_yes_no)
        result.loc[yesno_mask, "phase2_parsed_label"] = classifications
        result.loc[yesno_mask, "phase2_is_conflict_aligned"] = classifications.eq("yes")
        result.loc[yesno_mask, "phase2_is_faithful"] = classifications.eq("no")
        result.loc[yesno_mask, "phase2_is_parse_error"] = classifications.eq("")
    return result


def add_reference_rows(family: str, config: dict, new_df: pd.DataFrame) -> pd.DataFrame:
    paths = paper_paths(config)
    frames = []
    main = load_bool_results(paths["main_dir"] / "main_combined_parsed_results.csv")
    main = main[main["condition_name"].isin(["C0_neutral", "C3_presupposition_correction_allowed", "C4_stronger_open_conflict"])].copy()
    for col in EXTRA_FIELDS:
        main[col] = ""
    main["diagnostic_family"] = family
    main["answer_format"] = "label_set"
    main["response_schema"] = "one_label_from_primary_space"
    main["factor_id"] = main["condition_name"].map(
        {
            "C0_neutral": "REF_C0_label_set",
            "C3_presupposition_correction_allowed": "REF_C3_original_label_set",
            "C4_stronger_open_conflict": "REF_C4_original_label_set",
        }
    )
    main["false_text_form"] = main["condition_name"].map(
        {
            "C0_neutral": "none",
            "C3_presupposition_correction_allowed": "presupposition_correction_allowed",
            "C4_stronger_open_conflict": "repeated_false_report",
        }
    )
    main["tone_strength"] = main["condition_name"].map(
        {
            "C0_neutral": "neutral",
            "C3_presupposition_correction_allowed": "presuppositional",
            "C4_stronger_open_conflict": "stronger_open_conflict",
        }
    )
    frames.append(main)

    if family == "factorization":
        variant_path = paths["robustness_dir"] / "prompt_variant_combined_parsed_results.csv"
        if variant_path.exists():
            variants = load_bool_results(variant_path)
            variants = variants[variants["robustness_variant"].astype(str).isin(["C3_v2", "C3_v3"])].copy()
            for col in EXTRA_FIELDS:
                if col not in variants.columns:
                    variants[col] = ""
            variants["diagnostic_family"] = family
            variants["factor_id"] = variants["robustness_variant"].map(
                {"C3_v2": "REF_C3_v2_wording", "C3_v3": "REF_C3_v3_wording"}
            )
            variants["tone_strength"] = "assertive"
            variants["injection_position"] = "question_body"
            variants["false_text_form"] = variants["robustness_variant"].map(
                {"C3_v2": "quoted_or_reported_claim", "C3_v3": "explicit_prompt_says"}
            )
            variants["answer_format"] = "label_set"
            variants["response_schema"] = "one_label_from_primary_space"
            frames.append(variants)
    frames.append(new_df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_phase2_outcomes(combined)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    low, high = proportion_confint(successes, total, method="wilson")
    return float(low), float(high)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(
        [
            "model_key",
            "model_name",
            "condition_name",
            "factor_id",
            "tone_strength",
            "injection_position",
            "false_text_form",
            "answer_format",
            "response_schema",
        ],
        dropna=False,
        observed=False,
    )
    for keys, subset in grouped:
        total = int(len(subset))
        conflict_n = int(subset["phase2_is_conflict_aligned"].sum())
        faithful_n = int(subset["phase2_is_faithful"].sum())
        parse_n = int(subset["phase2_is_parse_error"].sum())
        conflict_low, conflict_high = wilson(conflict_n, total)
        faithful_low, faithful_high = wilson(faithful_n, total)
        rows.append(
            {
                "model_key": keys[0],
                "model": MODEL_DISPLAY.get(keys[0], keys[1]),
                "condition_name": keys[2],
                "factor_id": keys[3],
                "tone_strength": keys[4],
                "injection_position": keys[5],
                "false_text_form": keys[6],
                "answer_format": keys[7],
                "response_schema": keys[8],
                "n": total,
                "conflict_aligned_n": conflict_n,
                "conflict_aligned_rate": conflict_n / total if total else 0.0,
                "conflict_aligned_ci_low": conflict_low,
                "conflict_aligned_ci_high": conflict_high,
                "faithful_n": faithful_n,
                "faithful_rate": faithful_n / total if total else 0.0,
                "faithful_ci_low": faithful_low,
                "faithful_ci_high": faithful_high,
                "parse_error_n": parse_n,
                "parse_error_rate": parse_n / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def paired_test(current: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object] | None:
    merged = current[["image_id", "phase2_is_conflict_aligned"]].merge(
        reference[["image_id", "phase2_is_conflict_aligned"]],
        on="image_id",
        suffixes=("_current", "_reference"),
    )
    if merged.empty:
        return None
    current_bool = merged["phase2_is_conflict_aligned_current"].astype(bool)
    reference_bool = merged["phase2_is_conflict_aligned_reference"].astype(bool)
    both_yes = int((current_bool & reference_bool).sum())
    current_only = int((current_bool & ~reference_bool).sum())
    reference_only = int((~current_bool & reference_bool).sum())
    both_no = int((~current_bool & ~reference_bool).sum())
    result = mcnemar([[both_yes, current_only], [reference_only, both_no]], exact=True)
    return {
        "n_pairs": int(len(merged)),
        "current_rate": float(current_bool.mean()),
        "reference_rate": float(reference_bool.mean()),
        "rate_diff_current_minus_reference": float(current_bool.mean() - reference_bool.mean()),
        "current_only": current_only,
        "reference_only": reference_only,
        "p_value_raw": float(result.pvalue),
    }


def build_key_tests(df: pd.DataFrame, family: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key].copy()
        c0 = model_df[model_df["factor_id"] == "REF_C0_label_set"].copy()
        if c0.empty:
            continue
        for factor_id, current in model_df.groupby("factor_id", observed=False):
            if factor_id == "REF_C0_label_set":
                continue
            payload = paired_test(current, c0)
            if payload is None:
                continue
            rows.append(
                {
                    "comparison_family": "within_model_vs_C0",
                    "model_key": model_key,
                    "model": MODEL_DISPLAY[model_key],
                    "condition_name": current["condition_name"].iloc[0],
                    "factor_id": factor_id,
                    "reference_factor_id": "REF_C0_label_set",
                    **payload,
                }
            )
        if family == "multiturn":
            mt1 = model_df[model_df["factor_id"] == "single_turn_false_note"].copy()
            for factor_id in ["two_turn_persuasion", "three_turn_persuasion"]:
                current = model_df[model_df["factor_id"] == factor_id].copy()
                if current.empty or mt1.empty:
                    continue
                payload = paired_test(current, mt1)
                if payload is None:
                    continue
                rows.append(
                    {
                        "comparison_family": "within_model_vs_MT1",
                        "model_key": model_key,
                        "model": MODEL_DISPLAY[model_key],
                        "condition_name": current["condition_name"].iloc[0],
                        "factor_id": factor_id,
                        "reference_factor_id": "single_turn_false_note",
                        **payload,
                    }
                )
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    tests["p_value_holm"] = np.nan
    tests["significant_holm"] = False
    for family_name, family_df in tests.groupby("comparison_family", observed=False):
        values = family_df["p_value_raw"].to_numpy(dtype=float)
        _, corrected, _, _ = multipletests(values, method="holm")
        tests.loc[family_df.index, "p_value_holm"] = corrected
        tests.loc[family_df.index, "significant_holm"] = corrected < 0.05
    return tests


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def table_md(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for row in df[columns].to_dict("records")]
    return "\n".join([header, divider, *rows])


def make_figure(metrics: pd.DataFrame, family: str, output_path: Path) -> None:
    plot = metrics[metrics["model_key"].isin(MODEL_ORDER)].copy()
    if family == "factorization":
        keep = [
            "REF_C0_label_set",
            "REF_C3_original_label_set",
            "REF_C4_original_label_set",
            "REF_C3_v2_wording",
            "REF_C3_v3_wording",
            "quoted_claim_user",
            "indirect_hint_user",
            "title_prefix_framing",
            "presupposition_no_correction",
            "previous_turn_false_context",
        ]
    elif family == "format_control":
        keep = [
            "REF_C0_label_set",
            "REF_C3_original_label_set",
            "REF_C4_original_label_set",
            "free_neutral",
            "free_c3_presupposition",
            "free_c4_stronger_open",
            "multiple_choice_neutral",
            "multiple_choice_c3_presupposition",
            "multiple_choice_c4_stronger_open",
            "yesno_false_claim",
            "yesno_report_correct",
        ]
    else:
        keep = [
            "REF_C0_label_set",
            "REF_C3_original_label_set",
            "single_turn_false_note",
            "two_turn_persuasion",
            "three_turn_persuasion",
        ]
    plot = plot[plot["factor_id"].isin(keep)].copy()
    plot["factor_id"] = pd.Categorical(plot["factor_id"], categories=keep, ordered=True)
    plot = plot.sort_values(["factor_id", "model_key"])
    x = np.arange(len(keep))
    width = 0.24
    palette = {"qwen2vl7b": "#355f8d", "llava15_7b": "#c16843", "internvl2_8b": "#4d8f5b"}
    fig, ax = plt.subplots(figsize=(max(10, len(keep) * 1.05), 5.2))
    for offset, model_key in zip(np.linspace(-width, width, len(MODEL_ORDER)), MODEL_ORDER):
        subset = plot[plot["model_key"] == model_key].set_index("factor_id").reindex(keep)
        ax.bar(
            x + offset,
            subset["conflict_aligned_rate"].fillna(0).to_numpy(dtype=float) * 100.0,
            width=width,
            label=MODEL_DISPLAY[model_key],
            color=palette[model_key],
            alpha=0.94,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(keep, rotation=35, ha="right")
    ax.set_ylabel("Conflict-aligned / false-claim-following rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_family_summary(metrics: pd.DataFrame, tests: pd.DataFrame, family: str, output_path: Path) -> None:
    title = {
        "factorization": "Factorized Prompt Summary",
        "format_control": "Answer Format Control Summary",
        "multiturn": "Multi-Turn Persuasion Summary",
    }[family]
    llava = metrics[metrics["model_key"] == "llava15_7b"].copy()
    llava_display = llava[
        [
            "factor_id",
            "condition_name",
            "tone_strength",
            "injection_position",
            "false_text_form",
            "answer_format",
            "n",
            "conflict_aligned_n",
            "conflict_aligned_rate",
            "faithful_n",
            "faithful_rate",
            "parse_error_n",
        ]
    ].copy()
    for col in ["conflict_aligned_rate", "faithful_rate"]:
        llava_display[col] = llava_display[col].map(fmt_pct)

    significant = tests[(tests["model_key"] == "llava15_7b") & (tests["significant_holm"])] if not tests.empty else pd.DataFrame()
    sig_lines = []
    if significant.empty:
        sig_lines.append("- No Holm-significant LLaVA phase-2 paired tests were detected in this family.")
    else:
        for row in significant.to_dict("records"):
            sig_lines.append(
                f"- {row['comparison_family']}: `{row['factor_id']}` vs `{row['reference_factor_id']}` "
                f"diff={row['rate_diff_current_minus_reference'] * 100:.2f} pp, "
                f"current-only={row['current_only']}, reference-only={row['reference_only']}, "
                f"Holm p={fmt_p(row['p_value_holm'])}."
            )

    nonref = metrics[~metrics["factor_id"].astype(str).str.startswith("REF_")].copy()
    overview_rows = []
    for model_key in MODEL_ORDER:
        subset = nonref[nonref["model_key"] == model_key].sort_values("conflict_aligned_rate", ascending=False)
        if subset.empty:
            continue
        top = subset.iloc[0]
        low_n = int((subset["conflict_aligned_rate"] <= 0.01).sum())
        overview_rows.append(
            {
                "model": MODEL_DISPLAY.get(model_key, model_key),
                "highest_new_cell": top["factor_id"],
                "highest_rate": fmt_pct(float(top["conflict_aligned_rate"])),
                "highest_count": f"{int(top['conflict_aligned_n'])}/{int(top['n'])}",
                "new_cells_at_or_below_1pct": f"{low_n}/{len(subset)}",
            }
        )
    overview = pd.DataFrame(overview_rows)

    if family == "factorization":
        get_rate = lambda model, factor: metrics[
            (metrics["model_key"] == model) & (metrics["factor_id"] == factor)
        ]["conflict_aligned_rate"].iloc[0]
        interpretation = [
            "The factorization module should be read as a prompt-form diagnostic. Compare the original C3 reference to quoted, indirect, title/prefix, presuppositional, and previous-turn forms; do not describe the result as a general prompt-engineering law.",
            "The strongest new factors are not the quoted or indirect hints. They are title/prefix framing and presupposition without an explicit correction cue, with model-specific ordering: Qwen is most affected by no-correction presupposition, while LLaVA and InternVL2 peak under title/prefix framing.",
            f"For LLaVA, original C3 is {fmt_pct(get_rate('llava15_7b', 'REF_C3_original_label_set'))}, but title/prefix is {fmt_pct(get_rate('llava15_7b', 'title_prefix_framing'))} and no-correction presupposition is {fmt_pct(get_rate('llava15_7b', 'presupposition_no_correction'))}. The original C3 is therefore a meaningful mainline cell, not the upper bound of prompt susceptibility.",
            "Quoted claims and indirect hints stay near zero for most models. This argues against a simple 'any false text works' story and supports a narrower account about framing, presupposition, and correction affordances.",
        ]
    elif family == "format_control":
        def llava_rate(factor: str) -> float:
            return metrics[(metrics["model_key"] == "llava15_7b") & (metrics["factor_id"] == factor)][
                "conflict_aligned_rate"
            ].iloc[0]

        interpretation = [
            "The format-control module separates false-text effects from answer-format effects. Yes/no rows measure acceptance of a false claim, not color-label production, and must not be pooled with C0-C4 label-set rates without that caveat.",
            f"For LLaVA, original label-set C3 is {fmt_pct(llava_rate('REF_C3_original_label_set'))}, while matched free color-word C3 is {fmt_pct(llava_rate('free_c3_presupposition'))}, multiple-choice C3 is {fmt_pct(llava_rate('multiple_choice_c3_presupposition'))}, and yes/no false-claim acceptance is {fmt_pct(llava_rate('yesno_false_claim'))}. The original open label-set framing therefore appears to amplify the observed shift.",
            "Qwen and InternVL2 remain near zero across most formal answer formats, so answer format does not create a broad cross-model conflict-following effect in this diagnostic.",
            "Use A1/A2 only as auxiliary stress tests; they are not formal answer-format-control cells.",
        ]
    else:
        mt = {
            (row["model_key"], row["factor_id"]): float(row["conflict_aligned_rate"])
            for row in metrics.to_dict("records")
        }
        interpretation = [
            "The multi-turn module is an extension diagnostic. It tests whether short context accumulation increases conflict following while keeping the final question neutral in MT2/MT3.",
            f"LLaVA and Qwen do not show a meaningful monotonic persuasion effect in this compact setting: LLaVA MT1/MT2/MT3 are {fmt_pct(mt.get(('llava15_7b', 'single_turn_false_note'), 0.0))}, {fmt_pct(mt.get(('llava15_7b', 'two_turn_persuasion'), 0.0))}, and {fmt_pct(mt.get(('llava15_7b', 'three_turn_persuasion'), 0.0))}; Qwen remains at or below {fmt_pct(max(mt.get(('qwen2vl7b', 'single_turn_false_note'), 0.0), mt.get(('qwen2vl7b', 'two_turn_persuasion'), 0.0), mt.get(('qwen2vl7b', 'three_turn_persuasion'), 0.0)))}.",
            f"InternVL2 is the clear exception: MT2 reaches {fmt_pct(mt.get(('internvl2_8b', 'two_turn_persuasion'), 0.0))} and MT3 reaches {fmt_pct(mt.get(('internvl2_8b', 'three_turn_persuasion'), 0.0))}, despite zero conflict following in the original single-turn C0-C4 mainline.",
            "This changes the boundary, not the mainline: multi-turn context accumulation can create a strong model-specific vulnerability, but it should remain an appendix/extension result rather than replacing the frozen single-turn C0-C4 evidence chain.",
        ]

    lines = [
        f"# {title}",
        "",
        "## All-Model Readout",
        "",
        table_md(overview, list(overview.columns)) if not overview.empty else "_No new phase-2 cells available._",
        "",
        "## LLaVA-1.5-7B Focus Metrics",
        "",
        table_md(llava_display, list(llava_display.columns)),
        "",
        "## Key Paired Tests",
        "",
        *sig_lines,
        "",
        "## Interpretation",
        "",
        *[f"- {text}" for text in interpretation],
    ]
    write_markdown(output_path, "\n".join(lines))


def analyze_family(family: str, config: dict, parsed_paths: list[Path]) -> dict[str, object]:
    result_dir = FAMILY_RESULTS_DIR[family]
    result_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_bool_results(path) for path in parsed_paths if path.exists()]
    if frames:
        new_df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        new_df = pd.DataFrame()
    combined = add_reference_rows(family, config, new_df)
    combined = combined.sort_values(["model_key", "factor_id", "image_id"]).reset_index(drop=True)
    metrics = summarize_metrics(combined)
    tests = build_key_tests(combined, family)

    prefix = FAMILY_PREFIX[family]
    combined.to_csv(result_dir / f"{prefix}_combined_parsed_results.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(result_dir / f"{prefix}_metrics.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(result_dir / f"{prefix}_key_tests.csv", index=False, encoding="utf-8-sig")
    figure_name = {
        "factorization": "figure_factorized_prompt_effects.png",
        "format_control": "figure_format_control.png",
        "multiturn": "figure_multiturn_effects.png",
    }[family]
    figure_path = result_dir / figure_name
    make_figure(metrics, family, figure_path)
    summary_path = result_dir / f"{prefix}_summary.md"
    write_family_summary(metrics, tests, family, summary_path)
    payload = {
        "combined": str(result_dir / f"{prefix}_combined_parsed_results.csv"),
        "metrics": str(result_dir / f"{prefix}_metrics.csv"),
        "key_tests": str(result_dir / f"{prefix}_key_tests.csv"),
        "summary": str(summary_path),
        "figure": str(figure_path),
        "rows": int(len(combined)),
    }
    dump_json(result_dir / f"{prefix}_summary.json", payload)
    return payload


def parsed_paths_for(family: str, model_keys: list[str]) -> list[Path]:
    result_dir = FAMILY_RESULTS_DIR[family]
    prefix = FAMILY_PREFIX[family]
    return [model_raw_dir(result_dir, model_key) / f"{prefix}_parsed_results.csv" for model_key in model_keys]


def main() -> int:
    args = parse_args()
    config = load_paper_config(args.config)
    model_keys = selected_model_keys(config, args.models)
    prompt_csv = prompt_csv_for(args.family)
    result_dir = FAMILY_RESULTS_DIR[args.family]
    result_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build_prompts and not args.analyze_only:
        prompt_count = build_prompt_csv(args.family, config, prompt_csv)
    else:
        prompt_count = sum(1 for _ in prompt_csv.open("r", encoding="utf-8-sig")) - 1 if prompt_csv.exists() else 0

    parsed_paths = parsed_paths_for(args.family, model_keys)
    if not args.skip_run and not args.analyze_only:
        for model_key in model_keys:
            output_dir = model_raw_dir(result_dir, model_key)
            parsed_path = run_and_parse_prompt_set(
                config=config,
                model_key=model_key,
                prompt_csv=prompt_csv,
                output_dir=output_dir,
                prefix=FAMILY_PREFIX[args.family],
                limit=args.limit,
            )
            if parsed_path not in parsed_paths:
                parsed_paths.append(parsed_path)

    analysis_payload = analyze_family(args.family, config, parsed_paths)
    payload = {
        "family": args.family,
        "prompt_csv": str(prompt_csv),
        "prompt_rows": int(prompt_count),
        "models": model_keys,
        "limit": args.limit,
        "analysis": analysis_payload,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
