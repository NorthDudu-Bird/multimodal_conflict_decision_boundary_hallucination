#!/usr/bin/env python
"""Paper-focused proportion analysis for baseline, main, and auxiliary experiments."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from scripts.utils.paper_mainline_utils import (
    AUXILIARY_CONDITION_ORDER,
    MODEL_ORDER,
    PRIMARY_CONDITION_ORDER,
    dump_json,
    format_ci,
    format_pct,
    load_bool_results,
    load_paper_config,
    write_markdown,
)


METRIC_FIELDS = [
    ("conflict_aligned", "is_conflict_aligned"),
    ("faithful", "is_faithful"),
    ("refusal", "is_refusal_or_correction"),
    ("other_wrong", "is_other_wrong"),
    ("parse_error", "is_parse_error"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paper experiment results.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=["baseline", "main", "aux"], required=True)
    parser.add_argument("--input-csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    low, high = proportion_confint(successes, total, method="wilson")
    return float(low), float(high)


def load_combined_df(paths: list[Path]) -> pd.DataFrame:
    frames = [load_bool_results(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    for field in METRIC_FIELDS:
        _, column = field
        if column not in df.columns:
            df[column] = False
    if "in_allowed_answer_space" not in df.columns:
        df["in_allowed_answer_space"] = False
    return df


def condition_order_for_mode(mode: str) -> list[str]:
    if mode == "baseline":
        return ["C0_neutral"]
    if mode == "main":
        return PRIMARY_CONDITION_ORDER
    return AUXILIARY_CONDITION_ORDER


def summarize_condition_metrics(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    condition_order = condition_order_for_mode(mode)
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key]
        if model_df.empty:
            continue
        model_name = model_df["model_name"].iloc[0]
        checkpoint_name = model_df["checkpoint_name"].iloc[0]
        for condition_name in condition_order:
            subset = model_df[model_df["condition_name"] == condition_name].copy()
            total = len(subset)
            row: dict[str, object] = {
                "model_key": model_key,
                "model_name": model_name,
                "checkpoint_name": checkpoint_name,
                "condition_name": condition_name,
                "n": int(total),
            }
            for metric_name, column in METRIC_FIELDS:
                count = int(subset[column].sum())
                rate = count / total if total else 0.0
                ci_low, ci_high = wilson_interval(count, total)
                row[f"{metric_name}_n"] = count
                row[f"{metric_name}_rate"] = rate
                row[f"{metric_name}_ci_low"] = ci_low
                row[f"{metric_name}_ci_high"] = ci_high
            if mode == "aux":
                compliance_n = int(subset["in_allowed_answer_space"].sum())
                compliance_rate = compliance_n / total if total else 0.0
                ci_low, ci_high = wilson_interval(compliance_n, total)
                row["answer_space_compliance_n"] = compliance_n
                row["answer_space_compliance_rate"] = compliance_rate
                row["answer_space_compliance_ci_low"] = ci_low
                row["answer_space_compliance_ci_high"] = ci_high
            rows.append(row)
    return pd.DataFrame(rows)


def paired_exact_test(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    outcome_column: str,
    id_columns: list[str],
) -> dict[str, object] | None:
    merged = left_df[id_columns + [outcome_column]].merge(
        right_df[id_columns + [outcome_column]],
        on=id_columns,
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return None
    left_values = merged[f"{outcome_column}_left"].astype(bool)
    right_values = merged[f"{outcome_column}_right"].astype(bool)
    both_yes = int((left_values & right_values).sum())
    left_only = int((left_values & ~right_values).sum())
    right_only = int((~left_values & right_values).sum())
    both_no = int((~left_values & ~right_values).sum())
    table = [[both_yes, left_only], [right_only, both_no]]
    result = mcnemar(table, exact=True)
    return {
        "n_pairs": int(len(merged)),
        "left_rate": float(left_values.mean()),
        "right_rate": float(right_values.mean()),
        "rate_diff_left_minus_right": float(left_values.mean() - right_values.mean()),
        "both_yes": both_yes,
        "left_only": left_only,
        "right_only": right_only,
        "both_no": both_no,
        "p_value_exact_mcnemar": float(result.pvalue),
    }


def build_stat_tests(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if mode == "baseline":
        return pd.DataFrame(rows)

    if mode == "main":
        for model_key in MODEL_ORDER:
            model_df = df[df["model_key"] == model_key]
            baseline_df = model_df[model_df["condition_name"] == "C0_neutral"]
            for condition_name in PRIMARY_CONDITION_ORDER[1:]:
                current_df = model_df[model_df["condition_name"] == condition_name]
                for metric_label, column in [("conflict_aligned", "is_conflict_aligned"), ("faithful", "is_faithful")]:
                    payload = paired_exact_test(
                        current_df,
                        baseline_df,
                        outcome_column=column,
                        id_columns=["image_id"],
                    )
                    if payload is None:
                        continue
                    rows.append(
                        {
                            "comparison_type": "within_model_vs_C0",
                            "model_key": model_key,
                            "condition_name": condition_name,
                            "reference_condition": "C0_neutral",
                            "metric": metric_label,
                            **payload,
                        }
                    )
        for condition_name in PRIMARY_CONDITION_ORDER:
            condition_df = df[df["condition_name"] == condition_name]
            for left_model, right_model in combinations(MODEL_ORDER, 2):
                left_df = condition_df[condition_df["model_key"] == left_model]
                right_df = condition_df[condition_df["model_key"] == right_model]
                for metric_label, column in [("conflict_aligned", "is_conflict_aligned"), ("faithful", "is_faithful")]:
                    payload = paired_exact_test(
                        left_df,
                        right_df,
                        outcome_column=column,
                        id_columns=["image_id"],
                    )
                    if payload is None:
                        continue
                    rows.append(
                        {
                            "comparison_type": "cross_model_same_condition",
                            "left_model": left_model,
                            "right_model": right_model,
                            "condition_name": condition_name,
                            "metric": metric_label,
                            **payload,
                        }
                    )
    else:
        for model_key in MODEL_ORDER:
            model_df = df[df["model_key"] == model_key]
            a1_df = model_df[model_df["condition_name"] == "A1_forced_choice_red_family"]
            a2_df = model_df[model_df["condition_name"] == "A2_counterfactual_assumption"]
            for metric_label, column in [
                ("conflict_aligned", "is_conflict_aligned"),
                ("answer_space_compliance", "in_allowed_answer_space"),
            ]:
                payload = paired_exact_test(
                    a2_df,
                    a1_df,
                    outcome_column=column,
                    id_columns=["image_id"],
                )
                if payload is None:
                    continue
                rows.append(
                    {
                        "comparison_type": "within_model_A2_vs_A1",
                        "model_key": model_key,
                        "condition_name": "A2_counterfactual_assumption",
                        "reference_condition": "A1_forced_choice_red_family",
                        "metric": metric_label,
                        **payload,
                    }
                )
        for condition_name in AUXILIARY_CONDITION_ORDER:
            condition_df = df[df["condition_name"] == condition_name]
            for left_model, right_model in combinations(MODEL_ORDER, 2):
                left_df = condition_df[condition_df["model_key"] == left_model]
                right_df = condition_df[condition_df["model_key"] == right_model]
                for metric_label, column in [
                    ("conflict_aligned", "is_conflict_aligned"),
                    ("answer_space_compliance", "in_allowed_answer_space"),
                ]:
                    payload = paired_exact_test(
                        left_df,
                        right_df,
                        outcome_column=column,
                        id_columns=["image_id"],
                    )
                    if payload is None:
                        continue
                    rows.append(
                        {
                            "comparison_type": "cross_model_same_condition",
                            "left_model": left_model,
                            "right_model": right_model,
                            "condition_name": condition_name,
                            "metric": metric_label,
                            **payload,
                        }
                    )
    return pd.DataFrame(rows)


def build_representative_cases(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "baseline":
        case_df = df[df["is_other_wrong"] | df["is_refusal_or_correction"] | df["is_parse_error"]].copy()
        case_df["case_bucket"] = np.where(case_df["is_faithful"], "faithful", "baseline_exception")
        return case_df

    frames: list[pd.DataFrame] = []
    if mode == "main":
        conflict_cases = df[df["is_conflict_aligned"]].copy()
        conflict_cases["case_bucket"] = "main_conflict_aligned"
        frames.append(conflict_cases)
    if mode == "aux":
        high_compliance = df[df["in_allowed_answer_space"]].copy()
        high_compliance["case_bucket"] = "aux_in_answer_space"
        frames.append(high_compliance)

    disagreement_keys = (
        df.groupby(["image_id", "condition_name"], observed=False)["parsed_label"]
        .nunique()
        .reset_index(name="unique_labels")
    )
    disagreement_keys = disagreement_keys[disagreement_keys["unique_labels"] > 1]
    if not disagreement_keys.empty:
        disagreement_cases = df.merge(disagreement_keys, on=["image_id", "condition_name"], how="inner")
        disagreement_cases["case_bucket"] = "cross_model_disagreement"
        frames.append(disagreement_cases)

    refusal_cases = df[df["is_refusal_or_correction"] | df["is_parse_error"]].copy()
    if not refusal_cases.empty:
        refusal_cases["case_bucket"] = "refusal_or_parse_issue"
        frames.append(refusal_cases)

    if not frames:
        return pd.DataFrame(columns=df.columns.tolist() + ["case_bucket"])
    case_df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["sample_id", "model_key", "case_bucket"]
    )
    keep_columns = [
        "case_bucket",
        "sample_id",
        "image_id",
        "condition_name",
        "model_key",
        "true_color",
        "conflict_color",
        "source_dataset",
        "parsed_label",
        "raw_output",
        "outcome_type",
        "notes",
    ]
    return case_df[[column for column in keep_columns if column in case_df.columns]]


def write_summary_markdown(mode: str, metrics_df: pd.DataFrame, tests_df: pd.DataFrame, output_path: Path) -> None:
    title = {
        "baseline": "C0 Baseline Summary",
        "main": "C0-C4 Main Experiment Summary",
        "aux": "A1/A2 Auxiliary Experiment Summary",
    }[mode]
    lines = [f"# {title}", ""]
    for row in metrics_df.to_dict(orient="records"):
        summary = (
            f"- {row['model_key']} | {row['condition_name']}: "
            f"faithful={format_pct(row['faithful_rate'])} {format_ci(row['faithful_ci_low'], row['faithful_ci_high'])}; "
            f"conflict_aligned={format_pct(row['conflict_aligned_rate'])} {format_ci(row['conflict_aligned_ci_low'], row['conflict_aligned_ci_high'])}; "
            f"refusal={format_pct(row['refusal_rate'])} {format_ci(row['refusal_ci_low'], row['refusal_ci_high'])}; "
            f"n={row['n']}"
        )
        if mode == "aux":
            summary += (
                f"; compliance={format_pct(row['answer_space_compliance_rate'])} "
                f"{format_ci(row['answer_space_compliance_ci_low'], row['answer_space_compliance_ci_high'])}"
            )
        lines.append(summary)
    if not tests_df.empty:
        lines.extend(["", "## Exact Paired Proportion Tests"])
        for row in tests_df.to_dict(orient="records"):
            if row["comparison_type"] == "cross_model_same_condition":
                lines.append(
                    f"- cross_model_same_condition | {row['metric']} | condition={row['condition_name']} | "
                    f"{row['left_model']} minus {row['right_model']}: "
                    f"diff={row['rate_diff_left_minus_right']:.4f}, "
                    f"p_exact={row['p_value_exact_mcnemar']:.6f}, "
                    f"discordant=({row['left_only']}, {row['right_only']})"
                )
            else:
                lines.append(
                    f"- {row['comparison_type']} | {row['model_key']} | {row['metric']} | "
                    f"{row['condition_name']} minus {row['reference_condition']}: "
                    f"diff={row['rate_diff_left_minus_right']:.4f}, "
                    f"p_exact={row['p_value_exact_mcnemar']:.6f}, "
                    f"discordant=({row['left_only']}, {row['right_only']})"
                )
    write_markdown(output_path, "\n".join(lines))


def main() -> int:
    args = parse_args()
    config = load_paper_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_df = load_combined_df(args.input_csvs)
    condition_order = condition_order_for_mode(args.mode)
    combined_df = combined_df[combined_df["condition_name"].isin(condition_order)].copy()
    combined_df["condition_name"] = pd.Categorical(combined_df["condition_name"], categories=condition_order, ordered=True)
    combined_df["model_key"] = pd.Categorical(combined_df["model_key"], categories=MODEL_ORDER, ordered=True)
    combined_df = combined_df.sort_values(["model_key", "condition_name", "image_id"]).reset_index(drop=True)

    combined_path = output_dir / f"{args.mode}_combined_parsed_results.csv"
    combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

    metrics_df = summarize_condition_metrics(combined_df, args.mode)
    metrics_path = output_dir / f"{args.mode}_condition_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    tests_df = build_stat_tests(combined_df, args.mode)
    tests_path = output_dir / f"{args.mode}_exact_tests.csv"
    tests_df.to_csv(tests_path, index=False, encoding="utf-8-sig")

    cases_df = build_representative_cases(combined_df, args.mode)
    cases_path = output_dir / f"{args.mode}_representative_cases.csv"
    cases_df.to_csv(cases_path, index=False, encoding="utf-8-sig")

    summary_md = output_dir / f"{args.mode}_summary.md"
    write_summary_markdown(args.mode, metrics_df, tests_df, summary_md)

    summary_payload = {
        "mode": args.mode,
        "combined_csv": str(combined_path),
        "metrics_csv": str(metrics_path),
        "tests_csv": str(tests_path),
        "cases_csv": str(cases_path),
        "summary_md": str(summary_md),
        "rows": int(len(combined_df)),
    }
    dump_json(output_dir / f"{args.mode}_summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
