#!/usr/bin/env python
"""Analyze C3 prompt-variant robustness results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from scripts.utils.paper_mainline_utils import (
    MODEL_ORDER,
    ROBUSTNESS_VARIANT_ORDER,
    dump_json,
    format_ci,
    format_pct,
    load_bool_results,
    write_markdown,
)


MODEL_DISPLAY_ORDER = {
    "qwen2vl7b": "Qwen2-VL-7B-Instruct",
    "llava15_7b": "LLaVA-1.5-7B",
    "internvl2_8b": "InternVL2-8B",
}
ROBUSTNESS_VARIANT_LABELS = {
    "C3_original": "Original C3",
    "C3_v2": "C3-v2",
    "C3_v3": "C3-v3",
}
METRIC_FIELDS = [
    ("conflict_aligned", "is_conflict_aligned"),
    ("faithful", "is_faithful"),
    ("refusal", "is_refusal_or_correction"),
    ("other_wrong", "is_other_wrong"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze C3 prompt-variant robustness runs.")
    parser.add_argument("--input-csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-main-csv", type=Path, required=True)
    return parser.parse_args()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    low, high = proportion_confint(successes, total, method="wilson")
    return float(low), float(high)


def load_combined_df(paths: list[Path]) -> pd.DataFrame:
    frames = [load_bool_results(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    for _, column in METRIC_FIELDS:
        if column not in df.columns:
            df[column] = False
    return df


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key]
        if model_df.empty:
            continue
        model_name = model_df["model_name"].iloc[0]
        checkpoint_name = model_df["checkpoint_name"].iloc[0]
        for variant_name in ROBUSTNESS_VARIANT_ORDER:
            subset = model_df[model_df["robustness_variant"] == variant_name].copy()
            total = len(subset)
            row: dict[str, object] = {
                "model_key": model_key,
                "model_name": model_name,
                "checkpoint_name": checkpoint_name,
                "robustness_variant": variant_name,
                "n": int(total),
            }
            for metric_name, column in METRIC_FIELDS:
                count = int(subset[column].sum())
                rate = count / total if total else 0.0
                low, high = wilson_interval(count, total)
                row[f"{metric_name}_n"] = count
                row[f"{metric_name}_rate"] = rate
                row[f"{metric_name}_ci_low"] = low
                row[f"{metric_name}_ci_high"] = high
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
    result = mcnemar([[both_yes, left_only], [right_only, both_no]], exact=True)
    return {
        "n_pairs": int(len(merged)),
        "left_rate": float(left_values.mean()),
        "right_rate": float(right_values.mean()),
        "rate_diff_left_minus_right": float(left_values.mean() - right_values.mean()),
        "left_only": left_only,
        "right_only": right_only,
        "p_value_raw": float(result.pvalue),
    }


def build_exact_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key]
        original_df = model_df[model_df["robustness_variant"] == "C3_original"]
        for variant_name in ["C3_v2", "C3_v3"]:
            current_df = model_df[model_df["robustness_variant"] == variant_name]
            payload = paired_exact_test(
                current_df,
                original_df,
                outcome_column="is_conflict_aligned",
                id_columns=["image_id"],
            )
            if payload is None:
                continue
            rows.append(
                {
                    "comparison_family": "within_model_vs_original",
                    "comparison_label": f"{MODEL_DISPLAY_ORDER[model_key]}: Original C3 vs {ROBUSTNESS_VARIANT_LABELS[variant_name]}",
                    "model_key": model_key,
                    "left_model": model_key,
                    "right_model": model_key,
                    "left_variant": variant_name,
                    "right_variant": "C3_original",
                    "higher_rate_model": model_key,
                    "higher_rate_variant": variant_name if payload["left_rate"] > payload["right_rate"] else "C3_original",
                    **payload,
                }
            )

    cross_pairs = [("llava15_7b", "qwen2vl7b"), ("llava15_7b", "internvl2_8b")]
    for variant_name in ROBUSTNESS_VARIANT_ORDER:
        variant_df = df[df["robustness_variant"] == variant_name]
        for left_model, right_model in cross_pairs:
            left_df = variant_df[variant_df["model_key"] == left_model]
            right_df = variant_df[variant_df["model_key"] == right_model]
            payload = paired_exact_test(
                left_df,
                right_df,
                outcome_column="is_conflict_aligned",
                id_columns=["image_id"],
            )
            if payload is None:
                continue
            rows.append(
                {
                    "comparison_family": "cross_model_same_variant",
                    "comparison_label": f"{ROBUSTNESS_VARIANT_LABELS[variant_name]}: {MODEL_DISPLAY_ORDER[left_model]} vs {MODEL_DISPLAY_ORDER[right_model]}",
                    "model_key": "",
                    "left_model": left_model,
                    "right_model": right_model,
                    "left_variant": variant_name,
                    "right_variant": variant_name,
                    "higher_rate_model": left_model if payload["left_rate"] > payload["right_rate"] else right_model,
                    "higher_rate_variant": variant_name,
                    **payload,
                }
            )

    tests_df = pd.DataFrame(rows)
    if tests_df.empty:
        return tests_df
    tests_df["p_value_holm"] = np.nan
    tests_df["significant_holm"] = False
    for family_name in ["within_model_vs_original", "cross_model_same_variant"]:
        family_mask = tests_df["comparison_family"] == family_name
        family_values = tests_df.loc[family_mask, "p_value_raw"].to_numpy(dtype=float)
        _, corrected, _, _ = multipletests(family_values, method="holm")
        tests_df.loc[family_mask, "p_value_holm"] = corrected
        tests_df.loc[family_mask, "significant_holm"] = corrected < 0.05
    return tests_df


def format_pvalue(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def build_llava_vs_c0_checks(robustness_df: pd.DataFrame, reference_main_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    llava_c0 = reference_main_df[
        (reference_main_df["model_key"] == "llava15_7b") & (reference_main_df["condition_name"] == "C0_neutral")
    ].copy()
    for variant_name in ROBUSTNESS_VARIANT_ORDER:
        variant_df = robustness_df[
            (robustness_df["model_key"] == "llava15_7b") & (robustness_df["robustness_variant"] == variant_name)
        ].copy()
        payload = paired_exact_test(
            variant_df,
            llava_c0,
            outcome_column="is_conflict_aligned",
            id_columns=["image_id"],
        )
        if payload is None:
            continue
        rows.append(
            {
                "variant": variant_name,
                "comparison_label": f"{ROBUSTNESS_VARIANT_LABELS[variant_name]} vs C0",
                **payload,
            }
        )
    checks_df = pd.DataFrame(rows)
    if checks_df.empty:
        return checks_df
    _, corrected, _, _ = multipletests(checks_df["p_value_raw"].to_numpy(dtype=float), method="holm")
    checks_df["p_value_holm"] = corrected
    checks_df["significant_holm"] = corrected < 0.05
    return checks_df


def robustness_conclusion(metrics_df: pd.DataFrame, exact_tests_df: pd.DataFrame, llava_vs_c0_df: pd.DataFrame) -> tuple[str, str]:
    llava_metrics = metrics_df[metrics_df["model_key"] == "llava15_7b"].set_index("robustness_variant")
    qwen_metrics = metrics_df[metrics_df["model_key"] == "qwen2vl7b"].set_index("robustness_variant")
    intern_metrics = metrics_df[metrics_df["model_key"] == "internvl2_8b"].set_index("robustness_variant")
    llava_higher_all = True
    for variant_name in ROBUSTNESS_VARIANT_ORDER:
        llava_rate = float(llava_metrics.loc[variant_name, "conflict_aligned_rate"])
        stable_max = max(
            float(qwen_metrics.loc[variant_name, "conflict_aligned_rate"]),
            float(intern_metrics.loc[variant_name, "conflict_aligned_rate"]),
        )
        if llava_rate <= stable_max:
            llava_higher_all = False
            break

    significant_cross = exact_tests_df[
        (exact_tests_df["comparison_family"] == "cross_model_same_variant") & (exact_tests_df["significant_holm"])
    ]
    cross_by_variant = significant_cross.groupby("left_variant", observed=False).size().to_dict()
    llava_sig_all_variants = all(cross_by_variant.get(variant_name, 0) == 2 for variant_name in ROBUSTNESS_VARIANT_ORDER)
    llava_sig_vs_c0_count = int(llava_vs_c0_df["significant_holm"].sum()) if not llava_vs_c0_df.empty else 0

    if llava_higher_all and llava_sig_all_variants and llava_sig_vs_c0_count >= 2:
        return (
            "对 tested C3-style wording 基本稳健",
            "LLaVA 在三个 tested C3 wording 下都保持高于两个稳定模型，且多数 wording 相对 `C0` 仍显著偏离。",
        )
    if llava_higher_all:
        return (
            "对 prompt wording 部分稳健，强度依赖具体措辞",
            "效应方向保留，但显著性或幅度随具体 wording 波动，正文应避免把该现象写成完全模板无关。",
        )
    return (
        "当前现象对原模板敏感，不应写成稳定规律",
        "至少一个新 wording 下，LLaVA 不再稳定高于两个视觉稳定模型，因此结论需要降级为模板依赖现象。",
    )


def write_summary_markdown(
    metrics_df: pd.DataFrame,
    exact_tests_df: pd.DataFrame,
    llava_vs_c0_df: pd.DataFrame,
    output_path: Path,
) -> None:
    conclusion_title, conclusion_text = robustness_conclusion(metrics_df, exact_tests_df, llava_vs_c0_df)
    lines = [
        "# Prompt Variant Robustness Summary",
        "",
        "## Metrics",
    ]
    for row in metrics_df.to_dict(orient="records"):
        lines.append(
            f"- {MODEL_DISPLAY_ORDER[row['model_key']]} | {ROBUSTNESS_VARIANT_LABELS[row['robustness_variant']]}: "
            f"conflict_aligned={format_pct(row['conflict_aligned_rate'])} {format_ci(row['conflict_aligned_ci_low'], row['conflict_aligned_ci_high'])}; "
            f"faithful={format_pct(row['faithful_rate'])} {format_ci(row['faithful_ci_low'], row['faithful_ci_high'])}; "
            f"refusal={format_pct(row['refusal_rate'])} {format_ci(row['refusal_ci_low'], row['refusal_ci_high'])}; "
            f"other_wrong={format_pct(row['other_wrong_rate'])} {format_ci(row['other_wrong_ci_low'], row['other_wrong_ci_high'])}; "
            f"n={row['n']}"
        )

    lines.extend(["", "## Exact Tests", ""])
    if exact_tests_df.empty:
        lines.append("- No prompt-variant tests were available.")
    else:
        display_df = exact_tests_df.copy()
        display_df["left_rate"] = display_df["left_rate"].map(format_pct)
        display_df["right_rate"] = display_df["right_rate"].map(format_pct)
        display_df["rate_diff_left_minus_right"] = display_df["rate_diff_left_minus_right"].map(
            lambda value: f"{value * 100:.2f} pp"
        )
        display_df["p_value_raw"] = display_df["p_value_raw"].map(format_pvalue)
        display_df["p_value_holm"] = display_df["p_value_holm"].map(format_pvalue)
        display_df["significant_holm"] = display_df["significant_holm"].map(lambda value: "yes" if value else "no")
        table_df = display_df[
            [
                "comparison_family",
                "comparison_label",
                "left_rate",
                "right_rate",
                "rate_diff_left_minus_right",
                "left_only",
                "right_only",
                "p_value_raw",
                "p_value_holm",
                "significant_holm",
            ]
        ].rename(
            columns={
                "comparison_family": "family",
                "comparison_label": "comparison",
                "left_rate": "left rate",
                "right_rate": "right rate",
                "rate_diff_left_minus_right": "diff",
                "left_only": "left-only",
                "right_only": "right-only",
                "p_value_raw": "raw p",
                "p_value_holm": "Holm p",
                "significant_holm": "Holm sig",
            }
        )
        header = "| " + " | ".join(table_df.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
        lines.append(header)
        lines.append(divider)
        for row in table_df.to_dict(orient="records"):
            lines.append("| " + " | ".join(str(row[column]) for column in table_df.columns) + " |")

    lines.extend(["", "## LLaVA Variant vs C0 Check", ""])
    if llava_vs_c0_df.empty:
        lines.append("- No `C0` reference checks were available.")
    else:
        for row in llava_vs_c0_df.to_dict(orient="records"):
            lines.append(
                f"- {row['comparison_label']}: diff={row['rate_diff_left_minus_right'] * 100:.2f} pp, "
                f"raw p={format_pvalue(row['p_value_raw'])}, Holm p={format_pvalue(row['p_value_holm'])}, "
                f"Holm significant={'yes' if row['significant_holm'] else 'no'}."
            )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- 结论：{conclusion_title}。",
            f"- 解释：{conclusion_text}",
        ]
    )
    write_markdown(output_path, "\n".join(lines))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    new_variant_df = load_combined_df(args.input_csvs)
    new_variant_df = new_variant_df[
        (new_variant_df["condition_name"] == "C3_presupposition_correction_allowed")
        & (new_variant_df["robustness_variant"].isin(["C3_v2", "C3_v3"]))
    ].copy()
    reference_main_df = load_bool_results(args.reference_main_csv)
    original_df = reference_main_df[
        reference_main_df["condition_name"] == "C3_presupposition_correction_allowed"
    ].copy()
    original_df["robustness_variant"] = "C3_original"
    original_df["robustness_variant_label"] = "C3_original"

    combined_df = pd.concat([original_df, new_variant_df], ignore_index=True, sort=False)
    combined_df["robustness_variant"] = pd.Categorical(
        combined_df["robustness_variant"],
        categories=ROBUSTNESS_VARIANT_ORDER,
        ordered=True,
    )
    combined_df["model_key"] = pd.Categorical(combined_df["model_key"], categories=MODEL_ORDER, ordered=True)
    combined_df = combined_df.sort_values(["model_key", "robustness_variant", "image_id"]).reset_index(drop=True)

    combined_path = output_dir / "prompt_variant_combined_parsed_results.csv"
    combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

    metrics_df = summarize_metrics(combined_df)
    metrics_df.to_csv(output_dir / "prompt_variant_metrics.csv", index=False, encoding="utf-8-sig")

    exact_tests_df = build_exact_tests(combined_df)
    exact_tests_df.to_csv(output_dir / "prompt_variant_exact_tests.csv", index=False, encoding="utf-8-sig")

    llava_vs_c0_df = build_llava_vs_c0_checks(combined_df, reference_main_df)

    summary_path = output_dir / "prompt_variant_summary.md"
    write_summary_markdown(metrics_df, exact_tests_df, llava_vs_c0_df, summary_path)

    payload = {
        "combined_csv": str(combined_path),
        "metrics_csv": str(output_dir / "prompt_variant_metrics.csv"),
        "tests_csv": str(output_dir / "prompt_variant_exact_tests.csv"),
        "summary_md": str(summary_path),
        "rows": int(len(combined_df)),
    }
    dump_json(output_dir / "prompt_variant_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
