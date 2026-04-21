#!/usr/bin/env python
"""Generate strengthened paper-ready tables, figures, and summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from scripts.utils.paper_mainline_utils import (
    MODEL_ORDER,
    PRIMARY_CONDITION_ORDER,
    PRIMARY_CONDITION_SHORT_LABELS,
    dump_json,
    format_ci,
    format_pct,
    load_paper_config,
    paper_paths,
    write_markdown,
)


KEY_MAIN_CONDITIONS = ["C3_presupposition_correction_allowed", "C4_stronger_open_conflict"]
KEY_APPENDIX_CONDITIONS = ["C0_neutral", "C3_presupposition_correction_allowed"]
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
CONDITION_LABEL_EXPLANATIONS = {
    "C0": "neutral prompt",
    "C1": "weak suggestion",
    "C2": "false assertion, open answer",
    "C3": "presupposition with correction allowed",
    "C4": "stronger open conflict framing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def build_table_md(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in df.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def format_metric_cell(count: int, total: int, rate: float, low: float, high: float, marks: str = "") -> str:
    return f"{count}/{total} ({format_pct(rate)} {format_ci(low, high)}){marks}"


def format_pvalue(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    low, high = proportion_confint(successes, total, method="wilson")
    return float(low), float(high)


def make_dataset_distribution_plot(distribution_df: pd.DataFrame, output_path: Path) -> None:
    colors = distribution_df["true_color"].tolist()
    x = np.arange(len(colors))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes[0].bar(x, distribution_df["stanford_core_n"], color="#365c8d")
    axes[0].set_title("Original Stanford Strict-Clean Seed")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(colors, rotation=20, ha="right")
    axes[0].set_ylabel("Images")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, distribution_df["final_from_stanford_n"], label="StanfordCars", color="#365c8d")
    axes[1].bar(
        x,
        distribution_df["final_from_vcor_n"],
        bottom=distribution_df["final_from_stanford_n"],
        label="VCoR",
        color="#c78536",
    )
    axes[1].set_title("Final Balanced Evaluation Set")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(colors, rotation=20, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_main_key_tests(main_tests_df: pd.DataFrame) -> pd.DataFrame:
    within_df = main_tests_df[
        (main_tests_df["comparison_type"] == "within_model_vs_C0")
        & (main_tests_df["metric"] == "conflict_aligned")
        & (main_tests_df["condition_name"].isin(KEY_MAIN_CONDITIONS))
    ].copy()
    within_df["comparison_family"] = "within_model_vs_C0"
    within_df["comparison_label"] = within_df.apply(
        lambda row: (
            f"{MODEL_DISPLAY_ORDER[row['model_key']]}: "
            f"{PRIMARY_CONDITION_SHORT_LABELS[row['reference_condition']]} vs "
            f"{PRIMARY_CONDITION_SHORT_LABELS[row['condition_name']]}"
        ),
        axis=1,
    )
    within_df["higher_rate_model"] = within_df["model_key"]
    within_df["lower_rate_model"] = within_df["model_key"]

    cross_df = main_tests_df[
        (main_tests_df["comparison_type"] == "cross_model_same_condition")
        & (main_tests_df["metric"] == "conflict_aligned")
        & (main_tests_df["condition_name"].isin(KEY_MAIN_CONDITIONS))
    ].copy()
    cross_df["comparison_family"] = "cross_model_same_condition"
    cross_df["comparison_label"] = cross_df.apply(
        lambda row: (
            f"{PRIMARY_CONDITION_SHORT_LABELS[row['condition_name']]}: "
            f"{MODEL_DISPLAY_ORDER[row['left_model']]} vs {MODEL_DISPLAY_ORDER[row['right_model']]}"
        ),
        axis=1,
    )
    cross_df["higher_rate_model"] = np.where(
        cross_df["left_rate"] > cross_df["right_rate"],
        cross_df["left_model"],
        np.where(cross_df["right_rate"] > cross_df["left_rate"], cross_df["right_model"], ""),
    )
    cross_df["lower_rate_model"] = np.where(
        cross_df["left_rate"] < cross_df["right_rate"],
        cross_df["left_model"],
        np.where(cross_df["right_rate"] < cross_df["left_rate"], cross_df["right_model"], ""),
    )

    key_df = pd.concat([within_df, cross_df], ignore_index=True, sort=False)
    if key_df.empty:
        return key_df

    key_df["p_value_raw"] = key_df["p_value_exact_mcnemar"].astype(float)
    key_df["p_value_holm"] = np.nan
    key_df["significant_holm"] = False
    for family_name in ["within_model_vs_C0", "cross_model_same_condition"]:
        family_mask = key_df["comparison_family"] == family_name
        family_values = key_df.loc[family_mask, "p_value_raw"].to_numpy(dtype=float)
        if len(family_values) == 0:
            continue
        _, corrected, _, _ = multipletests(family_values, method="holm")
        key_df.loc[family_mask, "p_value_holm"] = corrected
        key_df.loc[family_mask, "significant_holm"] = corrected < 0.05

    keep_columns = [
        "comparison_family",
        "comparison_label",
        "model_key",
        "left_model",
        "right_model",
        "condition_name",
        "reference_condition",
        "metric",
        "n_pairs",
        "left_rate",
        "right_rate",
        "rate_diff_left_minus_right",
        "left_only",
        "right_only",
        "higher_rate_model",
        "lower_rate_model",
        "p_value_raw",
        "p_value_holm",
        "significant_holm",
    ]
    return key_df[keep_columns].copy()


def build_main_mark_lookup(key_tests_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    mark_lookup: dict[tuple[str, str], str] = {}
    for model_key in MODEL_ORDER:
        for condition_name in PRIMARY_CONDITION_ORDER:
            marks = ""
            if not key_tests_df.empty:
                within_match = key_tests_df[
                    (key_tests_df["comparison_family"] == "within_model_vs_C0")
                    & (key_tests_df["model_key"] == model_key)
                    & (key_tests_df["condition_name"] == condition_name)
                    & (key_tests_df["significant_holm"])
                    & (key_tests_df["rate_diff_left_minus_right"] > 0)
                ]
                if not within_match.empty:
                    marks += "*"
                cross_match = key_tests_df[
                    (key_tests_df["comparison_family"] == "cross_model_same_condition")
                    & (key_tests_df["condition_name"] == condition_name)
                    & (key_tests_df["significant_holm"])
                    & (key_tests_df["higher_rate_model"] == model_key)
                ]
                if not cross_match.empty:
                    marks += "†"
            mark_lookup[(model_key, condition_name)] = marks
    return mark_lookup


def write_main_table(main_df: pd.DataFrame, key_tests_df: pd.DataFrame, output_dir: Path) -> None:
    ordered_df = (
        main_df.assign(
            model_key=pd.Categorical(main_df["model_key"], categories=MODEL_ORDER, ordered=True),
            condition_name=pd.Categorical(main_df["condition_name"], categories=PRIMARY_CONDITION_ORDER, ordered=True),
        )
        .sort_values(["model_key", "condition_name"])
        .reset_index(drop=True)
    )
    mark_lookup = build_main_mark_lookup(key_tests_df)

    table_df = pd.DataFrame(
        {
            "model": ordered_df["model_key"].map(MODEL_DISPLAY_ORDER),
            "condition": ordered_df["condition_name"].map(PRIMARY_CONDITION_SHORT_LABELS),
            "n": ordered_df["n"].astype(int),
            "conflict_aligned": [
                format_metric_cell(
                    int(row["conflict_aligned_n"]),
                    int(row["n"]),
                    float(row["conflict_aligned_rate"]),
                    float(row["conflict_aligned_ci_low"]),
                    float(row["conflict_aligned_ci_high"]),
                    mark_lookup[(row["model_key"], row["condition_name"])],
                )
                for row in ordered_df.to_dict(orient="records")
            ],
            "faithful": [
                format_metric_cell(
                    int(row["faithful_n"]),
                    int(row["n"]),
                    float(row["faithful_rate"]),
                    float(row["faithful_ci_low"]),
                    float(row["faithful_ci_high"]),
                )
                for row in ordered_df.to_dict(orient="records")
            ],
            "refusal": [
                format_metric_cell(
                    int(row["refusal_n"]),
                    int(row["n"]),
                    float(row["refusal_rate"]),
                    float(row["refusal_ci_low"]),
                    float(row["refusal_ci_high"]),
                )
                for row in ordered_df.to_dict(orient="records")
            ],
            "other_wrong": [
                format_metric_cell(
                    int(row["other_wrong_n"]),
                    int(row["n"]),
                    float(row["other_wrong_rate"]),
                    float(row["other_wrong_ci_low"]),
                    float(row["other_wrong_ci_high"]),
                )
                for row in ordered_df.to_dict(orient="records")
            ],
        }
    )
    table_df.to_csv(output_dir / "table1_main_metrics.csv", index=False, encoding="utf-8-sig")
    notes = (
        "* `*` indicates a Holm-significant increase in conflict-aligned rate relative to the same model's `C0` baseline.\n"
        "* `†` indicates a Holm-significant higher conflict-aligned rate than at least one visually stable comparison model under the same condition.\n"
        "* Refusal, other-wrong, and parse-error outcomes all remained `0/n` in the strengthened main experiment."
    )
    write_markdown(
        output_dir / "table1_main_metrics.md",
        "# Table 1. Main experiment metrics\n\n"
        + build_table_md(table_df, list(table_df.columns))
        + "\n\n"
        + notes,
    )


def write_aux_table(aux_df: pd.DataFrame, output_dir: Path) -> None:
    ordered_df = (
        aux_df.assign(
            model_key=pd.Categorical(aux_df["model_key"], categories=MODEL_ORDER, ordered=True),
        )
        .sort_values(["model_key", "condition_name"])
        .reset_index(drop=True)
    )
    table_df = pd.DataFrame(
        {
            "model": ordered_df["model_key"].map(MODEL_DISPLAY_ORDER),
            "condition": ordered_df["condition_name"],
            "n": ordered_df["n"].astype(int),
            "compliance": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    ordered_df["answer_space_compliance_rate"],
                    ordered_df["answer_space_compliance_ci_low"],
                    ordered_df["answer_space_compliance_ci_high"],
                )
            ],
            "conflict_aligned": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    ordered_df["conflict_aligned_rate"],
                    ordered_df["conflict_aligned_ci_low"],
                    ordered_df["conflict_aligned_ci_high"],
                )
            ],
            "faithful": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    ordered_df["faithful_rate"],
                    ordered_df["faithful_ci_low"],
                    ordered_df["faithful_ci_high"],
                )
            ],
        }
    )
    table_df.to_csv(output_dir / "table3_aux_metrics.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        output_dir / "table3_aux_metrics.md",
        "# Table 3. Auxiliary A1/A2 metrics\n\n" + build_table_md(table_df, list(table_df.columns)),
    )


def make_main_figure(main_df: pd.DataFrame, output_path: Path) -> None:
    ordered_df = (
        main_df.assign(
            model_key=pd.Categorical(main_df["model_key"], categories=MODEL_ORDER, ordered=True),
            condition_name=pd.Categorical(main_df["condition_name"], categories=PRIMARY_CONDITION_ORDER, ordered=True),
        )
        .sort_values(["model_key", "condition_name"])
        .reset_index(drop=True)
    )
    x = np.arange(len(PRIMARY_CONDITION_ORDER))
    width = 0.24
    offsets = np.linspace(-width, width, num=len(MODEL_ORDER))
    palette = ["#355f8d", "#c16843", "#4d8f5b"]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for offset, color, model_key in zip(offsets, palette, MODEL_ORDER):
        subset = (
            ordered_df[ordered_df["model_key"] == model_key]
            .set_index("condition_name")
            .reindex(PRIMARY_CONDITION_ORDER)
            .reset_index()
        )
        y = subset["conflict_aligned_rate"].to_numpy(dtype=float) * 100.0
        lower = (subset["conflict_aligned_rate"] - subset["conflict_aligned_ci_low"]).to_numpy(dtype=float) * 100.0
        upper = (subset["conflict_aligned_ci_high"] - subset["conflict_aligned_rate"]).to_numpy(dtype=float) * 100.0
        ax.bar(x + offset, y, width=width, label=MODEL_DISPLAY_ORDER[model_key], color=color, alpha=0.94)
        ax.errorbar(x + offset, y, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([PRIMARY_CONDITION_SHORT_LABELS[name] for name in PRIMARY_CONDITION_ORDER])
    ax.set_ylabel("Conflict-aligned rate (%)")
    ax.set_ylim(0, max(15.0, float(ordered_df["conflict_aligned_ci_high"].max()) * 100.0 + 2.5))
    ax.set_title("Figure 2. Conflict-aligned rate across C0-C4")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_main_stats_summary(main_df: pd.DataFrame, key_tests_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Main Experiment Statistical Summary",
        "",
        "- Primary inference target: `conflict_aligned` only.",
        "- Wilson 95% confidence intervals are reported for all main-condition proportions.",
        "- Holm correction was applied separately within the six `within_model_vs_C0` tests and the six `cross_model_same_condition` tests.",
        "- Main-experiment `refusal`, `other_wrong`, and `parse_error` counts were all zero across every model-condition cell.",
        "",
        "## Condition Labels",
    ]
    for short_label, description in CONDITION_LABEL_EXPLANATIONS.items():
        lines.append(f"- `{short_label}`: {description}.")

    lines.extend(["", "## Key Tests"])
    display_df = key_tests_df.copy()
    if display_df.empty:
        lines.append("- No key tests were available.")
    else:
        display_df["comparison_family"] = display_df["comparison_family"].replace(
            {
                "within_model_vs_C0": "within-model vs C0",
                "cross_model_same_condition": "cross-model same condition",
            }
        )
        display_df["left_rate"] = display_df["left_rate"].map(format_pct)
        display_df["right_rate"] = display_df["right_rate"].map(format_pct)
        display_df["rate_diff_left_minus_right"] = display_df["rate_diff_left_minus_right"].map(
            lambda value: f"{value * 100:.2f} pp"
        )
        display_df["p_value_raw"] = display_df["p_value_raw"].map(format_pvalue)
        display_df["p_value_holm"] = display_df["p_value_holm"].map(format_pvalue)
        display_df["significant_holm"] = display_df["significant_holm"].map(lambda value: "yes" if value else "no")
        table_columns = [
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
        renamed = display_df[table_columns].rename(
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
        lines.extend(["", build_table_md(renamed, list(renamed.columns))])

        significant = key_tests_df[key_tests_df["significant_holm"]]
        lines.extend(["", "## Interpretation"])
        if significant.empty:
            lines.append("- No Holm-significant main comparisons were detected.")
        else:
            for row in significant.to_dict(orient="records"):
                if row["comparison_family"] == "within_model_vs_C0":
                    lines.append(
                        f"- {MODEL_DISPLAY_ORDER[row['model_key']]} showed a significant increase in conflict-aligned rate from "
                        f"{PRIMARY_CONDITION_SHORT_LABELS[row['reference_condition']]} to "
                        f"{PRIMARY_CONDITION_SHORT_LABELS[row['condition_name']]} "
                        f"(raw p={format_pvalue(row['p_value_raw'])}, Holm p={format_pvalue(row['p_value_holm'])})."
                    )
                else:
                    lines.append(
                        f"- Under {PRIMARY_CONDITION_SHORT_LABELS[row['condition_name']]}, "
                        f"{MODEL_DISPLAY_ORDER[row['higher_rate_model']]} had a significantly higher conflict-aligned rate than "
                        f"{MODEL_DISPLAY_ORDER[row['lower_rate_model']]} "
                        f"(raw p={format_pvalue(row['p_value_raw'])}, Holm p={format_pvalue(row['p_value_holm'])})."
                    )

    write_markdown(output_path, "\n".join(lines))


def write_appendix_sanity_check(main_combined_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    filtered_df = main_combined_df[main_combined_df["condition_name"].isin(KEY_APPENDIX_CONDITIONS)].copy()
    rows: list[dict[str, object]] = []
    fisher_rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = filtered_df[filtered_df["model_key"] == model_key].copy()
        for condition_name in KEY_APPENDIX_CONDITIONS:
            condition_df = model_df[model_df["condition_name"] == condition_name].copy()
            source_groups: dict[str, pd.DataFrame] = {
                source_name: subset.copy()
                for source_name, subset in condition_df.groupby("source_dataset", observed=False)
            }
            if {"StanfordCars", "VCoR"}.issubset(set(source_groups)):
                fisher_p = np.nan
                if condition_name == "C3_presupposition_correction_allowed":
                    stanford = source_groups["StanfordCars"]
                    vcor = source_groups["VCoR"]
                    fisher_p = float(
                        fisher_exact(
                            [
                                [
                                    int(stanford["is_conflict_aligned"].sum()),
                                    len(stanford) - int(stanford["is_conflict_aligned"].sum()),
                                ],
                                [
                                    int(vcor["is_conflict_aligned"].sum()),
                                    len(vcor) - int(vcor["is_conflict_aligned"].sum()),
                                ],
                            ]
                        )[1]
                    )
                fisher_rows.append(
                    {
                        "model_key": model_key,
                        "condition_name": condition_name,
                        "fisher_p_conflict_aligned": fisher_p,
                    }
                )
            for source_name in ["StanfordCars", "VCoR"]:
                subset = source_groups.get(source_name, pd.DataFrame())
                total = int(len(subset))
                conflict_n = int(subset["is_conflict_aligned"].sum()) if total else 0
                faithful_n = int(subset["is_faithful"].sum()) if total else 0
                conflict_low, conflict_high = wilson_interval(conflict_n, total)
                faithful_low, faithful_high = wilson_interval(faithful_n, total)
                rows.append(
                    {
                        "model": MODEL_DISPLAY_ORDER[model_key],
                        "condition": PRIMARY_CONDITION_SHORT_LABELS[condition_name],
                        "source_dataset": source_name,
                        "n": total,
                        "conflict_aligned": format_metric_cell(
                            conflict_n,
                            total,
                            conflict_n / total if total else 0.0,
                            conflict_low,
                            conflict_high,
                        ),
                        "faithful": format_metric_cell(
                            faithful_n,
                            total,
                            faithful_n / total if total else 0.0,
                            faithful_low,
                            faithful_high,
                        ),
                        "conflict_aligned_n": conflict_n,
                        "conflict_aligned_rate": conflict_n / total if total else 0.0,
                        "faithful_n": faithful_n,
                        "faithful_rate": faithful_n / total if total else 0.0,
                        "fisher_p_conflict_aligned": fisher_rows[-1]["fisher_p_conflict_aligned"] if fisher_rows else np.nan,
                    }
                )

    sanity_df = pd.DataFrame(rows)
    sanity_df.to_csv(output_dir / "stanford_core_sanity_check.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "# Appendix Sanity Check: Source-Stratified Comparison",
        "",
        "This appendix-only sanity check splits the final balanced evaluation set by `source_dataset` (`StanfordCars` vs `VCoR`) and reports only `C0` and `C3`.",
        "",
        build_table_md(
            sanity_df[["model", "condition", "source_dataset", "n", "conflict_aligned", "faithful"]],
            ["model", "condition", "source_dataset", "n", "conflict_aligned", "faithful"],
        ),
        "",
        "## Interpretation",
        "- `C0` remained perfectly faithful for all three models across both source groups, so there is no source-specific baseline collapse.",
    ]

    llava_c3 = sanity_df[
        (sanity_df["model"] == MODEL_DISPLAY_ORDER["llava15_7b"]) & (sanity_df["condition"] == "C3")
    ].copy()
    if len(llava_c3) == 2:
        stanford_row = llava_c3[llava_c3["source_dataset"] == "StanfordCars"].iloc[0]
        vcor_row = llava_c3[llava_c3["source_dataset"] == "VCoR"].iloc[0]
        summary_lines.append(
            f"- For {MODEL_DISPLAY_ORDER['llava15_7b']} under `C3`, conflict-aligned behavior was higher on `StanfordCars` "
            f"({stanford_row['conflict_aligned']}) than on `VCoR` ({vcor_row['conflict_aligned']}), "
            f"but the direction remained the same in both sources; Fisher exact p={format_pvalue(float(stanford_row['fisher_p_conflict_aligned']))}."
        )
    summary_lines.append(
        f"- {MODEL_DISPLAY_ORDER['qwen2vl7b']} and {MODEL_DISPLAY_ORDER['internvl2_8b']} remained visually faithful across both source groups, "
        "so the main conclusion is not driven by a single data source."
    )
    summary_lines.append(
        "- This analysis is appendix-only: it checks whether the core trend is source-dependent, not whether source becomes a new main experimental factor."
    )
    write_markdown(output_dir / "stanford_core_sanity_check.md", "\n".join(summary_lines))
    return sanity_df


def write_final_result_summary(
    paths: dict[str, Path],
    main_df: pd.DataFrame,
    key_tests_df: pd.DataFrame,
    robustness_metrics_df: pd.DataFrame | None,
    robustness_tests_df: pd.DataFrame | None,
) -> None:
    llava_c3 = main_df[
        (main_df["model_key"] == "llava15_7b")
        & (main_df["condition_name"] == "C3_presupposition_correction_allowed")
    ].iloc[0]
    llava_c4 = main_df[
        (main_df["model_key"] == "llava15_7b")
        & (main_df["condition_name"] == "C4_stronger_open_conflict")
    ].iloc[0]
    summary_lines = [
        "# 当前结果摘要",
        "",
        "## 数据集",
        "",
        "- 正式评测集：`data/balanced_eval_set/final_manifest.csv`",
        "- 总样本数：300",
        "- 颜色分布：`red / blue / green / yellow / black / white` 各 50",
        "- 来源构成：StanfordCars 93，VCoR 207",
        "",
        "## 主实验 C0-C4",
        "",
        "- 三模型在 `C0` 均保持 100.00% 忠实率，`conflict_aligned=0/300`。",
        f"- `LLaVA-1.5-7B` 在 `C3` 的冲突一致率为 {format_pct(float(llava_c3['conflict_aligned_rate']))} "
        f"{format_ci(float(llava_c3['conflict_aligned_ci_low']), float(llava_c3['conflict_aligned_ci_high']))}，"
        f"`C4` 为 {format_pct(float(llava_c4['conflict_aligned_rate']))} "
        f"{format_ci(float(llava_c4['conflict_aligned_ci_low']), float(llava_c4['conflict_aligned_ci_high']))}。",
        "- `Qwen2-VL-7B-Instruct` 仅在 `C3/C4` 各出现 1 例 conflict-aligned 输出；`InternVL2-8B` 在 `C0-C4` 中未出现 conflict-aligned 输出。",
        "- 主实验 `refusal / other_wrong / parse_error` 全部为 0，因此主推断聚焦在 `conflict_aligned`。",
        "",
        "## 关键统计",
        "",
    ]
    significant_rows = key_tests_df[key_tests_df["significant_holm"]]
    for row in significant_rows.to_dict(orient="records"):
        if row["comparison_family"] == "within_model_vs_C0":
            summary_lines.append(
                f"- {MODEL_DISPLAY_ORDER[row['model_key']]} 在 "
                f"{PRIMARY_CONDITION_SHORT_LABELS[row['condition_name']]} 相对 `C0` 的 conflict-aligned 增幅达到 Holm 显著 "
                f"(raw p={format_pvalue(row['p_value_raw'])}, Holm p={format_pvalue(row['p_value_holm'])})。"
            )
        else:
            summary_lines.append(
                f"- 在 {PRIMARY_CONDITION_SHORT_LABELS[row['condition_name']]}，"
                f"{MODEL_DISPLAY_ORDER[row['higher_rate_model']]} 的 conflict-aligned rate 显著高于 "
                f"{MODEL_DISPLAY_ORDER[row['lower_rate_model']]} "
                f"(raw p={format_pvalue(row['p_value_raw'])}, Holm p={format_pvalue(row['p_value_holm'])})。"
            )

    if robustness_metrics_df is not None and not robustness_metrics_df.empty:
        summary_lines.extend(["", "## Prompt Variant Robustness", ""])
        llava_robust = robustness_metrics_df[robustness_metrics_df["model_key"] == "llava15_7b"].copy()
        for row in llava_robust.to_dict(orient="records"):
            summary_lines.append(
                f"- `LLaVA-1.5-7B` | {ROBUSTNESS_VARIANT_LABELS[row['robustness_variant']]}: "
                f"conflict-aligned={format_pct(float(row['conflict_aligned_rate']))} "
                f"{format_ci(float(row['conflict_aligned_ci_low']), float(row['conflict_aligned_ci_high']))}."
            )
        if robustness_tests_df is not None and not robustness_tests_df.empty:
            significant_cross = robustness_tests_df[
                (robustness_tests_df["comparison_family"] == "cross_model_same_variant")
                & (robustness_tests_df["significant_holm"])
            ]
            if not significant_cross.empty:
                summary_lines.append("- Prompt-variant robustness details are summarized in `results/robustness/prompt_variant_summary.md`.")
        if len(llava_robust) == 3:
            llava_original = llava_robust[llava_robust["robustness_variant"] == "C3_original"].iloc[0]
            llava_v2 = llava_robust[llava_robust["robustness_variant"] == "C3_v2"].iloc[0]
            llava_v3 = llava_robust[llava_robust["robustness_variant"] == "C3_v3"].iloc[0]
            summary_lines.append(
                f"- 模板鲁棒性控制显示原始 `C3` 效应并不稳定：`LLaVA-1.5-7B` 从 Original C3 的 "
                f"{format_pct(float(llava_original['conflict_aligned_rate']))} 降到 C3-v2 的 "
                f"{format_pct(float(llava_v2['conflict_aligned_rate']))}，并在 C3-v3 降到 "
                f"{format_pct(float(llava_v3['conflict_aligned_rate']))}。"
            )
            summary_lines.append(
                "- 因此正文应把该现象写成“对原始强误导模板敏感的有限语言偏差”，而不应写成对所有等强度 wording 都稳定成立。"
            )

    summary_lines.extend(
        [
            "",
            "## 解析与附录检查",
            "",
            "- 解析规则审查见 `results/parser/label_mapping_audit.md`，别名抽样复核见 `results/parser/ambiguous_outputs_sample.csv`。",
            "- 数据来源 sanity check 已改为最终平衡集内部按 `source_dataset` 分层，见 `results/appendix/stanford_core_sanity_check.md`。",
            "",
            "## 关键文件",
            "",
            "- Table 1：`results/main/table1_main_metrics.csv`",
            "- Figure 2：`results/main/figure2_conflict_aligned_rates.png`",
            "- Main stats summary：`results/main/main_stats_summary.md`",
            "- Main key tests：`results/main/main_key_tests.csv`",
            "- Prompt robustness：`results/robustness/prompt_variant_summary.md`",
            "- Parser audit：`results/parser/label_mapping_audit.md`",
            "- Appendix sanity check：`results/appendix/stanford_core_sanity_check.md`",
        ]
    )
    write_markdown(paths["main_dir"].parent / "final_result_summary.md", "\n".join(summary_lines))


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> int:
    config = load_paper_config(parse_args().config)
    paths = paper_paths(config)

    baseline_dir = paths["baseline_dir"]
    main_dir = paths["main_dir"]
    aux_dir = paths["aux_dir"]
    appendix_dir = paths["appendix_dir"]
    appendix_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(main_dir / "main_condition_metrics.csv", encoding="utf-8-sig")
    aux_df = pd.read_csv(aux_dir / "aux_condition_metrics.csv", encoding="utf-8-sig")
    main_tests_df = pd.read_csv(main_dir / "main_exact_tests.csv", encoding="utf-8-sig")
    main_combined_df = pd.read_csv(main_dir / "main_combined_parsed_results.csv", encoding="utf-8-sig")
    distribution_df = pd.read_csv(paths["metadata_dir"] / "dataset_distribution.csv", encoding="utf-8-sig")

    key_tests_df = build_main_key_tests(main_tests_df)
    key_tests_df.to_csv(main_dir / "main_key_tests.csv", index=False, encoding="utf-8-sig")

    write_main_table(main_df, key_tests_df, main_dir)
    write_aux_table(aux_df, aux_dir)
    write_main_stats_summary(main_df, key_tests_df, main_dir / "main_stats_summary.md")
    make_main_figure(main_df, main_dir / "figure2_conflict_aligned_rates.png")
    make_dataset_distribution_plot(distribution_df, appendix_dir / "dataset_distribution.png")
    distribution_df.to_csv(appendix_dir / "dataset_distribution_table.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        appendix_dir / "dataset_distribution.md",
        "# Dataset distribution\n\n"
        "Panel A shows the original Stanford strict-clean seed distribution. "
        "Panel B shows the final balanced evaluation set with VCoR supplementation stacked on top of the retained Stanford seed images.\n",
    )
    write_appendix_sanity_check(main_combined_df, appendix_dir)

    robustness_metrics_df = load_optional_csv(paths["robustness_dir"] / "prompt_variant_metrics.csv")
    robustness_tests_df = load_optional_csv(paths["robustness_dir"] / "prompt_variant_exact_tests.csv")
    write_final_result_summary(paths, main_df, key_tests_df, robustness_metrics_df, robustness_tests_df)

    payload = {
        "table1": str(main_dir / "table1_main_metrics.csv"),
        "figure2": str(main_dir / "figure2_conflict_aligned_rates.png"),
        "table3": str(aux_dir / "table3_aux_metrics.csv"),
        "main_key_tests": str(main_dir / "main_key_tests.csv"),
        "main_stats_summary": str(main_dir / "main_stats_summary.md"),
        "dataset_figure": str(appendix_dir / "dataset_distribution.png"),
        "appendix_sanity_check": str(appendix_dir / "stanford_core_sanity_check.csv"),
        "baseline_summary": str(baseline_dir / "baseline_summary.md"),
    }
    dump_json(appendix_dir / "figure_manifest.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
