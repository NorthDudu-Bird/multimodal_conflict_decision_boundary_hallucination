#!/usr/bin/env python
"""Generate per-color main-experiment metrics and paired flip diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from scripts.utils.paper_mainline_utils import (
    MODEL_ORDER,
    PRIMARY_CONDITION_ORDER,
    PRIMARY_CONDITION_SHORT_LABELS,
    format_ci,
    format_pct,
    load_bool_results,
    write_markdown,
)


MODEL_DISPLAY = {
    "qwen2vl7b": "Qwen2-VL-7B-Instruct",
    "llava15_7b": "LLaVA-1.5-7B",
    "internvl2_8b": "InternVL2-8B",
}
COLOR_ORDER = ["black", "blue", "green", "red", "white", "yellow"]
FOCUS_CONDITIONS = ["C0_neutral", "C3_presupposition_correction_allowed", "C4_stronger_open_conflict"]
PAIR_FAMILY = {
    ("black", "white"): "achromatic_black_white",
    ("white", "black"): "achromatic_black_white",
    ("red", "blue"): "red_blue",
    ("blue", "red"): "red_blue",
    ("green", "yellow"): "green_yellow",
    ("yellow", "red"): "yellow_red",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate color-split main-experiment analyses.")
    parser.add_argument("--input-csv", type=Path, default=REPO_ROOT / "results" / "main" / "main_combined_parsed_results.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "color_split")
    return parser.parse_args()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    low, high = proportion_confint(successes, total, method="wilson")
    return float(low), float(high)


def pct_cell(count: int, total: int, rate: float, low: float, high: float) -> str:
    return f"{count}/{total} ({format_pct(rate)} {format_ci(low, high)})"


def exact_mcnemar(current: pd.Series, reference: pd.Series) -> dict[str, object]:
    current_bool = current.astype(bool)
    reference_bool = reference.astype(bool)
    both_yes = int((current_bool & reference_bool).sum())
    current_only = int((current_bool & ~reference_bool).sum())
    reference_only = int((~current_bool & reference_bool).sum())
    both_no = int((~current_bool & ~reference_bool).sum())
    result = mcnemar([[both_yes, current_only], [reference_only, both_no]], exact=True)
    return {
        "both_yes": both_yes,
        "current_only": current_only,
        "c0_only": reference_only,
        "both_no": both_no,
        "p_value_exact_mcnemar": float(result.pvalue),
    }


def build_color_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        for condition_name in PRIMARY_CONDITION_ORDER:
            for true_color in COLOR_ORDER:
                subset = df[
                    (df["model_key"] == model_key)
                    & (df["condition_name"] == condition_name)
                    & (df["true_color"] == true_color)
                ].copy()
                total = int(len(subset))
                faithful_n = int(subset["is_faithful"].sum())
                conflict_n = int(subset["is_conflict_aligned"].sum())
                other_n = int(subset["is_other_wrong"].sum()) if "is_other_wrong" in subset else 0
                refusal_n = int(subset["is_refusal_or_correction"].sum()) if "is_refusal_or_correction" in subset else 0
                parse_error_n = int(subset["is_parse_error"].sum()) if "is_parse_error" in subset else 0
                faithful_low, faithful_high = wilson_interval(faithful_n, total)
                conflict_low, conflict_high = wilson_interval(conflict_n, total)
                pair = ""
                if total:
                    conflicts = sorted(set(zip(subset["true_color"], subset["conflict_color"])))
                    if len(conflicts) == 1:
                        pair = f"{conflicts[0][0]}->{conflicts[0][1]}"
                rows.append(
                    {
                        "model_key": model_key,
                        "model": MODEL_DISPLAY[model_key],
                        "condition_name": condition_name,
                        "condition": PRIMARY_CONDITION_SHORT_LABELS[condition_name],
                        "true_color": true_color,
                        "false_prompt_pair": pair,
                        "n": total,
                        "faithful_n": faithful_n,
                        "faithful_rate": faithful_n / total if total else 0.0,
                        "faithful_ci_low": faithful_low,
                        "faithful_ci_high": faithful_high,
                        "conflict_aligned_n": conflict_n,
                        "conflict_aligned_rate": conflict_n / total if total else 0.0,
                        "conflict_aligned_ci_low": conflict_low,
                        "conflict_aligned_ci_high": conflict_high,
                        "conflict_following_n": conflict_n,
                        "conflict_following_rate": conflict_n / total if total else 0.0,
                        "other_wrong_n": other_n,
                        "refusal_n": refusal_n,
                        "parse_error_n": parse_error_n,
                    }
                )
    return pd.DataFrame(rows)


def build_paired_color_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key].copy()
        c0 = model_df[model_df["condition_name"] == "C0_neutral"].copy()
        for condition_name in PRIMARY_CONDITION_ORDER[1:]:
            current = model_df[model_df["condition_name"] == condition_name].copy()
            merged = c0[
                ["image_id", "true_color", "conflict_color", "parsed_label", "is_conflict_aligned", "is_faithful"]
            ].merge(
                current[
                    ["image_id", "true_color", "conflict_color", "parsed_label", "is_conflict_aligned", "is_faithful"]
                ],
                on=["image_id", "true_color", "conflict_color"],
                suffixes=("_c0", "_current"),
            )
            for true_color in COLOR_ORDER:
                subset = merged[merged["true_color"] == true_color].copy()
                n_pairs = int(len(subset))
                if n_pairs == 0:
                    continue
                c0_faithful = subset["is_faithful_c0"].astype(bool)
                current_conflict = subset["is_conflict_aligned_current"].astype(bool)
                current_faithful = subset["is_faithful_current"].astype(bool)
                answer_flip = (subset["parsed_label_c0"].astype(str) != subset["parsed_label_current"].astype(str)) | (
                    subset["is_conflict_aligned_c0"].astype(bool) != subset["is_conflict_aligned_current"].astype(bool)
                )
                payload = exact_mcnemar(subset["is_conflict_aligned_current"], subset["is_conflict_aligned_c0"])
                rows.append(
                    {
                        "model_key": model_key,
                        "model": MODEL_DISPLAY[model_key],
                        "reference_condition": "C0_neutral",
                        "condition_name": condition_name,
                        "condition": PRIMARY_CONDITION_SHORT_LABELS[condition_name],
                        "true_color": true_color,
                        "false_prompt_color": str(subset["conflict_color"].iloc[0]),
                        "pair_family": PAIR_FAMILY.get((true_color, str(subset["conflict_color"].iloc[0])), "other_pair"),
                        "n_pairs": n_pairs,
                        "c0_faithful_n": int(c0_faithful.sum()),
                        "current_faithful_n": int(current_faithful.sum()),
                        "faithful_to_faithful_n": int((c0_faithful & current_faithful).sum()),
                        "faithful_to_conflict_aligned_n": int((c0_faithful & current_conflict).sum()),
                        "answer_flip_n": int(answer_flip.sum()),
                        "answer_flip_rate": float(answer_flip.mean()),
                        "conflict_following_rate": int((c0_faithful & current_conflict).sum()) / int(c0_faithful.sum())
                        if int(c0_faithful.sum())
                        else 0.0,
                        "paired_discordant_current_only": payload["current_only"],
                        "paired_discordant_c0_only": payload["c0_only"],
                        "p_value_exact_mcnemar": payload["p_value_exact_mcnemar"],
                    }
                )
    return pd.DataFrame(rows)


def build_false_prompt_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        for condition_name in PRIMARY_CONDITION_ORDER:
            subset = df[(df["model_key"] == model_key) & (df["condition_name"] == condition_name)].copy()
            if subset.empty:
                continue
            grouped = (
                subset.groupby(["true_color", "conflict_color"], observed=False)
                .agg(
                    n=("image_id", "count"),
                    conflict_aligned_n=("is_conflict_aligned", "sum"),
                    faithful_n=("is_faithful", "sum"),
                )
                .reset_index()
            )
            for row in grouped.to_dict("records"):
                total = int(row["n"])
                conflict_n = int(row["conflict_aligned_n"])
                rows.append(
                    {
                        "model_key": model_key,
                        "model": MODEL_DISPLAY[model_key],
                        "condition_name": condition_name,
                        "condition": PRIMARY_CONDITION_SHORT_LABELS[condition_name],
                        "true_color": row["true_color"],
                        "false_prompt_color": row["conflict_color"],
                        "pair_family": PAIR_FAMILY.get((row["true_color"], row["conflict_color"]), "other_pair"),
                        "n": total,
                        "conflict_aligned_n": conflict_n,
                        "conflict_aligned_rate": conflict_n / total if total else 0.0,
                        "faithful_n": int(row["faithful_n"]),
                    }
                )
    return pd.DataFrame(rows)


def build_pair_family_metrics(paired_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        paired_df.groupby(["model_key", "model", "condition_name", "condition", "pair_family"], observed=False)
        .agg(
            n_pairs=("n_pairs", "sum"),
            faithful_to_conflict_aligned_n=("faithful_to_conflict_aligned_n", "sum"),
            answer_flip_n=("answer_flip_n", "sum"),
            paired_discordant_current_only=("paired_discordant_current_only", "sum"),
            paired_discordant_c0_only=("paired_discordant_c0_only", "sum"),
        )
        .reset_index()
    )
    grouped["conflict_following_rate"] = grouped["faithful_to_conflict_aligned_n"] / grouped["n_pairs"]
    grouped["answer_flip_rate"] = grouped["answer_flip_n"] / grouped["n_pairs"]
    return grouped


def make_rates_figure(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df[
        (metrics_df["condition_name"].isin(["C3_presupposition_correction_allowed", "C4_stronger_open_conflict"]))
        & (metrics_df["model_key"].isin(MODEL_ORDER))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    palette = {"qwen2vl7b": "#355f8d", "llava15_7b": "#c16843", "internvl2_8b": "#4d8f5b"}
    x = np.arange(len(COLOR_ORDER))
    width = 0.24
    for ax, condition_name in zip(axes, ["C3_presupposition_correction_allowed", "C4_stronger_open_conflict"]):
        condition_df = plot_df[plot_df["condition_name"] == condition_name]
        for offset, model_key in zip(np.linspace(-width, width, len(MODEL_ORDER)), MODEL_ORDER):
            subset = condition_df[condition_df["model_key"] == model_key].set_index("true_color").reindex(COLOR_ORDER)
            ax.bar(
                x + offset,
                subset["conflict_aligned_rate"].to_numpy(dtype=float) * 100.0,
                width=width,
                label=MODEL_DISPLAY[model_key],
                color=palette[model_key],
                alpha=0.94,
            )
        ax.set_title(PRIMARY_CONDITION_SHORT_LABELS[condition_name])
        ax.set_xticks(x)
        ax.set_xticklabels(COLOR_ORDER, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Conflict-aligned rate by true color (%)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_heatmap(matrix_df: pd.DataFrame, output_path: Path) -> None:
    llava_c3 = matrix_df[
        (matrix_df["model_key"] == "llava15_7b")
        & (matrix_df["condition_name"] == "C3_presupposition_correction_allowed")
    ].copy()
    pivot = (
        llava_c3.pivot_table(
            index="true_color",
            columns="false_prompt_color",
            values="conflict_aligned_n",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(index=COLOR_ORDER, columns=COLOR_ORDER, fill_value=0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(len(COLOR_ORDER)))
    ax.set_yticks(np.arange(len(COLOR_ORDER)))
    ax.set_xticklabels(COLOR_ORDER, rotation=35, ha="right")
    ax.set_yticklabels(COLOR_ORDER)
    ax.set_xlabel("False prompt color")
    ax.set_ylabel("True color")
    ax.set_title("LLaVA C3 conflict-aligned counts by color pair")
    for i in range(len(COLOR_ORDER)):
        for j in range(len(COLOR_ORDER)):
            value = int(pivot.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def table_md(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for row in df.to_dict("records")]
    return "\n".join([header, divider, *rows])


def write_summary(metrics_df: pd.DataFrame, paired_df: pd.DataFrame, pair_df: pd.DataFrame, output_path: Path) -> None:
    llava_c3 = paired_df[
        (paired_df["model_key"] == "llava15_7b")
        & (paired_df["condition_name"] == "C3_presupposition_correction_allowed")
    ].copy()
    llava_c4 = paired_df[
        (paired_df["model_key"] == "llava15_7b")
        & (paired_df["condition_name"] == "C4_stronger_open_conflict")
    ].copy()

    def concentration_text(subset: pd.DataFrame, label: str) -> tuple[str, pd.DataFrame]:
        total = int(subset["faithful_to_conflict_aligned_n"].sum())
        display = subset[
            [
                "true_color",
                "false_prompt_color",
                "pair_family",
                "n_pairs",
                "faithful_to_conflict_aligned_n",
                "conflict_following_rate",
                "paired_discordant_current_only",
                "paired_discordant_c0_only",
            ]
        ].copy()
        display["conflict_following_rate"] = display["conflict_following_rate"].map(format_pct)
        nonzero = subset[subset["faithful_to_conflict_aligned_n"] > 0].sort_values(
            "faithful_to_conflict_aligned_n", ascending=False
        )
        if nonzero.empty or total == 0:
            return f"{label}: no conflict-following flips.", display
        top = nonzero.iloc[0]
        share = int(top["faithful_to_conflict_aligned_n"]) / total
        return (
            f"{label}: {total} total flips; largest route is `{top['true_color']} -> {top['false_prompt_color']}` "
            f"with {int(top['faithful_to_conflict_aligned_n'])}/{total} ({format_pct(share)}).",
            display,
        )

    c3_text, c3_table = concentration_text(llava_c3, "LLaVA C3")
    c4_text, c4_table = concentration_text(llava_c4, "LLaVA C4")

    focus = metrics_df[
        (metrics_df["condition_name"].isin(FOCUS_CONDITIONS))
        & (metrics_df["model_key"] == "llava15_7b")
    ].copy()
    focus_display = focus[
        [
            "condition",
            "true_color",
            "false_prompt_pair",
            "n",
            "faithful_n",
            "faithful_rate",
            "conflict_aligned_n",
            "conflict_aligned_rate",
        ]
    ].copy()
    focus_display["faithful_rate"] = focus_display["faithful_rate"].map(format_pct)
    focus_display["conflict_aligned_rate"] = focus_display["conflict_aligned_rate"].map(format_pct)

    llava_pair = pair_df[
        (pair_df["model_key"] == "llava15_7b")
        & (pair_df["condition_name"].isin(["C3_presupposition_correction_allowed", "C4_stronger_open_conflict"]))
    ].copy()
    pair_display = llava_pair[
        ["condition", "pair_family", "n_pairs", "faithful_to_conflict_aligned_n", "conflict_following_rate"]
    ].copy()
    pair_display["conflict_following_rate"] = pair_display["conflict_following_rate"].map(format_pct)

    lines = [
        "# Color-Split Main Experiment Analysis",
        "",
        "## Role",
        "",
        "This analysis checks whether the main conflict-aligned effect is distributed across the six balanced true-color classes or driven by a small subset of color pairs. It is an attribution and boundary-control module, not a new color-perception paper.",
        "",
        "## Key Findings",
        "",
        f"- {c3_text}",
        f"- {c4_text}",
        "- C0 remains fully faithful in every true-color stratum for all three models.",
        "- The LLaVA effect is therefore not evenly dispersed across the six colors. It is concentrated mainly in the achromatic `white -> black` route, with smaller contributions from `black -> white` and `blue -> red` under C3.",
        "- This pattern supports a contracted interpretation: the phenomenon is best described as template sensitivity plus partial color-pair vulnerability, not a general color-task shift and not mainly a neighboring-hue confusion effect.",
        "",
        "## LLaVA Focus: C0/C3/C4 By True Color",
        "",
        table_md(focus_display, list(focus_display.columns)),
        "",
        "## LLaVA Paired Flips By True Color",
        "",
        "### C3",
        "",
        table_md(c3_table, list(c3_table.columns)),
        "",
        "### C4",
        "",
        table_md(c4_table, list(c4_table.columns)),
        "",
        "## Pair-Family Summary",
        "",
        table_md(pair_display, list(pair_display.columns)),
        "",
        "## Paper Boundary",
        "",
        "The original 9% LLaVA C3 effect should not be written as a uniformly distributed susceptibility across all colors. The more accurate wording is that a limited same-image conflict-following shift appears under the original strong misleading templates, and the shift is strongly concentrated in specific true-color/false-color pairings, especially `white -> black`.",
    ]
    write_markdown(output_path, "\n".join(lines))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_bool_results(args.input_csv)
    df = df[df["condition_name"].isin(PRIMARY_CONDITION_ORDER)].copy()

    metrics_df = build_color_metrics(df)
    paired_df = build_paired_color_metrics(df)
    matrix_df = build_false_prompt_matrix(df)
    pair_df = build_pair_family_metrics(paired_df)

    metrics_df.to_csv(args.output_dir / "color_split_main_metrics.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(args.output_dir / "color_split_paired_flip_metrics.csv", index=False, encoding="utf-8-sig")
    matrix_df.to_csv(args.output_dir / "color_by_false_prompt_matrix.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(args.output_dir / "color_pair_family_metrics.csv", index=False, encoding="utf-8-sig")
    make_rates_figure(metrics_df, args.output_dir / "figure_color_split_rates.png")
    make_heatmap(matrix_df, args.output_dir / "figure_color_flip_heatmap.png")
    write_summary(metrics_df, paired_df, pair_df, args.output_dir / "color_split_summary.md")

    payload = {
        "metrics": str(args.output_dir / "color_split_main_metrics.csv"),
        "paired": str(args.output_dir / "color_split_paired_flip_metrics.csv"),
        "matrix": str(args.output_dir / "color_by_false_prompt_matrix.csv"),
        "pair_family": str(args.output_dir / "color_pair_family_metrics.csv"),
        "summary": str(args.output_dir / "color_split_summary.md"),
        "figures": [
            str(args.output_dir / "figure_color_split_rates.png"),
            str(args.output_dir / "figure_color_flip_heatmap.png"),
        ],
        "rows": {
            "metrics": int(len(metrics_df)),
            "paired": int(len(paired_df)),
            "matrix": int(len(matrix_df)),
            "pair_family": int(len(pair_df)),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
