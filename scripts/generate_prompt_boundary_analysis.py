#!/usr/bin/env python
"""Repackage C3 prompt variants as a boundary-control analysis module."""

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
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from scripts.utils.paper_mainline_utils import MODEL_ORDER, format_ci, format_pct, load_bool_results, write_markdown


MODEL_DISPLAY = {
    "qwen2vl7b": "Qwen2-VL-7B-Instruct",
    "llava15_7b": "LLaVA-1.5-7B",
    "internvl2_8b": "InternVL2-8B",
}
VARIANT_ORDER = ["C3_original", "C3_v2", "C3_v3"]
VARIANT_DISPLAY = {
    "C3_original": "Original C3",
    "C3_v2": "C3-v2",
    "C3_v3": "C3-v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prompt-boundary robustness outputs.")
    parser.add_argument("--main-csv", type=Path, default=REPO_ROOT / "results" / "main" / "main_combined_parsed_results.csv")
    parser.add_argument(
        "--robustness-csv",
        type=Path,
        default=REPO_ROOT / "results" / "robustness" / "prompt_variant_combined_parsed_results.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "robustness")
    return parser.parse_args()


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    low, high = proportion_confint(successes, total, method="wilson")
    return float(low), float(high)


def paired_exact(left: pd.DataFrame, right: pd.DataFrame, outcome: str = "is_conflict_aligned") -> dict[str, object]:
    merged = left[["image_id", outcome]].merge(right[["image_id", outcome]], on="image_id", suffixes=("_left", "_right"))
    left_values = merged[f"{outcome}_left"].astype(bool)
    right_values = merged[f"{outcome}_right"].astype(bool)
    both_yes = int((left_values & right_values).sum())
    left_only = int((left_values & ~right_values).sum())
    right_only = int((~left_values & right_values).sum())
    both_no = int((~left_values & ~right_values).sum())
    result = mcnemar([[both_yes, left_only], [right_only, both_no]], exact=True)
    return {
        "n_pairs": int(len(merged)),
        "left_rate": float(left_values.mean()) if len(merged) else 0.0,
        "right_rate": float(right_values.mean()) if len(merged) else 0.0,
        "rate_diff_left_minus_right": float(left_values.mean() - right_values.mean()) if len(merged) else 0.0,
        "left_only": left_only,
        "right_only": right_only,
        "p_value_raw": float(result.pvalue),
    }


def load_boundary_df(main_csv: Path, robustness_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_df = load_bool_results(main_csv)
    robustness_df = load_bool_results(robustness_csv)
    if "robustness_variant" not in robustness_df.columns:
        raise ValueError("robustness CSV must contain robustness_variant")
    c3_variants = robustness_df[robustness_df["robustness_variant"].isin(VARIANT_ORDER)].copy()
    if "C3_original" not in set(c3_variants["robustness_variant"].astype(str)):
        original = main_df[main_df["condition_name"] == "C3_presupposition_correction_allowed"].copy()
        original["robustness_variant"] = "C3_original"
        original["robustness_variant_label"] = "C3_original"
        c3_variants = pd.concat([original, c3_variants], ignore_index=True, sort=False)
    c3_variants = c3_variants[c3_variants["robustness_variant"].isin(VARIANT_ORDER)].copy()
    return main_df, c3_variants


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key].copy()
        for variant in VARIANT_ORDER:
            subset = model_df[model_df["robustness_variant"] == variant].copy()
            total = int(len(subset))
            conflict_n = int(subset["is_conflict_aligned"].sum())
            faithful_n = int(subset["is_faithful"].sum())
            refusal_n = int(subset["is_refusal_or_correction"].sum()) if "is_refusal_or_correction" in subset else 0
            other_n = int(subset["is_other_wrong"].sum()) if "is_other_wrong" in subset else 0
            parse_n = int(subset["is_parse_error"].sum()) if "is_parse_error" in subset else 0
            conflict_low, conflict_high = wilson(conflict_n, total)
            faithful_low, faithful_high = wilson(faithful_n, total)
            rows.append(
                {
                    "model_key": model_key,
                    "model": MODEL_DISPLAY[model_key],
                    "robustness_variant": variant,
                    "variant": VARIANT_DISPLAY[variant],
                    "n": total,
                    "conflict_aligned_n": conflict_n,
                    "conflict_aligned_rate": conflict_n / total if total else 0.0,
                    "conflict_aligned_ci_low": conflict_low,
                    "conflict_aligned_ci_high": conflict_high,
                    "faithful_n": faithful_n,
                    "faithful_rate": faithful_n / total if total else 0.0,
                    "faithful_ci_low": faithful_low,
                    "faithful_ci_high": faithful_high,
                    "other_wrong_n": other_n,
                    "refusal_n": refusal_n,
                    "parse_error_n": parse_n,
                }
            )
    return pd.DataFrame(rows)


def build_tests(main_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        model_df = variant_df[variant_df["model_key"] == model_key].copy()
        original = model_df[model_df["robustness_variant"] == "C3_original"].copy()
        for variant in ["C3_v2", "C3_v3"]:
            current = model_df[model_df["robustness_variant"] == variant].copy()
            payload = paired_exact(current, original)
            rows.append(
                {
                    "comparison_family": "within_model_vs_original_c3",
                    "comparison_label": f"{MODEL_DISPLAY[model_key]}: {VARIANT_DISPLAY[variant]} vs Original C3",
                    "model_key": model_key,
                    "left_model": model_key,
                    "right_model": model_key,
                    "left_variant": variant,
                    "right_variant": "C3_original",
                    "metric": "conflict_aligned",
                    **payload,
                }
            )

    for variant in VARIANT_ORDER:
        subset = variant_df[variant_df["robustness_variant"] == variant].copy()
        llava = subset[subset["model_key"] == "llava15_7b"].copy()
        for right_model in ["qwen2vl7b", "internvl2_8b"]:
            right = subset[subset["model_key"] == right_model].copy()
            payload = paired_exact(llava, right)
            rows.append(
                {
                    "comparison_family": "cross_model_same_variant",
                    "comparison_label": f"{VARIANT_DISPLAY[variant]}: LLaVA-1.5-7B vs {MODEL_DISPLAY[right_model]}",
                    "model_key": "",
                    "left_model": "llava15_7b",
                    "right_model": right_model,
                    "left_variant": variant,
                    "right_variant": variant,
                    "metric": "conflict_aligned",
                    **payload,
                }
            )

    llava_c0 = main_df[(main_df["model_key"] == "llava15_7b") & (main_df["condition_name"] == "C0_neutral")].copy()
    for variant in VARIANT_ORDER:
        current = variant_df[(variant_df["model_key"] == "llava15_7b") & (variant_df["robustness_variant"] == variant)].copy()
        payload = paired_exact(current, llava_c0)
        rows.append(
            {
                "comparison_family": "llava_variant_vs_c0",
                "comparison_label": f"LLaVA-1.5-7B: {VARIANT_DISPLAY[variant]} vs C0",
                "model_key": "llava15_7b",
                "left_model": "llava15_7b",
                "right_model": "llava15_7b",
                "left_variant": variant,
                "right_variant": "C0_neutral",
                "metric": "conflict_aligned",
                **payload,
            }
        )

    tests_df = pd.DataFrame(rows)
    tests_df["p_value_holm"] = np.nan
    tests_df["significant_holm"] = False
    for family in tests_df["comparison_family"].unique():
        mask = tests_df["comparison_family"] == family
        _, corrected, _, _ = multipletests(tests_df.loc[mask, "p_value_raw"].to_numpy(dtype=float), method="holm")
        tests_df.loc[mask, "p_value_holm"] = corrected
        tests_df.loc[mask, "significant_holm"] = corrected < 0.05
    return tests_df


def format_p(value: float) -> str:
    return f"{value:.2e}" if value < 1e-4 else f"{value:.4f}"


def write_summary(metrics_df: pd.DataFrame, tests_df: pd.DataFrame, output_path: Path) -> None:
    llava = metrics_df[metrics_df["model_key"] == "llava15_7b"].set_index("robustness_variant")
    original = llava.loc["C3_original"]
    v2 = llava.loc["C3_v2"]
    v3 = llava.loc["C3_v3"]
    lines = [
        "# Prompt Wording Boundary-Control Summary",
        "",
        "## Role",
        "",
        "- This module limits the claim boundary. It is not used to expand the paper into a cross-template law.",
        "- The relevant question is whether the original `C3` effect survives semantically close wording changes.",
        "",
        "## Metrics",
        "",
    ]
    for row in metrics_df.to_dict("records"):
        lines.append(
            f"- {row['model']} | {row['variant']}: conflict_aligned={row['conflict_aligned_n']}/{row['n']} "
            f"({format_pct(row['conflict_aligned_rate'])} {format_ci(row['conflict_aligned_ci_low'], row['conflict_aligned_ci_high'])}); "
            f"faithful={row['faithful_n']}/{row['n']} ({format_pct(row['faithful_rate'])})."
        )

    lines.extend(["", "## Key Tests", ""])
    for row in tests_df[tests_df["comparison_family"].isin(["within_model_vs_original_c3", "cross_model_same_variant", "llava_variant_vs_c0"])].to_dict("records"):
        if row["comparison_family"] == "within_model_vs_original_c3" and row["model_key"] != "llava15_7b":
            continue
        if row["comparison_family"] == "cross_model_same_variant" and row["left_variant"] == "C3_original":
            keep = True
        elif row["comparison_family"] == "cross_model_same_variant" and row["left_variant"] in {"C3_v2", "C3_v3"}:
            keep = True
        elif row["comparison_family"] in {"within_model_vs_original_c3", "llava_variant_vs_c0"}:
            keep = True
        else:
            keep = False
        if keep:
            lines.append(
                f"- {row['comparison_label']}: diff={row['rate_diff_left_minus_right'] * 100:.2f} pp, "
                f"left-only={row['left_only']}, right-only={row['right_only']}, "
                f"raw p={format_p(row['p_value_raw'])}, Holm p={format_p(row['p_value_holm'])}, "
                f"Holm significant={'yes' if row['significant_holm'] else 'no'}."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Original `C3` remains the strongest observation: LLaVA-1.5-7B shows "
                f"{int(original['conflict_aligned_n'])}/300 conflict-aligned outputs "
                f"({format_pct(float(original['conflict_aligned_rate']))})."
            ),
            (
                f"Under revised wording, the same model drops to {int(v2['conflict_aligned_n'])}/300 in `C3-v2` "
                f"and {int(v3['conflict_aligned_n'])}/300 in `C3-v3`. The within-model decreases from Original C3 "
                "to both variants are Holm-significant."
            ),
            (
                "The new wording variants no longer show Holm-significant LLaVA-vs-stable-model differences. "
                "Therefore, the paper should state that the observed language-aligned behavior is template-sensitive, "
                "not stable across equivalent wording variants."
            ),
            "",
            "## Paper Paragraph",
            "",
            (
                "The prompt wording control provides an explicit upper bound on the main claim. Although LLaVA-1.5-7B "
                "shows a significant conflict-aligned shift under the original `C3` wording, semantically similar rewrites "
                "substantially weaken or remove the effect. This pattern supports a conservative interpretation: the observed "
                "shift is a limited, wording-sensitive behavior in one model under the original strong misleading template, "
                "rather than evidence for a stable cross-template rule."
            ),
        ]
    )
    write_markdown(output_path, "\n".join(lines))


def make_figure(metrics_df: pd.DataFrame, output_path: Path) -> None:
    x_labels = [VARIANT_DISPLAY[v] for v in VARIANT_ORDER]
    x = np.arange(len(x_labels))
    width = 0.24
    palette = ["#355f8d", "#c16843", "#4d8f5b"]
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    for offset, model_key, color in zip(np.linspace(-width, width, len(MODEL_ORDER)), MODEL_ORDER, palette):
        subset = metrics_df[metrics_df["model_key"] == model_key].set_index("robustness_variant").reindex(VARIANT_ORDER)
        y = subset["conflict_aligned_rate"].to_numpy(dtype=float) * 100.0
        low = (subset["conflict_aligned_rate"] - subset["conflict_aligned_ci_low"]).to_numpy(dtype=float) * 100.0
        high = (subset["conflict_aligned_ci_high"] - subset["conflict_aligned_rate"]).to_numpy(dtype=float) * 100.0
        ax.bar(x + offset, y, width=width, color=color, label=MODEL_DISPLAY[model_key], alpha=0.94)
        ax.errorbar(x + offset, y, yerr=[low, high], fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Conflict-aligned rate (%)")
    ax.set_title("C3 wording boundary control")
    ax.set_ylim(0, max(15.0, float(metrics_df["conflict_aligned_ci_high"].max()) * 100 + 2.5))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_df, boundary_df = load_boundary_df(args.main_csv, args.robustness_csv)
    metrics_df = build_metrics(boundary_df)
    tests_df = build_tests(main_df, boundary_df)
    metrics_df.to_csv(args.output_dir / "prompt_boundary_metrics.csv", index=False, encoding="utf-8-sig")
    tests_df.to_csv(args.output_dir / "prompt_boundary_key_tests.csv", index=False, encoding="utf-8-sig")
    write_summary(metrics_df, tests_df, args.output_dir / "prompt_boundary_summary.md")
    make_figure(metrics_df, args.output_dir / "figure_prompt_boundary.png")
    print(
        json.dumps(
            {
                "metrics": str(args.output_dir / "prompt_boundary_metrics.csv"),
                "key_tests": str(args.output_dir / "prompt_boundary_key_tests.csv"),
                "summary": str(args.output_dir / "prompt_boundary_summary.md"),
                "figure": str(args.output_dir / "figure_prompt_boundary.png"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
