#!/usr/bin/env python
"""Generate paper-ready tables and figures from the new mainline results."""

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

from scripts.utils.paper_mainline_utils import (
    MODEL_ORDER,
    PRIMARY_CONDITION_ORDER,
    dump_json,
    format_ci,
    format_pct,
    load_paper_config,
    paper_paths,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def build_table_md(df: pd.DataFrame, columns: list[str], value_map: dict[str, str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in df.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if column in value_map:
                value = value_map[column].format(**row)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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


def make_main_figure(main_df: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(PRIMARY_CONDITION_ORDER))
    width = 0.24
    offsets = np.linspace(-width, width, num=len(MODEL_ORDER))
    palette = ["#355f8d", "#c16843", "#4d8f5b"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, color, model_key in zip(offsets, palette, MODEL_ORDER):
        subset = (
            main_df[main_df["model_key"] == model_key]
            .set_index("condition_name")
            .reindex(PRIMARY_CONDITION_ORDER)
            .reset_index()
        )
        y = subset["conflict_aligned_rate"].to_numpy(dtype=float)
        lower = y - subset["conflict_aligned_ci_low"].to_numpy(dtype=float)
        upper = subset["conflict_aligned_ci_high"].to_numpy(dtype=float) - y
        ax.bar(x + offset, y, width=width, label=model_key, color=color, alpha=0.92)
        ax.errorbar(x + offset, y, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(PRIMARY_CONDITION_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Conflict-aligned rate")
    ax.set_ylim(0, max(0.12, float(main_df["conflict_aligned_ci_high"].max()) + 0.03))
    ax.set_title("Figure 2. Conflict-aligned rate across C0-C4")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_main_table(main_df: pd.DataFrame, output_dir: Path) -> None:
    table_df = pd.DataFrame(
        {
            "model": main_df["model_key"],
            "condition": main_df["condition_name"],
            "n": main_df["n"].astype(int),
            "conflict_aligned": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    main_df["conflict_aligned_rate"],
                    main_df["conflict_aligned_ci_low"],
                    main_df["conflict_aligned_ci_high"],
                )
            ],
            "faithful": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    main_df["faithful_rate"],
                    main_df["faithful_ci_low"],
                    main_df["faithful_ci_high"],
                )
            ],
            "refusal": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    main_df["refusal_rate"],
                    main_df["refusal_ci_low"],
                    main_df["refusal_ci_high"],
                )
            ],
        }
    )
    table_df.to_csv(output_dir / "table1_main_metrics.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        output_dir / "table1_main_metrics.md",
        "# Table 1. Main experiment metrics\n\n" + build_table_md(table_df, list(table_df.columns), {}),
    )


def write_aux_table(aux_df: pd.DataFrame, output_dir: Path) -> None:
    table_df = pd.DataFrame(
        {
            "model": aux_df["model_key"],
            "condition": aux_df["condition_name"],
            "n": aux_df["n"].astype(int),
            "compliance": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    aux_df["answer_space_compliance_rate"],
                    aux_df["answer_space_compliance_ci_low"],
                    aux_df["answer_space_compliance_ci_high"],
                )
            ],
            "conflict_aligned": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    aux_df["conflict_aligned_rate"],
                    aux_df["conflict_aligned_ci_low"],
                    aux_df["conflict_aligned_ci_high"],
                )
            ],
            "faithful": [
                f"{format_pct(rate)} {format_ci(low, high)}"
                for rate, low, high in zip(
                    aux_df["faithful_rate"],
                    aux_df["faithful_ci_low"],
                    aux_df["faithful_ci_high"],
                )
            ],
        }
    )
    table_df.to_csv(output_dir / "table3_aux_metrics.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        output_dir / "table3_aux_metrics.md",
        "# Table 3. Auxiliary A1/A2 metrics\n\n" + build_table_md(table_df, list(table_df.columns), {}),
    )


def write_appendix_sanity_check(config: dict, output_dir: Path) -> None:
    legacy_path = Path("analysis/current/stanford_core_primary/model_condition_metrics.csv")
    if not legacy_path.exists():
        return
    legacy_df = pd.read_csv(legacy_path, encoding="utf-8-sig")
    current_df = pd.read_csv(output_dir.parent / "main" / "main_condition_metrics.csv", encoding="utf-8-sig")
    keep_columns = ["model_key", "condition_name", "conflict_aligned_rate", "faithful_rate"]
    merged = legacy_df[keep_columns].merge(
        current_df[keep_columns],
        on=["model_key", "condition_name"],
        suffixes=("_stanford_core", "_balanced_eval"),
    )
    merged["conflict_aligned_delta"] = merged["conflict_aligned_rate_balanced_eval"] - merged["conflict_aligned_rate_stanford_core"]
    merged["faithful_delta"] = merged["faithful_rate_balanced_eval"] - merged["faithful_rate_stanford_core"]
    merged.to_csv(output_dir / "stanford_core_sanity_check.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        output_dir / "stanford_core_sanity_check.md",
        "# Appendix sanity check\n\n"
        "This table is a light appendix-only comparison between the old Stanford-only control and the final balanced evaluation set.\n",
    )


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
    distribution_df = pd.read_csv(paths["metadata_dir"] / "dataset_distribution.csv", encoding="utf-8-sig")

    write_main_table(main_df, main_dir)
    write_aux_table(aux_df, aux_dir)
    make_main_figure(main_df, main_dir / "figure2_conflict_aligned_rates.png")
    make_dataset_distribution_plot(distribution_df, appendix_dir / "dataset_distribution.png")
    distribution_df.to_csv(appendix_dir / "dataset_distribution_table.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        appendix_dir / "dataset_distribution.md",
        "# Dataset distribution\n\n"
        "Panel A shows the original Stanford strict-clean seed distribution. "
        "Panel B shows the final balanced evaluation set with VCoR supplementation stacked on top of the retained Stanford seed images.\n",
    )
    write_appendix_sanity_check(config, appendix_dir)

    payload = {
        "table1": str(main_dir / "table1_main_metrics.csv"),
        "figure2": str(main_dir / "figure2_conflict_aligned_rates.png"),
        "table3": str(aux_dir / "table3_aux_metrics.csv"),
        "dataset_figure": str(appendix_dir / "dataset_distribution.png"),
        "baseline_summary": str(baseline_dir / "baseline_summary.md"),
    }
    dump_json(appendix_dir / "figure_manifest.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
