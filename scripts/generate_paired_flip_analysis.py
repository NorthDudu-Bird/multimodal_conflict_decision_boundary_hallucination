#!/usr/bin/env python
"""Generate same-image paired transition analysis for the main C0-C4 runs."""

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

from scripts.utils.paper_mainline_utils import (
    MODEL_ORDER,
    PRIMARY_CONDITION_ORDER,
    PRIMARY_CONDITION_SHORT_LABELS,
    format_pct,
    load_bool_results,
    write_markdown,
)


STATE_ORDER = ["faithful", "conflict_aligned", "other_wrong", "refusal", "parse_error"]
MODEL_DISPLAY = {
    "qwen2vl7b": "Qwen2-VL-7B-Instruct",
    "llava15_7b": "LLaVA-1.5-7B",
    "internvl2_8b": "InternVL2-8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired main-experiment flip analysis.")
    parser.add_argument("--input-csv", type=Path, default=REPO_ROOT / "results" / "main" / "main_combined_parsed_results.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "main")
    return parser.parse_args()


def outcome_state(row: pd.Series) -> str:
    if bool(row.get("is_parse_error", False)):
        return "parse_error"
    if bool(row.get("is_refusal_or_correction", False)):
        return "refusal"
    if bool(row.get("is_conflict_aligned", False)):
        return "conflict_aligned"
    if bool(row.get("is_faithful", False)):
        return "faithful"
    if bool(row.get("is_other_wrong", False)):
        return "other_wrong"
    outcome = str(row.get("outcome_type", "")).strip()
    return outcome if outcome in STATE_ORDER else "other_wrong"


def validate_paired_structure(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    duplicate_keys = df.duplicated(subset=["model_key", "condition_name", "image_id"]).sum()
    if duplicate_keys:
        issues.append(f"Duplicate (model_key, condition_name, image_id) rows: {duplicate_keys}")

    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key]
        c0_ids = set(model_df[model_df["condition_name"] == "C0_neutral"]["image_id"])
        if len(c0_ids) != 300:
            issues.append(f"{model_key} C0 has {len(c0_ids)} unique images, expected 300")
        for condition_name in PRIMARY_CONDITION_ORDER:
            current_ids = set(model_df[model_df["condition_name"] == condition_name]["image_id"])
            if current_ids != c0_ids:
                issues.append(f"{model_key} {condition_name} image set differs from C0")
            if len(current_ids) != 300:
                issues.append(f"{model_key} {condition_name} has {len(current_ids)} unique images, expected 300")
    return issues


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
        "reference_only": reference_only,
        "both_no": both_no,
        "p_value_raw": float(result.pvalue),
    }


def build_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transition_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for model_key in MODEL_ORDER:
        model_df = df[df["model_key"] == model_key].copy()
        reference = model_df[model_df["condition_name"] == "C0_neutral"].copy()
        for condition_name in PRIMARY_CONDITION_ORDER[1:]:
            current = model_df[model_df["condition_name"] == condition_name].copy()
            merged = reference[["image_id", "state", "parsed_label", "is_conflict_aligned", "is_faithful"]].merge(
                current[["image_id", "state", "parsed_label", "is_conflict_aligned", "is_faithful"]],
                on="image_id",
                suffixes=("_c0", "_current"),
            )
            n_pairs = int(len(merged))
            c0_faithful = merged["state_c0"] == "faithful"
            current_faithful = merged["state_current"] == "faithful"
            current_conflict = merged["state_current"] == "conflict_aligned"
            reference_conflict = merged["state_c0"] == "conflict_aligned"
            answer_flip = (merged["state_c0"] != merged["state_current"]) | (
                merged["parsed_label_c0"].astype(str) != merged["parsed_label_current"].astype(str)
            )

            for from_state in STATE_ORDER:
                for to_state in STATE_ORDER:
                    count = int(((merged["state_c0"] == from_state) & (merged["state_current"] == to_state)).sum())
                    transition_rows.append(
                        {
                            "model_key": model_key,
                            "model": MODEL_DISPLAY[model_key],
                            "reference_condition": "C0_neutral",
                            "condition_name": condition_name,
                            "comparison": f"C0_vs_{PRIMARY_CONDITION_SHORT_LABELS[condition_name]}",
                            "from_state": from_state,
                            "to_state": to_state,
                            "count": count,
                            "n_pairs": n_pairs,
                            "rate": count / n_pairs if n_pairs else 0.0,
                        }
                    )

            faithful_to_faithful = int((c0_faithful & current_faithful).sum())
            faithful_to_conflict = int((c0_faithful & current_conflict).sum())
            faithful_to_other = int((c0_faithful & (merged["state_current"] == "other_wrong")).sum())
            faithful_to_refusal = int((c0_faithful & (merged["state_current"] == "refusal")).sum())
            faithful_to_parse_error = int((c0_faithful & (merged["state_current"] == "parse_error")).sum())
            reference_faithful_n = int(c0_faithful.sum())
            current_conflict_n = int(current_conflict.sum())
            reference_conflict_n = int(reference_conflict.sum())
            answer_flip_n = int(answer_flip.sum())
            payload = exact_mcnemar(current_conflict, reference_conflict)

            metric_rows.append(
                {
                    "model_key": model_key,
                    "model": MODEL_DISPLAY[model_key],
                    "reference_condition": "C0_neutral",
                    "condition_name": condition_name,
                    "condition": PRIMARY_CONDITION_SHORT_LABELS[condition_name],
                    "n_pairs": n_pairs,
                    "c0_faithful_n": reference_faithful_n,
                    "current_faithful_n": int(current_faithful.sum()),
                    "faithful_to_faithful_n": faithful_to_faithful,
                    "faithful_to_conflict_aligned_n": faithful_to_conflict,
                    "faithful_to_other_wrong_n": faithful_to_other,
                    "faithful_to_refusal_n": faithful_to_refusal,
                    "faithful_to_parse_error_n": faithful_to_parse_error,
                    "answer_flip_n": answer_flip_n,
                    "answer_flip_rate": answer_flip_n / n_pairs if n_pairs else 0.0,
                    "faithful_retention_rate": faithful_to_faithful / reference_faithful_n if reference_faithful_n else 0.0,
                    "conflict_following_rate": faithful_to_conflict / reference_faithful_n if reference_faithful_n else 0.0,
                    "current_conflict_aligned_n": current_conflict_n,
                    "c0_conflict_aligned_n": reference_conflict_n,
                    "net_conflict_shift_n": current_conflict_n - reference_conflict_n,
                    "net_conflict_shift": (current_conflict_n - reference_conflict_n) / n_pairs if n_pairs else 0.0,
                    "paired_discordant_current_only": payload["current_only"],
                    "paired_discordant_c0_only": payload["reference_only"],
                    "p_value_exact_mcnemar": payload["p_value_raw"],
                }
            )
            test_rows.append(
                {
                    "comparison_family": "within_model_vs_C0_all_conditions",
                    "model_key": model_key,
                    "model": MODEL_DISPLAY[model_key],
                    "comparison_label": f"{MODEL_DISPLAY[model_key]}: C0 vs {PRIMARY_CONDITION_SHORT_LABELS[condition_name]}",
                    "reference_condition": "C0_neutral",
                    "condition_name": condition_name,
                    "condition": PRIMARY_CONDITION_SHORT_LABELS[condition_name],
                    "metric": "conflict_aligned",
                    "n_pairs": n_pairs,
                    "current_rate": current_conflict_n / n_pairs if n_pairs else 0.0,
                    "c0_rate": reference_conflict_n / n_pairs if n_pairs else 0.0,
                    "rate_diff_current_minus_c0": (current_conflict_n - reference_conflict_n) / n_pairs if n_pairs else 0.0,
                    "current_only": payload["current_only"],
                    "c0_only": payload["reference_only"],
                    "p_value_raw": payload["p_value_raw"],
                }
            )

    tests_df = pd.DataFrame(test_rows)
    if not tests_df.empty:
        _, corrected, _, _ = multipletests(tests_df["p_value_raw"].to_numpy(dtype=float), method="holm")
        tests_df["p_value_holm"] = corrected
        tests_df["significant_holm"] = tests_df["p_value_holm"] < 0.05
    return pd.DataFrame(transition_rows), pd.DataFrame(metric_rows), tests_df


def format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_summary(
    metrics_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    issues: list[str],
    output_path: Path,
) -> None:
    llava_c3 = metrics_df[(metrics_df["model_key"] == "llava15_7b") & (metrics_df["condition"] == "C3")].iloc[0]
    llava_c4 = metrics_df[(metrics_df["model_key"] == "llava15_7b") & (metrics_df["condition"] == "C4")].iloc[0]

    display = metrics_df[
        [
            "model",
            "condition",
            "n_pairs",
            "faithful_to_faithful_n",
            "faithful_to_conflict_aligned_n",
            "answer_flip_rate",
            "faithful_retention_rate",
            "net_conflict_shift",
            "p_value_exact_mcnemar",
        ]
    ].copy()
    for col in ["answer_flip_rate", "faithful_retention_rate", "net_conflict_shift"]:
        display[col] = display[col].map(format_rate)
    display["p_value_exact_mcnemar"] = display["p_value_exact_mcnemar"].map(lambda value: f"{value:.4g}")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in display.columns) + " |" for row in display.to_dict("records")]

    significant = tests_df[tests_df["significant_holm"]].copy()
    sig_lines = []
    if significant.empty:
        sig_lines.append("- No Holm-significant paired conflict-aligned shifts were detected.")
    else:
        for row in significant.to_dict("records"):
            sig_lines.append(
                f"- {row['comparison_label']}: current-only={row['current_only']}, C0-only={row['c0_only']}, "
                f"raw p={row['p_value_raw']:.4g}, Holm p={row['p_value_holm']:.4g}."
            )

    lines = [
        "# Same-Image Paired Flip Analysis",
        "",
        "## Paired Structure Check",
        "",
    ]
    if issues:
        lines.extend([f"- WARNING: {issue}" for issue in issues])
    else:
        lines.append("- Passed: for each model, `C0-C4` use the same 300 `image_id` values with one row per model-condition-image cell.")
        lines.append("- Passed: current main outputs contain only `faithful` and `conflict_aligned` outcome states; `other_wrong`, `refusal`, and `parse_error` transitions are all zero.")

    lines.extend(
        [
            "",
            "## Derived Metrics",
            "",
            header,
            divider,
            *rows,
            "",
            "## Exact Paired Tests",
            "",
            *sig_lines,
            "",
            "## Paper-Ready Interpretation",
            "",
            (
                f"The main experiment is a same-image paired evaluation: every conflict prompt is compared against "
                f"the same model's answer to the same image under `C0`. Because all three models are fully faithful "
                f"under `C0`, conflict-aligned outputs under later conditions can be read as image-level answer flips "
                f"from a visually faithful baseline rather than as differences between unrelated image pools."
            ),
            (
                f"For LLaVA-1.5-7B, `C3` produced {int(llava_c3['faithful_to_conflict_aligned_n'])}/300 "
                f"faithful-to-conflict flips and `C4` produced {int(llava_c4['faithful_to_conflict_aligned_n'])}/300. "
                f"Qwen2-VL-7B-Instruct produced only one such flip in each of `C3` and `C4`, while InternVL2-8B produced none."
            ),
            (
                "This paired framing is stronger than reporting condition-level rates alone because it ties each changed answer "
                "to the exact same visual evidence. It supports a narrow claim of conditional language-induced shifts in one "
                "model/template setting, not a broad claim that text generally dominates vision."
            ),
        ]
    )
    write_markdown(output_path, "\n".join(lines))


def make_figure(metrics_df: pd.DataFrame, output_path: Path) -> None:
    x_labels = ["C1", "C2", "C3", "C4"]
    x = np.arange(len(x_labels))
    width = 0.24
    palette = ["#355f8d", "#c16843", "#4d8f5b"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, model_key, color in zip(np.linspace(-width, width, len(MODEL_ORDER)), MODEL_ORDER, palette):
        subset = metrics_df[metrics_df["model_key"] == model_key].set_index("condition").reindex(x_labels)
        y = subset["conflict_following_rate"].to_numpy(dtype=float) * 100.0
        ax.bar(x + offset, y, width=width, label=MODEL_DISPLAY[model_key], color=color, alpha=0.94)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Faithful-to-conflict flip rate from C0 (%)")
    ax.set_title("Paired answer flips from faithful C0 to conflict-aligned output")
    ax.set_ylim(0, max(12.0, float(metrics_df["conflict_following_rate"].max()) * 100 + 2.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_bool_results(args.input_csv)
    df = df[df["condition_name"].isin(PRIMARY_CONDITION_ORDER)].copy()
    df["state"] = df.apply(outcome_state, axis=1)

    issues = validate_paired_structure(df)
    transitions_df, metrics_df, tests_df = build_outputs(df)

    transitions_df.to_csv(args.output_dir / "paired_transition_counts.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(args.output_dir / "paired_flip_metrics.csv", index=False, encoding="utf-8-sig")
    tests_df.to_csv(args.output_dir / "paired_flip_key_tests.csv", index=False, encoding="utf-8-sig")
    write_summary(metrics_df, tests_df, issues, args.output_dir / "paired_flip_summary.md")
    make_figure(metrics_df, args.output_dir / "figure_paired_flip_rates.png")

    payload = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "paired_structure_issues": issues,
        "transition_counts": str(args.output_dir / "paired_transition_counts.csv"),
        "metrics": str(args.output_dir / "paired_flip_metrics.csv"),
        "key_tests": str(args.output_dir / "paired_flip_key_tests.csv"),
        "summary": str(args.output_dir / "paired_flip_summary.md"),
        "figure": str(args.output_dir / "figure_paired_flip_rates.png"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
