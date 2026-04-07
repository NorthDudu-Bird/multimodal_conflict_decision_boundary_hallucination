#!/usr/bin/env python
"""Analyze current multimodel car-color results with rare-event aware fallbacks."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils._local_deps import ensure_local_deps

ensure_local_deps()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.proportion import proportion_confint

from scripts.utils.restructured_experiment_utils import (
    AUXILIARY_CONDITION_NAMES_V2,
    PRIMARY_CONDITION_NAMES_V2,
    build_logger,
    ensure_dirs,
    load_config,
    relative_str,
)


ROOT = REPO_ROOT
OUTCOME_ORDER = ["faithful", "conflict_aligned", "other_wrong", "refusal_or_correction", "parse_error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multimodel parsed outputs for the current Stanford Cars study.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--family", choices=["primary", "auxiliary"], required=True)
    parser.add_argument("--input-csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-primary-csvs", type=Path, nargs="*", default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def load_df(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in paths]
    df = pd.concat(frames, ignore_index=True)
    for field in [
        "parse_success",
        "is_conflict_aligned",
        "is_faithful",
        "is_other_wrong",
        "is_refusal_or_correction",
        "is_parse_error",
        "in_allowed_answer_space",
    ]:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip().isin(["1", "true", "True"])
    return df


def condition_order_for_family(family: str) -> list[str]:
    return PRIMARY_CONDITION_NAMES_V2 if family == "primary" else AUXILIARY_CONDITION_NAMES_V2


def model_order_from_config(config: dict, df: pd.DataFrame) -> list[str]:
    configured = [model["model_key"] for model in config.get("models", [])]
    present = [model_key for model_key in configured if model_key in set(df["model_key"])]
    remainder = sorted(set(df["model_key"]) - set(present))
    return present + remainder


def rare_event_interval(successes: int, total: int, method: str) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    low, high = proportion_confint(count=successes, nobs=total, alpha=0.05, method=method)
    return float(low), float(high)


def build_model_condition_metrics(df: pd.DataFrame, condition_order: list[str], model_order: list[str], ci_method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in model_order:
        model_df = df[df["model_key"] == model_key].copy()
        model_name = model_df["model_name"].iloc[0]
        checkpoint_name = model_df["checkpoint_name"].iloc[0]
        for condition_name in condition_order:
            subset = model_df[model_df["condition_name"] == condition_name].copy()
            total = len(subset)
            conflict_aligned_n = int(subset["is_conflict_aligned"].sum())
            faithful_n = int(subset["is_faithful"].sum())
            other_wrong_n = int(subset["is_other_wrong"].sum())
            refusal_or_correction_n = int(subset["is_refusal_or_correction"].sum())
            parse_error_n = int(subset["is_parse_error"].sum())

            conflict_ci_low, conflict_ci_high = rare_event_interval(conflict_aligned_n, total, ci_method)
            faithful_ci_low, faithful_ci_high = rare_event_interval(faithful_n, total, ci_method)
            other_wrong_ci_low, other_wrong_ci_high = rare_event_interval(other_wrong_n, total, ci_method)

            row = {
                "model_key": model_key,
                "model_name": model_name,
                "checkpoint_name": checkpoint_name,
                "condition_name": condition_name,
                "n": total,
                "conflict_aligned_n": conflict_aligned_n,
                "conflict_aligned_rate": conflict_aligned_n / total if total else 0.0,
                "conflict_aligned_rate_ci_low": conflict_ci_low,
                "conflict_aligned_rate_ci_high": conflict_ci_high,
                "hallucination_rate": conflict_aligned_n / total if total else 0.0,
                "hallucination_rate_ci_low": conflict_ci_low,
                "hallucination_rate_ci_high": conflict_ci_high,
                "faithful_n": faithful_n,
                "faithful_rate": faithful_n / total if total else 0.0,
                "faithful_rate_ci_low": faithful_ci_low,
                "faithful_rate_ci_high": faithful_ci_high,
                "other_wrong_n": other_wrong_n,
                "other_wrong_rate": other_wrong_n / total if total else 0.0,
                "other_wrong_rate_ci_low": other_wrong_ci_low,
                "other_wrong_rate_ci_high": other_wrong_ci_high,
                "refusal_or_correction_n": refusal_or_correction_n,
                "refusal_or_correction_rate": refusal_or_correction_n / total if total else 0.0,
                "parse_error_n": parse_error_n,
                "parse_error_rate": parse_error_n / total if total else 0.0,
            }
            if "in_allowed_answer_space" in subset.columns:
                in_space_n = int(subset["in_allowed_answer_space"].sum())
                out_space_n = total - in_space_n
                in_space_ci_low, in_space_ci_high = rare_event_interval(in_space_n, total, ci_method)
                in_space_conflict_n = int((subset["in_allowed_answer_space"] & subset["is_conflict_aligned"]).sum())
                out_of_space_faithful_n = int((~subset["in_allowed_answer_space"] & subset["is_faithful"]).sum())
                other_behavior_n = total - in_space_conflict_n - out_of_space_faithful_n
                row.update(
                    {
                        "in_allowed_answer_space_n": in_space_n,
                        "answer_space_compliance_rate": in_space_n / total if total else 0.0,
                        "answer_space_compliance_ci_low": in_space_ci_low,
                        "answer_space_compliance_ci_high": in_space_ci_high,
                        "out_of_allowed_answer_space_n": out_space_n,
                        "out_of_allowed_answer_space_rate": out_space_n / total if total else 0.0,
                        "in_space_conflict_aligned_n": in_space_conflict_n,
                        "in_space_conflict_aligned_rate": in_space_conflict_n / total if total else 0.0,
                        "out_of_space_faithful_n": out_of_space_faithful_n,
                        "out_of_space_faithful_rate": out_of_space_faithful_n / total if total else 0.0,
                        "other_answer_space_behavior_n": other_behavior_n,
                        "other_answer_space_behavior_rate": other_behavior_n / total if total else 0.0,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def pivot_for_model(df: pd.DataFrame, model_key: str, condition_order: list[str]) -> pd.DataFrame:
    model_df = df[df["model_key"] == model_key].copy()
    return (
        model_df.pivot_table(index="image_id", columns="condition_name", values="is_conflict_aligned", aggfunc="max")
        .reindex(columns=condition_order)
        .fillna(0.0)
    )


def paired_bootstrap_difference(values_left: np.ndarray, values_right: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    observed = float(values_left.mean() - values_right.mean())
    if len(values_left) == 0:
        return (observed, 0.0, 0.0)
    draws = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(values_left), size=len(values_left))
        draws.append(float(values_left[sample_idx].mean() - values_right[sample_idx].mean()))
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    return observed, float(ci_low), float(ci_high)


def build_primary_summary_metrics(df: pd.DataFrame, model_order: list[str], bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model_key in model_order:
        pivot = pivot_for_model(df, model_key=model_key, condition_order=PRIMARY_CONDITION_NAMES_V2)
        if pivot.empty:
            continue
        c0 = pivot["C0_neutral"].to_numpy(dtype=float)
        for condition_name in PRIMARY_CONDITION_NAMES_V2:
            observed, ci_low, ci_high = paired_bootstrap_difference(
                pivot[condition_name].to_numpy(dtype=float),
                c0,
                rng=rng,
                n_bootstrap=bootstrap_samples,
            )
            rows.append(
                {
                    "model_key": model_key,
                    "metric_name": f"RPE({condition_name})",
                    "estimate": observed,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

        stronger = pivot[["C2_false_assertion_open", "C3_presupposition_correction_allowed", "C4_stronger_open_conflict"]].mean(axis=1).to_numpy(dtype=float)
        weaker = pivot[["C0_neutral", "C1_weak_suggestion"]].mean(axis=1).to_numpy(dtype=float)
        observed, ci_low, ci_high = paired_bootstrap_difference(stronger, weaker, rng=rng, n_bootstrap=bootstrap_samples)
        rows.append(
            {
                "model_key": model_key,
                "metric_name": "LDI",
                "estimate": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def build_auxiliary_summary_metrics(auxiliary_df: pd.DataFrame, primary_df: pd.DataFrame, model_order: list[str], bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model_key in model_order:
        left = auxiliary_df[(auxiliary_df["model_key"] == model_key) & (auxiliary_df["condition_name"] == "A1_forced_choice_red_family")][["image_id", "is_conflict_aligned"]].copy()
        right = primary_df[(primary_df["model_key"] == model_key) & (primary_df["condition_name"] == "C2_false_assertion_open")][["image_id", "is_conflict_aligned"]].copy()
        merged = left.merge(right, on="image_id", suffixes=("_a1", "_c2"))
        if merged.empty:
            continue
        observed, ci_low, ci_high = paired_bootstrap_difference(
            merged["is_conflict_aligned_a1"].to_numpy(dtype=float),
            merged["is_conflict_aligned_c2"].to_numpy(dtype=float),
            rng=rng,
            n_bootstrap=bootstrap_samples,
        )
        rows.append(
            {
                "model_key": model_key,
                "metric_name": "FSS",
                "estimate": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def build_cross_model_pairwise(df: pd.DataFrame, condition_order: list[str], model_order: list[str], bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for condition_name in condition_order:
        condition_df = df[df["condition_name"] == condition_name].copy()
        for left_model, right_model in combinations(model_order, 2):
            left = condition_df[condition_df["model_key"] == left_model][["image_id", "is_conflict_aligned"]].copy()
            right = condition_df[condition_df["model_key"] == right_model][["image_id", "is_conflict_aligned"]].copy()
            merged = left.merge(right, on="image_id", suffixes=("_left", "_right"))
            if merged.empty:
                continue
            observed, ci_low, ci_high = paired_bootstrap_difference(
                merged["is_conflict_aligned_left"].to_numpy(dtype=float),
                merged["is_conflict_aligned_right"].to_numpy(dtype=float),
                rng=rng,
                n_bootstrap=bootstrap_samples,
            )
            rows.append(
                {
                    "condition_name": condition_name,
                    "left_model": left_model,
                    "right_model": right_model,
                    "estimate_left_minus_right": observed,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def build_auxiliary_answer_space_tables(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    compliance_columns = [
        "model_key",
        "model_name",
        "checkpoint_name",
        "condition_name",
        "n",
        "in_allowed_answer_space_n",
        "answer_space_compliance_rate",
        "answer_space_compliance_ci_low",
        "answer_space_compliance_ci_high",
        "out_of_allowed_answer_space_n",
        "out_of_allowed_answer_space_rate",
        "in_space_conflict_aligned_n",
        "in_space_conflict_aligned_rate",
        "out_of_space_faithful_n",
        "out_of_space_faithful_rate",
        "other_answer_space_behavior_n",
        "other_answer_space_behavior_rate",
    ]
    available_columns = [column for column in compliance_columns if column in metrics_df.columns]
    compliance_df = metrics_df[available_columns].copy()

    breakdown_rows: list[dict[str, object]] = []
    for row in compliance_df.to_dict(orient="records"):
        for label, count_key, rate_key in [
            ("in_space_conflict_aligned", "in_space_conflict_aligned_n", "in_space_conflict_aligned_rate"),
            ("out_of_space_faithful", "out_of_space_faithful_n", "out_of_space_faithful_rate"),
            ("other_behavior", "other_answer_space_behavior_n", "other_answer_space_behavior_rate"),
        ]:
            breakdown_rows.append(
                {
                    "model_key": row["model_key"],
                    "model_name": row["model_name"],
                    "condition_name": row["condition_name"],
                    "category": label,
                    "count": row.get(count_key, 0),
                    "rate": row.get(rate_key, 0.0),
                }
            )
    return compliance_df, pd.DataFrame(breakdown_rows)


def fit_mixed_or_regularized_model(df: pd.DataFrame, family: str, model_order: list[str]) -> tuple[str, pd.DataFrame, str]:
    if df.empty or df["is_conflict_aligned"].nunique() <= 1:
        return ("proportion_only", pd.DataFrame(), "Outcome is constant; only exact proportions and confidence intervals are reported.")

    condition_order = condition_order_for_family(family)
    reference_condition = condition_order[0]
    reference_model = model_order[0]
    working_df = df[["image_id", "model_key", "condition_name", "is_conflict_aligned"]].copy()
    working_df["condition_name"] = pd.Categorical(working_df["condition_name"], categories=condition_order, ordered=True)
    working_df["model_key"] = pd.Categorical(working_df["model_key"], categories=model_order, ordered=True)
    working_df["is_conflict_aligned"] = working_df["is_conflict_aligned"].astype(int)

    try:
        formula = (
            f"is_conflict_aligned ~ C(model_key, Treatment(reference='{reference_model}')) * "
            f"C(condition_name, Treatment(reference='{reference_condition}'))"
        )
        model = BinomialBayesMixedGLM.from_formula(formula, {"image_re": "0 + C(image_id)"}, working_df)
        fit = model.fit_vb()
        rows = []
        for name, estimate, std_error in zip(model.exog_names, fit.fe_mean, fit.fe_sd):
            ci_low = float(estimate - 1.96 * std_error)
            ci_high = float(estimate + 1.96 * std_error)
            rows.append(
                {
                    "term": name,
                    "estimate": float(estimate),
                    "std_error": float(std_error),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "odds_ratio": float(np.exp(estimate)),
                    "odds_ratio_ci_low": float(np.exp(ci_low)),
                    "odds_ratio_ci_high": float(np.exp(ci_high)),
                    "fit_method": "BinomialBayesMixedGLM.fit_vb",
                }
            )
        return ("glmm", pd.DataFrame(rows), "")
    except Exception as exc:
        formula = (
            f"is_conflict_aligned ~ C(model_key, Treatment(reference='{reference_model}')) * "
            f"C(condition_name, Treatment(reference='{reference_condition}')) + C(image_id)"
        )
        try:
            y, x = dmatrices(formula, working_df, return_type="dataframe")
            fit = sm.Logit(y, x).fit_regularized(alpha=1.0, L1_wt=0.0, disp=False)
            rows = []
            for term, estimate in fit.params.items():
                rows.append(
                    {
                        "term": term,
                        "estimate": float(estimate),
                        "std_error": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "odds_ratio": float(np.exp(estimate)),
                        "odds_ratio_ci_low": np.nan,
                        "odds_ratio_ci_high": np.nan,
                        "fit_method": "Logit.fit_regularized",
                    }
                )
            note = (
                "GLMM failed and the analysis fell back to a regularized fixed-effects logistic model "
                f"with image indicators. Failure: {type(exc).__name__}: {exc}"
            )
            return ("regularized_logit_fallback", pd.DataFrame(rows), note)
        except Exception as second_exc:
            note = (
                "Both the preferred mixed-effects model and the regularized fallback failed. "
                f"GLMM failure: {type(exc).__name__}: {exc}. "
                f"Regularized failure: {type(second_exc).__name__}: {second_exc}"
            )
            return ("proportion_only", pd.DataFrame(), note)


def plot_model_panels(metrics_df: pd.DataFrame, model_order: list[str], condition_order: list[str], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(model_order), 1, figsize=(10, 3.4 * len(model_order)), sharex=True)
    if len(model_order) == 1:
        axes = [axes]

    for ax, model_key in zip(axes, model_order):
        subset = metrics_df[metrics_df["model_key"] == model_key].set_index("condition_name").reindex(condition_order).reset_index()
        x = np.arange(len(condition_order))
        y = subset["hallucination_rate"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - subset["hallucination_rate_ci_low"].to_numpy(dtype=float),
                subset["hallucination_rate_ci_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color="#1F5A7A", ecolor="#1F5A7A", capsize=4)
        ax.set_title(model_key)
        ax.set_ylabel("HR")
        ax.set_ylim(0, min(1.0, max(0.08, subset["hallucination_rate_ci_high"].max() + 0.08)))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(condition_order, rotation=15, ha="right")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cross_model_comparison(metrics_df: pd.DataFrame, model_order: list[str], condition_order: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(condition_order))
    offsets = np.linspace(-0.22, 0.22, num=max(1, len(model_order)))
    for offset, model_key in zip(offsets, model_order):
        subset = metrics_df[metrics_df["model_key"] == model_key].set_index("condition_name").reindex(condition_order).reset_index()
        y = subset["hallucination_rate"].to_numpy(dtype=float)
        ax.plot(x + offset, y, marker="o", linewidth=1.8, label=model_key)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_order, rotation=15, ha="right")
    ax.set_ylabel("Hallucination Rate")
    ax.set_ylim(0, min(1.0, max(0.08, metrics_df["hallucination_rate_ci_high"].max() + 0.08)))
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_outcome_distribution(df: pd.DataFrame, model_order: list[str], condition_order: list[str], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(model_order), 1, figsize=(12, 3.8 * len(model_order)), sharex=True)
    if len(model_order) == 1:
        axes = [axes]
    palette = {
        "faithful": "#2F7D32",
        "conflict_aligned": "#C0392B",
        "other_wrong": "#C6860A",
        "refusal_or_correction": "#6C5B7B",
        "parse_error": "#7F8C8D",
    }
    for ax, model_key in zip(axes, model_order):
        summary = (
            df[df["model_key"] == model_key]
            .groupby(["condition_name", "outcome_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=condition_order, columns=OUTCOME_ORDER, fill_value=0)
        )
        proportions = summary.div(summary.sum(axis=1), axis=0).fillna(0.0)
        bottom = np.zeros(len(proportions))
        for outcome in OUTCOME_ORDER:
            values = proportions[outcome].to_numpy(dtype=float)
            ax.bar(proportions.index, values, bottom=bottom, label=outcome, color=palette[outcome], alpha=0.92)
            bottom += values
        ax.set_title(model_key)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[-1].tick_params(axis="x", rotation=15)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_hr_heatmap(metrics_df: pd.DataFrame, model_order: list[str], condition_order: list[str], output_path: Path, title: str) -> None:
    heatmap = (
        metrics_df.pivot(index="model_key", columns="condition_name", values="hallucination_rate")
        .reindex(index=model_order, columns=condition_order)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(heatmap.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(0.1, float(heatmap.to_numpy(dtype=float).max())))
    ax.set_xticks(np.arange(len(condition_order)))
    ax.set_xticklabels(condition_order, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order)
    for row_idx, model_key in enumerate(model_order):
        for col_idx, condition_name in enumerate(condition_order):
            value = heatmap.loc[model_key, condition_name]
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_answer_space_compliance(metrics_df: pd.DataFrame, model_order: list[str], condition_order: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(condition_order))
    offsets = np.linspace(-0.22, 0.22, num=max(1, len(model_order)))
    for offset, model_key in zip(offsets, model_order):
        subset = metrics_df[metrics_df["model_key"] == model_key].set_index("condition_name").reindex(condition_order).reset_index()
        y = subset["answer_space_compliance_rate"].to_numpy(dtype=float)
        ax.plot(x + offset, y, marker="o", linewidth=1.8, label=model_key)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_order, rotation=15, ha="right")
    ax.set_ylabel("Compliance Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Auxiliary Answer-Space Compliance")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_answer_space_breakdown(breakdown_df: pd.DataFrame, model_order: list[str], condition_order: list[str], output_path: Path) -> None:
    if breakdown_df.empty:
        return
    fig, axes = plt.subplots(len(model_order), 1, figsize=(11, 3.6 * len(model_order)), sharex=True)
    if len(model_order) == 1:
        axes = [axes]
    palette = {
        "in_space_conflict_aligned": "#C0392B",
        "out_of_space_faithful": "#0F766E",
        "other_behavior": "#6B7280",
    }
    category_order = ["in_space_conflict_aligned", "out_of_space_faithful", "other_behavior"]
    for ax, model_key in zip(axes, model_order):
        subset = (
            breakdown_df[breakdown_df["model_key"] == model_key]
            .pivot_table(index="condition_name", columns="category", values="rate", aggfunc="sum")
            .reindex(index=condition_order, columns=category_order, fill_value=0.0)
        )
        bottom = np.zeros(len(subset))
        for category in category_order:
            values = subset[category].to_numpy(dtype=float)
            ax.bar(subset.index, values, bottom=bottom, color=palette[category], label=category, alpha=0.92)
            bottom += values
        ax.set_title(model_key)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Proportion")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[-1].tick_params(axis="x", rotation=15)
    fig.suptitle("Auxiliary Answer-Space Behavior Breakdown", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_markdown_summary(output_dir: Path, family: str, metrics_df: pd.DataFrame, summary_df: pd.DataFrame, model_status: str, model_note: str) -> Path:
    lines = [
        f"# Multimodel {'Primary' if family == 'primary' else 'Auxiliary'} Analysis",
        "",
        f"- model_status: {model_status}",
    ]
    if model_note:
        lines.append(f"- model_note: {model_note}")
    if family == "primary":
        lines.extend(["", "## Primary Descriptive Rates by Model and Condition"])
    else:
        lines.extend(["", "## Auxiliary Descriptive Rates by Model and Condition"])
    for row in metrics_df.sort_values(["model_key", "condition_name"]).to_dict(orient="records"):
        description = (
            f"- {row['model_key']} | {row['condition_name']}: "
            f"HR={row['conflict_aligned_rate']:.4f} "
            f"(95% CI {row['conflict_aligned_rate_ci_low']:.4f}, {row['conflict_aligned_rate_ci_high']:.4f}); "
            f"faithful={row['faithful_rate']:.4f} "
            f"(95% CI {row['faithful_rate_ci_low']:.4f}, {row['faithful_rate_ci_high']:.4f}); "
            f"other_wrong={row['other_wrong_rate']:.4f} "
            f"(95% CI {row['other_wrong_rate_ci_low']:.4f}, {row['other_wrong_rate_ci_high']:.4f}); "
            f"n={row['n']}"
        )
        if family == "auxiliary" and "answer_space_compliance_rate" in row:
            description += (
                f"; compliance={row['answer_space_compliance_rate']:.4f} "
                f"(95% CI {row['answer_space_compliance_ci_low']:.4f}, {row['answer_space_compliance_ci_high']:.4f}); "
                f"in-space+conflict={row['in_space_conflict_aligned_rate']:.4f}; "
                f"out-of-space+faithful={row['out_of_space_faithful_rate']:.4f}; "
                f"other={row['other_answer_space_behavior_rate']:.4f}"
            )
        lines.append(description)
    if not summary_df.empty:
        lines.extend(["", "## Derived Metrics"])
        for row in summary_df.sort_values(["model_key", "metric_name"]).to_dict(orient="records"):
            lines.append(
                f"- {row['model_key']} | {row['metric_name']}: {row['estimate']:.4f} "
                f"(95% CI {row['ci_low']:.4f}, {row['ci_high']:.4f})"
            )
    output_path = output_dir / "analysis_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    ensure_dirs([output_dir, output_dir / "plots"])
    log_path = args.log_path or output_dir / "analysis.log"
    logger = build_logger("analyze_multimodel_car_color_results", log_path)

    combined_df = load_df(paths=args.input_csvs)
    combined_df = combined_df[combined_df["condition_name"].isin(condition_order_for_family(args.family))].copy()
    condition_order = condition_order_for_family(args.family)
    model_order = model_order_from_config(config=config, df=combined_df)
    logger.info("Loaded %s rows across models=%s", len(combined_df), model_order)

    combined_csv = output_dir / f"{args.family}_combined_parsed_results.csv"
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")

    ci_method = str(config["analysis"].get("rare_event_ci_method", "beta"))
    metrics_df = build_model_condition_metrics(combined_df, condition_order=condition_order, model_order=model_order, ci_method=ci_method)
    metrics_path = output_dir / "model_condition_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    answer_space_metrics_path = None
    answer_space_breakdown_path = None
    answer_space_compliance_plot = None
    answer_space_breakdown_plot = None
    if args.family == "auxiliary" and "answer_space_compliance_rate" in metrics_df.columns:
        answer_space_metrics_df, answer_space_breakdown_df = build_auxiliary_answer_space_tables(metrics_df)
        answer_space_metrics_path = output_dir / "answer_space_compliance_metrics.csv"
        answer_space_breakdown_path = output_dir / "answer_space_behavior_breakdown.csv"
        answer_space_metrics_df.to_csv(answer_space_metrics_path, index=False, encoding="utf-8-sig")
        answer_space_breakdown_df.to_csv(answer_space_breakdown_path, index=False, encoding="utf-8-sig")

    bootstrap_samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["analysis"]["random_seed"])
    if args.family == "primary":
        summary_df = build_primary_summary_metrics(combined_df, model_order=model_order, bootstrap_samples=bootstrap_samples, seed=seed)
    else:
        reference_primary_paths = args.reference_primary_csvs or []
        if not reference_primary_paths:
            raise ValueError("Auxiliary analysis requires --reference-primary-csvs to compute FSS.")
        primary_df = load_df(paths=reference_primary_paths)
        summary_df = build_auxiliary_summary_metrics(
            auxiliary_df=combined_df,
            primary_df=primary_df,
            model_order=model_order,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    pairwise_df = build_cross_model_pairwise(
        combined_df,
        condition_order=condition_order,
        model_order=model_order,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    pairwise_path = output_dir / "cross_model_pairwise.csv"
    pairwise_df.to_csv(pairwise_path, index=False, encoding="utf-8-sig")

    model_status, model_results_df, model_note = fit_mixed_or_regularized_model(combined_df, family=args.family, model_order=model_order)
    model_results_path = output_dir / "model_fit_results.csv"
    model_results_df.to_csv(model_results_path, index=False, encoding="utf-8-sig")

    plot_model_panels(
        metrics_df=metrics_df,
        model_order=model_order,
        condition_order=condition_order,
        output_path=output_dir / "plots" / f"{args.family}_hr_by_model.png",
        title=f"{args.family.capitalize()} Hallucination Rate by Model",
    )
    plot_cross_model_comparison(
        metrics_df=metrics_df,
        model_order=model_order,
        condition_order=condition_order,
        output_path=output_dir / "plots" / f"{args.family}_cross_model_comparison.png",
        title=f"{args.family.capitalize()} Cross-Model Hallucination Comparison",
    )
    plot_outcome_distribution(
        df=combined_df,
        model_order=model_order,
        condition_order=condition_order,
        output_path=output_dir / "plots" / f"{args.family}_outcome_distribution.png",
        title=f"{args.family.capitalize()} Outcome Distribution",
    )
    plot_hr_heatmap(
        metrics_df=metrics_df,
        model_order=model_order,
        condition_order=condition_order,
        output_path=output_dir / "plots" / f"{args.family}_hr_heatmap.png",
        title=f"{args.family.capitalize()} HR Heatmap",
    )
    if args.family == "auxiliary" and answer_space_metrics_path is not None and answer_space_breakdown_path is not None:
        answer_space_compliance_plot = output_dir / "plots" / "auxiliary_answer_space_compliance.png"
        answer_space_breakdown_plot = output_dir / "plots" / "auxiliary_answer_space_breakdown.png"
        plot_answer_space_compliance(
            metrics_df=metrics_df,
            model_order=model_order,
            condition_order=condition_order,
            output_path=answer_space_compliance_plot,
        )
        plot_answer_space_breakdown(
            breakdown_df=pd.read_csv(answer_space_breakdown_path, encoding="utf-8-sig"),
            model_order=model_order,
            condition_order=condition_order,
            output_path=answer_space_breakdown_plot,
        )

    summary_md = write_markdown_summary(
        output_dir=output_dir,
        family=args.family,
        metrics_df=metrics_df,
        summary_df=summary_df,
        model_status=model_status,
        model_note=model_note,
    )

    summary_json = output_dir / "analysis_summary.json"
    payload = {
        "family": args.family,
        "combined_csv": relative_str(combined_csv),
        "metrics_csv": relative_str(metrics_path),
        "summary_metrics_csv": relative_str(summary_path),
        "pairwise_csv": relative_str(pairwise_path),
        "model_fit_results_csv": relative_str(model_results_path),
        "summary_md": relative_str(summary_md),
        "model_status": model_status,
        "model_note": model_note,
    }
    if answer_space_metrics_path is not None:
        payload["answer_space_compliance_csv"] = relative_str(answer_space_metrics_path)
    if answer_space_breakdown_path is not None:
        payload["answer_space_breakdown_csv"] = relative_str(answer_space_breakdown_path)
    if answer_space_compliance_plot is not None:
        payload["answer_space_compliance_plot"] = relative_str(answer_space_compliance_plot)
    if answer_space_breakdown_plot is not None:
        payload["answer_space_breakdown_plot"] = relative_str(answer_space_breakdown_plot)
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Analysis complete: %s", json.dumps(payload, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
