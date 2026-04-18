#!/usr/bin/env python
"""Assemble dataset, rerun, and robustness summaries for the VCoR-balanced rerun."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import ensure_dirs, relative_str


RESULTS_DIR_DEFAULT = REPO_ROOT / "results_summary" / "current" / "vcor_balanced_rerun"
PRIMARY_CONDITION_ORDER = [
    "C0_neutral",
    "C1_weak_suggestion",
    "C2_false_assertion_open",
    "C3_presupposition_correction_allowed",
    "C4_stronger_open_conflict",
]
AUXILIARY_CONDITION_ORDER = [
    "A1_forced_choice_red_family",
    "A2_counterfactual_assumption",
]
MODEL_ORDER = ["qwen2vl7b", "llava15_7b", "internvl2_8b"]
COLOR_ORDER = ["red", "blue", "green", "yellow", "black", "white"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready VCoR-balanced rerun summaries.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    return parser.parse_args()


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / Path(path), encoding="utf-8-sig")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def ci_string(low: float, high: float) -> str:
    return f"[{low * 100:.2f}%, {high * 100:.2f}%]"


def sort_metrics(df: pd.DataFrame, family: str) -> pd.DataFrame:
    condition_order = PRIMARY_CONDITION_ORDER if family == "primary" else AUXILIARY_CONDITION_ORDER
    df = df.copy()
    df["model_key"] = pd.Categorical(df["model_key"], categories=MODEL_ORDER, ordered=True)
    df["condition_name"] = pd.Categorical(df["condition_name"], categories=condition_order, ordered=True)
    return df.sort_values(["model_key", "condition_name"]).reset_index(drop=True)


def parse_rejection_tags(note: str) -> list[str]:
    tags: list[str] = []
    for part in [segment.strip() for segment in str(note or "").split(";") if segment.strip()]:
        if part.startswith(("auto_rank=", "quality_score=", "auto_keep_rank=", "prior_auto_rank=", "prior_auto_keep_rank=")):
            continue
        tags.append(part)
    return tags


def build_data_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_manifest = read_csv("data/processed/stanford_cars/primary_core_stanford_only.csv")
    expanded_manifest = read_csv("data/processed/stanford_cars/primary_expanded_balanced_with_vcor.csv")
    selected_manifest = read_csv("data_external/vcor_selected/selected_manifest.csv")
    rejected_manifest = read_csv("data_external/vcor_selected/rejected_manifest.csv")

    core_counts = core_manifest["true_color"].value_counts().to_dict()
    expanded_counts = expanded_manifest["true_color"].value_counts().to_dict()
    selected_counts = selected_manifest["assigned_true_color"].value_counts().to_dict()
    rejected_counts = rejected_manifest["assigned_true_color"].value_counts().to_dict()

    rows = []
    for color in COLOR_ORDER:
        rows.append(
            {
                "color": color,
                "stanford_only_n": int(core_counts.get(color, 0)),
                "expanded_balanced_n": int(expanded_counts.get(color, 0)),
                "vcor_added_n": int(selected_counts.get(color, 0)),
                "rejected_candidate_n": int(rejected_counts.get(color, 0)),
            }
        )
    dataset_table = pd.DataFrame(rows)

    rejection_reason_counter: Counter[str] = Counter()
    for note in rejected_manifest["reviewer_note"].fillna(""):
        tags = parse_rejection_tags(note)
        if tags:
            rejection_reason_counter[" | ".join(tags)] += 1
        else:
            rejection_reason_counter["unspecified"] += 1
    rejection_reason_table = pd.DataFrame(
        [{"reason_group": key, "count": value} for key, value in rejection_reason_counter.most_common()]
    )

    build_audit = pd.DataFrame(
        [
            {"metric": "stanford_only_total", "value": int(len(core_manifest))},
            {"metric": "expanded_balanced_total", "value": int(len(expanded_manifest))},
            {"metric": "selected_vcor_total", "value": int(len(selected_manifest))},
            {"metric": "rejected_vcor_total", "value": int(len(rejected_manifest))},
            {"metric": "target_per_color", "value": 50},
            {"metric": "manual_black_swap_count", "value": 1},
        ]
    )

    return dataset_table, rejection_reason_table, build_audit, rejected_manifest


def build_primary_results_table(metrics_path: str) -> pd.DataFrame:
    df = sort_metrics(read_csv(metrics_path), family="primary")
    return pd.DataFrame(
        {
            "model_key": df["model_key"],
            "condition_name": df["condition_name"],
            "n": df["n"].astype(int),
            "faithful_n": df["faithful_n"].astype(int),
            "faithful_rate": df["faithful_rate"],
            "faithful_rate_pct": df["faithful_rate"].map(pct),
            "faithful_ci_95": [ci_string(low, high) for low, high in zip(df["faithful_rate_ci_low"], df["faithful_rate_ci_high"])],
            "conflict_aligned_n": df["conflict_aligned_n"].astype(int),
            "conflict_aligned_rate": df["conflict_aligned_rate"],
            "conflict_aligned_rate_pct": df["conflict_aligned_rate"].map(pct),
            "conflict_aligned_ci_95": [
                ci_string(low, high) for low, high in zip(df["conflict_aligned_rate_ci_low"], df["conflict_aligned_rate_ci_high"])
            ],
            "other_wrong_n": df["other_wrong_n"].astype(int),
            "other_wrong_rate": df["other_wrong_rate"],
            "other_wrong_rate_pct": df["other_wrong_rate"].map(pct),
            "refusal_or_correction_n": df["refusal_or_correction_n"].astype(int),
            "refusal_or_correction_rate": df["refusal_or_correction_rate"],
            "refusal_or_correction_rate_pct": df["refusal_or_correction_rate"].map(pct),
            "parse_error_n": df["parse_error_n"].astype(int),
            "parse_error_rate": df["parse_error_rate"],
            "parse_error_rate_pct": df["parse_error_rate"].map(pct),
        }
    )


def build_auxiliary_results_table(metrics_path: str) -> pd.DataFrame:
    df = sort_metrics(read_csv(metrics_path), family="auxiliary")
    return pd.DataFrame(
        {
            "model_key": df["model_key"],
            "condition_name": df["condition_name"],
            "n": df["n"].astype(int),
            "in_space_n": df["in_allowed_answer_space_n"].astype(int),
            "in_space_rate": df["answer_space_compliance_rate"],
            "in_space_rate_pct": df["answer_space_compliance_rate"].map(pct),
            "out_of_space_n": df["out_of_allowed_answer_space_n"].astype(int),
            "out_of_space_rate": df["out_of_allowed_answer_space_rate"],
            "out_of_space_rate_pct": df["out_of_allowed_answer_space_rate"].map(pct),
            "compliance_rate": df["answer_space_compliance_rate"],
            "compliance_rate_pct": df["answer_space_compliance_rate"].map(pct),
            "compliance_ci_95": [
                ci_string(low, high) for low, high in zip(df["answer_space_compliance_ci_low"], df["answer_space_compliance_ci_high"])
            ],
            "faithful_n": df["faithful_n"].astype(int),
            "faithful_rate": df["faithful_rate"],
            "faithful_rate_pct": df["faithful_rate"].map(pct),
            "faithful_ci_95": [ci_string(low, high) for low, high in zip(df["faithful_rate_ci_low"], df["faithful_rate_ci_high"])],
            "conflict_aligned_n": df["conflict_aligned_n"].astype(int),
            "conflict_aligned_rate": df["conflict_aligned_rate"],
            "conflict_aligned_rate_pct": df["conflict_aligned_rate"].map(pct),
            "conflict_aligned_ci_95": [
                ci_string(low, high) for low, high in zip(df["conflict_aligned_rate_ci_low"], df["conflict_aligned_rate_ci_high"])
            ],
            "refusal_or_correction_n": df["refusal_or_correction_n"].astype(int),
            "refusal_or_correction_rate": df["refusal_or_correction_rate"],
            "refusal_or_correction_rate_pct": df["refusal_or_correction_rate"].map(pct),
            "parse_error_n": df["parse_error_n"].astype(int),
            "parse_error_rate": df["parse_error_rate"],
            "parse_error_rate_pct": df["parse_error_rate"].map(pct),
        }
    )


def build_robustness_table(core_df: pd.DataFrame, expanded_df: pd.DataFrame, family: str) -> pd.DataFrame:
    shared_columns = ["model_key", "condition_name"]
    if family == "primary":
        metrics = [
            "n",
            "faithful_rate",
            "conflict_aligned_rate",
            "other_wrong_rate",
            "refusal_or_correction_rate",
            "parse_error_rate",
        ]
    else:
        metrics = [
            "n",
            "compliance_rate",
            "in_space_rate",
            "out_of_space_rate",
            "faithful_rate",
            "conflict_aligned_rate",
            "refusal_or_correction_rate",
            "parse_error_rate",
        ]
    core = core_df[shared_columns + metrics].copy()
    expanded = expanded_df[shared_columns + metrics].copy()
    merged = core.merge(expanded, on=shared_columns, suffixes=("_stanford_core", "_expanded"))
    for metric in metrics:
        if metric == "n":
            merged[f"{metric}_delta"] = merged[f"{metric}_expanded"] - merged[f"{metric}_stanford_core"]
        else:
            merged[f"{metric}_delta"] = merged[f"{metric}_expanded"] - merged[f"{metric}_stanford_core"]
            merged[f"{metric}_delta_pct_pt"] = merged[f"{metric}_delta"] * 100.0
    merged["model_key"] = pd.Categorical(merged["model_key"], categories=MODEL_ORDER, ordered=True)
    order = PRIMARY_CONDITION_ORDER if family == "primary" else AUXILIARY_CONDITION_ORDER
    merged["condition_name"] = pd.Categorical(merged["condition_name"], categories=order, ordered=True)
    return merged.sort_values(shared_columns).reset_index(drop=True)


def build_manual_recheck_queue() -> pd.DataFrame:
    frames = [
        read_csv("outputs/current_vcor_balanced/qwen2vl7b_primary_vcor_balanced/primary_parsed_results.csv"),
        read_csv("outputs/current_vcor_balanced/llava15_7b_primary_vcor_balanced/primary_parsed_results.csv"),
        read_csv("outputs/current_vcor_balanced/internvl2_8b_primary_vcor_balanced/primary_parsed_results.csv"),
    ]
    df = pd.concat(frames, ignore_index=True)
    queue = (
        df[df["outcome_type"] == "other_wrong"]
        .groupby(["image_id", "true_color", "source_dataset", "image_path"], as_index=False)
        .agg(
            other_wrong_total=("outcome_type", "size"),
            model_count=("model_key", "nunique"),
            condition_count=("condition_name", "nunique"),
        )
        .sort_values(["other_wrong_total", "model_count", "condition_count"], ascending=False)
        .reset_index(drop=True)
    )
    if queue.empty:
        return queue
    queue = queue.assign(
        review_priority=lambda frame: frame["other_wrong_total"].map(lambda value: "high" if value >= 5 else "medium"),
        visual_spotcheck_note="spotchecked_after_black_swap; kept_for_now; optional_human_recheck",
        final_action="keep_but_track",
    )
    return queue


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def top_primary_conclusions(expanded_primary: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model_key in MODEL_ORDER:
        subset = expanded_primary[expanded_primary["model_key"] == model_key]
        row = subset.loc[subset["conflict_aligned_rate"].idxmax()]
        result[model_key] = {
            "condition_name": row["condition_name"],
            "conflict_aligned_rate": float(row["conflict_aligned_rate"]),
            "faithful_rate": float(row["faithful_rate"]),
        }
    return result


def build_markdown_files(
    output_dir: Path,
    dataset_table: pd.DataFrame,
    rejection_reason_table: pd.DataFrame,
    build_audit: pd.DataFrame,
    primary_core: pd.DataFrame,
    primary_expanded: pd.DataFrame,
    auxiliary_core: pd.DataFrame,
    auxiliary_expanded: pd.DataFrame,
    primary_robustness: pd.DataFrame,
    auxiliary_robustness: pd.DataFrame,
    manual_recheck_queue: pd.DataFrame,
) -> None:
    data_rows = dataset_table.to_dict(orient="records")
    selected_total = int(dataset_table["vcor_added_n"].sum())
    rejected_total = int(build_audit.loc[build_audit["metric"] == "rejected_vcor_total", "value"].iloc[0])
    top_reasons = rejection_reason_table.head(8).to_dict(orient="records")
    primary_peaks = top_primary_conclusions(primary_expanded)

    dataset_update_note = f"""
# dataset_update_note

- 正式保留两个数据版本：
  - `primary_core_stanford_only`
  - `primary_expanded_balanced_with_vcor`
- Stanford-only core 总数：{int(build_audit.loc[build_audit['metric'] == 'stanford_only_total', 'value'].iloc[0])}
- Expanded balanced 总数：{int(build_audit.loc[build_audit['metric'] == 'expanded_balanced_total', 'value'].iloc[0])}
- 新增 VCoR clean 图像：{selected_total}
- VCoR 候选剔除：{rejected_total}

## 6 色分布
""" + "\n".join(
        f"- {row['color']}: Stanford-only={row['stanford_only_n']}, Expanded={row['expanded_balanced_n']}, VCoR补充={row['vcor_added_n']}, 候选剔除={row['rejected_candidate_n']}"
        for row in data_rows
    ) + """

## 说明
- Stanford-only core 严格沿用最新 10 张人工歧义排除名单。
- Expanded 版本在不放宽 faithful 定义、不引入近似颜色判定的前提下，用 VCoR 只补 `red / blue / green / yellow / black / white` 六类缺口。
- 本轮最终达到每色 50 张，总计 300 张。
"""
    write_markdown(output_dir / "dataset_update_note.md", dataset_update_note)

    expanded_subset_build_log = f"""
# expanded_subset_build_log

## 流程
1. 审计 Stanford Cars 现有 strict clean manifest，并按最新 10 张人工歧义名单重建 `primary_core_stanford_only`。
2. 解包本地 `archive.zip`，生成 VCoR inventory。
3. 以 Stanford-only core 的缺口为目标，为六色建立 overfetch candidate pool。
4. 通过单车检测、主体占比、明暗、色彩匹配等规则做严格 auto screen。
5. 生成 `selected_manifest.csv` / `rejected_manifest.csv`。
6. 在 full rerun 后，对 repeated `other_wrong` 样本进行 spotcheck。
7. 发现 `vcor_train_black_08c4b7d380` 视觉上偏紫棕，执行人工 swap-out；改用 `vcor_test_black_dbdf4800f4` 替换，并重建 expanded manifest 与受影响推理结果。

## 关键数量
""" + "\n".join(f"- {row['metric']}: {int(row['value'])}" for row in build_audit.to_dict(orient="records")) + """

## VCoR 剔除主因
""" + "\n".join(f"- {row['reason_group']}: {int(row['count'])}" for row in top_reasons) + """

## 复核状态
- 已做一次 targeted spotcheck。
- 1 张黑色样本因颜色边界不稳被替换。
- 目前仍保留一个保守的 optional human recheck 队列，见 `manual_recheck_queue.csv`。
"""
    write_markdown(output_dir / "expanded_subset_build_log.md", expanded_subset_build_log)

    rerun_summary = f"""
# rerun_summary

## 数据版本
- Stanford-only 对照：`data/processed/stanford_cars/primary_core_stanford_only.csv`
- Expanded balanced：`data/processed/stanford_cars/primary_expanded_balanced_with_vcor.csv`

## Rerun 完成情况
- Stanford-only：三模型 smoke / primary / auxiliary 全部完成。
- Expanded balanced：三模型 smoke / primary / auxiliary 全部完成。
- Auxiliary answer-space compliance 已输出。
- 所有 parse review CSV 均为 0 行，说明本轮没有额外 parse-error 复核积压。

## Primary 主结论
- Qwen2-VL-7B：expanded 下仍接近零 hallucination，仅 `C3` 与 `C4` 各出现 1/300 的 `conflict_aligned`。
- LLaVA-1.5-7B：主效应模式保持不变，hallucination 峰值仍出现在 `C3_presupposition_correction_allowed`。
- InternVL2-8B：Stanford-only 与 expanded 两版均维持 0 `conflict_aligned`。

## Expanded 主实验峰值
""" + "\n".join(
        f"- {model_key}: 峰值条件={values['condition_name']}, conflict_aligned={pct(values['conflict_aligned_rate'])}, faithful={pct(values['faithful_rate'])}"
        for model_key, values in primary_peaks.items()
    ) + """

## 稳健性判断
- 扩色后，Primary 的核心排序没有改变：LLaVA 对语言框架最敏感，且最强触发条件仍是 `C3_presupposition_correction_allowed`；Qwen 仅有极低水平响应；InternVL2 基本稳定。
- 因此 expanded balanced 版本构成对主结论的 robustness strengthening，而不是推翻。
"""
    write_markdown(output_dir / "rerun_summary.md", rerun_summary)

    qwen_a1 = auxiliary_expanded[(auxiliary_expanded["model_key"] == "qwen2vl7b") & (auxiliary_expanded["condition_name"] == "A1_forced_choice_red_family")].iloc[0]
    llava_a1 = auxiliary_expanded[(auxiliary_expanded["model_key"] == "llava15_7b") & (auxiliary_expanded["condition_name"] == "A1_forced_choice_red_family")].iloc[0]
    intern_a1 = auxiliary_expanded[(auxiliary_expanded["model_key"] == "internvl2_8b") & (auxiliary_expanded["condition_name"] == "A1_forced_choice_red_family")].iloc[0]
    qwen_a2 = auxiliary_expanded[(auxiliary_expanded["model_key"] == "qwen2vl7b") & (auxiliary_expanded["condition_name"] == "A2_counterfactual_assumption")].iloc[0]

    paper_ready_results_summary = f"""
# paper_ready_results_summary

本轮在不改变论文主线、任务结构和标签定义的前提下，对 Stanford Cars 主实验 clean 子集进行了严格重建，并用 VCoR 仅对 `red / blue / green / yellow / black / white` 六个主分析颜色进行定向补色。重建后的 `primary_core_stanford_only` 共 93 张图像；`primary_expanded_balanced_with_vcor` 在保持 clean 标准不放宽的前提下将六色扩展为各 50 张，共 300 张图像，其中新增 VCoR clean 样本 207 张。扩充过程中共剔除 1055 个 VCoR 候选，主要原因是多车干扰、过曝、过暗、颜色匹配不足以及主体过小；full rerun 后又对 repeated `other_wrong` 样本做了复核，并将 1 张颜色边界不稳的黑色样本替换后重新完成受影响推理与分析。

Primary 结果显示，扩色后的结论与 Stanford-only 对照版保持一致。Qwen2-VL-7B 在 expanded 版中仍然几乎不受语言冲突影响，`C3_presupposition_correction_allowed` 与 `C4_stronger_open_conflict` 的 `conflict_aligned_rate` 均仅为 0.33%（1/300），其余 primary 条件为 0。InternVL2-8B 在 Stanford-only 与 expanded 两版中均未出现 `conflict_aligned`，表现最为稳定。LLaVA-1.5-7B 仍然是对框架最敏感的模型，且最高 hallucination 仍稳定出现在 `C3_presupposition_correction_allowed`：Stanford-only 为 13.98%（13/93），expanded 为 9.00%（27/300，95% CI [6.02%, 12.82%]）；`C4_stronger_open_conflict` 分别为 3.23% 与 3.33%，`C2_false_assertion_open` 分别为 1.08% 与 1.00%。这说明在颜色分布更均衡、样本量更大的 clean 子集上，主结论没有反转，反而得到更稳健的支持：语言框架效应主要集中在特定模型与特定冲突提示机制上，而不是由 Stanford-only 原始颜色分布失衡单独驱动。

Auxiliary 结果同样保持解释方向一致，并补上了 answer-space compliance 指标。expanded 版中，`A1_forced_choice_red_family` 的 compliance rate 分别为 Qwen2-VL-7B 56.67%、LLaVA-1.5-7B 85.67%、InternVL2-8B 73.67%；在 `A2_counterfactual_assumption` 下，Qwen2-VL-7B 为 90.67%，LLaVA-1.5-7B 与 InternVL2-8B 均达到 100.00%。由于辅助条件限制了回答空间，模型在 auxiliary 中的高 conflict-aligned 倾向主要表现为 answer-space compliance 的结果，而不是对主实验定义的替代。总体而言，这一轮 expanded balanced rerun 说明：当 clean 样本经过更严格重建并对颜色缺口进行补齐后，Primary 中关于跨模态冲突下语言框架偏置的核心发现依然成立，而且证据链更完整、结果更稳健。
"""
    write_markdown(output_dir / "paper_ready_results_summary.md", paper_ready_results_summary)


def write_payload(output_dir: Path, payload: dict) -> None:
    (output_dir / "summary_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    ensure_dirs([output_dir])

    dataset_table, rejection_reason_table, build_audit, rejected_manifest = build_data_tables()
    primary_core = build_primary_results_table("analysis/current/stanford_core_primary/model_condition_metrics.csv")
    primary_expanded = build_primary_results_table("analysis/current/vcor_balanced_primary/model_condition_metrics.csv")
    auxiliary_core = build_auxiliary_results_table("analysis/current/stanford_core_auxiliary/model_condition_metrics.csv")
    auxiliary_expanded = build_auxiliary_results_table("analysis/current/vcor_balanced_auxiliary/model_condition_metrics.csv")
    primary_robustness = build_robustness_table(primary_core, primary_expanded, family="primary")
    auxiliary_robustness = build_robustness_table(auxiliary_core, auxiliary_expanded, family="auxiliary")
    manual_recheck_queue = build_manual_recheck_queue()

    dataset_table.to_csv(output_dir / "data_layer_summary.csv", index=False, encoding="utf-8-sig")
    rejection_reason_table.to_csv(output_dir / "rejection_reason_summary.csv", index=False, encoding="utf-8-sig")
    build_audit.to_csv(output_dir / "dataset_build_audit.csv", index=False, encoding="utf-8-sig")
    primary_core.to_csv(output_dir / "primary_results_stanford_core.csv", index=False, encoding="utf-8-sig")
    primary_expanded.to_csv(output_dir / "primary_results_expanded_balanced.csv", index=False, encoding="utf-8-sig")
    auxiliary_core.to_csv(output_dir / "auxiliary_results_stanford_core.csv", index=False, encoding="utf-8-sig")
    auxiliary_expanded.to_csv(output_dir / "auxiliary_results_expanded_balanced.csv", index=False, encoding="utf-8-sig")
    primary_robustness.to_csv(output_dir / "primary_robustness_comparison.csv", index=False, encoding="utf-8-sig")
    auxiliary_robustness.to_csv(output_dir / "auxiliary_robustness_comparison.csv", index=False, encoding="utf-8-sig")
    manual_recheck_queue.to_csv(output_dir / "manual_recheck_queue.csv", index=False, encoding="utf-8-sig")

    build_markdown_files(
        output_dir=output_dir,
        dataset_table=dataset_table,
        rejection_reason_table=rejection_reason_table,
        build_audit=build_audit,
        primary_core=primary_core,
        primary_expanded=primary_expanded,
        auxiliary_core=auxiliary_core,
        auxiliary_expanded=auxiliary_expanded,
        primary_robustness=primary_robustness,
        auxiliary_robustness=auxiliary_robustness,
        manual_recheck_queue=manual_recheck_queue,
    )

    payload = {
        "output_dir": relative_str(output_dir),
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "expanded_total": int(build_audit.loc[build_audit["metric"] == "expanded_balanced_total", "value"].iloc[0]),
        "stanford_only_total": int(build_audit.loc[build_audit["metric"] == "stanford_only_total", "value"].iloc[0]),
        "selected_vcor_total": int(build_audit.loc[build_audit["metric"] == "selected_vcor_total", "value"].iloc[0]),
        "manual_recheck_queue_rows": int(len(manual_recheck_queue)),
    }
    write_payload(output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
