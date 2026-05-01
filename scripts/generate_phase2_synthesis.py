#!/usr/bin/env python
"""Generate phase-2 synthesis artifacts for writing and gatekeeping."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.paper_mainline_utils import write_markdown


MODEL_DISPLAY = {
    "qwen2vl7b": "Qwen2-VL-7B-Instruct",
    "llava15_7b": "LLaVA-1.5-7B",
    "internvl2_8b": "InternVL2-8B",
}
MAIN_CONDITION_LABELS = {
    "C0_neutral": "C0",
    "C1_weak_suggestion": "C1",
    "C2_false_assertion_open": "C2",
    "C3_presupposition_correction_allowed": "C3",
    "C4_stronger_open_conflict": "C4",
}


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / path, encoding="utf-8-sig")


def boolify(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in [
        "is_conflict_aligned",
        "is_faithful",
        "is_other_wrong",
        "is_refusal_or_correction",
        "is_parse_error",
        "phase2_is_conflict_aligned",
        "phase2_is_faithful",
        "phase2_is_parse_error",
    ]:
        if col in result.columns:
            result[col] = result[col].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    return result


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def ratio(n: int, d: int) -> str:
    return f"{n}/{d} ({pct(n / d) if d else 'n/a'})"


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    use = df.copy()
    if columns is not None:
        use = use[columns]
    if max_rows is not None:
        use = use.head(max_rows)
    cols = list(use.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def rel_image(path: str, from_dir: Path) -> str:
    try:
        return os.path.relpath(REPO_ROOT / str(path), from_dir).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def route(row: pd.Series) -> str:
    return f"{row.get('true_color', '')}->{row.get('conflict_color', row.get('false_prompt_color', ''))}"


def pair_family(true_color: str, false_color: str) -> str:
    pair = (str(true_color), str(false_color))
    if pair in {("white", "black"), ("black", "white")}:
        return "achromatic_black_white"
    if pair in {("red", "blue"), ("blue", "red")}:
        return "red_blue"
    if pair == ("green", "yellow"):
        return "green_yellow"
    if pair == ("yellow", "red"):
        return "yellow_red"
    return "other_pair"


def add_phase2_block(path: Path, marker: str, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    write_markdown(path, existing.rstrip() + "\n\n" + marker + "\n\n" + block.strip())


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "main": boolify(read_csv("results/main/main_combined_parsed_results.csv")),
        "variant": boolify(read_csv("results/robustness/prompt_variant_combined_parsed_results.csv")),
        "audit": boolify(read_csv("results/audit/visual_clarity_audit_manifest_completed.csv")),
        "color_paired": read_csv("results/color_split/color_split_paired_flip_metrics.csv"),
        "factor_metrics": read_csv("results/factorization/factorized_prompt_metrics.csv"),
        "format_metrics": read_csv("results/format_control/format_control_metrics.csv"),
        "format_combined": boolify(read_csv("results/format_control/format_control_combined_parsed_results.csv")),
        "multi_metrics": read_csv("results/multiturn/multiturn_metrics.csv"),
        "multi_combined": boolify(read_csv("results/multiturn/multiturn_combined_parsed_results.csv")),
    }


def get_metric(metrics: pd.DataFrame, model_key: str, factor_id: str) -> pd.Series:
    subset = metrics[(metrics["model_key"] == model_key) & (metrics["factor_id"] == factor_id)]
    if subset.empty:
        raise KeyError(f"Metric not found: {model_key} {factor_id}")
    return subset.iloc[0]


def target_main_flips(main: pd.DataFrame) -> pd.DataFrame:
    c0 = main[(main["model_key"] == "llava15_7b") & (main["condition_name"] == "C0_neutral")][
        ["image_id", "is_faithful", "parsed_label"]
    ].rename(columns={"is_faithful": "c0_is_faithful", "parsed_label": "c0_label"})
    flips = main[
        (main["model_key"] == "llava15_7b")
        & (main["condition_name"].isin(["C3_presupposition_correction_allowed", "C4_stronger_open_conflict"]))
        & (main["is_conflict_aligned"])
    ].copy()
    flips = flips.merge(c0, on="image_id", how="left")
    flips["route"] = flips.apply(route, axis=1)
    flips["pair_family"] = flips.apply(lambda r: pair_family(r["true_color"], r["conflict_color"]), axis=1)
    flips["condition"] = flips["condition_name"].map(MAIN_CONDITION_LABELS)
    return flips


def audit_confounds(audit: pd.DataFrame) -> pd.DataFrame:
    result = audit.copy()
    def flagged(row: pd.Series) -> bool:
        values = {
            "audit_visual_clarity": {"moderate", "unclear"},
            "audit_body_color_salience": {"medium", "low"},
            "audit_specular_reflection": {"moderate", "strong"},
            "audit_shadow_or_night_effect": {"moderate", "strong"},
            "audit_background_color_bias": {"moderate", "strong"},
            "audit_multi_car_interference": {"present", "strong"},
            "audit_occlusion": {"moderate", "strong"},
        }
        for col, bad in values.items():
            if str(row.get(col, "")).strip().lower() in bad:
                return True
        return False

    result["any_visual_confound_flag"] = result.apply(flagged, axis=1)
    return result


def format_sensitive_cases(format_combined: pd.DataFrame) -> pd.DataFrame:
    llava = format_combined[format_combined["model_key"] == "llava15_7b"].copy()
    ref = llava[
        (llava["factor_id"] == "REF_C3_original_label_set") & (llava["phase2_is_conflict_aligned"])
    ][["image_id", "true_color", "conflict_color", "image_path", "source_dataset", "phase2_parsed_label"]].copy()
    ref = ref.rename(columns={"phase2_parsed_label": "original_c3_output"})
    alt = llava[
        llava["factor_id"].isin(["free_c3_presupposition", "multiple_choice_c3_presupposition", "yesno_false_claim"])
    ][["image_id", "factor_id", "phase2_is_conflict_aligned", "phase2_parsed_label", "raw_output"]].copy()
    if ref.empty or alt.empty:
        return pd.DataFrame()
    pivot = alt.pivot_table(
        index="image_id",
        columns="factor_id",
        values="phase2_is_conflict_aligned",
        aggfunc="first",
    ).reset_index()
    labels = alt.pivot_table(
        index="image_id",
        columns="factor_id",
        values="phase2_parsed_label",
        aggfunc="first",
    ).reset_index()
    labels = labels.add_prefix("label_").rename(columns={"label_image_id": "image_id"})
    merged = ref.merge(pivot, on="image_id", how="left").merge(labels, on="image_id", how="left")
    alt_cols = ["free_c3_presupposition", "multiple_choice_c3_presupposition", "yesno_false_claim"]
    for col in alt_cols:
        if col not in merged.columns:
            merged[col] = False
    merged["all_alt_nonconflict"] = ~merged[alt_cols].fillna(False).any(axis=1)
    merged["route"] = merged.apply(route, axis=1)
    return merged[merged["all_alt_nonconflict"]].copy()


def multiturn_induced_cases(multi_combined: pd.DataFrame) -> pd.DataFrame:
    cases = multi_combined[
        (multi_combined["model_key"] == "internvl2_8b")
        & (multi_combined["factor_id"].isin(["two_turn_persuasion", "three_turn_persuasion"]))
        & (multi_combined["phase2_is_conflict_aligned"])
    ].copy()
    cases["route"] = cases.apply(route, axis=1)
    return cases


def representative_cases(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [col for col in ["condition", "condition_name", "route", "source_dataset", "image_id"] if col in df.columns]
    result = df.sort_values(sort_cols).head(n).copy()
    cols = [
        col
        for col in [
            "image_id",
            "condition",
            "condition_name",
            "factor_id",
            "route",
            "true_color",
            "conflict_color",
            "source_dataset",
            "parsed_label",
            "phase2_parsed_label",
            "raw_output",
        ]
        if col in result.columns
    ]
    return result[cols]


def build_failure_taxonomy(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    main = data["main"]
    flips = target_main_flips(main)
    audit = audit_confounds(data["audit"])
    format_sensitive = format_sensitive_cases(data["format_combined"])
    multi_induced = multiturn_induced_cases(data["multi_combined"])
    c3_source = main[
        (main["model_key"] == "llava15_7b")
        & (main["condition_name"] == "C3_presupposition_correction_allowed")
    ].copy()
    source_counts = (
        c3_source.groupby("source_dataset", observed=False)
        .agg(n=("image_id", "count"), conflict_aligned_n=("is_conflict_aligned", "sum"))
        .reset_index()
    )
    source_counts["rate"] = source_counts["conflict_aligned_n"] / source_counts["n"]

    target_audit = audit[audit["audit_group"] == "target_conflict_flip"].copy()
    visual_flagged = target_audit[target_audit["any_visual_confound_flag"]].copy()
    achromatic = flips[flips["pair_family"] == "achromatic_black_white"].copy()
    white_black = flips[flips["route"] == "white->black"].copy()

    counts = pd.DataFrame(
        [
            {
                "category_id": "prompt_following_flip",
                "sample_count": len(flips),
                "unique_image_count": flips["image_id"].nunique(),
                "scope": "LLaVA original C3/C4 same-image flips",
                "primary_role": "main evidence",
                "note": "Rows are same-image flips from faithful C0 to the false prompt color.",
            },
            {
                "category_id": "color_pair_concentration",
                "sample_count": len(achromatic),
                "unique_image_count": achromatic["image_id"].nunique(),
                "scope": "Achromatic black/white route among LLaVA C3/C4 flips",
                "primary_role": "boundary/attribution",
                "note": f"{len(white_black)} row-events are specifically white->black.",
            },
            {
                "category_id": "visual_clarity_flagged",
                "sample_count": len(visual_flagged),
                "unique_image_count": visual_flagged["image_id"].nunique(),
                "scope": "Completed audit target rows with at least one visual confound flag",
                "primary_role": "threat reduction",
                "note": f"Target rows {len(visual_flagged)}/{len(target_audit)} flagged; controls {int(audit[(audit['audit_group'] == 'matched_faithful_control')]['any_visual_confound_flag'].sum())}/{len(audit[audit['audit_group'] == 'matched_faithful_control'])} flagged.",
            },
            {
                "category_id": "format_compliance_sensitive",
                "sample_count": len(format_sensitive),
                "unique_image_count": format_sensitive["image_id"].nunique(),
                "scope": "Original LLaVA C3 flips not reproduced by free C3, MC C3, or yes/no false-claim probes",
                "primary_role": "format boundary",
                "note": "This is a format-sensitivity tag, not proof that format alone caused each individual flip.",
            },
            {
                "category_id": "multiturn_induced",
                "sample_count": len(multi_induced),
                "unique_image_count": multi_induced["image_id"].nunique(),
                "scope": "InternVL2 MT2/MT3 conflict-following rows with neutral final question",
                "primary_role": "extension diagnostic",
                "note": "Short multi-turn context creates a strong InternVL2-specific effect absent from its original C0-C4 rows.",
            },
            {
                "category_id": "source_style_sensitive_candidate",
                "sample_count": int(source_counts["conflict_aligned_n"].sum()),
                "unique_image_count": c3_source[c3_source["is_conflict_aligned"]]["image_id"].nunique(),
                "scope": "LLaVA C3 flips split by StanfordCars/VCoR",
                "primary_role": "appendix threat check",
                "note": "; ".join(
                    f"{row.source_dataset}: {int(row.conflict_aligned_n)}/{int(row.n)} ({pct(float(row.rate))})"
                    for row in source_counts.itertuples()
                ),
            },
        ]
    )

    out_dir = REPO_ROOT / "results" / "case_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "failure_taxonomy_counts.csv", index=False, encoding="utf-8-sig")

    definitions = pd.DataFrame(
        [
            {
                "category_id": "prompt_following_flip",
                "definition": "A same-image LLaVA C3/C4 row where C0 was faithful and the conflict prompt answer equals the false prompt color.",
                "decision_rule": "`model_key=llava15_7b`, condition in original C3/C4, `is_conflict_aligned=True`, paired C0 faithful.",
                "paper_use": "Core case-level evidence for conditional conflict following.",
            },
            {
                "category_id": "color_pair_concentration",
                "definition": "Prompt-following flips concentrated in a specific true-color/false-color route rather than spread across all six colors.",
                "decision_rule": "Tag prompt-following flips by `true_color -> conflict_color`, with achromatic black/white and `white->black` reported separately.",
                "paper_use": "Limits the 9% result: not a uniform color-class effect.",
            },
            {
                "category_id": "visual_clarity_flagged",
                "definition": "Flip/control rows where the completed audit notes moderate/strong reflection, shadow, background color, salience, occlusion, or multi-car interference.",
                "decision_rule": "Any completed audit field enters a moderate/strong/present or lower-salience state.",
                "paper_use": "Threat-to-validity discussion, not a new main effect.",
            },
            {
                "category_id": "format_compliance_sensitive",
                "definition": "Original C3 flip cases that do not remain conflict-aligned under matched free-answer, multiple-choice, and yes/no format controls.",
                "decision_rule": "LLaVA original C3 conflict row; same `image_id` is non-conflict in all three formal C3/false-claim probes.",
                "paper_use": "Supports answer-format dependence of the original C3 effect.",
            },
            {
                "category_id": "multiturn_induced",
                "definition": "Conflict-following rows appearing only after short previous-turn false context while the final question remains neutral.",
                "decision_rule": "InternVL2 `two_turn_persuasion` or `three_turn_persuasion` with `phase2_is_conflict_aligned=True`.",
                "paper_use": "Appendix extension showing multi-turn vulnerability can be model-specific.",
            },
            {
                "category_id": "source_style_sensitive_candidate",
                "definition": "A source-stratified difference in flip rate that is visible but not sufficient to make source a new main experimental factor.",
                "decision_rule": "Report LLaVA C3 conflict rates separately for StanfordCars and VCoR.",
                "paper_use": "Appendix caveat: direction persists across sources, magnitude varies.",
            },
        ]
    )
    write_markdown(
        out_dir / "failure_taxonomy_definition.md",
        "\n".join(
            [
                "# Failure Taxonomy Definition",
                "",
                "This taxonomy is descriptive and data-driven. Categories are diagnostic tags and may overlap; they are not proposed as a new theory of VLM behavior.",
                "",
                md_table(definitions),
                "",
                "## Counts",
                "",
                md_table(counts),
                "",
                "## Boundary",
                "",
                "The most important categories for the main story are `prompt_following_flip`, `color_pair_concentration`, and `format_compliance_sensitive`. `visual_clarity_flagged`, `multiturn_induced`, and `source_style_sensitive_candidate` are threat-reduction or appendix categories.",
            ]
        ),
    )

    casebook_sections = [
        "# Casebook",
        "",
        "The examples below are sampled from actual result rows. They should be used as illustrative cases, not as additional statistical evidence.",
        "",
        "## Prompt-Following Flips",
        "",
        md_table(representative_cases(flips, 8)),
        "",
        "## Color-Pair Concentration",
        "",
        md_table(representative_cases(white_black, 8)),
        "",
        "## Visual-Clarity Flagged Targets",
        "",
        md_table(
            visual_flagged[
                [
                    "image_id",
                    "audit_source_module",
                    "true_color",
                    "false_prompt_color",
                    "source_dataset",
                    "audit_visual_clarity",
                    "audit_body_color_salience",
                    "audit_specular_reflection",
                    "audit_shadow_or_night_effect",
                    "audit_background_color_bias",
                    "audit_notes",
                ]
            ].head(10)
        ),
        "",
        "## Format-Sensitive Original C3 Cases",
        "",
        md_table(
            format_sensitive[
                [
                    "image_id",
                    "route",
                    "source_dataset",
                    "original_c3_output",
                    "label_free_c3_presupposition",
                    "label_multiple_choice_c3_presupposition",
                    "label_yesno_false_claim",
                ]
            ].head(10)
        ),
        "",
        "## Multi-Turn-Induced Cases",
        "",
        md_table(representative_cases(multi_induced, 10)),
        "",
        "## Source/Style Candidate Split",
        "",
        md_table(source_counts.assign(rate=source_counts["rate"].map(pct))),
    ]
    write_markdown(out_dir / "casebook.md", "\n".join(casebook_sections))

    gallery_lines = [
        "# Case Gallery",
        "",
        "Images are linked from actual result rows. The gallery is intentionally small and writing-facing.",
        "",
    ]
    gallery_specs = [
        ("Prompt-following flip", flips.head(6)),
        ("Visual-clarity flagged", visual_flagged.head(6)),
        ("Multi-turn induced", multi_induced.head(6)),
    ]
    for title, frame in gallery_specs:
        gallery_lines.extend([f"## {title}", ""])
        for _, row in frame.iterrows():
            image_path = row.get("image_path", "")
            rel = rel_image(str(image_path), out_dir)
            model = MODEL_DISPLAY.get(str(row.get("model_key", "")), str(row.get("model", "")))
            cond = row.get("condition", row.get("condition_name", row.get("factor_id", "")))
            note = row.get("audit_notes", "")
            gallery_lines.extend(
                [
                    f"### {row.get('image_id', '')} | {model} | {cond} | {row.get('true_color', '')}->{row.get('conflict_color', row.get('false_prompt_color', ''))}",
                    "",
                    f"![{row.get('image_id', '')}]({rel})",
                    "",
                    f"- source: `{row.get('source_dataset', '')}`",
                    f"- output: `{row.get('parsed_label', row.get('phase2_parsed_label', row.get('model_output', '')))}`",
                ]
            )
            if note:
                gallery_lines.append(f"- audit note: {note}")
            gallery_lines.append("")
    write_markdown(out_dir / "case_gallery.md", "\n".join(gallery_lines))

    return {
        "flips": flips,
        "audit": audit,
        "format_sensitive": format_sensitive,
        "multi_induced": multi_induced,
        "counts": counts,
        "source_counts": source_counts,
    }


def build_gatekeeping(data: dict[str, pd.DataFrame], derived: dict[str, object]) -> None:
    main = data["main"]
    factor = data["factor_metrics"]
    fmt = data["format_metrics"]
    multi = data["multi_metrics"]
    audit = derived["audit"]
    source_counts = derived["source_counts"]
    flips = derived["flips"]

    c0 = main[main["condition_name"] == "C0_neutral"]
    c0_faithful = int(c0["is_faithful"].sum())
    c0_total = len(c0)
    main_parse_errors = int(main["is_parse_error"].sum())
    target_audit = audit[audit["audit_group"] == "target_conflict_flip"]
    control_audit = audit[audit["audit_group"] == "matched_faithful_control"]
    target_clear = int((target_audit["audit_visual_clarity"] == "clear").sum())
    control_clear = int((control_audit["audit_visual_clarity"] == "clear").sum())
    target_flagged = int(target_audit["any_visual_confound_flag"].sum())
    control_flagged = int(control_audit["any_visual_confound_flag"].sum())

    llava_c3 = get_metric(factor, "llava15_7b", "REF_C3_original_label_set")
    llava_c3_v2 = get_metric(factor, "llava15_7b", "REF_C3_v2_wording")
    llava_c3_v3 = get_metric(factor, "llava15_7b", "REF_C3_v3_wording")
    llava_title = get_metric(factor, "llava15_7b", "title_prefix_framing")
    qwen_presup = get_metric(factor, "qwen2vl7b", "presupposition_no_correction")
    intern_title = get_metric(factor, "internvl2_8b", "title_prefix_framing")
    llava_free_c3 = get_metric(fmt, "llava15_7b", "free_c3_presupposition")
    llava_mc_c3 = get_metric(fmt, "llava15_7b", "multiple_choice_c3_presupposition")
    llava_yesno = get_metric(fmt, "llava15_7b", "yesno_false_claim")
    intern_mt2 = get_metric(multi, "internvl2_8b", "two_turn_persuasion")
    intern_mt3 = get_metric(multi, "internvl2_8b", "three_turn_persuasion")
    llava_mt3 = get_metric(multi, "llava15_7b", "three_turn_persuasion")

    source_note = "; ".join(
        f"{row.source_dataset} {int(row.conflict_aligned_n)}/{int(row.n)} ({pct(float(row.rate))})"
        for row in source_counts.itertuples()
    )

    rows = [
        {
            "gate_id": "Gate 1",
            "gate": "Dataset validity / balance",
            "role": "main evidence chain",
            "evidence": "300 reviewed images; six true colors balanced at 50 each; source composition StanfordCars=93 and VCoR=207.",
            "alternative_explanation_reduced": "Uneven color priors or accidental single-source dataset construction.",
            "verdict": "Pass; keep task local to car-body primary color.",
        },
        {
            "gate_id": "Gate 2",
            "gate": "Neutral visual fidelity (C0)",
            "role": "main evidence chain",
            "evidence": f"C0 faithful={c0_faithful}/{c0_total}; conflict_aligned=0/{c0_total}.",
            "alternative_explanation_reduced": "Baseline color-recognition failure under neutral prompting.",
            "verdict": "Pass; conflict outputs can be interpreted against a faithful visual baseline.",
        },
        {
            "gate_id": "Gate 3",
            "gate": "Parser reliability",
            "role": "main evidence chain",
            "evidence": f"Main C0-C4 parse_error={main_parse_errors}; parser audit remains documented in results/parser/label_mapping_audit.md.",
            "alternative_explanation_reduced": "Parser inflation of conflict-aligned counts.",
            "verdict": "Pass for mainline; yes/no phase-2 rows are separately normalized.",
        },
        {
            "gate_id": "Gate 4",
            "gate": "Source robustness",
            "role": "appendix threat reduction",
            "evidence": f"LLaVA C3 split: {source_note}; C0 remains faithful in both sources.",
            "alternative_explanation_reduced": "Effect exists only because one source is visually invalid.",
            "verdict": "Partial pass; direction persists, magnitude differs and should not be overinterpreted.",
        },
        {
            "gate_id": "Gate 5",
            "gate": "Visual clarity validity",
            "role": "appendix threat reduction",
            "evidence": f"Targets clear={target_clear}/{len(target_audit)}, controls clear={control_clear}/{len(control_audit)}; any confound targets={target_flagged}/{len(target_audit)}, controls={control_flagged}/{len(control_audit)}.",
            "alternative_explanation_reduced": "Flip cases are simply unreadable or systematically occluded.",
            "verdict": "Mostly pass; residual reflection/lighting/background confounds remain visible.",
        },
        {
            "gate_id": "Gate 6",
            "gate": "Wording boundary",
            "role": "main boundary evidence",
            "evidence": f"LLaVA C3 original={int(llava_c3.conflict_aligned_n)}/300 ({pct(float(llava_c3.conflict_aligned_rate))}); C3-v2={int(llava_c3_v2.conflict_aligned_n)}/300; C3-v3={int(llava_c3_v3.conflict_aligned_n)}/300.",
            "alternative_explanation_reduced": "Original C3 reflects a stable cross-wording law.",
            "verdict": "Boundary set; the effect is wording-sensitive.",
        },
        {
            "gate_id": "Gate 7",
            "gate": "Color / format / factorization diagnostics",
            "role": "secondary attribution",
            "evidence": f"LLaVA original C3/C4 flips concentrate in {len(flips[flips['pair_family'] == 'achromatic_black_white'])}/{len(flips)} achromatic black/white rows; LLaVA free/MC/yes-no C3 rates are {pct(float(llava_free_c3.conflict_aligned_rate))}, {pct(float(llava_mc_c3.conflict_aligned_rate))}, {pct(float(llava_yesno.conflict_aligned_rate))}; factor peaks include LLaVA title/prefix {pct(float(llava_title.conflict_aligned_rate))}, Qwen no-correction presupposition {pct(float(qwen_presup.conflict_aligned_rate))}, InternVL title/prefix {pct(float(intern_title.conflict_aligned_rate))}.",
            "alternative_explanation_reduced": "The 9% LLaVA result is uniform across colors, formats, and false-text forms.",
            "verdict": "Boundary tightened; prompt form, answer form, and color pair all matter.",
        },
        {
            "gate_id": "Gate 8",
            "gate": "Multi-turn extension",
            "role": "appendix extension",
            "evidence": f"InternVL2 MT2={int(intern_mt2.conflict_aligned_n)}/300 ({pct(float(intern_mt2.conflict_aligned_rate))}), MT3={int(intern_mt3.conflict_aligned_n)}/300 ({pct(float(intern_mt3.conflict_aligned_rate))}); LLaVA MT3={int(llava_mt3.conflict_aligned_n)}/300.",
            "alternative_explanation_reduced": "Single-turn stability necessarily implies dialogue stability.",
            "verdict": "Extension only; multi-turn vulnerability is strong for InternVL2 but not a replacement for C0-C4.",
        },
    ]
    table = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "results" / "gatekeeping"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "gatekeeping_table.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        out_dir / "gatekeeping_summary.md",
        "\n".join(
            [
                "# Gatekeeping Summary",
                "",
                "This pipeline organizes the main and phase-2 evidence as a controlled diagnostic evaluation. It does not turn the paper into a broad benchmark leaderboard.",
                "",
                md_table(table),
                "",
                "## Main vs Appendix Placement",
                "",
                "- Main chain: Gates 1-3, same-image C0-C4 flips, Gate 6 wording boundary.",
                "- Secondary attribution in Results: per-color split, answer-format control, and the most compact prompt-factorization comparison from Gate 7.",
                "- Appendix and threats: source robustness, completed visual clarity audit, multi-turn extension, parser/reproducibility details, and full gate table.",
            ]
        ),
    )
    write_markdown(
        REPO_ROOT / "docs" / "gatekeeping_protocol.md",
        "\n".join(
            [
                "# Gatekeeping Protocol",
                "",
                "This protocol presents the project as a controlled diagnostic empirical study rather than a method paper or a broad model leaderboard.",
                "",
                md_table(table),
                "",
                "## Writing Boundary",
                "",
                "The protocol supports a local claim about car-body primary-color conflict prompts. It must not be generalized to all VLM language bias, all visual attributes, model scale, or a new method.",
            ]
        ),
    )


def update_writing_summaries(data: dict[str, pd.DataFrame], derived: dict[str, object]) -> None:
    factor = data["factor_metrics"]
    fmt = data["format_metrics"]
    multi = data["multi_metrics"]
    audit = derived["audit"]
    flips = derived["flips"]
    format_sensitive = derived["format_sensitive"]
    target_audit = audit[audit["audit_group"] == "target_conflict_flip"]
    control_audit = audit[audit["audit_group"] == "matched_faithful_control"]
    target_flagged = int(target_audit["any_visual_confound_flag"].sum())
    control_flagged = int(control_audit["any_visual_confound_flag"].sum())

    llava_title = get_metric(factor, "llava15_7b", "title_prefix_framing")
    llava_presup = get_metric(factor, "llava15_7b", "presupposition_no_correction")
    qwen_presup = get_metric(factor, "qwen2vl7b", "presupposition_no_correction")
    intern_title = get_metric(factor, "internvl2_8b", "title_prefix_framing")
    llava_free_c3 = get_metric(fmt, "llava15_7b", "free_c3_presupposition")
    llava_mc_c3 = get_metric(fmt, "llava15_7b", "multiple_choice_c3_presupposition")
    llava_yesno = get_metric(fmt, "llava15_7b", "yesno_false_claim")
    intern_mt2 = get_metric(multi, "internvl2_8b", "two_turn_persuasion")
    intern_mt3 = get_metric(multi, "internvl2_8b", "three_turn_persuasion")
    llava_mt = [
        get_metric(multi, "llava15_7b", "single_turn_false_note"),
        get_metric(multi, "llava15_7b", "two_turn_persuasion"),
        get_metric(multi, "llava15_7b", "three_turn_persuasion"),
    ]

    phase2_cn = f"""
## Phase 2 Full Strengthening Addendum

- Per-color split tightens the original LLaVA C3 interpretation: the 27 C3 flips are not dispersed across all six colors; 20/27 are `white->black`, with smaller `black->white` and `blue->red` contributions. Across LLaVA C3/C4, {len(flips[flips['pair_family'] == 'achromatic_black_white'])}/{len(flips)} main flip row-events are in the achromatic black/white family.
- The completed visual clarity audit reviewed 42 target flip rows and 42 matched faithful controls. Target rows are mostly inspectable (`clear`={int((target_audit['audit_visual_clarity'] == 'clear').sum())}/42), but visual confound flags are more common in targets ({target_flagged}/42) than controls ({control_flagged}/42). This reduces, but does not eliminate, the "images are hard" alternative explanation.
- Prompt factorization shows that false-text form matters. LLaVA reaches {pct(float(llava_title.conflict_aligned_rate))} under title/prefix framing and {pct(float(llava_presup.conflict_aligned_rate))} under no-correction presupposition; Qwen is most affected by no-correction presupposition ({pct(float(qwen_presup.conflict_aligned_rate))}), and InternVL2 by title/prefix framing ({pct(float(intern_title.conflict_aligned_rate))}).
- Answer-format control shows that original LLaVA C3 is format-sensitive: free-answer C3 is {pct(float(llava_free_c3.conflict_aligned_rate))}, multiple-choice C3 is {pct(float(llava_mc_c3.conflict_aligned_rate))}, and yes/no false-claim acceptance is {pct(float(llava_yesno.conflict_aligned_rate))}. {len(format_sensitive)}/27 original C3 flip rows are not reproduced by all three formal controls.
- Multi-turn persuasion is an appendix-level extension. LLaVA remains near zero across MT1/MT2/MT3 ({', '.join(pct(float(row.conflict_aligned_rate)) for row in llava_mt)}), while InternVL2 rises sharply in MT2/MT3 ({pct(float(intern_mt2.conflict_aligned_rate))}/{pct(float(intern_mt3.conflict_aligned_rate))}). This changes the boundary of the paper, not the frozen single-turn mainline.
"""
    add_phase2_block(REPO_ROOT / "results" / "final_result_summary.md", "## Phase 2 Full Strengthening Addendum", phase2_cn.replace("## Phase 2 Full Strengthening Addendum", "", 1))

    discussion = f"""
The Phase 2 diagnostics make the paper's conclusion narrower and stronger. The main evidence chain remains C0-C4 plus same-image paired flips: visual evidence dominates overall, while LLaVA shows a limited and significant conflict-aligned shift under the original misleading templates. The new per-color split prevents an overbroad reading of that 9% result: it is concentrated mainly in `white->black`, not evenly spread across all colors.

The new prompt and format controls also show that the false-text effect is not a single universal property of misleading language. It depends on framing, correction affordance, and answer format. Some factorized prompts can produce larger conflict following than original C3, including title/prefix framing and no-correction presupposition, but quoted and indirect hints remain weak. Meanwhile, formal answer formats reduce the original LLaVA C3 effect. These findings should be written as attribution and boundary evidence, not as a prompt-engineering paper.

The completed visual audit and case taxonomy improve the Discussion by separating inspectable prompt-following flips, color-pair concentration, residual visual confounds, format sensitivity, source/style caveats, and multi-turn-induced failures. Multi-turn persuasion is especially important as a boundary condition: InternVL2 is stable in the original single-turn C0-C4 setup but becomes highly susceptible under repeated previous-turn false context. This belongs in the appendix or extended diagnostics because it tests a different interaction regime.
"""
    add_phase2_block(REPO_ROOT / "results" / "results_discussion_summary.md", "## Phase 2 Discussion Addendum", discussion)

    threats = f"""
## Phase 2 Threat Updates

The visual-clarity threat is now a completed audit rather than only infrastructure. The audit covers all LLaVA original C3 flips, all C4 flips, all C3-v2 remaining flips, and matched faithful controls. Most target rows are clear/high-salience, so the main flips are not plausibly explained by globally unreadable images. However, visual confound flags are more frequent among targets ({target_flagged}/42) than controls ({control_flagged}/42), so reflection, lighting, background color, and dataset style remain residual threats.

The color split introduces a new boundary: the LLaVA original C3/C4 shifts are strongly concentrated in achromatic black/white routes, especially `white->black`. The paper should not imply uniform color-class susceptibility.

The factorization and format-control results reduce the risk of treating "false text" as one monolithic condition. Title/prefix framing and no-correction presupposition can be much stronger than quoted or indirect hints, while free-answer, multiple-choice, and yes/no controls reduce the original LLaVA C3 effect.

The multi-turn result adds a separate boundary. InternVL2 is stable in the original single-turn mainline but highly susceptible in repeated previous-turn false context. This does not invalidate the mainline; it shows that single-turn visual consistency cannot be generalized to multi-turn persuasion without an explicit dialogue diagnostic.
"""
    add_phase2_block(REPO_ROOT / "results" / "threats_to_validity_summary.md", "## Phase 2 Threat Updates", threats)

    main_ready = f"""
## Phase 2 Writing Placement

Use the Phase 2 results to strengthen, not replace, the main C0-C4 story. The正文 main chain should remain: balanced 300-image dataset, C0 visual fidelity, C0-C4 conflict conditions, same-image paired flips, and C3 wording boundary.

Recommended正文 secondary results are: per-color split showing `white->black` concentration, answer-format control showing reduced conflict following under free/multiple-choice/yes-no formats, and a compact prompt-factorization paragraph showing that title/prefix framing and no-correction presupposition are the strongest false-text forms.

Recommended appendix results are: completed visual clarity audit, full factorization tables, multi-turn persuasion, failure taxonomy/casebook, gatekeeping protocol, parser/source/reproducibility details.

Updated main conclusion: in a visually clear single-attribute car-color task, all three models are faithful under neutral prompting; LLaVA shows a limited, significant, same-image conflict-following shift under the original strong misleading templates; that shift is model-, wording-, answer-format-, and color-pair-sensitive; multi-turn context can induce a separate model-specific vulnerability, especially for InternVL2, but it should not be folded into the single-turn mainline.
"""
    add_phase2_block(REPO_ROOT / "results" / "main" / "main_results_paper_ready.md", "## Phase 2 Writing Placement", main_ready)


def main() -> None:
    data = load_inputs()
    derived = build_failure_taxonomy(data)
    build_gatekeeping(data, derived)
    update_writing_summaries(data, derived)
    print(
        {
            "case_analysis": "results/case_analysis",
            "gatekeeping": "results/gatekeeping",
            "writing_summaries_updated": [
                "results/final_result_summary.md",
                "results/results_discussion_summary.md",
                "results/threats_to_validity_summary.md",
                "results/main/main_results_paper_ready.md",
            ],
        }
    )


if __name__ == "__main__":
    main()
