from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "conference"
SOURCE_DIR = OUT_DIR / "source_data"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["figure.facecolor"] = "white"


PALETTE = {
    "llava": "#484878",
    "qwen": "#7884B4",
    "internvl": "#B4C0E4",
    "signal": "#B64342",
    "signal_soft": "#E9A6A1",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8A8A8A",
    "neutral_dark": "#4D4D4D",
    "panel_bg": "#F6F6F8",
    "blue_soft": "#E0F0F0",
    "peach": "#F0E0D0",
}

MODEL_ORDER = ["LLaVA-1.5-7B", "Qwen2-VL-7B-Instruct", "InternVL2-8B"]
MODEL_KEY_ORDER = ["llava15_7b", "qwen2vl7b", "internvl2_8b"]
MODEL_COLORS = {
    "LLaVA-1.5-7B": PALETTE["llava"],
    "Qwen2-VL-7B-Instruct": PALETTE["qwen"],
    "InternVL2-8B": PALETTE["internvl"],
}
CONDITION_ORDER = [
    "C0_neutral",
    "C1_weak_suggestion",
    "C2_false_assertion_open",
    "C3_presupposition_correction_allowed",
    "C4_stronger_open_conflict",
]
CONDITION_LABELS = {
    "C0_neutral": "C0",
    "C1_weak_suggestion": "C1",
    "C2_false_assertion_open": "C2",
    "C3_presupposition_correction_allowed": "C3",
    "C4_stronger_open_conflict": "C4",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def save_all(fig, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=600)
    fig.savefig(OUT_DIR / f"{stem}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def pct(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def short_condition(condition_name: str) -> str:
    return CONDITION_LABELS.get(condition_name, condition_name.split("_", 1)[0])


def wrap_label(label: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def load_main_metrics() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "main" / "main_condition_metrics.csv")
    df["condition_short"] = df["condition_name"].map(short_condition)
    df["model_name"] = pd.Categorical(df["model_name"], MODEL_ORDER, ordered=True)
    df["condition_name"] = pd.Categorical(df["condition_name"], CONDITION_ORDER, ordered=True)
    return df.sort_values(["model_name", "condition_name"])


def load_paired_metrics() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "main" / "paired_flip_metrics.csv")
    df["condition_short"] = df["condition"].map(lambda x: str(x))
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], ["C1", "C2", "C3", "C4"], ordered=True)
    return df.sort_values(["model", "condition"])


def figure1_evidence_chain() -> None:
    rows = [
        {"step": "Image set", "summary": "300 real car images\n6 colours, 50 each"},
        {"step": "Neutral baseline", "summary": "C0 faithful\n300/300 for all models"},
        {"step": "Conflict prompts", "summary": "Same images under\nC1-C4 false colour cues"},
        {"step": "Paired flip test", "summary": "Faithful C0 ->\nfalse-text-aligned output"},
        {"step": "Boundary checks", "summary": "Wording, format,\ncolour route, factors"},
        {"step": "Bounded claim", "summary": "Local LLaVA C3/C4 shift;\nno general text-over-vision claim"},
    ]
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "figure1_evidence_chain_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.set_axis_off()

    xs = np.linspace(0.06, 0.94, len(rows))
    y = 0.58
    w = 0.132
    h = 0.46
    for i, (x, row) in enumerate(zip(xs, rows)):
        face = PALETTE["panel_bg"] if i not in (3, 5) else PALETTE["blue_soft"]
        edge = PALETTE["neutral_dark"] if i not in (3, 5) else PALETTE["llava"]
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.13, row["step"], ha="center", va="center", fontsize=7.5, fontweight="bold", transform=ax.transAxes)
        ax.text(x, y - 0.05, row["summary"], ha="center", va="center", fontsize=6.6, linespacing=1.2, transform=ax.transAxes)
        if i < len(rows) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w / 2 + 0.006, y),
                    (xs[i + 1] - w / 2 - 0.006, y),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.8,
                    color=PALETTE["neutral_mid"],
                    transform=ax.transAxes,
                )
            )

    ax.text(0.06, 0.95, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    callouts = [
        "Main baseline: C0 faithful = 300/300 for LLaVA, Qwen, InternVL2",
        "Primary shift: LLaVA C3 = 27/300; C4 = 10/300",
        "Stable comparators: Qwen C3/C4 = 1/300 each; InternVL2 C0-C4 = 0/300",
    ]
    for i, text in enumerate(callouts):
        ax.text(0.07 + i * 0.31, 0.12, textwrap.fill(text, 34), transform=ax.transAxes, ha="left", va="center", fontsize=6.4)

    save_all(fig, "figure1_evidence_chain")


def figure2_main_conflict_rates() -> None:
    df = load_main_metrics()
    src = df[
        [
            "model_key",
            "model_name",
            "condition_name",
            "condition_short",
            "n",
            "conflict_aligned_n",
            "conflict_aligned_rate",
            "conflict_aligned_ci_low",
            "conflict_aligned_ci_high",
        ]
    ].copy()
    src.to_csv(SOURCE_DIR / "figure2_main_conflict_rates_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    x = np.arange(len(CONDITION_ORDER))
    for model in MODEL_ORDER:
        sub = df[df["model_name"] == model].sort_values("condition_name")
        y = pct(sub["conflict_aligned_rate"])
        low = y - pct(sub["conflict_aligned_ci_low"])
        high = pct(sub["conflict_aligned_ci_high"]) - y
        ax.errorbar(
            x,
            y,
            yerr=[low, high],
            marker="o",
            markersize=4.2,
            linewidth=1.6 if model == "LLaVA-1.5-7B" else 1.2,
            capsize=2.5,
            color=MODEL_COLORS[model],
            label=model,
            zorder=3 if model == "LLaVA-1.5-7B" else 2,
        )
        for xi, yi, n in zip(x, y, sub["conflict_aligned_n"]):
            if n > 0:
                ax.text(xi, yi + 0.9, f"{int(n)}/300", ha="center", va="bottom", fontsize=6.2, color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(["C0", "C1", "C2", "C3", "C4"])
    ax.set_ylabel("False-colour aligned outputs (%)")
    ax.set_xlabel("Prompt condition")
    ax.set_ylim(-0.5, 14.5)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=6.3)
    add_panel_label(ax, "a")
    ax.text(0.62, 0.94, "n = 300 images per model-condition", transform=ax.transAxes, fontsize=6.4, ha="left")
    save_all(fig, "figure2_main_conflict_rates")


def figure3_paired_flips() -> None:
    df = load_paired_metrics()
    src = df[
        [
            "model_key",
            "model",
            "condition",
            "n_pairs",
            "faithful_to_faithful_n",
            "faithful_to_conflict_aligned_n",
            "answer_flip_n",
            "answer_flip_rate",
            "faithful_retention_rate",
            "p_value_exact_mcnemar",
        ]
    ].copy()
    src.to_csv(SOURCE_DIR / "figure3_paired_flips_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.25))
    conditions = ["C1", "C2", "C3", "C4"]
    y_positions = []
    labels = []
    values = []
    colors = []
    for mi, model in enumerate(MODEL_ORDER):
        for ci, cond in enumerate(conditions):
            row = df[(df["model"] == model) & (df["condition"] == cond)].iloc[0]
            y_positions.append(mi * 5 + ci)
            labels.append(f"{model.replace('-Instruct', '')} {cond}")
            values.append(row["faithful_to_conflict_aligned_n"])
            colors.append(MODEL_COLORS[model])

    ax.barh(y_positions, values, color=colors, height=0.72)
    for y, v in zip(y_positions, values):
        ax.text(v + 0.5, y, f"{int(v)}/300", va="center", ha="left", fontsize=6.4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlabel("Same-image faithful-to-false-colour flips (count)")
    ax.set_xlim(0, 31)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    for boundary in (3.5, 8.5):
        ax.axhline(boundary, color="#E0E0E0", linewidth=0.8)
    add_panel_label(ax, "a")
    ax.text(
        0.53,
        0.08,
        "All C0 answers are faithful;\ntherefore each bar is a paired C0-to-conflict shift.",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    save_all(fig, "figure3_paired_flips")


def figure4_boundary_diagnostics() -> None:
    robust = pd.read_csv(ROOT / "results" / "robustness" / "prompt_boundary_metrics.csv")
    format_df = pd.read_csv(ROOT / "results" / "format_control" / "format_control_metrics.csv")
    color_df = pd.read_csv(ROOT / "results" / "color_split" / "color_pair_family_metrics.csv")
    factor_df = pd.read_csv(ROOT / "results" / "factorization" / "factorized_prompt_metrics.csv")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    axs = axs.ravel()

    # a. Wording robustness for LLaVA.
    ax = axs[0]
    sub = robust[robust["model"] == "LLaVA-1.5-7B"].copy()
    order = ["Original C3", "C3-v2", "C3-v3"]
    sub["variant"] = pd.Categorical(sub["variant"], order, ordered=True)
    sub = sub.sort_values("variant")
    x = np.arange(len(sub))
    y = pct(sub["conflict_aligned_rate"])
    ax.bar(x, y, color=[PALETTE["llava"], PALETTE["qwen"], PALETTE["internvl"]], width=0.68)
    for xi, row, yi in zip(x, sub.itertuples(), y):
        ax.text(xi, yi + 0.65, f"{int(row.conflict_aligned_n)}/300", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("False-colour aligned (%)")
    ax.set_title("C3 wording weakens the LLaVA shift", fontsize=7.5)
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    add_panel_label(ax, "a")

    # b. Answer format controls for LLaVA C3-like probes.
    ax = axs[1]
    labels = [
        ("Canonical C3", "REF_C3_original_label_set"),
        ("Free C3", "free_c3_presupposition"),
        ("MC C3", "multiple_choice_c3_presupposition"),
        ("Yes/no false", "yesno_false_claim"),
    ]
    rows = []
    for label, factor in labels:
        row = format_df[(format_df["model"] == "LLaVA-1.5-7B") & (format_df["factor_id"] == factor)].iloc[0]
        rows.append({"label": label, "n": row["conflict_aligned_n"], "rate": row["conflict_aligned_rate"]})
    fmt_src = pd.DataFrame(rows)
    x = np.arange(len(fmt_src))
    y = pct(fmt_src["rate"])
    ax.bar(x, y, color=[PALETTE["llava"], PALETTE["qwen"], PALETTE["qwen"], PALETTE["qwen"]], width=0.68)
    for xi, row, yi in zip(x, fmt_src.itertuples(), y):
        ax.text(xi, yi + 0.65, f"{int(row.n)}/300", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(v, 10) for v in fmt_src["label"]])
    ax.set_title("Answer format changes the observed rate", fontsize=7.5)
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    add_panel_label(ax, "b")

    # c. Colour-pair concentration for LLaVA C3.
    ax = axs[2]
    sub = color_df[(color_df["model"] == "LLaVA-1.5-7B") & (color_df["condition"] == "C3")].copy()
    pair_order = ["achromatic_black_white", "red_blue", "green_yellow", "yellow_red"]
    pair_labels = ["black/white", "red/blue", "green/yellow", "yellow/red"]
    sub["pair_family"] = pd.Categorical(sub["pair_family"], pair_order, ordered=True)
    sub = sub.sort_values("pair_family")
    x = np.arange(len(sub))
    y = pct(sub["conflict_following_rate"])
    ax.bar(x, y, color=[PALETTE["signal"], PALETTE["signal_soft"], PALETTE["neutral_light"], PALETTE["neutral_light"]], width=0.68)
    for xi, row, yi in zip(x, sub.itertuples(), y):
        ax.text(xi, yi + 1.1, f"{int(row.faithful_to_conflict_aligned_n)}/{int(row.n_pairs)}", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(v, 10) for v in pair_labels])
    ax.set_ylabel("C3 paired flip rate (%)")
    ax.set_title("C3 flips concentrate in achromatic routes", fontsize=7.5)
    ax.set_ylim(0, 26)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    add_panel_label(ax, "c")

    # d. Factorized prompts show a separate, stronger regime.
    ax = axs[3]
    factors = [
        ("F1", "quoted_claim_user"),
        ("F2", "indirect_hint_user"),
        ("F3", "title_prefix_framing"),
        ("F4", "presupposition_no_correction"),
        ("F5", "previous_turn_false_context"),
    ]
    factor_rows = factor_df[factor_df["factor_id"].isin([f for _, f in factors])].copy()
    x = np.arange(len(factors))
    width = 0.22
    for i, model in enumerate(MODEL_ORDER):
        sub = factor_rows[factor_rows["model"] == model].copy()
        sub["factor_id"] = pd.Categorical(sub["factor_id"], [f for _, f in factors], ordered=True)
        sub = sub.sort_values("factor_id")
        ax.bar(x + (i - 1) * width, pct(sub["conflict_aligned_rate"]), width=width, label=model, color=MODEL_COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in factors])
    ax.set_ylabel("False-colour aligned (%)")
    ax.set_title("Factorized prompts are a separate boundary", fontsize=7.5)
    ax.set_ylim(0, 42)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend(fontsize=5.8, loc="upper left")
    add_panel_label(ax, "d")

    for ax in axs:
        ax.tick_params(axis="both", labelsize=6.2)

    plt.tight_layout(w_pad=2.2, h_pad=2.4)

    robust[robust["model"] == "LLaVA-1.5-7B"].to_csv(SOURCE_DIR / "figure4a_wording_source.csv", index=False)
    fmt_src.to_csv(SOURCE_DIR / "figure4b_answer_format_source.csv", index=False)
    sub.to_csv(SOURCE_DIR / "figure4c_color_pair_source.csv", index=False)
    factor_rows.to_csv(SOURCE_DIR / "figure4d_factorization_source.csv", index=False)
    save_all(fig, "figure4_boundary_diagnostics")


def write_manifest() -> None:
    rows = [
        {
            "figure": "figure1_evidence_chain",
            "claim": "The study uses a short evidence chain from identical images and faithful C0 baselines to paired conflict flips and boundary checks.",
            "archetype": "schematic-led composite",
        },
        {
            "figure": "figure2_main_conflict_rates",
            "claim": "Only LLaVA-1.5-7B shows a clear primary C3/C4 false-colour alignment shift.",
            "archetype": "quantitative grid",
        },
        {
            "figure": "figure3_paired_flips",
            "claim": "The LLaVA C3/C4 outputs are same-image faithful-to-conflict paired flips, not unpaired aggregate differences.",
            "archetype": "quantitative grid",
        },
        {
            "figure": "figure4_boundary_diagnostics",
            "claim": "The observed shift is bounded by wording, answer format, colour route, and prompt factor.",
            "archetype": "asymmetric mixed-modality figure",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "conference_figure_manifest.csv", index=False)


def main() -> None:
    ensure_dirs()
    figure1_evidence_chain()
    figure2_main_conflict_rates()
    figure3_paired_flips()
    figure4_boundary_diagnostics()
    write_manifest()
    print(f"Wrote conference figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
