from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "conference_narrative"
SOURCE_DIR = OUT_DIR / "source_data"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.facecolor"] = "white"

COLORS = {
    "ink": "#272727",
    "muted": "#696969",
    "line": "#B8B8B8",
    "paper": "#F7F7F9",
    "blue": "#484878",
    "blue_mid": "#7884B4",
    "blue_soft": "#E0E4F6",
    "red": "#B64342",
    "red_soft": "#F4D5D2",
    "green": "#2E7D4F",
    "green_soft": "#DDF3DE",
    "gold": "#D99B2B",
    "gold_soft": "#F4E6C8",
    "teal": "#42949E",
    "teal_soft": "#DCEFF1",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def save_all(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=600)
    fig.savefig(OUT_DIR / f"{stem}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def add_box(ax, xy, wh, title, body, face, edge=None, title_color=None, fontsize=7.2, wrap_width=32):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        facecolor=face,
        edgecolor=edge or COLORS["line"],
        linewidth=0.9,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + h - 0.055,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize + 0.5,
        fontweight="bold",
        color=title_color or COLORS["ink"],
    )
    ax.text(
        x + 0.02,
        y + h - 0.12,
        "\n".join(textwrap.wrap(body, width=wrap_width, break_long_words=False)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.25,
    )


def arrow(ax, start, end, color=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=color or COLORS["muted"],
            transform=ax.transAxes,
        )
    )


def select_example_case() -> pd.Series:
    df = pd.read_csv(ROOT / "results" / "main" / "main_combined_parsed_results.csv")
    flips = df[
        (df["model_key"] == "llava15_7b")
        & (df["condition_name"] == "C3_presupposition_correction_allowed")
        & (df["is_conflict_aligned"] == True)
    ].copy()
    for _, row in flips.iterrows():
        path = ROOT / str(row["image_path"])
        if path.exists():
            c0 = df[
                (df["model_key"] == "llava15_7b")
                & (df["condition_name"] == "C0_neutral")
                & (df["image_id"] == row["image_id"])
            ].iloc[0]
            out = pd.DataFrame([row.to_dict() | {"c0_raw_output": c0["raw_output"], "c0_parsed_label": c0["parsed_label"]}])
            out.to_csv(SOURCE_DIR / "graphical_abstract_example_case.csv", index=False)
            return out.iloc[0]
    raise RuntimeError("No local LLaVA C3 flip image found")


def draw_graphical_abstract() -> None:
    case = select_example_case()
    image_path = ROOT / str(case["image_path"])
    img = Image.open(image_path).convert("RGB")

    fig = plt.figure(figsize=(7.3, 4.25))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(
        0.04,
        0.95,
        "A same image can support a faithful answer or a false-text-aligned flip",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.04,
        0.90,
        "Real example from the primary evaluation set; shown as an illustration, not as a stand-alone result.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )

    img_ax = fig.add_axes([0.045, 0.22, 0.28, 0.58])
    img_ax.imshow(img)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_color(COLORS["line"])
        spine.set_linewidth(1.2)
    img_ax.set_title(f"Image {case['image_id']} ({case['source_dataset']})", fontsize=7.2, pad=5)

    add_box(
        ax,
        (0.38, 0.58),
        (0.23, 0.20),
        "C0 neutral prompt",
        f"Question asks for the car-body colour. LLaVA answers: {str(case['c0_parsed_label']).lower()}.",
        COLORS["green_soft"],
        COLORS["green"],
        wrap_width=28,
    )
    add_box(
        ax,
        (0.38, 0.25),
        (0.23, 0.22),
        "C3 conflict prompt",
        f"Prompt embeds the false colour {case['conflict_color']}. LLaVA answers: {str(case['parsed_label']).lower()}.",
        COLORS["red_soft"],
        COLORS["red"],
        wrap_width=28,
    )
    add_box(
        ax,
        (0.68, 0.43),
        (0.26, 0.22),
        "Paired flip",
        f"Same image, same model: C0 is faithful ({case['true_color']}), C3 follows false text ({case['conflict_color']}).",
        COLORS["blue_soft"],
        COLORS["blue"],
        wrap_width=30,
    )
    add_box(
        ax,
        (0.68, 0.14),
        (0.26, 0.18),
        "Boundary",
        "This illustrates the evidence unit. The paper's claim still depends on aggregate rates and diagnostics.",
        COLORS["paper"],
        COLORS["line"],
        wrap_width=30,
    )

    arrow(ax, (0.33, 0.56), (0.38, 0.68), COLORS["green"])
    arrow(ax, (0.33, 0.43), (0.38, 0.36), COLORS["red"])
    arrow(ax, (0.61, 0.68), (0.68, 0.56), COLORS["blue"])
    arrow(ax, (0.61, 0.36), (0.68, 0.50), COLORS["blue"])
    arrow(ax, (0.81, 0.43), (0.81, 0.32), COLORS["muted"])

    ax.text(
        0.045,
        0.13,
        f"True colour: {case['true_color']}  |  false cue: {case['conflict_color']}  |  parsed C3 label: {case['parsed_label']}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["muted"],
    )
    save_all(fig, "graphical_abstract_real_case")


def draw_manuscript_roadmap() -> None:
    rows = [
        ("Question", "Can false text move a simple visual judgement?", COLORS["blue_soft"], COLORS["blue"]),
        ("Primary evidence", "C0 baseline and C1-C4 paired conflict prompts", COLORS["green_soft"], COLORS["green"]),
        ("Observed shift", "LLaVA C3/C4 shows limited same-image flips", COLORS["red_soft"], COLORS["red"]),
        ("Boundary tests", "Wording, format, colour route, prompt factor", COLORS["gold_soft"], COLORS["gold"]),
        ("Validity checks", "Parser, source sanity, visual clarity, reproducibility", COLORS["teal_soft"], COLORS["teal"]),
        ("Claim", "Local, conditional shift rather than general text-over-vision", COLORS["paper"], COLORS["ink"]),
    ]
    pd.DataFrame(rows, columns=["stage", "message", "face", "edge"]).drop(columns=["face", "edge"]).to_csv(
        SOURCE_DIR / "manuscript_roadmap_source.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_axis_off()
    ax.text(0.04, 0.94, "How the paper's argument is meant to be read", transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    y = 0.64
    for i, (stage, message, face, edge) in enumerate(rows):
        x = 0.05 + i * 0.155
        box = FancyBboxPatch(
            (x, y - 0.20),
            0.125,
            0.32,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(x + 0.0625, y + 0.06, stage, transform=ax.transAxes, ha="center", va="center", fontsize=6.8, fontweight="bold")
        ax.text(
            x + 0.0625,
            y - 0.07,
            "\n".join(textwrap.wrap(message, 18, break_long_words=False)),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.3,
            linespacing=1.15,
        )
        if i < len(rows) - 1:
            arrow(ax, (x + 0.128, y - 0.04), (x + 0.152, y - 0.04), COLORS["muted"])

    ax.add_patch(Rectangle((0.06, 0.15), 0.88, 0.07, transform=ax.transAxes, color=COLORS["paper"], ec=COLORS["line"], lw=0.8))
    ax.text(
        0.50,
        0.185,
        "Use as an introductory or talk-slide bridge: it links motivation, evidence, diagnostics, and claim boundary.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["muted"],
    )
    save_all(fig, "manuscript_argument_roadmap")


def draw_claim_boundary_summary() -> None:
    rows = [
        ("Supported", "All models are faithful under C0; LLaVA shows C3/C4 paired shifts.", COLORS["green_soft"], COLORS["green"]),
        ("Bounded", "The shift changes with wording, format, prompt factor, and colour pair.", COLORS["gold_soft"], COLORS["gold"]),
        ("Not claimed", "The evidence does not show general text-over-vision behaviour.", COLORS["red_soft"], COLORS["red"]),
    ]
    pd.DataFrame(rows, columns=["zone", "message", "face", "edge"]).drop(columns=["face", "edge"]).to_csv(
        SOURCE_DIR / "claim_boundary_summary_source.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(6.7, 3.0))
    ax.set_axis_off()
    ax.text(0.04, 0.93, "Claim boundary at a glance", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.04, 0.84, "A compact reader guide for keeping the result strong without overclaiming.", transform=ax.transAxes, fontsize=7.4, color=COLORS["muted"], va="top")
    for i, (zone, message, face, edge) in enumerate(rows):
        x = 0.06 + i * 0.31
        add_box(ax, (x, 0.30), (0.25, 0.38), zone, message, face, edge, fontsize=6.6, wrap_width=24)
    ax.text(
        0.50,
        0.14,
        "Best use: final Discussion slide, graphical takeaway, or optional summary panel after Figure 4.",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    save_all(fig, "claim_boundary_summary")


def write_manifest() -> None:
    rows = [
        {
            "figure": "graphical_abstract_real_case",
            "type": "real-image workflow",
            "suggested_use": "Graphical abstract or opening transition before Figure 1",
        },
        {
            "figure": "manuscript_argument_roadmap",
            "type": "argument roadmap",
            "suggested_use": "Introductory bridge between Introduction and Related Work or for slides",
        },
        {
            "figure": "claim_boundary_summary",
            "type": "claim-boundary summary",
            "suggested_use": "Discussion transition or graphical takeaway",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "narrative_figure_manifest.csv", index=False)


def main() -> None:
    ensure_dirs()
    draw_graphical_abstract()
    draw_manuscript_roadmap()
    draw_claim_boundary_summary()
    write_manifest()
    print(f"Wrote narrative figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
