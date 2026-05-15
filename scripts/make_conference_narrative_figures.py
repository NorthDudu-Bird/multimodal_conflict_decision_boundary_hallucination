from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
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
    "cream": "#FFF8E9",
    "blue": "#484878",
    "blue_soft": "#E0E4F6",
    "red": "#B64342",
    "red_soft": "#F4D5D2",
    "green": "#2E7D4F",
    "green_soft": "#DDF3DE",
    "gold": "#D99B2B",
    "gold_soft": "#F4E6C8",
    "teal": "#42949E",
    "teal_soft": "#DCEFF1",
    "pink": "#F7D5E0",
    "pink_soft": "#FCECF1",
    "purple": "#76569A",
    "purple_soft": "#EDE3F5",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def save_all(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=600)
    svg_path = OUT_DIR / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(OUT_DIR / f"{stem}.tiff", bbox_inches="tight", dpi=450)
    plt.close(fig)


def arrow(ax, start, end, color=None, scale=12, curve=0.0, lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color or COLORS["muted"],
            transform=ax.transAxes,
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def add_comic_box(
    ax,
    xy,
    wh,
    title,
    body,
    face,
    edge,
    fontsize=7.0,
    wrap_width=30,
    lw=1.2,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
        path_effects=[pe.SimplePatchShadow(offset=(1.6, -1.6), alpha=0.16), pe.Normal()],
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.048,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize + 0.6,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + 0.018,
        y + h - 0.105,
        "\n".join(textwrap.wrap(body, width=wrap_width, break_long_words=False)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.18,
    )


def add_prompt_bubble(ax, xy, wh, label, prompt, face, edge, wrap_width=53):
    x, y = xy
    w, h = wh
    bubble = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.3,
        transform=ax.transAxes,
        path_effects=[pe.SimplePatchShadow(offset=(2, -2), alpha=0.14), pe.Normal()],
    )
    ax.add_patch(bubble)
    tail = Polygon(
        [(x + 0.025, y + 0.035), (x - 0.035, y - 0.005), (x + 0.075, y + 0.015)],
        closed=True,
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
    )
    ax.add_patch(tail)
    ax.text(
        x + 0.02,
        y + h - 0.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + 0.02,
        y + h - 0.095,
        '"' + "\n".join(textwrap.wrap(prompt, width=wrap_width, break_long_words=False)) + '"',
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=COLORS["ink"],
        linespacing=1.17,
    )


def add_doodle_background(ax) -> None:
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=COLORS["cream"], edgecolor="none", zorder=-10))
    for cx, cy, r, color in [
        (0.08, 0.86, 0.030, COLORS["pink"]),
        (0.95, 0.86, 0.038, COLORS["teal_soft"]),
        (0.18, 0.10, 0.025, COLORS["gold_soft"]),
        (0.93, 0.12, 0.027, COLORS["purple_soft"]),
    ]:
        ax.add_patch(Circle((cx, cy), r, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.85))


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
            out = pd.DataFrame(
                [
                    row.to_dict()
                    | {
                        "c0_raw_output": c0["raw_output"],
                        "c0_parsed_label": c0["parsed_label"],
                        "c0_prompt_text": c0["prompt_text"],
                    }
                ]
            )
            out.to_csv(SOURCE_DIR / "graphical_abstract_example_case.csv", index=False)
            return out.iloc[0]
    raise RuntimeError("No local LLaVA C3 flip image found")


def draw_graphical_abstract() -> None:
    case = select_example_case()
    image_path = ROOT / str(case["image_path"])
    img = Image.open(image_path).convert("RGB")

    fig = plt.figure(figsize=(7.5, 4.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    add_doodle_background(ax)

    ax.text(
        0.04,
        0.965,
        "One car, two prompts: where the answer flips",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.04,
        0.905,
        "A real paired example from the evaluation set; playful layout, audit-traceable content.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )

    img_ax = fig.add_axes([0.045, 0.23, 0.285, 0.56])
    img_ax.imshow(img)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_color(COLORS["line"])
        spine.set_linewidth(1.2)
    img_ax.set_title(f"real crop: {case['image_id']} ({case['source_dataset']})", fontsize=7.2, pad=5)

    ax.text(
        0.205,
        0.175,
        f"true: {case['true_color']}    false cue: {case['conflict_color']}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["muted"],
    )
    ax.text(
        0.20,
        0.78,
        "image",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.18", fc="#FFFFFF", ec=COLORS["line"], lw=0.8),
    )

    add_prompt_bubble(
        ax,
        (0.39, 0.60),
        (0.36, 0.22),
        "C0 prompt shown to the model",
        case["c0_prompt_text"],
        "#FFFFFF",
        COLORS["green"],
        wrap_width=53,
    )
    add_prompt_bubble(
        ax,
        (0.39, 0.27),
        (0.36, 0.25),
        "C3 prompt shown to the model",
        case["prompt_text"],
        "#FFFFFF",
        COLORS["red"],
        wrap_width=53,
    )
    add_comic_box(
        ax,
        (0.80, 0.60),
        (0.155, 0.16),
        "Answer",
        str(case["c0_parsed_label"]).lower(),
        COLORS["green_soft"],
        COLORS["green"],
        fontsize=8.2,
        wrap_width=18,
    )
    add_comic_box(
        ax,
        (0.80, 0.33),
        (0.155, 0.16),
        "Answer",
        str(case["parsed_label"]).lower(),
        COLORS["red_soft"],
        COLORS["red"],
        fontsize=8.2,
        wrap_width=18,
    )
    add_comic_box(
        ax,
        (0.70, 0.075),
        (0.255, 0.145),
        "Takeaway",
        "Illustrative example only; aggregate rates and diagnostics carry the claim.",
        COLORS["purple_soft"],
        COLORS["purple"],
        fontsize=6.4,
        wrap_width=36,
    )

    arrow(ax, (0.32, 0.55), (0.39, 0.70), COLORS["green"], curve=0.10)
    arrow(ax, (0.32, 0.45), (0.39, 0.39), COLORS["red"], curve=-0.08)
    arrow(ax, (0.75, 0.70), (0.80, 0.68), COLORS["green"])
    arrow(ax, (0.75, 0.39), (0.80, 0.41), COLORS["red"])
    arrow(ax, (0.875, 0.33), (0.82, 0.22), COLORS["purple"], curve=-0.15)

    ax.text(
        0.045,
        0.075,
        f"Parsed labels: C0={case['c0_parsed_label']} | C3={case['parsed_label']} | model=LLaVA-1.5-7B",
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
        ("Evidence", "C0 baseline and C1-C4 paired conflict prompts", COLORS["green_soft"], COLORS["green"]),
        ("Shift", "LLaVA C3/C4 shows limited same-image flips", COLORS["red_soft"], COLORS["red"]),
        ("Diagnostics", "Wording, format, colour route, prompt factor", COLORS["gold_soft"], COLORS["gold"]),
        ("Checks", "Parser, source sanity, visual clarity, reproducibility", COLORS["teal_soft"], COLORS["teal"]),
        ("Claim", "Local, conditional shift rather than general text-over-vision", COLORS["paper"], COLORS["ink"]),
    ]
    pd.DataFrame(rows, columns=["stage", "message", "face", "edge"]).drop(columns=["face", "edge"]).to_csv(
        SOURCE_DIR / "manuscript_roadmap_source.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#FFFBF2", edgecolor="none", zorder=-10))
    ax.text(0.04, 0.945, "A reader's map through the paper", transform=ax.transAxes, fontsize=11.2, fontweight="bold", va="top")
    ax.text(
        0.04,
        0.875,
        "A lighter bridge figure for the introduction, not a replacement for quantitative evidence.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLORS["muted"],
        va="top",
    )

    y_positions = [0.65, 0.50, 0.64, 0.47, 0.62, 0.48]
    for i, (stage, message, face, edge) in enumerate(rows):
        x = 0.055 + i * 0.154
        y = y_positions[i]
        add_comic_box(ax, (x, y - 0.17), (0.126, 0.27), stage, message, face, edge, fontsize=5.85, wrap_width=17, lw=1.1)
        ax.add_patch(Circle((x + 0.017, y + 0.082), 0.018, transform=ax.transAxes, facecolor="#FFFFFF", edgecolor=edge, lw=1.0))
        ax.text(x + 0.017, y + 0.082, str(i + 1), transform=ax.transAxes, ha="center", va="center", fontsize=6.0, fontweight="bold", color=edge)
        if i < len(rows) - 1:
            arrow(ax, (x + 0.13, y - 0.015), (x + 0.152, y_positions[i + 1] - 0.015), COLORS["muted"], curve=0.08)

    ax.add_patch(Rectangle((0.06, 0.13), 0.88, 0.075, transform=ax.transAxes, color="#FFFFFF", ec=COLORS["line"], lw=0.8))
    ax.text(
        0.50,
        0.168,
        "Suggested use: introduce the story once, then let Figures 1-4 carry the measured evidence.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["muted"],
    )
    save_all(fig, "manuscript_argument_roadmap")


def draw_claim_boundary_summary() -> None:
    rows = [
        ("Supported", "All models are faithful under C0; LLaVA shows C3/C4 paired shifts.", COLORS["green_soft"], COLORS["green"], "OK"),
        ("Bounded", "The shift changes with wording, format, prompt factor, and colour pair.", COLORS["gold_soft"], COLORS["gold"], "IF"),
        ("Not claimed", "The evidence does not show general text-over-vision behaviour.", COLORS["red_soft"], COLORS["red"], "NO"),
    ]
    pd.DataFrame(rows, columns=["zone", "message", "face", "edge", "tag"]).drop(columns=["face", "edge"]).to_csv(
        SOURCE_DIR / "claim_boundary_summary_source.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(6.9, 3.15))
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#FBFCFD", edgecolor="none", zorder=-10))
    ax.text(0.04, 0.94, "Claim boundary at a glance", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(
        0.04,
        0.855,
        "A compact reader guide for keeping the result strong without overclaiming.",
        transform=ax.transAxes,
        fontsize=7.4,
        color=COLORS["muted"],
        va="top",
    )
    for i, (zone, message, face, edge, tag) in enumerate(rows):
        x = 0.06 + i * 0.31
        add_comic_box(ax, (x, 0.29), (0.25, 0.39), zone, message, face, edge, fontsize=6.4, wrap_width=25)
        cx = x + 0.125
        ax.add_patch(Circle((cx, 0.71), 0.033, transform=ax.transAxes, facecolor=edge, edgecolor="#FFFFFF", lw=1.5))
        ax.text(cx, 0.71, tag, transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#FFFFFF")
    ax.text(
        0.50,
        0.135,
        "Best use: Discussion transition, graphical takeaway, or slide summary after the evidence figures.",
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
            "type": "real-image workflow with actual prompts",
            "suggested_use": "Graphical abstract or opening transition before Figure 1",
        },
        {
            "figure": "manuscript_argument_roadmap",
            "type": "cartoon argument roadmap",
            "suggested_use": "Introductory bridge between Introduction and Related Work or for slides",
        },
        {
            "figure": "claim_boundary_summary",
            "type": "cartoon claim-boundary summary",
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
