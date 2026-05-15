from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, RegularPolygon
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "conference_narrative"
SOURCE_DIR = OUT_DIR / "source_data"

FONT_DIR = Path("C:/Windows/Fonts")
COMIC = FontProperties(fname=str(FONT_DIR / "comic.ttf")) if (FONT_DIR / "comic.ttf").exists() else FontProperties(family="DejaVu Sans")
COMIC_BOLD = FontProperties(fname=str(FONT_DIR / "comicbd.ttf")) if (FONT_DIR / "comicbd.ttf").exists() else FontProperties(family="DejaVu Sans", weight="bold")
HAND = FontProperties(fname=str(FONT_DIR / "Inkfree.ttf")) if (FONT_DIR / "Inkfree.ttf").exists() else COMIC

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Comic Sans MS", "Ink Free", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.facecolor"] = "white"

COLORS = {
    "ink": "#262626",
    "muted": "#6C6C6C",
    "paper": "#FFF8E6",
    "paper2": "#FFFDF7",
    "line": "#2E2E2E",
    "shadow": "#D8CAB0",
    "blue": "#4257A0",
    "blue_soft": "#DDE7FF",
    "red": "#BA3F3F",
    "red_soft": "#FFD7D1",
    "green": "#238456",
    "green_soft": "#D9F5DE",
    "gold": "#D89C22",
    "gold_soft": "#FFE6AD",
    "teal": "#27919E",
    "teal_soft": "#D8F4F5",
    "purple": "#7651A3",
    "purple_soft": "#EEE0FF",
    "pink": "#E85D90",
    "pink_soft": "#FFE0EC",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def save_all(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=600)
    svg_path = OUT_DIR / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n", encoding="utf-8")
    fig.savefig(OUT_DIR / f"{stem}.tiff", bbox_inches="tight", dpi=450)
    plt.close(fig)


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def sketch(patch, scale=0.9, length=90, randomness=2):
    patch.set_sketch_params(scale=scale, length=length, randomness=randomness)
    return patch


def arrow(ax, start, end, color, scale=16, curve=0.0, lw=2.2):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        connectionstyle=f"arc3,rad={curve}",
        path_effects=[pe.withStroke(linewidth=lw + 2.3, foreground="#FFFFFF", alpha=0.85)],
    )
    arr.set_sketch_params(scale=1.0, length=90, randomness=2)
    ax.add_patch(arr)


def add_halftone(ax, color="#F2D79B") -> None:
    xs = np.linspace(0.04, 0.96, 19)
    ys = np.linspace(0.06, 0.92, 11)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if (i + j) % 3 == 0:
                ax.add_patch(Circle((x, y), 0.0045, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.22, zorder=-8))


def add_tape(ax, x, y, w, h, color="#F6D17C", angle=0):
    tape = Rectangle((x, y), w, h, angle=angle, transform=ax.transAxes, facecolor=color, edgecolor="#B78C2E", lw=0.8, alpha=0.72)
    tape.set_sketch_params(scale=0.8, length=70, randomness=2)
    ax.add_patch(tape)


def add_panel(ax, xy, wh, face, edge, lw=2.6, radius=0.04, shadow=True):
    x, y = xy
    w, h = wh
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.008, y - 0.008),
            w,
            h,
            boxstyle=f"round,pad=0.016,rounding_size={radius}",
            facecolor=COLORS["shadow"],
            edgecolor="none",
            transform=ax.transAxes,
            alpha=0.55,
            zorder=0,
        )
        ax.add_patch(sh)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.016,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=1,
    )
    sketch(patch)
    ax.add_patch(patch)
    return patch


def add_label(ax, x, y, text, color, size=8.5, ha="left", va="top", weight="bold", font=None, zorder=8):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontproperties=font or (COMIC_BOLD if weight == "bold" else COMIC),
        fontweight=weight if font is None else None,
        zorder=zorder,
    )


def add_body(ax, x, y, text, size=7.2, color=None, width=30, ha="left", va="top"):
    ax.text(
        x,
        y,
        wrap(text, width),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        color=color or COLORS["ink"],
        fontproperties=COMIC,
        linespacing=1.12,
        zorder=5,
    )


def add_prompt_bubble(ax, xy, wh, title, prompt, edge, face="#FFFFFF", width=54):
    x, y = xy
    w, h = wh
    add_panel(ax, (x, y), (w, h), face, edge, lw=2.4, radius=0.045)
    tail = Polygon(
        [(x + 0.038, y + 0.045), (x - 0.040, y - 0.012), (x + 0.094, y + 0.016)],
        closed=True,
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=2.0,
        zorder=2,
    )
    sketch(tail)
    ax.add_patch(tail)
    add_label(ax, x + 0.022, y + h - 0.050, title, edge, size=8.0)
    add_body(ax, x + 0.022, y + h - 0.107, '"' + prompt + '"', size=6.35, width=width)


def add_answer_card(ax, xy, title, answer, color, face):
    x, y = xy
    add_panel(ax, (x, y), (0.145, 0.145), face, color, lw=2.4, radius=0.035)
    add_label(ax, x + 0.026, y + 0.104, title, color, size=8.1)
    ax.text(
        x + 0.072,
        y + 0.045,
        answer,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["ink"],
        fontproperties=HAND,
        zorder=5,
    )


def cartoonize_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.25)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    poster = ImageOps.posterize(img, bits=4)
    edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
    edges = ImageOps.autocontrast(edges)
    mask = np.array(edges) > 55
    arr = np.array(poster)
    arr[mask] = (35, 35, 35)
    return Image.fromarray(arr)


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


def draw_robot(ax, x, y, scale=1.0):
    head = FancyBboxPatch(
        (x, y),
        0.090 * scale,
        0.070 * scale,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="#F7FAFF",
        edgecolor=COLORS["blue"],
        linewidth=2.0,
        zorder=4,
    )
    sketch(head)
    ax.add_patch(head)
    ax.add_patch(Circle((x + 0.028 * scale, y + 0.044 * scale), 0.008 * scale, transform=ax.transAxes, facecolor=COLORS["blue"], edgecolor="none", zorder=5))
    ax.add_patch(Circle((x + 0.063 * scale, y + 0.044 * scale), 0.008 * scale, transform=ax.transAxes, facecolor=COLORS["blue"], edgecolor="none", zorder=5))
    ax.plot([x + 0.032 * scale, x + 0.060 * scale], [y + 0.023 * scale, y + 0.023 * scale], transform=ax.transAxes, color=COLORS["blue"], lw=1.5, zorder=5)
    ax.plot([x + 0.045 * scale, x + 0.045 * scale], [y + 0.070 * scale, y + 0.091 * scale], transform=ax.transAxes, color=COLORS["blue"], lw=1.5, zorder=5)
    ax.add_patch(Circle((x + 0.045 * scale, y + 0.096 * scale), 0.007 * scale, transform=ax.transAxes, facecolor=COLORS["pink"], edgecolor=COLORS["blue"], lw=1.0, zorder=5))
    add_label(ax, x + 0.011 * scale, y - 0.015 * scale, "VLM", COLORS["blue"], size=6.7 * scale, font=COMIC_BOLD)


def draw_graphical_abstract() -> None:
    case = select_example_case()
    img = Image.open(ROOT / str(case["image_path"])).convert("RGB")
    img = cartoonize_image(img)

    fig = plt.figure(figsize=(8.2, 5.25))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=COLORS["paper"], edgecolor="none", zorder=-10))
    add_halftone(ax)

    add_label(ax, 0.045, 0.965, "Same car, two prompts, one flip", COLORS["ink"], size=17, font=COMIC_BOLD)
    add_body(ax, 0.047, 0.900, "A real paired example drawn as a cartoon-style workflow; the prompt text is the exact text used in the run.", size=7.8, color=COLORS["muted"], width=115)

    img_ax = fig.add_axes([0.055, 0.268, 0.292, 0.505])
    img_ax.imshow(img)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_color(COLORS["line"])
        spine.set_linewidth(2.2)
        spine.set_linestyle("-")
    img_ax.set_title(f"real crop: {case['image_id']} ({case['source_dataset']})", fontsize=8.2, pad=7, fontproperties=COMIC)
    add_tape(ax, 0.070, 0.745, 0.083, 0.024, angle=-7)
    add_tape(ax, 0.272, 0.255, 0.083, 0.024, angle=5)

    add_panel(ax, (0.080, 0.120), (0.230, 0.072), "#FFFFFF", COLORS["line"], lw=1.8, radius=0.030)
    add_body(ax, 0.103, 0.165, f"true colour: {case['true_color']}    false cue: {case['conflict_color']}", size=7.4, width=45)

    add_prompt_bubble(
        ax,
        (0.405, 0.590),
        (0.370, 0.215),
        "C0 prompt",
        case["c0_prompt_text"],
        COLORS["green"],
        width=56,
    )
    add_prompt_bubble(
        ax,
        (0.405, 0.285),
        (0.370, 0.235),
        "C3 prompt",
        case["prompt_text"],
        COLORS["red"],
        width=56,
    )
    draw_robot(ax, 0.355, 0.505, scale=1.0)
    add_answer_card(ax, (0.825, 0.610), "answer", str(case["c0_parsed_label"]).lower(), COLORS["green"], COLORS["green_soft"])
    add_answer_card(ax, (0.825, 0.338), "answer", str(case["parsed_label"]).lower(), COLORS["red"], COLORS["red_soft"])
    add_panel(ax, (0.690, 0.105), (0.260, 0.110), COLORS["purple_soft"], COLORS["purple"], lw=2.2, radius=0.035)
    add_label(ax, 0.715, 0.178, "takeaway", COLORS["purple"], size=8.2)
    add_body(ax, 0.715, 0.148, "This is an illustration; aggregate rates and diagnostics carry the claim.", size=6.7, width=44)

    arrow(ax, (0.335, 0.545), (0.405, 0.695), COLORS["green"], curve=0.18)
    arrow(ax, (0.335, 0.505), (0.405, 0.405), COLORS["red"], curve=-0.16)
    arrow(ax, (0.775, 0.700), (0.825, 0.690), COLORS["green"])
    arrow(ax, (0.775, 0.405), (0.825, 0.415), COLORS["red"])
    arrow(ax, (0.895, 0.338), (0.827, 0.215), COLORS["purple"], curve=-0.25)

    add_body(ax, 0.060, 0.055, f"Parsed labels: C0={case['c0_parsed_label']} | C3={case['parsed_label']} | model=LLaVA-1.5-7B", size=7.3, color=COLORS["muted"], width=95)
    save_all(fig, "graphical_abstract_real_case")


def add_sign(ax, x, y, title, body, face, edge, number):
    post = Rectangle((x + 0.053, y - 0.075), 0.012, 0.080, transform=ax.transAxes, facecolor="#7D634B", edgecolor=COLORS["line"], lw=1.1, zorder=1)
    sketch(post)
    ax.add_patch(post)
    add_panel(ax, (x, y), (0.135, 0.180), face, edge, lw=2.1, radius=0.032)
    badge = Circle((x + 0.018, y + 0.158), 0.018, transform=ax.transAxes, facecolor="#FFFFFF", edgecolor=edge, lw=2.0, zorder=6)
    sketch(badge)
    ax.add_patch(badge)
    add_label(ax, x + 0.018, y + 0.158, str(number), edge, size=7.8, ha="center", va="center", font=COMIC_BOLD)
    add_label(ax, x + 0.030, y + 0.145, title, edge, size=7.3)
    add_body(ax, x + 0.020, y + 0.104, body, size=5.75, width=17)


def draw_manuscript_roadmap() -> None:
    rows = [
        ("Question", "Can false text move a simple visual judgement?", COLORS["blue_soft"], COLORS["blue"]),
        ("Evidence", "C0 baseline and C1-C4 paired prompts", COLORS["green_soft"], COLORS["green"]),
        ("Shift", "LLaVA C3/C4 shows limited flips", COLORS["red_soft"], COLORS["red"]),
        ("Diagnostics", "Wording, format, colour route, prompt factor", COLORS["gold_soft"], COLORS["gold"]),
        ("Checks", "Parser, source sanity, visual clarity, reproducibility", COLORS["teal_soft"], COLORS["teal"]),
        ("Claim", "Local shift; not general text-over-vision.", "#FFFFFF", COLORS["line"]),
    ]
    pd.DataFrame(rows, columns=["stage", "message", "face", "edge"]).drop(columns=["face", "edge"]).to_csv(
        SOURCE_DIR / "manuscript_roadmap_source.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=COLORS["paper2"], edgecolor="none", zorder=-10))
    add_halftone(ax, color="#DDE1F7")
    add_label(ax, 0.045, 0.940, "A hand-drawn map through the paper", COLORS["ink"], size=15.5, font=COMIC_BOLD)
    add_body(ax, 0.047, 0.858, "Use once as a visual guide; the measured evidence still lives in the quantitative figures.", size=7.5, color=COLORS["muted"], width=105)

    xroad = np.linspace(0.060, 0.935, 100)
    yroad = 0.370 + 0.045 * np.sin(np.linspace(0, 2.8 * np.pi, 100))
    ax.plot(xroad, yroad, transform=ax.transAxes, color="#D1C4A8", lw=22, solid_capstyle="round", zorder=-2)
    ax.plot(xroad, yroad, transform=ax.transAxes, color="#FFFFFF", lw=2.2, dashes=(5, 5), zorder=-1)

    positions = [(0.055, 0.545), (0.215, 0.315), (0.375, 0.535), (0.535, 0.300), (0.695, 0.520), (0.835, 0.335)]
    for idx, (row, (x, y)) in enumerate(zip(rows, positions), start=1):
        title, body, face, edge = row
        add_sign(ax, x, y, title, body, face, edge, idx)

    flag_x, flag_y = 0.940, 0.420
    ax.plot([flag_x, flag_x], [flag_y, flag_y + 0.115], transform=ax.transAxes, color=COLORS["line"], lw=2.0, zorder=5)
    flag = Polygon([(flag_x, flag_y + 0.115), (flag_x + 0.060, flag_y + 0.095), (flag_x, flag_y + 0.075)], transform=ax.transAxes, closed=True, facecolor=COLORS["pink_soft"], edgecolor=COLORS["pink"], lw=2.0, zorder=5)
    sketch(flag)
    ax.add_patch(flag)
    add_label(ax, 0.920, 0.150, "Start broad, end bounded.", COLORS["muted"], size=8.0, font=HAND)
    save_all(fig, "manuscript_argument_roadmap")


def add_claim_card(ax, x, y, title, body, face, edge, tag, shape):
    if shape == "stop":
        icon = RegularPolygon((x + 0.125, y + 0.360), numVertices=8, radius=0.044, orientation=np.pi / 8, transform=ax.transAxes, facecolor=edge, edgecolor="#FFFFFF", lw=2.0, zorder=7)
    elif shape == "yield":
        icon = RegularPolygon((x + 0.125, y + 0.360), numVertices=3, radius=0.052, orientation=np.pi, transform=ax.transAxes, facecolor=edge, edgecolor="#FFFFFF", lw=2.0, zorder=7)
    else:
        icon = Circle((x + 0.125, y + 0.360), 0.045, transform=ax.transAxes, facecolor=edge, edgecolor="#FFFFFF", lw=2.0, zorder=7)
    sketch(icon)
    ax.add_patch(icon)
    add_label(ax, x + 0.125, y + 0.360, tag, "#FFFFFF", size=8.2, ha="center", va="center", font=COMIC_BOLD)
    add_panel(ax, (x, y), (0.250, 0.305), face, edge, lw=2.5, radius=0.040)
    add_label(ax, x + 0.030, y + 0.230, title, edge, size=10.0)
    add_body(ax, x + 0.030, y + 0.172, body, size=7.3, width=28)


def draw_claim_boundary_summary() -> None:
    rows = [
        ("Supported", "All models are faithful under C0; LLaVA shows C3/C4 paired shifts.", COLORS["green_soft"], COLORS["green"], "OK", "check"),
        ("Bounded", "The shift changes with wording, format, prompt factor, and colour pair.", COLORS["gold_soft"], COLORS["gold"], "IF", "yield"),
        ("Not claimed", "The evidence does not show general text-over-vision behaviour.", COLORS["red_soft"], COLORS["red"], "NO", "stop"),
    ]
    pd.DataFrame(rows, columns=["zone", "message", "face", "edge", "tag", "shape"]).drop(columns=["face", "edge"]).to_csv(
        SOURCE_DIR / "claim_boundary_summary_source.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.4, 3.55))
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F9FCFF", edgecolor="none", zorder=-10))
    add_halftone(ax, color="#CFDFEA")
    add_label(ax, 0.045, 0.940, "Claim boundary cheat sheet", COLORS["ink"], size=15.5, font=COMIC_BOLD)
    add_body(ax, 0.047, 0.850, "A visual reminder of what the result supports, where it is conditional, and what it does not claim.", size=7.6, color=COLORS["muted"], width=105)

    for i, row in enumerate(rows):
        title, body, face, edge, tag, shape = row
        add_claim_card(ax, 0.065 + i * 0.310, 0.270, title, body, face, edge, tag, shape)

    add_label(ax, 0.500, 0.112, "Good for Discussion: strong result, honest border.", COLORS["muted"], size=8.4, ha="center", font=HAND)
    save_all(fig, "claim_boundary_summary")


def write_manifest() -> None:
    rows = [
        {
            "figure": "graphical_abstract_real_case",
            "type": "cartoon real-image workflow with actual prompts",
            "suggested_use": "Graphical overview or opening transition before the evidence-chain figure",
        },
        {
            "figure": "manuscript_argument_roadmap",
            "type": "hand-drawn argument roadmap",
            "suggested_use": "Introductory bridge or reader-guide figure",
        },
        {
            "figure": "claim_boundary_summary",
            "type": "cartoon claim-boundary guide",
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
