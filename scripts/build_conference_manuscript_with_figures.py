from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_MD = ROOT / "docs" / "conference_manuscript_polished.md"
OUT_DIR = ROOT / "deliverables" / "conference_submission_draft"
IMG_DIR = OUT_DIR / "images"

FIGURES = {
    "Figure 1": {
        "png": ROOT / "figures" / "conference" / "figure1_evidence_chain.png",
        "alt": "Figure 1. Evidence chain for same-image conflict-following analysis.",
    },
    "Figure 2": {
        "png": ROOT / "figures" / "conference" / "figure2_main_conflict_rates.png",
        "alt": "Figure 2. Primary C0-C4 conflict-following rates.",
    },
    "Figure 3": {
        "png": ROOT / "figures" / "conference" / "figure3_paired_flips.png",
        "alt": "Figure 3. Same-image paired faithful-to-conflict flips.",
    },
    "Figure 4": {
        "png": ROOT / "figures" / "conference" / "figure4_boundary_diagnostics.png",
        "alt": "Figure 4. Boundary diagnostics for the primary LLaVA shift.",
    },
}

NARRATIVE_FIGURES = {
    "Graphical overview": {
        "png": ROOT / "figures" / "conference_narrative" / "graphical_abstract_real_case.png",
        "alt": "Graphical overview. Real-prompt paired workflow for one LLaVA example.",
    },
    "Reader map": {
        "png": ROOT / "figures" / "conference_narrative" / "manuscript_argument_roadmap.png",
        "alt": "Reader map. Hand-drawn argument path of the manuscript.",
    },
    "Claim boundary guide": {
        "png": ROOT / "figures" / "conference_narrative" / "claim_boundary_summary.png",
        "alt": "Claim boundary guide. Supported, bounded, and not-claimed conclusions.",
    },
}


def strip_working_notes(text: str) -> str:
    marker = "\n## Polishing Notes\n"
    if marker in text:
        return text.split(marker, 1)[0].rstrip() + "\n"
    return text


def copy_figures() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for spec in list(NARRATIVE_FIGURES.values()) + list(FIGURES.values()):
        src = spec["png"]
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, IMG_DIR / src.name)


def insert_narrative_figures(text: str) -> str:
    for fig_label, spec in NARRATIVE_FIGURES.items():
        filename = spec["png"].name
        image_md = f"![](images/{filename}){{width=100%}}\n\n"
        pattern = rf"(\*\*{re.escape(fig_label)}\.[\s\S]*?)(?=\n\n)"

        def repl(match: re.Match[str]) -> str:
            block = match.group(1)
            if f"](images/{filename})" in block:
                return block
            return image_md + block

        text, n = re.subn(pattern, repl, text, count=1)
        if n != 1:
            raise RuntimeError(f"Could not insert {fig_label}")
    return text


def insert_figures(text: str) -> str:
    for fig_label, spec in FIGURES.items():
        filename = spec["png"].name
        image_md = f"![](images/{filename}){{width=100%}}\n\n"
        pattern = rf"(\*\*{re.escape(fig_label)}\.[\s\S]*?)(?=\n\n## |\n\n### |\Z)"

        def repl(match: re.Match[str]) -> str:
            block = match.group(1)
            if f"](images/{filename})" in block:
                return block
            return image_md + block

        text, n = re.subn(pattern, repl, text, count=1)
        if n != 1:
            raise RuntimeError(f"Could not insert {fig_label}")
    return text


def make_table1() -> str:
    src = pd.read_csv(ROOT / "data" / "metadata" / "balanced_eval_set" / "balanced_eval_set_color_by_source.csv")
    pivot = src.pivot(index="true_color", columns="source_dataset", values="count").fillna(0).astype(int)
    rows = []
    for colour in ["black", "blue", "green", "red", "white", "yellow"]:
        stanford = int(pivot.loc[colour, "StanfordCars"])
        vcor = int(pivot.loc[colour, "VCoR"])
        rows.append([colour, stanford, vcor, stanford + vcor])
    rows.append(["Total", sum(r[1] for r in rows), sum(r[2] for r in rows), sum(r[3] for r in rows)])
    df = pd.DataFrame(rows, columns=["True colour", "StanfordCars", "VCoR", "Total"])
    return df.to_markdown(index=False)


def make_table2() -> str:
    rows = [
        ["C0", "Neutral baseline", "No false colour cue", "Estimate faithful visual colour recognition"],
        ["C1", "Weak suggestion", "Low-strength erroneous colour cue", "Test weak textual influence"],
        ["C2", "False assertion", "Open prompt with a false colour assertion", "Test direct false-text conflict"],
        ["C3", "Presupposition, correction allowed", "False colour embedded as a presupposition while correction remains possible", "Primary conflict condition"],
        ["C4", "Stronger open conflict", "Repeated or stronger false-colour framing", "Primary stronger conflict condition"],
    ]
    df = pd.DataFrame(rows, columns=["Condition", "Role", "False-text form", "Purpose in analysis"])
    return df.to_markdown(index=False)


def make_table3() -> str:
    src = pd.read_csv(ROOT / "results" / "main" / "table1_main_metrics.csv")
    order = ["LLaVA-1.5-7B", "Qwen2-VL-7B-Instruct", "InternVL2-8B"]
    src["model"] = pd.Categorical(src["model"], order, ordered=True)
    src["condition"] = pd.Categorical(src["condition"], ["C0", "C1", "C2", "C3", "C4"], ordered=True)
    src = src.sort_values(["model", "condition"])
    keep = src[["model", "condition", "n", "conflict_aligned", "faithful"]].copy()
    keep.columns = ["Model", "Condition", "n", "False-colour aligned", "Faithful"]
    return keep.to_markdown(index=False)


def replace_table_block(text: str, table_label: str, replacement: str) -> str:
    pattern = rf"(\*\*{re.escape(table_label)}\.[\s\S]*?)(?=\n\n### |\n\n## |\Z)"
    text, n = re.subn(pattern, replacement, text, count=1)
    if n != 1:
        raise RuntimeError(f"Could not replace {table_label}")
    return text


def insert_tables(text: str) -> str:
    table1 = (
        "**Table 1. Evaluation set composition.** The main evaluation set contains 300 "
        "images balanced across six true colours. Source identity is retained for "
        "provenance and sanity checks, not as a primary comparison axis.\n\n"
        f"{make_table1()}\n"
    )
    table2 = (
        "**Table 2. Prompt condition roles.** C0 is the neutral reference condition. "
        "C1-C4 introduce erroneous colour information with increasing or different "
        "forms of textual conflict; full prompt text is reserved for supplementary "
        "material.\n\n"
        f"{make_table2()}\n"
    )
    table3 = (
        "**Table 3. Main C0-C4 metrics.** Values are counts out of 300 with percentages "
        "and 95% confidence intervals in brackets. Asterisks and daggers follow the "
        "locked source table and mark the primary LLaVA C3/C4 findings.\n\n"
        f"{make_table3()}\n"
    )
    text = replace_table_block(text, "Table 1", table1)
    text = replace_table_block(text, "Table 2", table2)
    text = replace_table_block(text, "Table 3", table3)
    return text


def clean_captions(text: str) -> str:
    text = re.sub(r"\*\*(Figure \d\.[^*]+)\*\* File stem:\n`[^`]+`\. ", r"**\1** ", text)
    return text


def build_with_pandoc(md_path: Path) -> None:
    docx_path = OUT_DIR / "conference_manuscript_with_figures.docx"
    html_path = OUT_DIR / "conference_manuscript_with_figures.html"
    common = [
        "pandoc",
        str(md_path),
        "--resource-path",
        str(OUT_DIR),
        "--metadata",
        "title=False Colour Cues Reveal Local Paired Shifts in Car-Colour VLMs",
    ]
    subprocess.run(common + ["-o", str(docx_path)], check=True, cwd=ROOT)
    subprocess.run(common + ["--standalone", "--toc", "-o", str(html_path)], check=True, cwd=ROOT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_figures()
    text = SRC_MD.read_text(encoding="utf-8")
    text = strip_working_notes(text)
    text = clean_captions(text)
    text = insert_tables(text)
    text = insert_narrative_figures(text)
    text = insert_figures(text)
    out_md = OUT_DIR / "conference_manuscript_with_figures.md"
    out_md.write_text(text, encoding="utf-8", newline="\n")
    build_with_pandoc(out_md)
    print(f"Wrote {out_md}")
    print(f"Wrote {OUT_DIR / 'conference_manuscript_with_figures.docx'}")
    print(f"Wrote {OUT_DIR / 'conference_manuscript_with_figures.html'}")


if __name__ == "__main__":
    main()
