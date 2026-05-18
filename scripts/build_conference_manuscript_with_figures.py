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
    "Claim-boundary guide": {
        "png": ROOT / "figures" / "conference_narrative" / "claim_boundary_summary.png",
        "alt": "Claim-boundary guide. Supported, bounded, and not-claimed conclusions.",
    },
}

REFERENCES_MD = """## References

[1] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in Proc. ICML, 2021.

[2] J.-B. Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning," arXiv:2204.14198, 2022.

[3] J. Li, D. Li, S. Savarese, and S. Hoi, "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," in Proc. ICML, 2023.

[4] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," arXiv:2304.08485, 2023.

[5] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved Baselines with Visual Instruction Tuning," arXiv:2310.03744, 2023.

[6] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering," in Proc. CVPR, 2017.

[7] A. Agrawal, D. Batra, D. Parikh, and A. Kembhavi, "Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering," in Proc. CVPR, 2018.

[8] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, "Evaluating Object Hallucination in Large Vision-Language Models," arXiv:2305.10355, 2023.

[9] T. Guan et al., "HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models," arXiv:2310.14566, 2023.

[10] K.-i. Lee, M. Kim, S. Yoon, M. Kim, D. Lee, H. Koh, and K. Jung, "VLind-Bench: Measuring Language Priors in Large Vision-Language Models," arXiv:2406.08702, 2024.

[11] Y. Liang et al., "ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness," arXiv:2504.10514, 2025.

[12] M. Yuksekgonul, F. Bianchi, P. Kalluri, D. Jurafsky, and J. Zou, "When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It?," in Proc. ICLR, 2023.

[13] T. Thrush et al., "Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality," in Proc. CVPR, 2022.

[14] J. Krause, M. Stark, J. Deng, and L. Fei-Fei, "3D Object Representations for Fine-Grained Categorization," in Proc. 4th International IEEE Workshop on 3D Representation and Recognition, 2013.

[15] K. Panetta, L. Kezebou, V. Oludare, J. Intriligator, and S. Agaian, "Artificial Intelligence for Text-Based Vehicle Search, Recognition, and Continuous Localization in Traffic Videos," AI, vol. 2, no. 4, pp. 684-704, 2021.

[16] P. Wang et al., "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution," arXiv:2409.12191, 2024.

[17] Z. Chen et al., "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks," arXiv:2312.14238, 2023.

[18] Z. Chen et al., "How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites," arXiv:2404.16821, 2024.
"""


def strip_working_notes(text: str) -> str:
    marker = "\n## Polishing Notes\n"
    if marker in text:
        return text.split(marker, 1)[0].rstrip() + "\n"
    return text


def insert_references(text: str) -> str:
    text = text.rstrip()
    references = REFERENCES_MD.rstrip() + "\n"
    if "\n## References\n" in text:
        return re.sub(r"\n## References\n[\s\S]*\Z", "\n\n" + references, text)
    return text + "\n\n" + references


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
    text = insert_references(text)
    out_md = OUT_DIR / "conference_manuscript_with_figures.md"
    out_md.write_text(text, encoding="utf-8", newline="\n")
    build_with_pandoc(out_md)
    print(f"Wrote {out_md}")
    print(f"Wrote {OUT_DIR / 'conference_manuscript_with_figures.docx'}")
    print(f"Wrote {OUT_DIR / 'conference_manuscript_with_figures.html'}")


if __name__ == "__main__":
    main()
