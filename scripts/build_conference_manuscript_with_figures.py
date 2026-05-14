from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


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


def strip_working_notes(text: str) -> str:
    marker = "\n## Polishing Notes\n"
    if marker in text:
        return text.split(marker, 1)[0].rstrip() + "\n"
    return text


def copy_figures() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for spec in FIGURES.values():
        src = spec["png"]
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, IMG_DIR / src.name)


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
    text = insert_figures(text)
    out_md = OUT_DIR / "conference_manuscript_with_figures.md"
    out_md.write_text(text, encoding="utf-8", newline="\n")
    build_with_pandoc(out_md)
    print(f"Wrote {out_md}")
    print(f"Wrote {OUT_DIR / 'conference_manuscript_with_figures.docx'}")
    print(f"Wrote {OUT_DIR / 'conference_manuscript_with_figures.html'}")


if __name__ == "__main__":
    main()
