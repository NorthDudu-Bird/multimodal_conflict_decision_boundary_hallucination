# Conference Submission Draft Deliverable

This directory contains a generated manuscript draft with embedded figures.

## Files

- `conference_manuscript_with_figures.md`: Markdown manuscript with inline PNG figures.
- `conference_manuscript_with_figures.docx`: DOCX generated from the Markdown via Pandoc.
- `conference_manuscript_with_figures.html`: standalone HTML generated from the Markdown via Pandoc.
- `images/`: PNG copies of Figures 1-4 used by the Markdown and HTML files.

## Regeneration

From the repository root:

```bash
python scripts/build_conference_manuscript_with_figures.py
```

The script reads `docs/conference_manuscript_polished.md`, removes internal polishing
notes, inserts the four conference figures, and writes the deliverable files here.

## Current QA Status

- Pandoc successfully generated DOCX and HTML.
- Pandoc DOCX round-trip extraction confirmed that all four figures are embedded.
- The repository reproducibility gate reports `blocking_failures=0`.
- Page-level DOCX render QA was attempted but could not be completed on this machine
  because LibreOffice/`soffice` is unavailable and the renderer could not infer page
  size from the Pandoc DOCX.

## Before Formal Submission

- Choose the target venue and apply its official LaTeX/Word template.
- Convert citation keys to the venue's bibliography format.
- Render final table bodies for Tables 1-3.
- Archive the repository or release package with a DOI-bearing service.
- Confirm whether cropped StanfordCars/VCoR images can be included in the public archive;
  otherwise archive manifests and derived outputs only.
- Re-run visual QA after the manuscript is placed in the final venue template.
