# Conference Figure Plan

This plan follows the `nature-figure` contract with Python as the selected backend. The
figures are designed for a computer-conference manuscript rather than a Nature-format
article page, but they keep Nature-style evidence discipline: each figure must defend a
specific claim, expose source data, and avoid overstating the experiment.

## Export Contract

- Backend: Python / matplotlib only.
- Output directory: `figures/conference/`.
- Export formats: SVG, PDF, TIFF, and PNG for every figure.
- Source data: `figures/conference/source_data/`.
- Text policy: SVG text remains editable through `svg.fonttype = none`.
- Statistical display: rates use existing Wilson confidence intervals where shown; paired
  flip figures use locked paired-count tables.

## Figure 1: Evidence Chain

- File stem: `figure1_evidence_chain`.
- Archetype: schematic-led composite.
- Core conclusion: the paper's claim is built from a short chain: same images, faithful
  C0 baseline, C1-C4 false-colour prompts, paired flips, and boundary diagnostics.
- Unique evidence role: defines the interpretation logic before quantitative panels.
- Review risk: readers may read the study as a general text-over-vision claim; the final
  box explicitly states the bounded claim.

## Figure 2: Main Conflict-Following Rates

- File stem: `figure2_main_conflict_rates`.
- Archetype: quantitative grid.
- Core conclusion: in the primary C0-C4 family, only LLaVA-1.5-7B shows a clear C3/C4
  false-colour-aligned shift.
- Source data: `results/main/main_condition_metrics.csv`.
- Review risk: C0-C4 rates are small in absolute terms, so the y-axis is kept low and
  count labels are shown for non-zero cells.

## Figure 3: Same-Image Paired Flips

- File stem: `figure3_paired_flips`.
- Archetype: quantitative grid.
- Core conclusion: because C0 is fully faithful, conflict-aligned C3/C4 rows can be read
  as faithful-to-conflict flips on the same images.
- Source data: `results/main/paired_flip_metrics.csv`.
- Review risk: this panel should not duplicate Figure 2; it reframes the same phenomenon
  around paired attribution rather than condition-level rates.

## Figure 4: Boundary Diagnostics

- File stem: `figure4_boundary_diagnostics`.
- Archetype: asymmetric mixed-modality figure.
- Core conclusion: the LLaVA shift is not a stable cross-template law; it is bounded by
  wording, answer format, colour route, and prompt factor.
- Source data:
  - `results/robustness/prompt_boundary_metrics.csv`
  - `results/format_control/format_control_metrics.csv`
  - `results/color_split/color_pair_family_metrics.csv`
  - `results/factorization/factorized_prompt_metrics.csv`
- Review risk: factorized prompts can look like a stronger main result. Panel d is
  labelled as a boundary regime so it does not overwrite the primary single-turn claim.

## Not Yet Included

- Visual clarity contact sheets are not included in the main figure set because they are
  better suited to supplementary material.
- Multi-turn diagnostics are not included in the first main figure set because they test
  a distinct dialogue-context regime. They can become a supplementary figure after the
  main story stabilizes.
