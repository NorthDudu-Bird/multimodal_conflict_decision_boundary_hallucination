# Conference Narrative Figures

This note records the optional narrative and transition figures generated with the
Python `nature-figure` workflow. These are not replacements for the quantitative Figures
1-4. They are meant to make the manuscript, talk, or graphical abstract easier to read.
The current version uses a lighter cartoon-like visual style for non-data figures while
keeping all scientific content traceable to source data.

## Generated Figures

| Figure stem | File location | Best use | Notes |
| --- | --- | --- | --- |
| `graphical_abstract_real_case` | `figures/conference_narrative/graphical_abstract_real_case.*` | Graphical abstract or opening transition before Figure 1 | Uses a real LLaVA C3 paired-flip example from the primary evaluation set and displays the actual C0/C3 prompt text shown to the model. Treat as illustrative, not as stand-alone evidence. |
| `manuscript_argument_roadmap` | `figures/conference_narrative/manuscript_argument_roadmap.*` | Introductory bridge or talk slide | Uses a playful reader-map layout to summarize the argument path from question to bounded claim. |
| `claim_boundary_summary` | `figures/conference_narrative/claim_boundary_summary.*` | Discussion transition, graphical takeaway, or optional summary panel after Figure 4 | Uses a compact visual-guide layout to keep the paper's claim strong without overgeneralizing. |

## Suggested Placement

- Manuscript opening: use `graphical_abstract_real_case` as an unnumbered graphical
  abstract if the venue allows it.
- Main text: use `manuscript_argument_roadmap` only if the paper needs a visual bridge
  between Introduction and Results. Otherwise keep it for slides.
- Discussion: use `claim_boundary_summary` if the final manuscript needs a compact
  visual reminder of what is supported, bounded, and not claimed.

## Source-Data Notes

- `figures/conference_narrative/source_data/graphical_abstract_example_case.csv` records
  the real example used in the graphical abstract, including the exact C0 and C3 prompts.
- The example image is from a reused third-party dataset. Before public archival release,
  confirm whether cropped StanfordCars/VCoR images can be redistributed. If not, use the
  figure only in manuscript-review materials where permitted by the target venue and
  dataset terms.

## Preview Links

![Graphical abstract real case](../figures/conference_narrative/graphical_abstract_real_case.png)

![Manuscript argument roadmap](../figures/conference_narrative/manuscript_argument_roadmap.png)

![Claim boundary summary](../figures/conference_narrative/claim_boundary_summary.png)
