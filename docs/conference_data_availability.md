# Conference Data and Code Availability Draft

This file is the first `nature-data` pass for the conference manuscript. It turns the
project files into a transparent availability package while keeping the StanfordCars and
VCoR source-image restrictions visible.

## Ready-To-Paste Manuscript Text

### Data Availability

The processed manifests, reviewed colour labels, prompt condition tables, parsed model
outputs, statistical result tables, figure source data, audit summaries, and
reproducibility records supporting this study are available in the project repository at
`https://github.com/NorthDudu-Bird/multimodal_conflict_decision_boundary_hallucination`
and should be archived in a DOI-bearing release before submission. The primary
300-image evaluation manifest is stored at `data/balanced_eval_set/final_manifest.csv`,
with the balanced-set summary at
`data/metadata/balanced_eval_set/balanced_eval_set_summary.json`. Figure source data for
the conference draft are provided under `figures/conference/source_data/`.

The underlying vehicle images are reused third-party data from StanfordCars and VCoR.
Readers should obtain the original images from the corresponding dataset providers and
respect their terms of use. The repository records source dataset identity, source paths,
cropped-image paths, reviewed labels, and selection metadata needed to reconstruct the
evaluation set, but the paper does not claim to newly release or relicense the original
third-party image collections.

### Code Availability

The analysis and figure-generation code used for this manuscript draft is available in
the same repository at
`https://github.com/NorthDudu-Bird/multimodal_conflict_decision_boundary_hallucination`.
The main reproduction entry points are documented in `docs/reproduction.md`; the current
conference-figure script is `scripts/make_conference_figures.py`. Model weights are not
redistributed by this repository and should be obtained from their respective model
providers.

## Data Inventory

| Dataset or artifact family | Location in project | Access route | Manuscript role |
| --- | --- | --- | --- |
| Primary 300-image evaluation manifest | `data/balanced_eval_set/final_manifest.csv` | Public repository metadata; source images are third-party | Defines image IDs, source dataset, source path, crop path, true colour, false colour, and inclusion flags |
| Balanced-set summaries | `data/metadata/balanced_eval_set/` | Public repository | Documents 300 total images, six colours, 93 StanfordCars images, and 207 VCoR images |
| StanfordCars manual/review labels | `data/annotations/stanford_cars/` | Public repository metadata; original images from provider | Supports reviewed car-body primary-colour labels for StanfordCars-derived rows |
| Main parsed outputs and metrics | `results/main/`; `results/baseline/` | Public repository | Supports C0-C4 rates, paired flips, exact tests, and primary tables |
| Auxiliary diagnostics | `results/auxiliary/` | Public repository | Supports A1/A2 stress-test interpretation |
| Controlled diagnostics | `results/robustness/`; `results/color_split/`; `results/format_control/`; `results/factorization/`; `results/multiturn/` | Public repository | Supports wording, colour-pair, answer-format, prompt-factor, and multi-turn boundary statements |
| Parser/source/visual/reproducibility audits | `results/parser/`; `results/appendix/`; `results/audit/`; `results/reproducibility_audit.md` | Public repository | Supports validity and scope claims |
| Conference figure source data | `figures/conference/source_data/` | Public repository | Source data for Figures 1-4 in the conference draft |
| Figure exports | `figures/conference/` | Public repository | SVG/PDF/TIFF/PNG figure outputs |
| Reproduction scripts | `scripts/`; `docs/reproduction.md` | Public repository | Recreates analyses, audits, and paper figures |
| Model weights | External model providers | Reused public/model-provider resources | Not redistributed; needed for fresh inference reruns |

## Third-Party Dataset Notes

- StanfordCars is cited through the original fine-grained car categorization work by
  Krause et al. The project uses a subset of car images after colour review and cropping;
  source identity is retained for provenance and sanity checks, not as a primary
  comparison axis.
- VCoR is cited through the public Kaggle dataset and the associated article by Panetta
  et al. The project uses VCoR as a source of real vehicle images with vehicle colour
  labels, then filters and balances rows for the six-colour task.
- Formal manuscript wording should avoid implying that this project owns or relicenses
  either source image collection.

## Repository And Citation Actions

- Create a versioned release before submission and archive it with a DOI-bearing service
  such as Zenodo, Figshare, OSF, or an institutional repository.
- Add the final DOI to the Data Availability and Code Availability sections.
- Confirm whether the formal public archive may include cropped third-party images. If
  not, archive the manifests, reviewed labels, parsed outputs, result tables, source data,
  and scripts, while directing readers to the dataset providers for image downloads.
- Add formal dataset/reference-list entries for StanfordCars and VCoR in the target venue
  style.
- Consider moving large TIFF exports to a release asset or data archive if repository
  size becomes a concern.

## FAIR Metadata Checklist

| Check | Status | Note |
| --- | --- | --- |
| Findable | Partial | GitHub URL exists; DOI-bearing release still needed |
| Accessible | Partial | Code, metadata, outputs, and source data are public in the repository; third-party images depend on provider access terms |
| Interoperable | Good | Core data are CSV, JSON, Markdown, SVG, PDF, TIFF, and PNG |
| Reusable | Partial | Reproduction guide and scripts exist; final archive should add release DOI, licence, and README/data dictionary fields |
| Figure source data mapped | Good | `figures/conference/source_data/` maps source CSVs to Figures 1-4 |
| Restrictions explicit | Good | Third-party image restriction is stated directly |
| Code route explicit | Partial | GitHub route exists; DOI-bearing code archive still needed |

## Chinese Author Check

- 现在不能写成 "all data are available in the paper"，因为支撑图表和复现的 CSV/JSON/脚本在仓库里。
- 不建议只写 "available upon request"。这里没有隐私/临床限制，主要限制来自第三方图片再分发权。
- 正式投稿前最关键的动作是给当前 GitHub release 归档 DOI，并确认裁剪后的 StanfordCars/VCoR 图片是否能随 archive 公开。
- Stanford/VCoR 只写成来源和限制，不要把它们写成论文主比较轴。
