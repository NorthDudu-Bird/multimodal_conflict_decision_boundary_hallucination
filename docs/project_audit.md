# Repository Audit

## Retained Public Components

- `configs/`: active experiment configuration
- `data/`: evaluation manifests, annotations, metadata, and processed samples
- `data_external/`: shareable source-selection metadata
- `prompts/`: prompts used by the experiments
- `scripts/`: dataset, inference, parsing, analysis, and audit code
- `results/`: structured outputs, metrics, statistical tests, audits, and plots

## Excluded Components

- Manuscripts and article drafts
- Submission and reviewer material
- Writing packages and document-formatting tools
- Personal notes and contact information
- Model weights and original restricted datasets
- Runtime logs, local caches, and temporary rendering output

## Maintenance Rule

Keep only material required to understand, execute, or audit the experiment. Before
every push, inspect the staged file list and scan for documents, archives, credentials,
email addresses, and absolute local paths.
