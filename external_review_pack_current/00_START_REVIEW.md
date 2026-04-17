# External Review Pack

This folder is the smallest current package for reviewing whether the experiment is rigorous.

## Reading order for a human reviewer

1. `01_paper_ready_results_summary.md`
2. `02_dataset_update_note.md`
3. `03_rerun_summary.md`
4. `04_primary_robustness_comparison.csv`
5. `05_auxiliary_robustness_comparison.csv`
6. `08_restructured_experiment_vcor_balanced.yaml`

Use this as the main example browser:

- `09_vcor_balanced_multimodel_results_viewer.html`

In that viewer, each image now shows:

- `source_dataset`
- `source_split`

So you do not need to inspect Stanford and VCoR in completely separate files just to know where an image came from.

Use this only if you specifically want the control viewer:

- `10_stanford_core_multimodel_results_viewer.html`

Use these if you want raw final tables:

- `06_primary_results_expanded_balanced.csv`
- `07_auxiliary_results_expanded_balanced.csv`
- `11_primary_expanded_balanced_with_vcor.csv`
- `12_primary_core_stanford_only.csv`

## If you are giving this to GPT or another LLM

Yes, GPT can read HTML source if the file contents are provided.

But GPT does not use the HTML like a browser:

- it does not interact with the page filters
- it does not get the same visual browsing experience as a human
- JavaScript-heavy viewers are less useful than raw markdown and CSV

So for model-based review, prioritize:

1. `01_paper_ready_results_summary.md`
2. `02_dataset_update_note.md`
3. `03_rerun_summary.md`
4. `04_primary_robustness_comparison.csv`
5. `05_auxiliary_robustness_comparison.csv`
6. `08_restructured_experiment_vcor_balanced.yaml`

Treat the HTML files as optional supporting material, mainly for human spot-checking.
If you do use the main HTML viewer, the dataset origin is shown directly on each image group.
