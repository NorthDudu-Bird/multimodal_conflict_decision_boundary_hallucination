# Multimodal Conflict Decision Boundary Hallucination

This workspace is now cleaned down to one main active experiment line plus one optional control line:

- `configs/current/restructured_experiment_vcor_balanced.yaml`
  Active main experiment: Stanford Cars + VCoR strict clean balanced rerun
- `configs/current/restructured_experiment_stanford_core_vcor_robustness.yaml`
  Stanford-only strict clean core control rerun for robustness only

Open these first:

- `results_summary/current/vcor_balanced_rerun/`
- `reports/current/vcor_balanced_multimodel_results_viewer.html`
- `data/processed/stanford_cars/primary_expanded_balanced_with_vcor.csv`

The main viewer and the main manifest already mix Stanford Cars and VCoR together.
Use `source_dataset` and `source_split` to see where each image came from.
You do not need separate dataset browsers as the default workflow.

Active results live in:

- `analysis/current/vcor_balanced_primary/`
- `analysis/current/vcor_balanced_auxiliary/`
- `analysis/current/stanford_core_primary/`
- `analysis/current/stanford_core_auxiliary/`
- `outputs/current_vcor_balanced/`
- `outputs/current_stanford_core/`

Active prompts live in:

- `prompts/current/vcor_balanced_primary_prompts.csv`
- `prompts/current/vcor_balanced_auxiliary_prompts.csv`
- `prompts/current/stanford_core_primary_prompts.csv`
- `prompts/current/stanford_core_auxiliary_prompts.csv`

Optional control-only entry points:

- `reports/current/stanford_core_multimodel_results_viewer.html`
- `data/processed/stanford_cars/primary_core_stanford_only.csv`

Cleanup note:

- Old experiments, old viewers, old prompts, old report images, old markdown files, and legacy archives were moved out of the active workspace locally.
- The current GitHub snapshot focuses on the active mainline and does not include local cleanup archives or raw external dataset packages.
