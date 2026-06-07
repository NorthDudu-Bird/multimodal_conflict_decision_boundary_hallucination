# Balanced Eval Set Cleaning Rules

1. The official paper dataset is the final balanced evaluation set in `data/balanced_eval_set/final_manifest.csv`.
2. Only the six paper colors are retained for the mainline: `red`, `blue`, `green`, `yellow`, `black`, `white`.
3. The Stanford Cars strict-clean subset is treated as the seed source, not the final paper benchmark.
4. Ten Stanford seed examples flagged by the latest manual review are excluded before balancing.
5. VCoR supplementation is used only to fill per-color shortages until each color reaches 50 examples.
6. The benchmark keeps the strict reviewed truth labels and does not relax the faithful-match rule.
7. `gray`, `silver`, and `other` remain excluded from the paper main analysis.
