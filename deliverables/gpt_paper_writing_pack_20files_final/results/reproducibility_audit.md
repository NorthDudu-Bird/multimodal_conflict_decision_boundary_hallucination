# Reproducibility Audit

- Snapshot root: `D:\multimodal_conflict_decision_boundary_hallucination\logs\reproducibility_snapshot\latest`
- Files checked: 51
- Exact/normalized matches: 50
- Blocking mismatches or missing files: 0
- Non-blocking writing-summary differences: 1
- Verdict: The rerun reproduced all locked experimental artifacts.

## Allowed Non-Canonical Differences

- `*.log`
- `*_runtime.csv`
- `*_run_metadata.json`
- `*_raw_results.csv`
- `*_parse_review.csv`
- `results/**/raw/**`

## Non-Blocking Writing-Facing Differences

- `results/final_result_summary.md`: different. Writing-facing summary differs from the locked snapshot; experimental artifacts remain gated separately.

## Result

- All tracked canonical manifests, prompts, parsed outputs, condition metrics, key tests, parser audit files, and appendix sanity files matched the locked snapshot.
- Any log/runtime/raw-output differences are outside the reproducibility gate.
