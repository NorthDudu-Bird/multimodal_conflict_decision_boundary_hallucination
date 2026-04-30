# Reproducibility Audit

- Snapshot root: `D:\multimodal_conflict_decision_boundary_hallucination\logs\reproducibility_snapshot\latest`
- Canonical files checked: 51
- Exact/normalized matches: 51
- Blocking mismatches or missing files: 0
- Verdict: The rerun reproduced all locked canonical artifacts.

## Allowed Non-Canonical Differences

- `*.log`
- `*_runtime.csv`
- `*_run_metadata.json`
- `*_raw_results.csv`
- `*_parse_review.csv`
- `results/**/raw/**`

## Result

- All tracked canonical manifests, prompts, parsed outputs, condition metrics, key tests, summary files, parser audit files, and appendix sanity files matched the locked snapshot.
- Any log/runtime/raw-output differences are outside the reproducibility gate.
