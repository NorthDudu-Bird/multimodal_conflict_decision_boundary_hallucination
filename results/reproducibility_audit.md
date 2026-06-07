# Reproducibility Audit

- Snapshot root: `logs/reproducibility_snapshot/latest`
- Files checked: 50
- Exact/normalized matches: 50
- Blocking mismatches or missing files: 0
- Verdict: The rerun reproduced all locked experimental artifacts.

## Allowed Non-Canonical Differences

- `*.log`
- `*_runtime.csv`
- `*_run_metadata.json`
- `*_raw_results.csv`
- `*_parse_review.csv`
- `results/**/raw/**`

## Result

- All tracked canonical manifests, prompts, parsed outputs, condition metrics, key tests, parser audit files, and appendix sanity files matched the locked snapshot.
- Any log/runtime/raw-output differences are outside the reproducibility gate.
