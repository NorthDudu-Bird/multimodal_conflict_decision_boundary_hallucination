# Reproducibility Audit

- Snapshot root: `D:\multimodal_conflict_decision_boundary_hallucination\logs\reproducibility_snapshot\latest`
- Canonical files checked: 51
- Exact/normalized matches: 50
- Blocking mismatches or missing files: 1
- Verdict: The rerun did not reproduce all locked canonical artifacts.

## Allowed Non-Canonical Differences

- `*.log`
- `*_runtime.csv`
- `*_run_metadata.json`
- `*_raw_results.csv`
- `*_parse_review.csv`
- `results/**/raw/**`

## Blocking Items

- `results/final_result_summary.md`: different. Canonical artifact differs from the locked snapshot.

## Phase 2 Interpretation Note

The single mismatch is an intentional writing-facing update: `results/final_result_summary.md`
now contains the Phase 2 diagnostic addendum. The locked canonical result tables,
statistics, parser/source audits, and figures remain reproducible against the snapshot.
Treat this as a summary-document drift after additional diagnostics, not as a change in
the original C0-C4/A1-A2 experimental results.
