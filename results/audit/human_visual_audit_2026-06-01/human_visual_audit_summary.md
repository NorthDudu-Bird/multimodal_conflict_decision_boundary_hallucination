# A2 Manual Human Visual Audit Summary

## Provenance

- Frozen workbook SHA-256: `d1e9bfc6cfc3a16598558bca5122f3e0a27c195a9e249e83bfd1876dcbf16635`
- Row-level reviewer identifier: `Yutong Liu`
- Protocol metadata fields were not used to claim reviewer independence.
- The frozen human workbook is archived unchanged. Analysis-level adjudications are logged separately.

## Results

- Target conflict-flip rows: `42`; clear: `24/42`; strict visual confound: `9/42`.
- Matched faithful controls: `42`; clear: `29/42`; strict visual confound: `6/42`.
- One matched-control row remained `unsure` for validity-analysis inclusion. It is retained in the aggregate report and is not analyzed as a separate case.

## Analysis-Level Adjudications

- `VC-029` and `VC-069` are duplicate appearances of the same source image. Their reflection field is harmonized to the more conservative human value, `moderate`.

## Wording Constraint

Use `manual human visual audit`, not `independent human visual audit`, unless reviewer independence metadata is separately documented.
