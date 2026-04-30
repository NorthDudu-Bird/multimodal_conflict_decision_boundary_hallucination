# Label Mapping Audit

## Current Parser Facts

- Main `C0-C4` parsed rows: 4500; `parse_error=0`, `refusal_or_correction=0`, `other_wrong=0`.
- Main raw outputs used only base labels: black, blue, green, red, white, yellow, plus `other` when applicable. Observed non-base exact outputs in main: none.
- Auxiliary `A1/A2` parsed rows: 1800; all parsed rows also used `exact_single_label` parsing. Observed non-base exact outputs in auxiliary: bright white, dark black, dark blue, dark red, dark yellow, light blue, light red, light yellow, off white.
- Because the main experiment produced only base single-label outputs, the audit sample is drawn from real alias outputs in the auxiliary runs rather than from fabricated main-experiment ambiguities.

## Mapping Table

| term | normalized | parser result | main exact count | aux exact count | risk note |
| --- | --- | --- | --- | --- | --- |
| silver | silver | silver | 0 | 0 | Recognized as a nonstandard standalone label; if emitted under the main prompt it would be parsed as `silver` and counted as non-faithful because the final evaluation set does not use `silver` as a ground-truth class. |
| gray | gray | gray | 0 | 0 | Recognized as a nonstandard standalone label; treated like `silver`, but it did not appear in the strengthened main results. |
| grey | grey | gray | 0 | 0 | Normalized to `gray`; same risk profile as `gray`. |
| white | white | white | 727 | 245 | Exact canonical label with no ambiguity in the current six-color task. |
| off-white | off white | white | 0 | 5 | Family alias that is deterministically collapsed to `white`; used in auxiliary prompts and sampled for audit. |
| beige | beige | other | 0 | 0 | Canonicalized to `other`; would not inflate `conflict_aligned` because it remains outside the six target labels. |
| dark | dark | (no exact alias) | 0 | 0 | No standalone mapping. The parser only uses `dark` when it is part of a known family alias such as `dark red` or `dark blue`. |
| metallic | metallic | (no exact alias) | 0 | 0 | No standalone mapping. A phrase such as `metallic blue` could still match the embedded color token `blue` through mention detection, so this remains a boundary case to document. |
| bluish black | bluish black | (no exact alias) | 0 | 0 | No exact alias. Mention detection would likely recover `black` from the trailing token, so this is documented as a potential composite-phrase edge case. |

## Review Sample

- `results/parser/ambiguous_outputs_sample.csv` contains 27 stratified alias-output checks, with three examples for each of the nine observed alias classes.
- All sampled rows were retained as `pass` under rule-based manual review: the canonical label matched the intended color family semantics of the original string.

## Risk Assessment

- The current main conclusion does not depend on heuristic mention recovery, because the strengthened main experiment never left the base single-label regime.
- Boundary cases such as `metallic blue` or `bluish black` should still be treated cautiously in future work because mention detection could recover only the embedded color token.
- In the present paper, parser-induced inflation of `conflict_aligned` is low risk: the main analysis cells remain unchanged even if all auxiliary-only alias mappings are excluded.
