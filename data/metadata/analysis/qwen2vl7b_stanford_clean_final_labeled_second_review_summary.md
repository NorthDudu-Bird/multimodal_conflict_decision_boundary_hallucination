# Stanford Cars Final-Labeled Second Review

## Scope
- reviewed images: 30
- reviewed rows: 240
- original/final-labeled source: data/metadata/outputs_labeled/qwen2vl7b_stanford_clean_final_labeled.csv
- image-level review table: data/metadata/analysis/stanford_clean_image_color_second_review.csv
- reviewed/final table: data/metadata/outputs_labeled/qwen2vl7b_stanford_clean_final_labeled_reviewed.csv
- audit table: data/metadata/analysis/qwen2vl7b_stanford_clean_final_labeled_review_audit.csv

## Second-Pass Label Counts
- faithful: 193
- hallucination: 47

## Agreement With Existing Final Labels
- matches: 155
- mismatches: 85
- images with corrected true_color: 13

## Mismatch Transitions
- faithful->hallucination: 2
- hallucination->faithful: 15
- needs_manual_review->faithful: 68

## Images With Corrected True Color
- test_00601
- test_01311
- test_03801
- test_05126
- test_06328
- test_06787
- test_07040
- train_00211
- train_01253
- train_01446
- train_01561
- train_04760
- train_05584

## Images With Any Mismatch
- test_00601
- test_01311
- test_03801
- test_05126
- test_06328
- test_06787
- test_07040
- train_00211
- train_01253
- train_01446
- train_01561
- train_04760
- train_05584
- train_07332
