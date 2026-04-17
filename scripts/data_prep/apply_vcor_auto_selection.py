#!/usr/bin/env python
"""Write auto-screened include/exclude decisions back to the VCoR review CSV."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import build_logger, ensure_dirs, load_config, read_rows, relative_str, repo_path, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply auto-screen recommendations to the VCoR review CSV.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def candidate_fieldnames() -> list[str]:
    return [
        "candidate_id",
        "candidate_rank",
        "image_id",
        "source_dataset",
        "split",
        "assigned_true_color",
        "source_path",
        "staged_path",
        "keep",
        "drop",
        "decision",
        "rejection_reason",
        "reviewer_note",
    ]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    vcor_cfg = config.get("vcor", {}) or {}
    review_csv = repo_path(vcor_cfg.get("candidate_review_csv", "data_external/vcor_selected/candidate_review.csv"))
    screen_csv = repo_path(vcor_cfg.get("screen_csv", "data_external/vcor_selected/candidate_auto_screen.csv"))
    log_path = args.log_path or repo_path(vcor_cfg.get("selection_log_path", "logs/apply_vcor_auto_selection.log"))
    ensure_dirs([review_csv.parent, log_path.parent])
    logger = build_logger("apply_vcor_auto_selection", log_path)

    review_rows = read_rows(review_csv)
    screen_rows = read_rows(screen_csv)
    by_id = {row["candidate_id"]: row for row in screen_rows}
    updated_rows = []
    keep_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()

    for row in review_rows:
        candidate_id = row.get("candidate_id", "")
        screen = by_id.get(candidate_id)
        updated = dict(row)
        if screen is None:
            updated["decision"] = "exclude"
            updated["keep"] = "0"
            updated["drop"] = "1"
            updated["rejection_reason"] = "missing_auto_screen"
            updated["reviewer_note"] = "No auto-screen record found."
            updated_rows.append(updated)
            continue

        color = screen["assigned_true_color"]
        rank = screen["auto_rank_within_color"]
        quality = screen["quality_score"]
        reason = screen["auto_reason"]
        if screen.get("auto_recommend_keep") == "1":
            updated["decision"] = "include"
            updated["keep"] = "1"
            updated["drop"] = "0"
            updated["rejection_reason"] = ""
            updated["reviewer_note"] = f"auto_keep_rank={rank}; quality_score={quality}; {reason}"
            keep_counts[color] += 1
        else:
            updated["decision"] = "exclude"
            updated["keep"] = "0"
            updated["drop"] = "1"
            updated["rejection_reason"] = "not_selected_after_strict_auto_screen"
            updated["reviewer_note"] = f"auto_rank={rank}; quality_score={quality}; {reason}"
            drop_counts[color] += 1
        updated_rows.append(updated)

    write_rows(review_csv, candidate_fieldnames(), updated_rows)
    payload = {
        "review_csv": relative_str(review_csv),
        "keep_counts": dict(keep_counts),
        "drop_counts": dict(drop_counts),
    }
    logger.info("Applied VCoR auto selection: %s", json.dumps(payload, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
