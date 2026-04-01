#!/usr/bin/env python
"""Run the full Qwen2-VL batch experiment with resume support."""

from __future__ import annotations

import argparse
from pathlib import Path

from metadata_paths import BASELINE_PROMPTS_CSV, BASELINE_RUNTIME_CSV, LEGACY_BASELINE_PROMPTS_CSV, resolve_existing_path
from qwen2vl_runtime import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_NAME,
    Qwen2VLRunner,
    append_result,
    build_logger,
    chunked,
    make_result_row,
    read_completed_ids,
    read_rows,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = resolve_existing_path(BASELINE_PROMPTS_CSV, LEGACY_BASELINE_PROMPTS_CSV)
DEFAULT_OUTPUT_CSV = BASELINE_RUNTIME_CSV
DEFAULT_LOG_PATH = ROOT / "logs" / "qwen2vl_batch.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2-VL full batch inference with resume support.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=1, help="Increase cautiously; larger batches use much more VRAM.")
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit loading. Disabled by default because the local 4-bit path produced severe visual mismatches in the main experiment.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Use >=128 for the existence-conflict main run to avoid truncating descriptive S0 responses.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Set >0 for sampling. Keep 0.0 for deterministic baseline runs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for debugging a subset before the full 200 rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = build_logger("qwen2vl_batch", args.log_path)
    logger.info("Starting Qwen2-VL batch run.")
    logger.info("Input CSV: %s", args.input_csv)
    logger.info("Output CSV: %s", args.output_csv)
    logger.info("Model dir: %s", args.model_dir)
    logger.info(
        "batch_size=%s | max_new_tokens=%s | temperature=%s | limit=%s | use_4bit=%s",
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
        args.limit,
        args.use_4bit,
    )

    rows = read_rows(args.input_csv, limit=args.limit)
    completed_ids = read_completed_ids(args.output_csv)
    pending_rows = [row for row in rows if row.get("sample_id") not in completed_ids]
    logger.info("Loaded %s rows, %s already completed, %s pending.", len(rows), len(completed_ids), len(pending_rows))

    if not pending_rows:
        logger.info("No pending rows. Batch run already complete.")
        return 0

    runner = Qwen2VLRunner(model_dir=args.model_dir, model_name=args.model_name, use_4bit=args.use_4bit)
    runner.load(logger=logger)
    logger.info("Final load mode: %s", runner.load_mode)

    processed = 0
    for batch_index, batch_rows in enumerate(chunked(pending_rows, args.batch_size), start=1):
        sample_ids = [row.get("sample_id", "") for row in batch_rows]
        logger.info("Running batch %s on sample_ids=%s", batch_index, sample_ids)
        try:
            outputs = runner.generate_batch(
                batch_rows,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            for source_row, raw_output in zip(batch_rows, outputs):
                append_result(args.output_csv, make_result_row(source_row, args.model_name, raw_output))
                processed += 1
                logger.info("Completed sample %s", source_row.get("sample_id", ""))
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            logger.exception("Batch %s failed: %s", batch_index, error_text)
            for source_row in batch_rows:
                append_result(
                    args.output_csv,
                    make_result_row(source_row, args.model_name, "", status="error", error=error_text),
                )

    logger.info("Batch run finished. Successful generations written this run: %s", processed)
    logger.info("To resume later, rerun the same command; completed sample_id rows are skipped automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
