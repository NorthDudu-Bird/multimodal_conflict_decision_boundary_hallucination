#!/usr/bin/env python
"""Run a 20-row smoke test with Qwen2-VL-7B-Instruct."""

from __future__ import annotations

from pathlib import Path

from metadata_paths import BASELINE_PROMPTS_CSV, LEGACY_BASELINE_PROMPTS_CSV, SMOKE_RAW_CSV, resolve_existing_path
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
INPUT_CSV = resolve_existing_path(BASELINE_PROMPTS_CSV, LEGACY_BASELINE_PROMPTS_CSV)
OUTPUT_CSV = SMOKE_RAW_CSV
LOG_PATH = ROOT / "logs" / "qwen2vl_smoke_test.log"
LIMIT = 20
BATCH_SIZE = 1
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.0
USE_4BIT = False


def main() -> int:
    logger = build_logger("qwen2vl_smoke_test", LOG_PATH)
    logger.info("Starting Qwen2-VL smoke test.")
    logger.info("Input CSV: %s", INPUT_CSV)
    logger.info("Output CSV: %s", OUTPUT_CSV)
    logger.info("Model dir: %s", DEFAULT_MODEL_DIR)

    rows = read_rows(INPUT_CSV, limit=LIMIT)
    completed_ids = read_completed_ids(OUTPUT_CSV)
    pending_rows = [row for row in rows if row.get("sample_id") not in completed_ids]
    logger.info("Loaded %s rows, %s already completed, %s pending.", len(rows), len(completed_ids), len(pending_rows))

    if not pending_rows:
        logger.info("Smoke test already completed. Nothing to do.")
        return 0

    runner = Qwen2VLRunner(model_dir=DEFAULT_MODEL_DIR, model_name=DEFAULT_MODEL_NAME, use_4bit=USE_4BIT)
    runner.load(logger=logger)
    logger.info("Final load mode: %s", runner.load_mode)

    success_count = 0
    for batch_index, batch_rows in enumerate(chunked(pending_rows, BATCH_SIZE), start=1):
        sample_ids = [row.get("sample_id", "") for row in batch_rows]
        logger.info("Running batch %s on sample_ids=%s", batch_index, sample_ids)
        try:
            outputs = runner.generate_batch(
                batch_rows,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            for source_row, raw_output in zip(batch_rows, outputs):
                append_result(OUTPUT_CSV, make_result_row(source_row, DEFAULT_MODEL_NAME, raw_output))
                success_count += 1
                logger.info("Completed sample %s", source_row.get("sample_id", ""))
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            logger.exception("Batch %s failed: %s", batch_index, error_text)
            for source_row in batch_rows:
                append_result(
                    OUTPUT_CSV,
                    make_result_row(source_row, DEFAULT_MODEL_NAME, "", status="error", error=error_text),
                )

    logger.info("Smoke test finished. Successful generations written this run: %s", success_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
