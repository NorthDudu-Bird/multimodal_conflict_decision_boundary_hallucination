#!/usr/bin/env python
"""Run sequential full-precision inference for one model on a prompt table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import ensure_dirs, load_config, repo_path
from scripts.inference.multimodel_vlm_runtime import (
    append_result,
    build_logger,
    create_runner,
    make_result_row,
    model_spec_from_config,
    read_completed_ids,
    read_rows,
    timed_generation,
)


ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential multimodel full-precision inference.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    spec = model_spec_from_config(config=config, model_key=args.model_key)
    inference_cfg = config["inference"]
    logger = build_logger(f"run_multimodel_batch_{args.model_key}", args.log_path)
    offload_dir = repo_path(config["outputs"]["offload_dir"]) / args.model_key
    ensure_dirs([offload_dir, args.output_csv.parent])

    rows = read_rows(args.input_csv, limit=args.limit)
    completed_ids = read_completed_ids(args.output_csv)
    pending_rows = [row for row in rows if row.get("sample_id") not in completed_ids]
    max_new_tokens = int(args.max_new_tokens or spec.max_new_tokens or inference_cfg["max_new_tokens"])
    temperature = float(args.temperature if args.temperature is not None else inference_cfg["temperature"])
    top_p = float(args.top_p if args.top_p is not None else inference_cfg["top_p"])

    logger.info(
        "Starting model_key=%s checkpoint=%s rows=%s pending=%s max_new_tokens=%s temperature=%s top_p=%s",
        spec.model_key,
        spec.checkpoint_name,
        len(rows),
        len(pending_rows),
        max_new_tokens,
        temperature,
        top_p,
    )

    if not pending_rows:
        logger.info("No pending rows left for %s.", args.model_key)
        return 0

    runner = create_runner(spec=spec, offload_dir=offload_dir)
    try:
        runner.load(logger=logger)
        metadata_json = args.metadata_json or args.output_csv.parent / "run_metadata.json"
        metadata_payload = {
            **runner.metadata(),
            "input_csv": str(args.input_csv),
            "output_csv": str(args.output_csv),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "limit": args.limit,
        }
        metadata_json.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        success_count = 0
        for row in pending_rows:
            sample_id = row.get("sample_id", "")
            logger.info("Running sample_id=%s", sample_id)
            try:
                raw_output, elapsed = timed_generation(
                    runner=runner,
                    row=row,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                append_result(
                    args.output_csv,
                    make_result_row(source_row=row, runner=runner, raw_output=raw_output, elapsed_seconds=elapsed),
                )
                success_count += 1
                logger.info("Completed sample_id=%s in %.2fs", sample_id, elapsed)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                logger.exception("Inference failed for sample_id=%s: %s", sample_id, error_text)
                append_result(
                    args.output_csv,
                    make_result_row(
                        source_row=row,
                        runner=runner,
                        raw_output="",
                        elapsed_seconds=0.0,
                        status="error",
                        error=error_text,
                    ),
                )

        logger.info("Finished model_key=%s with %s successful generations.", args.model_key, success_count)
        return 0
    finally:
        runner.unload()


if __name__ == "__main__":
    raise SystemExit(main())
