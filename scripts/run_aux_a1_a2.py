#!/usr/bin/env python
"""Run the A1/A2 auxiliary experiment on the balanced evaluation set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.paper_mainline_utils import (
    load_paper_config,
    model_raw_dir,
    paper_paths,
    run_and_parse_prompt_set,
    run_command,
    selected_model_keys,
)

ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper auxiliary A1/A2 experiment.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_paper_config(args.config)
    paths = paper_paths(config)
    model_keys = selected_model_keys(config, args.models)

    if not args.skip_build:
        run_command([sys.executable, str(ROOT / "scripts" / "build_dataset.py"), "--config", str(config["_config_path"])])

    parsed_paths: list[Path] = []
    for model_key in model_keys:
        output_dir = model_raw_dir(paths["aux_dir"], model_key)
        parsed_paths.append(
            run_and_parse_prompt_set(
                config=config,
                model_key=model_key,
                prompt_csv=paths["aux_prompt_csv"],
                output_dir=output_dir,
                prefix="aux_a1_a2",
                limit=args.limit,
            )
        )

    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_results.py"),
            "--config",
            str(config["_config_path"]),
            "--mode",
            "aux",
            "--output-dir",
            str(paths["aux_dir"]),
            "--input-csvs",
            *[str(path) for path in parsed_paths],
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
