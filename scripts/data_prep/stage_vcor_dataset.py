#!/usr/bin/env python
"""Download or unpack the VCoR dataset and write a basic inventory."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import build_logger, ensure_dirs, load_config, relative_str, repo_path


DEFAULT_DATASET_HANDLE = "landrykezebou/vcor-vehicle-color-recognition-dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the VCoR dataset into repo-managed directories.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--download-mode", choices=["kaggle_api", "local_zip", "skip_download"], default="kaggle_api")
    parser.add_argument("--dataset-handle", default=DEFAULT_DATASET_HANDLE)
    parser.add_argument("--local-zip", type=Path, default=None)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def inventory_fieldnames() -> list[str]:
    return ["source_dataset", "split", "assigned_true_color", "source_path", "file_name", "image_id"]


def run_command(command: list[str], logger) -> None:
    logger.info("Running command: %s", " ".join(str(part) for part in command))
    env = dict(subprocess.os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def ensure_kaggle_download(dataset_handle: str, raw_dir: Path, force_redownload: bool, logger) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    kaggle_exe = shutil.which("kaggle")
    if not kaggle_exe:
        user_scripts = Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle.exe"
        if user_scripts.exists():
            kaggle_exe = str(user_scripts)
    if not kaggle_exe:
        raise FileNotFoundError("Kaggle CLI was not found. Install the `kaggle` package or provide a local VCoR zip.")
    command = [
        kaggle_exe,
        "datasets",
        "download",
        "-d",
        dataset_handle,
        "-p",
        str(raw_dir),
    ]
    if force_redownload:
        command.append("--force")
    else:
        command.append("--quiet")
    try:
        run_command(command, logger)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle API download failed. This environment needs both Kaggle authentication "
            "and working TLS access to kaggle.com/api.kaggle.com. Either provide Kaggle "
            "credentials or place a local VCoR zip file and rerun with "
            "--download-mode local_zip --local-zip <path>."
        ) from exc


def find_zip_files(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.glob("*.zip") if path.is_file())


def unpack_zip(zip_path: Path, raw_dir: Path, logger) -> Path:
    extract_dir = raw_dir / "vcor_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Unpacking %s to %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return extract_dir


def resolve_dataset_root(search_root: Path) -> Path:
    if all((search_root / split).exists() for split in ("train", "val", "test")):
        return search_root
    candidates = []
    for path in search_root.rglob("*"):
        if path.is_dir() and all((path / split).exists() for split in ("train", "val", "test")):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not locate a VCoR root containing train/val/test under {search_root}.")
    candidates.sort(key=lambda item: (len(item.parts), str(item)))
    return candidates[0]


def inventory_rows(dataset_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        split = split_dir.name
        for color_dir in sorted(path for path in split_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            color = color_dir.name.strip().lower()
            for image_path in sorted(path for path in color_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
                rows.append(
                    {
                        "source_dataset": "VCoR",
                        "split": split,
                        "assigned_true_color": color,
                        "source_path": relative_str(image_path),
                        "file_name": image_path.name,
                        "image_id": f"vcor_{split}_{color}_{image_path.stem}",
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    vcor_cfg = config.get("vcor", {}) or {}
    raw_dir = repo_path(vcor_cfg.get("raw_dir", "data_external/vcor_raw"))
    selected_dir = repo_path(vcor_cfg.get("selected_dir", "data_external/vcor_selected"))
    inventory_csv = repo_path(vcor_cfg.get("inventory_csv", "data_external/vcor_selected/vcor_inventory.csv"))
    dataset_root_marker = repo_path(vcor_cfg.get("dataset_root_marker", "data_external/vcor_selected/vcor_dataset_root.json"))
    log_path = args.log_path or repo_path(vcor_cfg.get("log_path", "logs/stage_vcor_dataset.log"))
    ensure_dirs([raw_dir, selected_dir, inventory_csv.parent, dataset_root_marker.parent, log_path.parent])
    logger = build_logger("stage_vcor_dataset", log_path)

    if args.download_mode == "kaggle_api":
        ensure_kaggle_download(args.dataset_handle, raw_dir=raw_dir, force_redownload=args.force_redownload, logger=logger)
        zip_files = find_zip_files(raw_dir)
        if not zip_files:
            raise FileNotFoundError(f"No zip files were found in {raw_dir} after the Kaggle download step.")
        extracted_root = unpack_zip(zip_files[0], raw_dir=raw_dir, logger=logger)
    elif args.download_mode == "local_zip":
        if args.local_zip is None:
            raise ValueError("--local-zip is required when --download-mode local_zip is used.")
        local_zip = repo_path(args.local_zip)
        if not local_zip.exists():
            raise FileNotFoundError(f"Local zip file not found: {local_zip}")
        copied_zip = raw_dir / local_zip.name
        if copied_zip.resolve() != local_zip.resolve():
            shutil.copy2(local_zip, copied_zip)
        extracted_root = unpack_zip(copied_zip, raw_dir=raw_dir, logger=logger)
    else:
        extracted_root = raw_dir

    dataset_root = resolve_dataset_root(extracted_root)
    rows = inventory_rows(dataset_root)
    if not rows:
        raise RuntimeError(f"No VCoR images were discovered under {dataset_root}.")

    with inventory_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=inventory_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "dataset_handle": args.dataset_handle,
        "download_mode": args.download_mode,
        "raw_dir": relative_str(raw_dir),
        "dataset_root": relative_str(dataset_root),
        "inventory_csv": relative_str(inventory_csv),
        "inventory_rows": len(rows),
    }
    dataset_root_marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("VCoR staging complete: %s", json.dumps(payload, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
