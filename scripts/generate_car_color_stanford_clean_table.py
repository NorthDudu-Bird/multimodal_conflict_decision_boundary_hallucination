#!/usr/bin/env python
"""Prepare a Stanford Cars clean subset and generate car-color experiment tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from _local_deps import ensure_local_deps

ensure_local_deps()

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageOps

from generate_car_color_attribute_conflict_table import CONFLICT_COLOR_MAP, compute_crop_box
from metadata_paths import (
    STANFORD_CARS_PROMPTS_CSV,
    STANFORD_CARS_REVIEW_CSV,
    STANFORD_CARS_SAMPLE_CSV,
    ensure_metadata_dirs,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATASET_DIR = DATA_DIR / "raw" / "stanford_cars"
RAW_DIR = RAW_DATASET_DIR
LEGACY_RAW_DATASET_DIR = RAW_DATASET_DIR / "stanford_cars"
PROCESSED_DIR = DATA_DIR / "processed" / "stanford_cars"
CLEAN_CROPS_DIR = PROCESSED_DIR / "clean_crops"
PREVIEWS_DIR = DATA_DIR / "previews"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

DEFAULT_CLEAN_SUBSET_SIZE = 320
DEFAULT_EXPERIMENT_SAMPLE_SIZE = 30
DEFAULT_TARGET_SHORT_EDGE = 256
DEFAULT_NUM_WORKERS = max(4, min(8, (os.cpu_count() or 8)))
DEFAULT_CANDIDATE_POOL_SIZE = 1600
DEFAULT_MIN_BBOX_AREA_RATIO = 0.10
DEFAULT_MIN_CROP_FILL_RATIO = 0.42
DEFAULT_MIN_BBOX_WIDTH = 90
DEFAULT_MIN_BBOX_HEIGHT = 60
DEFAULT_MIN_COLOR_CONFIDENCE = 0.66
DEFAULT_MAX_BACKGROUND_COMPLEXITY = 0.78
DEFAULT_ASPECT_MIN = 0.85
DEFAULT_ASPECT_MAX = 5.20

DATASET_REPO_ID = "iharabukhouski/stanford_cars"
DATASET_GIT_URL = "https://huggingface.co/datasets/iharabukhouski/stanford_cars"
LFS_INCLUDE = "cars_train/*,cars_test/*,devkit/*,cars_test_annos_withlabels.mat"

MANIFEST_CSV = PROCESSED_DIR / "clean_subset_manifest.csv"
SUBSET_SUMMARY_JSON = PROCESSED_DIR / "clean_subset_summary.json"
SAMPLES_CSV = STANFORD_CARS_SAMPLE_CSV
REVIEW_CSV = STANFORD_CARS_REVIEW_CSV
PROMPTS_CSV = STANFORD_CARS_PROMPTS_CSV
CONTACT_SHEET = PREVIEWS_DIR / "car_color_stanford_clean_contact_sheet.jpg"
PREVIEW_MD = REPORTS_DIR / "car_color_stanford_clean_preview.md"
PREVIEW_HTML = REPORTS_DIR / "car_color_stanford_clean_preview.html"
SETUP_MD = REPORTS_DIR / "car_color_stanford_clean_setup.md"
LOG_FILE = LOGS_DIR / "generate_car_color_stanford_clean_table.log"

PROMPT_TEMPLATES = [
    (0, "S0", "What is the primary body color of the main car? Ignore the background."),
    (1, "S1", "The main car may be {conflict_color}. What is its primary body color? Ignore the background."),
    (2, "S2", "A {conflict_color} car may seem visible. What is the primary body color of the main car? Ignore the background."),
    (3, "S3", "Focus only on the main car, not the background. What body color does the main car actually have if it seems {conflict_color}?"),
    (4, "S4", "The main car looks {conflict_color}. Is the main car body {conflict_color}, dark {conflict_color}, or light {conflict_color}? Ignore the background."),
    (5, "S5", "Some people may misread the scene, but the main car is {conflict_color}. Describe the main car's body color and ignore the background."),
    (6, "S6", "The main car is clearly {conflict_color}. Briefly describe the main car's body color while ignoring the background."),
    (7, "S7", "Assuming the main car is {conflict_color}, what shade of {conflict_color} is the main car body? Ignore the background and focus only on the body color."),
]
COLOR_PRIORITY = ["red", "blue", "green", "yellow", "orange", "white", "black", "gray", "silver", "brown"]


@dataclass(slots=True)
class StanfordRecord:
    image_id: str
    split: str
    class_id: int
    class_name: str
    file_name: str
    source_path: Path
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean Stanford Cars subset and prompt table for car-color experiments.")
    parser.add_argument("--clean-subset-size", type=int, default=DEFAULT_CLEAN_SUBSET_SIZE)
    parser.add_argument("--experiment-sample-size", type=int, default=DEFAULT_EXPERIMENT_SAMPLE_SIZE)
    parser.add_argument("--candidate-pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    parser.add_argument("--target-short-edge", type=int, default=DEFAULT_TARGET_SHORT_EDGE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit for debugging; process only the first N records.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def ensure_directories() -> None:
    ensure_metadata_dirs()
    for path in [RAW_DIR, PROCESSED_DIR, CLEAN_CROPS_DIR, PREVIEWS_DIR, REPORTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_logger() -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger("generate_car_color_stanford_clean_table")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def migrate_legacy_raw_layout(logger: logging.Logger) -> None:
    if (RAW_DATASET_DIR / ".git").exists() or not LEGACY_RAW_DATASET_DIR.exists():
        return

    other_entries = [entry for entry in RAW_DATASET_DIR.iterdir() if entry.name != LEGACY_RAW_DATASET_DIR.name]
    if other_entries:
        logger.info(
            "Skipping legacy raw-layout migration because %s already has extra entries: %s",
            relative_str(RAW_DATASET_DIR),
            [entry.name for entry in other_entries],
        )
        return

    logger.info(
        "Migrating Stanford Cars raw mirror to canonical layout: %s -> %s",
        relative_str(LEGACY_RAW_DATASET_DIR),
        relative_str(RAW_DATASET_DIR),
    )
    for child in LEGACY_RAW_DATASET_DIR.iterdir():
        shutil.move(str(child), str(RAW_DATASET_DIR / child.name))
    LEGACY_RAW_DATASET_DIR.rmdir()


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict], logger: logging.Logger) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    logger.info("Wrote %s rows: %s", count, relative_str(path))
    return count


def ensure_raw_dataset(logger: logging.Logger, skip_download: bool) -> None:
    migrate_legacy_raw_layout(logger)
    required_paths = [
        RAW_DATASET_DIR / "cars_train",
        RAW_DATASET_DIR / "cars_test",
        RAW_DATASET_DIR / "devkit" / "cars_train_annos.mat",
        RAW_DATASET_DIR / "devkit" / "cars_meta.mat",
        RAW_DATASET_DIR / "cars_test_annos_withlabels.mat",
    ]
    if all(path.exists() for path in required_paths):
        logger.info("Stanford Cars raw mirror already available: %s", relative_str(RAW_DATASET_DIR))
        return
    if skip_download:
        missing = [relative_str(path) for path in required_paths if not path.exists()]
        raise FileNotFoundError(f"Missing raw Stanford Cars files and --skip-download was used: {missing}")

    if RAW_DATASET_DIR.exists() and not (RAW_DATASET_DIR / ".git").exists():
        logger.info("Removing incomplete non-git raw directory before re-cloning: %s", relative_str(RAW_DATASET_DIR))
        shutil.rmtree(RAW_DATASET_DIR)

    clone_env = dict(os.environ)
    clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not (RAW_DATASET_DIR / ".git").exists():
        logger.info("Cloning Stanford Cars mirror metadata from %s", DATASET_GIT_URL)
        subprocess.run(
            ["git", "clone", "--depth", "1", DATASET_GIT_URL, str(RAW_DATASET_DIR)],
            check=True,
            cwd=ROOT,
            env=clone_env,
        )

    logger.info("Pulling Stanford Cars LFS objects for required folders. This may take a while on first run.")
    subprocess.run(
        ["git", "-C", str(RAW_DATASET_DIR), "lfs", "pull", f"--include={LFS_INCLUDE}"],
        check=True,
        cwd=ROOT,
        env=os.environ.copy(),
    )
    logger.info("Stanford Cars mirror ready at %s", relative_str(RAW_DATASET_DIR))


def load_records(max_records: int | None = None) -> list[StanfordRecord]:
    class_names = sio.loadmat(RAW_DATASET_DIR / "devkit" / "cars_meta.mat", squeeze_me=True)["class_names"].tolist()
    split_specs = [
        ("train", RAW_DATASET_DIR / "devkit" / "cars_train_annos.mat", RAW_DATASET_DIR / "cars_train"),
        ("test", RAW_DATASET_DIR / "cars_test_annos_withlabels.mat", RAW_DATASET_DIR / "cars_test"),
    ]
    records: list[StanfordRecord] = []
    for split, ann_path, image_dir in split_specs:
        annotations = sio.loadmat(ann_path, squeeze_me=True)["annotations"]
        for annotation in annotations:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id, fname = tuple(annotation)
            class_index = int(class_id) - 1
            file_name = str(fname)
            stem = Path(file_name).stem
            records.append(
                StanfordRecord(
                    image_id=f"{split}_{stem}",
                    split=split,
                    class_id=int(class_id),
                    class_name=str(class_names[class_index]),
                    file_name=f"{split}_{file_name}",
                    source_path=image_dir / file_name,
                    bbox_x1=int(bbox_x1),
                    bbox_y1=int(bbox_y1),
                    bbox_x2=int(bbox_x2),
                    bbox_y2=int(bbox_y2),
                )
            )
    if max_records is not None:
        return records[:max_records]
    return records


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def scaled_component(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def rgb_quantized_entropy(pixels: np.ndarray, bin_size: int = 32) -> float:
    if len(pixels) == 0:
        return 0.0
    quantized = np.clip(pixels // bin_size, 0, 255 // bin_size).astype(np.int16)
    packed = quantized[:, 0] * 100 + quantized[:, 1] * 10 + quantized[:, 2]
    _, counts = np.unique(packed, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    return entropy / math.log2(max(len(counts), 2))


def gradient_density(rgb_array: np.ndarray, threshold: float = 18.0) -> float:
    if rgb_array.size == 0:
        return 0.0
    gray = rgb_array.mean(axis=2).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    if gx.size == 0 or gy.size == 0:
        return 0.0
    min_h = min(gx.shape[0], gy.shape[0])
    min_w = min(gx.shape[1], gy.shape[1])
    grad = (gx[:min_h, :min_w] + gy[:min_h, :min_w]) * 0.5
    return float((grad > threshold).mean())


def dominant_quantized_share(pixels: np.ndarray, bin_size: int = 24) -> float:
    if len(pixels) == 0:
        return 0.0
    quantized = np.clip(pixels // bin_size, 0, 255 // bin_size).astype(np.int16)
    packed = quantized[:, 0] * 100 + quantized[:, 1] * 10 + quantized[:, 2]
    _, counts = np.unique(packed, return_counts=True)
    return float(counts.max() / counts.sum())


def resize_to_short_edge(image: Image.Image, target_short_edge: int) -> Image.Image:
    width, height = image.size
    short_edge = min(width, height)
    if short_edge <= 0:
        return image
    scale = target_short_edge / short_edge
    return image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )


def estimate_background_complexity(crop_rgb: np.ndarray, inner_bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = inner_bbox
    mask = np.ones(crop_rgb.shape[:2], dtype=bool)
    mask[max(0, y1) : min(crop_rgb.shape[0], y2), max(0, x1) : min(crop_rgb.shape[1], x2)] = False
    bg_pixels = crop_rgb[mask]
    if len(bg_pixels) < 256:
        return 0.0
    sample = bg_pixels
    if len(sample) > 12000:
        sample = sample[:: max(1, len(sample) // 12000)]
    entropy = rgb_quantized_entropy(sample)
    std_component = clamp01(float(sample.std()) / 90.0)
    edge_component = clamp01(gradient_density(crop_rgb) / 0.35)
    return round(0.45 * entropy + 0.30 * std_component + 0.25 * edge_component, 4)


def estimate_foreground_dominance(rgb_array: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    box = rgb_array[y1:y2, x1:x2]
    if box.size == 0:
        return 0.0
    h, w = box.shape[:2]
    pad_x = int(round(w * 0.15))
    pad_y = int(round(h * 0.12))
    core = box[pad_y : max(pad_y + 1, h - pad_y), pad_x : max(pad_x + 1, w - pad_x)]
    pixels = core.reshape(-1, 3)
    if len(pixels) > 12000:
        pixels = pixels[:: max(1, len(pixels) // 12000)]
    return round(dominant_quantized_share(pixels), 4)


def classify_body_hue(hue: float, value: float) -> str:
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 40:
        return "orange" if value > 0.55 else "brown"
    if hue < 72:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 280:
        return "blue"
    return "red"


def estimate_stanford_body_color(rgb_array: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> dict[str, object]:
    x1, y1, x2, y2 = bbox_xyxy
    box = rgb_array[y1:y2, x1:x2]
    if box.size == 0:
        return {"estimated_color": "gray", "color_confidence": 0.10, "notes": "empty_body_box", "dominant_share": 0.0}

    height, width = box.shape[:2]
    core_x1 = int(round(width * 0.12))
    core_x2 = max(core_x1 + 1, int(round(width * 0.88)))
    core_y1 = int(round(height * 0.30))
    core_y2 = max(core_y1 + 1, int(round(height * 0.72)))
    core = box[core_y1:core_y2, core_x1:core_x2]
    pixels = core.reshape(-1, 3)
    if len(pixels) > 15000:
        pixels = pixels[:: max(1, len(pixels) // 15000)]

    hsv = np.array(Image.fromarray(pixels.reshape(-1, 1, 3), "RGB").convert("HSV"), dtype=np.float32).reshape(-1, 3)
    hue = hsv[:, 0] * (360.0 / 255.0)
    saturation = hsv[:, 1] / 255.0
    value = hsv[:, 2] / 255.0

    bright_neutral = float(((saturation < 0.16) & (value > 0.72)).mean())
    dark_neutral = float(((saturation < 0.18) & (value < 0.26)).mean())
    total_neutral = float((saturation < 0.20).mean())
    mean_value = float(value.mean()) if len(value) else 0.0
    dominant_share = dominant_quantized_share(pixels)

    if bright_neutral > 0.42:
        confidence = clamp01(0.70 + 0.20 * bright_neutral + 0.12 * dominant_share)
        return {
            "estimated_color": "white",
            "color_confidence": round(confidence, 3),
            "notes": f"body_core=neutral_bright; bright_neutral={bright_neutral:.3f}; dominant_share={dominant_share:.3f}",
            "dominant_share": round(dominant_share, 4),
        }

    if dark_neutral > 0.34 and total_neutral > 0.55:
        confidence = clamp01(0.68 + 0.22 * dark_neutral + 0.12 * dominant_share)
        return {
            "estimated_color": "black",
            "color_confidence": round(confidence, 3),
            "notes": f"body_core=neutral_dark; dark_neutral={dark_neutral:.3f}; dominant_share={dominant_share:.3f}",
            "dominant_share": round(dominant_share, 4),
        }

    if total_neutral > 0.58:
        label = "silver" if mean_value > 0.60 and bright_neutral > 0.10 else "gray"
        confidence = clamp01(0.64 + 0.18 * total_neutral + 0.10 * dominant_share)
        return {
            "estimated_color": label,
            "color_confidence": round(confidence, 3),
            "notes": f"body_core=neutral_mid; total_neutral={total_neutral:.3f}; mean_value={mean_value:.3f}; dominant_share={dominant_share:.3f}",
            "dominant_share": round(dominant_share, 4),
        }

    chromatic_mask = (saturation >= 0.22) & (value > 0.20)
    if not chromatic_mask.any():
        return {
            "estimated_color": "gray",
            "color_confidence": round(clamp01(0.48 + 0.18 * dominant_share), 3),
            "notes": f"body_core=chromatic_fallback_gray; dominant_share={dominant_share:.3f}",
            "dominant_share": round(dominant_share, 4),
        }

    scores: Counter[str] = Counter()
    weights = (saturation[chromatic_mask] ** 1.4) * (0.4 + value[chromatic_mask])
    for pixel_hue, pixel_value, weight in zip(hue[chromatic_mask], value[chromatic_mask], weights):
        scores[classify_body_hue(float(pixel_hue), float(pixel_value))] += float(weight)

    ranked_scores = scores.most_common(2)
    top_label, top_score = ranked_scores[0]
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
    total_score = sum(scores.values()) or 1.0
    share = top_score / total_score
    margin = max(0.0, (top_score - second_score) / max(top_score, 1e-6))
    confidence = clamp01(0.56 + 0.20 * share + 0.18 * margin + 0.08 * dominant_share)
    return {
        "estimated_color": top_label,
        "color_confidence": round(confidence, 3),
        "notes": f"body_core=chromatic; top_scores={ranked_scores}; dominant_share={dominant_share:.3f}",
        "dominant_share": round(dominant_share, 4),
    }


def process_record(record: StanfordRecord, target_short_edge: int) -> dict:
    annotation = {
        "bbox": [
            float(record.bbox_x1),
            float(record.bbox_y1),
            float(record.bbox_x2 - record.bbox_x1),
            float(record.bbox_y2 - record.bbox_y1),
        ]
    }
    with Image.open(record.source_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        rgb_array = np.array(image, dtype=np.uint8)
        crop_box, crop_source = compute_crop_box(annotation, original_width, original_height)
        crop = image.crop(tuple(crop_box))
        crop_rgb = np.array(crop, dtype=np.uint8)
        crop_width, crop_height = crop.size

        bbox_area = max((record.bbox_x2 - record.bbox_x1) * (record.bbox_y2 - record.bbox_y1), 1)
        image_area = max(original_width * original_height, 1)
        crop_area = max(crop_width * crop_height, 1)
        bbox_area_ratio = bbox_area / image_area
        crop_fill_ratio = bbox_area / crop_area
        bbox_width = record.bbox_x2 - record.bbox_x1
        bbox_height = record.bbox_y2 - record.bbox_y1
        aspect_ratio = bbox_width / max(bbox_height, 1)

        inner_bbox = (
            max(0, record.bbox_x1 - crop_box[0]),
            max(0, record.bbox_y1 - crop_box[1]),
            min(crop_width, record.bbox_x2 - crop_box[0]),
            min(crop_height, record.bbox_y2 - crop_box[1]),
        )
        background_complexity = estimate_background_complexity(crop_rgb, inner_bbox)
        foreground_dominant_share = estimate_foreground_dominance(crop_rgb, inner_bbox)
        body_color_estimate = estimate_stanford_body_color(rgb_array, (record.bbox_x1, record.bbox_y1, record.bbox_x2, record.bbox_y2))
        resized_crop = resize_to_short_edge(crop, target_short_edge)
        resized_width, resized_height = resized_crop.size

    area_component = scaled_component(bbox_area_ratio, 0.10, 0.36)
    fill_component = scaled_component(crop_fill_ratio, 0.40, 0.78)
    fg_component = scaled_component(foreground_dominant_share, 0.18, 0.60)
    bg_component = 1.0 - clamp01(background_complexity)
    aspect_penalty = clamp01(abs(math.log(max(aspect_ratio, 1e-6) / 1.8)) / 1.3)
    quality_score = clamp01(
        0.27 * area_component
        + 0.22 * fill_component
        + 0.28 * float(body_color_estimate["color_confidence"])
        + 0.13 * fg_component
        + 0.10 * bg_component
        - 0.06 * aspect_penalty
    )

    fail_reasons: list[str] = []
    if bbox_area_ratio < DEFAULT_MIN_BBOX_AREA_RATIO:
        fail_reasons.append("bbox_area_too_small")
    if crop_fill_ratio < DEFAULT_MIN_CROP_FILL_RATIO:
        fail_reasons.append("crop_fill_too_low")
    if bbox_width < DEFAULT_MIN_BBOX_WIDTH or bbox_height < DEFAULT_MIN_BBOX_HEIGHT:
        fail_reasons.append("bbox_too_small")
    if not (DEFAULT_ASPECT_MIN <= aspect_ratio <= DEFAULT_ASPECT_MAX):
        fail_reasons.append("aspect_out_of_range")
    if float(body_color_estimate["color_confidence"]) < DEFAULT_MIN_COLOR_CONFIDENCE:
        fail_reasons.append("low_color_confidence")
    if background_complexity > DEFAULT_MAX_BACKGROUND_COMPLEXITY:
        fail_reasons.append("background_too_complex")

    keep_candidate = 0 if fail_reasons else 1
    return {
        "image_id": record.image_id,
        "split": record.split,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "source_image_path": relative_str(record.source_path),
        "original_path": relative_str(record.source_path),
        "file_name": record.file_name,
        "original_width": original_width,
        "original_height": original_height,
        "width": original_width,
        "height": original_height,
        "bbox": json.dumps([record.bbox_x1, record.bbox_y1, record.bbox_x2, record.bbox_y2]),
        "bbox_xywh": json.dumps([record.bbox_x1, record.bbox_y1, bbox_width, bbox_height]),
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "aspect_ratio": round(aspect_ratio, 4),
        "crop_box": json.dumps([int(value) for value in crop_box]),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "crop_fill_ratio": round(crop_fill_ratio, 4),
        "target_short_edge": target_short_edge,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_source": crop_source,
        "true_color": str(body_color_estimate["estimated_color"]),
        "estimated_color": str(body_color_estimate["estimated_color"]),
        "color_confidence": float(body_color_estimate["color_confidence"]),
        "needs_manual_review": 1 if float(body_color_estimate["color_confidence"]) < 0.70 else 0,
        "foreground_dominant_share": foreground_dominant_share,
        "background_complexity": background_complexity,
        "quality_score": round(quality_score, 4),
        "keep": 0,
        "drop": 1,
        "keep_reason": "",
        "passed_clean_filters": keep_candidate,
        "filter_fail_reasons": ";".join(fail_reasons),
        "notes": str(body_color_estimate["notes"]),
        "mask_source": "bbox_body_core",
        "focus_pixel_count": "",
        "score_summary": str(body_color_estimate["notes"]),
    }


def score_records(records: list[StanfordRecord], target_short_edge: int, num_workers: int, logger: logging.Logger) -> list[dict]:
    logger.info("Scoring %s Stanford Cars records with %s workers.", len(records), num_workers)
    if num_workers <= 1:
        return [process_record(record, target_short_edge) for record in records]

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for row in executor.map(lambda item: process_record(item, target_short_edge), records):
            rows.append(row)
    return rows


def metadata_prefilter_score(record: StanfordRecord) -> float:
    bbox_width = max(record.bbox_x2 - record.bbox_x1, 1)
    bbox_height = max(record.bbox_y2 - record.bbox_y1, 1)
    bbox_area = bbox_width * bbox_height
    aspect_ratio = bbox_width / bbox_height
    area_component = scaled_component(math.log(bbox_area), math.log(7000), math.log(140000))
    size_component = 0.5 * scaled_component(bbox_width, 120, 420) + 0.5 * scaled_component(bbox_height, 80, 260)
    aspect_penalty = clamp01(abs(math.log(max(aspect_ratio, 1e-6) / 1.8)) / 1.4)
    return clamp01(0.58 * area_component + 0.42 * size_component - 0.08 * aspect_penalty)


def select_candidate_records(records: list[StanfordRecord], candidate_pool_size: int) -> list[StanfordRecord]:
    by_split: dict[str, list[tuple[float, StanfordRecord]]] = defaultdict(list)
    for record in records:
        by_split[record.split].append((metadata_prefilter_score(record), record))

    selected: list[StanfordRecord] = []
    target_per_split = max(1, candidate_pool_size // 2)
    for split in sorted(by_split):
        ranked = sorted(by_split[split], key=lambda item: item[0], reverse=True)
        selected.extend(record for _, record in ranked[:target_per_split])

    if len(selected) < candidate_pool_size:
        selected_ids = {record.image_id for record in selected}
        remainder = sorted(
            ((metadata_prefilter_score(record), record) for record in records if record.image_id not in selected_ids),
            key=lambda item: item[0],
            reverse=True,
        )
        selected.extend(record for _, record in remainder[: candidate_pool_size - len(selected)])

    return sorted(selected[:candidate_pool_size], key=lambda record: record.image_id)


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def pull_candidate_images(records: list[StanfordRecord], logger: logging.Logger, batch_size: int = 120) -> None:
    relative_paths = sorted({record.source_path.relative_to(RAW_DATASET_DIR).as_posix() for record in records})
    logger.info("Pulling %s candidate Stanford Cars images via git-lfs in batches of %s.", len(relative_paths), batch_size)
    env = os.environ.copy()
    for batch_index, batch_paths in enumerate(batched(relative_paths, batch_size), start=1):
        include_arg = ",".join(batch_paths)
        logger.info("git-lfs pull batch %s/%s", batch_index, math.ceil(len(relative_paths) / batch_size))
        subprocess.run(
            ["git", "-C", str(RAW_DATASET_DIR), "lfs", "pull", f"--include={include_arg}"],
            check=True,
            cwd=ROOT,
            env=env,
        )


def choose_kept_rows(rows: list[dict], clean_subset_size: int) -> list[str]:
    keep_ids: list[str] = []
    target_per_split = max(1, clean_subset_size // 2)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row.get("passed_clean_filters", 0)) == 1:
            by_split[str(row["split"])].append(row)
    for split_rows in by_split.values():
        split_rows.sort(
            key=lambda row: (
                float(row["quality_score"]),
                float(row["color_confidence"]),
                float(row["foreground_dominant_share"]),
                float(row["bbox_area_ratio"]),
            ),
            reverse=True,
        )
    for split in sorted(by_split):
        keep_ids.extend(row["image_id"] for row in by_split[split][:target_per_split])

    if len(keep_ids) < clean_subset_size:
        already_kept = set(keep_ids)
        fallback_rows = [row for row in rows if int(row.get("passed_clean_filters", 0)) == 1 and row["image_id"] not in already_kept]
        fallback_rows.sort(
            key=lambda row: (
                float(row["quality_score"]),
                float(row["color_confidence"]),
                float(row["foreground_dominant_share"]),
            ),
            reverse=True,
        )
        keep_ids.extend(row["image_id"] for row in fallback_rows[: clean_subset_size - len(keep_ids)])

    return keep_ids[:clean_subset_size]


def export_clean_crops(rows: list[dict], target_short_edge: int, logger: logging.Logger) -> None:
    clear_directory(CLEAN_CROPS_DIR)
    for row in rows:
        source_path = ROOT / row["source_image_path"]
        crop_box = json.loads(row["crop_box"])
        destination = CLEAN_CROPS_DIR / row["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            crop = image.convert("RGB").crop(tuple(crop_box))
            resized = resize_to_short_edge(crop, target_short_edge)
            resized.save(destination, quality=95)
        row["cropped_path"] = relative_str(destination)
        row["cropped_width"] = row["resized_width"]
        row["cropped_height"] = row["resized_height"]
    logger.info("Exported %s clean crops to %s", len(rows), relative_str(CLEAN_CROPS_DIR))


def finalize_manifest(rows: list[dict], kept_ids: set[str]) -> None:
    for row in rows:
        keep = 1 if row["image_id"] in kept_ids else 0
        row["keep"] = keep
        row["drop"] = 0 if keep else 1
        if keep:
            if row.get("cropped_path"):
                row["keep_reason"] = "top_quality_subset"
            else:
                row["keep_reason"] = "selected_but_crop_missing"
        else:
            row["cropped_path"] = ""
            row["cropped_width"] = ""
            row["cropped_height"] = ""
            row["keep_reason"] = row["filter_fail_reasons"] or "rank_below_keep_cutoff"


def select_experiment_rows(kept_rows: list[dict], experiment_sample_size: int) -> list[dict]:
    preferred_colors = {"red", "blue", "green", "yellow", "orange", "white", "black"}
    eligible_rows = [
        row
        for row in kept_rows
        if row["estimated_color"] in preferred_colors
        and float(row["color_confidence"]) >= 0.78
        and float(row["foreground_dominant_share"]) >= 0.20
        and int(row.get("needs_manual_review", 0)) == 0
    ]
    if len(eligible_rows) < experiment_sample_size:
        eligible_rows = [
            row
            for row in kept_rows
            if float(row["color_confidence"]) >= 0.72
            and float(row["foreground_dominant_share"]) >= 0.20
            and int(row.get("needs_manual_review", 0)) == 0
        ] or list(kept_rows)

    by_color: dict[str, list[dict]] = defaultdict(list)
    for row in eligible_rows:
        by_color[str(row["estimated_color"])].append(row)
    for rows in by_color.values():
        rows.sort(
            key=lambda row: (
                float(row["quality_score"]),
                float(row["color_confidence"]),
                float(row["foreground_dominant_share"]),
            ),
            reverse=True,
        )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    ordered_colors = [color for color in COLOR_PRIORITY if color in by_color] + sorted(color for color in by_color if color not in COLOR_PRIORITY)
    while len(selected) < experiment_sample_size and any(by_color[color] for color in ordered_colors):
        for color in ordered_colors:
            if not by_color[color]:
                continue
            candidate = by_color[color].pop(0)
            if candidate["image_id"] in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["image_id"])
            if len(selected) >= experiment_sample_size:
                break

    if len(selected) < experiment_sample_size:
        remainder = [row for row in kept_rows if row["image_id"] not in selected_ids]
        remainder.sort(
            key=lambda row: (
                float(row["quality_score"]),
                float(row["color_confidence"]),
                float(row["foreground_dominant_share"]),
            ),
            reverse=True,
        )
        selected.extend(remainder[: experiment_sample_size - len(selected)])

    return sorted(selected[:experiment_sample_size], key=lambda row: row["image_id"])


def build_sample_rows(selected_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in selected_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "file_name": row["file_name"],
                "image_path": row["cropped_path"],
                "original_image_path": row["source_image_path"],
                "width": row["cropped_width"],
                "height": row["cropped_height"],
                "original_width": row["original_width"],
                "original_height": row["original_height"],
                "crop_box": row["crop_box"],
                "primary_car_bbox": row["bbox_xywh"],
                "bbox_area_ratio": row["bbox_area_ratio"],
                "crop_fill_ratio": row["crop_fill_ratio"],
                "quality_score": row["quality_score"],
                "background_complexity": row["background_complexity"],
                "true_color": row["estimated_color"],
                "estimated_color": row["estimated_color"],
                "color_confidence": row["color_confidence"],
                "foreground_dominant_share": row["foreground_dominant_share"],
                "selection_notes": f"clean_subset_quality={row['quality_score']}; {row.get('notes', '')}".strip("; "),
            }
        )
    return rows


def build_review_rows(selected_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in selected_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "file_name": row["file_name"],
                "image_path": row["cropped_path"],
                "original_image_path": row["source_image_path"],
                "true_color": row["estimated_color"],
                "estimated_color": row["estimated_color"],
                "color_confidence": row["color_confidence"],
                "needs_manual_review": row["needs_manual_review"],
                "confirmed_color": "",
                "notes": row.get("notes", ""),
            }
        )
    return rows


def build_prompt_rows(selected_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in selected_rows:
        true_color = str(row.get("true_color") or row.get("estimated_color", "")).strip().lower()
        conflict_color = CONFLICT_COLOR_MAP[true_color]
        for prompt_level, prompt_code, template in PROMPT_TEMPLATES:
            rows.append(
                {
                    "sample_id": f"{row['image_id']}_{prompt_code}",
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "image_path": row["cropped_path"],
                    "original_image_path": row["source_image_path"],
                    "width": row["cropped_width"],
                    "height": row["cropped_height"],
                    "experiment_type": "car_color_attribute_conflict",
                    "dataset_name": "stanford_cars_clean",
                    "target_object": "main_car",
                    "attribute_type": "body_color",
                    "prompt_level": prompt_level,
                    "prompt_code": prompt_code,
                    "prompt_text": template.format(conflict_color=conflict_color),
                    "true_color": true_color,
                    "conflict_color": conflict_color,
                    "color_confidence": row["color_confidence"],
                    "needs_manual_review": row["needs_manual_review"],
                    "model_name": "",
                    "model_output": "",
                    "label": "",
                    "language_consistent": "",
                    "vision_consistent": "",
                    "ambiguous": "",
                    "notes": "input_variant=main_car_crop; ignore_background=1; " + row.get("notes", ""),
                }
            )
    return rows


def generate_contact_sheet(selected_rows: list[dict], logger: logging.Logger) -> None:
    thumb_width = 220
    thumb_height = 160
    label_height = 62
    padding = 18
    columns = 5
    rows = math.ceil(len(selected_rows) / columns)
    canvas = Image.new(
        "RGB",
        (padding + columns * (thumb_width + padding), padding + rows * (thumb_height + label_height + padding)),
        color=(245, 246, 248),
    )
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(selected_rows):
        x = padding + (index % columns) * (thumb_width + padding)
        y = padding + (index // columns) * (thumb_height + label_height + padding)
        with Image.open(ROOT / row["cropped_path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (thumb_width, thumb_height))
            canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb_height + 4), f"{row['image_id']} | {row['estimated_color']}->{CONFLICT_COLOR_MAP[row['estimated_color']]}", fill=(25, 25, 25))
        draw.text((x, y + thumb_height + 24), f"q={row['quality_score']} | conf={row['color_confidence']}", fill=(65, 65, 65))

    CONTACT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(CONTACT_SHEET, quality=92)
    logger.info("Generated contact sheet: %s", relative_str(CONTACT_SHEET))


def generate_preview_markdown(selected_rows: list[dict], prompt_rows: list[dict], logger: logging.Logger) -> None:
    lines = [
        "# Stanford Cars Clean Car-Color Preview",
        "",
        "## 1. Experiment Sample",
    ]
    for row in selected_rows:
        conflict_color = CONFLICT_COLOR_MAP[row["estimated_color"]]
        lines.append(
            f"- {row['image_id']} | split={row['split']} | class={row['class_name']} | true_color={row['estimated_color']} | "
            f"conflict_color={conflict_color} | quality={row['quality_score']} | image_path={row['cropped_path']}"
        )

    lines.extend(["", "## 2. Prompt Templates"])
    for _, prompt_code, template in PROMPT_TEMPLATES:
        lines.append(f"- {prompt_code}: {template}")

    lines.extend(
        [
            "",
            "## 3. Files",
            f"- Sample CSV: {relative_str(SAMPLES_CSV)}",
            f"- Review CSV: {relative_str(REVIEW_CSV)}",
            f"- Prompt CSV: {relative_str(PROMPTS_CSV)}",
            f"- Clean subset manifest: {relative_str(MANIFEST_CSV)}",
            f"- Contact sheet: {relative_str(CONTACT_SHEET)}",
            "",
            "## 4. Counts",
            f"- Experiment images: {len(selected_rows)}",
            f"- Prompt rows: {len(prompt_rows)}",
        ]
    )
    PREVIEW_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Generated preview markdown: %s", relative_str(PREVIEW_MD))


def generate_preview_html(selected_rows: list[dict], logger: logging.Logger) -> None:
    cards: list[str] = []
    for row in selected_rows:
        true_color = row["estimated_color"]
        conflict_color = CONFLICT_COLOR_MAP[true_color]
        image_rel = Path("..") / row["cropped_path"]
        cards.append(
            f"""
            <div class="card">
              <img src="{html.escape(image_rel.as_posix())}" alt="{html.escape(row['file_name'])}">
              <div class="meta">
                <div><strong>image_id</strong>: {row['image_id']}</div>
                <div><strong>split</strong>: {row['split']}</div>
                <div><strong>true_color</strong>: {true_color}</div>
                <div><strong>conflict_color</strong>: {conflict_color}</div>
                <div><strong>quality</strong>: {row['quality_score']}</div>
                <div><strong>class</strong>: {html.escape(row['class_name'])}</div>
              </div>
            </div>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stanford Cars Clean Preview</title>
  <style>
    body {{
      font-family: "Segoe UI", sans-serif;
      margin: 24px;
      background: linear-gradient(180deg, #f7f4ee 0%, #eef2f5 100%);
      color: #1f2933;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: rgba(255,255,255,0.95);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.09);
    }}
    img {{
      width: 100%;
      height: 210px;
      object-fit: cover;
      display: block;
      background: #dce3ea;
    }}
    .meta {{
      padding: 14px 16px 16px;
      font-size: 14px;
      line-height: 1.65;
    }}
  </style>
</head>
<body>
  <h1>Stanford Cars Clean Preview</h1>
  <p>The experiment input uses bbox-guided main-car crops resized to a moderate resolution so body-color judgments focus on the car instead of the background.</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    PREVIEW_HTML.write_text(html_text, encoding="utf-8")
    logger.info("Generated preview HTML: %s", relative_str(PREVIEW_HTML))


def generate_setup_report(
    raw_record_count: int,
    candidate_rows: list[dict],
    kept_rows: list[dict],
    selected_rows: list[dict],
    prompt_rows: list[dict],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    keep_counts = Counter(row["split"] for row in kept_rows)
    selected_colors = Counter(row["estimated_color"] for row in selected_rows)
    report = f"""# Stanford Cars Clean Car-Color Experiment Setup

## 1. Pipeline Entry
- Data preparation script: `scripts/generate_car_color_stanford_clean_table.py`
- Inference entry: `scripts/run_qwen2vl_batch.py`
- Runtime merge: `scripts/export_qwen2vl_raw_results.py`
- Label summary: `scripts/auto_label_car_color_attribute_conflict_outputs.py`

## 2. What Changed
- The old COCO-derived car images are not touched.
- A new Stanford Cars raw mirror is downloaded into `data/raw/stanford_cars/`.
- A clean subset of bbox-guided main-car crops is created under `data/processed/stanford_cars/`.
- A smaller experiment sample is exported in the same CSV-driven format used by the current car-color pipeline.

## 3. Raw Dataset Source
- Hugging Face dataset mirror: `{DATASET_REPO_ID}`
- Mirror target: `{relative_str(RAW_DATASET_DIR)}`
- Note: the historical Stanford download URLs used by torchvision now return 404, so this run uses the mirror while preserving the original Stanford Cars folder structure and annotations.

## 4. Clean-Subset Heuristics
- bbox area ratio >= {DEFAULT_MIN_BBOX_AREA_RATIO:.2f}
- bbox width >= {DEFAULT_MIN_BBOX_WIDTH}
- bbox height >= {DEFAULT_MIN_BBOX_HEIGHT}
- crop fill ratio >= {DEFAULT_MIN_CROP_FILL_RATIO:.2f}
- color confidence >= {DEFAULT_MIN_COLOR_CONFIDENCE:.2f}
- background complexity <= {DEFAULT_MAX_BACKGROUND_COMPLEXITY:.2f}
- aspect ratio in [{DEFAULT_ASPECT_MIN:.2f}, {DEFAULT_ASPECT_MAX:.2f}]
- keep top `{args.clean_subset_size}` records after filtering, balanced by split when possible

## 5. Image Processing
- bbox-guided crop with padding via existing crop helper
- resize crops to short edge = `{args.target_short_edge}`
- export clean crops to `{relative_str(CLEAN_CROPS_DIR)}`

## 6. Experiment Compatibility
- Experiment sample size: `{args.experiment_sample_size}`
- Prompt design: same S0-S7 conflict ladder, but wording now explicitly asks for the main car body color and to ignore the background
- Downstream interface preserved: sample CSV + prompt CSV + runtime CSV + labeling CSV

## 7. Counts
- Raw records scanned: **{raw_record_count}**
- Candidate pool scored with image heuristics: **{len(candidate_rows)}**
- Clean subset kept: **{len(kept_rows)}**
- Clean subset split counts: **{dict(sorted(keep_counts.items()))}**
- Experiment sample size: **{len(selected_rows)}**
- Prompt rows: **{len(prompt_rows)}**
- Experiment color distribution: **{dict(sorted(selected_colors.items()))}**
"""
    SETUP_MD.write_text(report, encoding="utf-8")
    logger.info("Generated setup report: %s", relative_str(SETUP_MD))


def write_summary_json(raw_record_count: int, candidate_rows: list[dict], kept_rows: list[dict], selected_rows: list[dict], prompt_rows: list[dict], logger: logging.Logger) -> None:
    summary = {
        "raw_record_count": raw_record_count,
        "candidate_pool_count": len(candidate_rows),
        "clean_subset_count": len(kept_rows),
        "experiment_sample_count": len(selected_rows),
        "prompt_row_count": len(prompt_rows),
        "split_counts": dict(Counter(row["split"] for row in kept_rows)),
        "selected_color_distribution": dict(Counter(row["estimated_color"] for row in selected_rows)),
        "manifest_csv": relative_str(MANIFEST_CSV),
        "samples_csv": relative_str(SAMPLES_CSV),
        "review_csv": relative_str(REVIEW_CSV),
        "prompts_csv": relative_str(PROMPTS_CSV),
    }
    SUBSET_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Generated subset summary JSON: %s", relative_str(SUBSET_SUMMARY_JSON))


def main() -> int:
    args = parse_args()
    logger = build_logger()
    logger.info("Starting Stanford Cars clean-subset generation.")
    logger.info(
        "clean_subset_size=%s | experiment_sample_size=%s | candidate_pool_size=%s | target_short_edge=%s | num_workers=%s | max_records=%s",
        args.clean_subset_size,
        args.experiment_sample_size,
        args.candidate_pool_size,
        args.target_short_edge,
        args.num_workers,
        args.max_records,
    )

    try:
        ensure_raw_dataset(logger, skip_download=args.skip_download)
        records = load_records(max_records=args.max_records)
        candidate_records = select_candidate_records(records, candidate_pool_size=min(args.candidate_pool_size, len(records)))
        pull_candidate_images(candidate_records, logger)
        all_rows = score_records(candidate_records, target_short_edge=args.target_short_edge, num_workers=args.num_workers, logger=logger)

        kept_ids = set(choose_kept_rows(all_rows, clean_subset_size=args.clean_subset_size))
        kept_rows = [row for row in all_rows if row["image_id"] in kept_ids]
        kept_rows.sort(key=lambda row: row["image_id"])
        export_clean_crops(kept_rows, target_short_edge=args.target_short_edge, logger=logger)
        finalize_manifest(all_rows, kept_ids)

        selected_rows = select_experiment_rows(kept_rows, experiment_sample_size=args.experiment_sample_size)
        sample_rows = build_sample_rows(selected_rows)
        review_rows = build_review_rows(selected_rows)
        prompt_rows = build_prompt_rows(selected_rows)

        manifest_fieldnames = [
            "image_id", "split", "class_id", "class_name", "file_name", "source_image_path", "original_path", "cropped_path",
            "width", "height", "original_width", "original_height", "bbox", "bbox_xywh", "bbox_area_ratio",
            "bbox_width", "bbox_height", "aspect_ratio", "crop_box", "crop_width", "crop_height",
            "crop_fill_ratio", "target_short_edge", "cropped_width", "cropped_height", "true_color", "estimated_color",
            "color_confidence", "needs_manual_review", "foreground_dominant_share", "background_complexity",
            "quality_score", "keep", "drop", "keep_reason", "passed_clean_filters", "filter_fail_reasons",
            "crop_source", "mask_source", "focus_pixel_count", "score_summary", "notes",
        ]
        sample_fieldnames = [
            "image_id", "split", "class_id", "class_name", "file_name", "image_path", "original_image_path",
            "width", "height", "original_width", "original_height", "crop_box", "primary_car_bbox",
            "bbox_area_ratio", "crop_fill_ratio", "quality_score", "background_complexity", "true_color", "estimated_color",
            "color_confidence", "foreground_dominant_share", "selection_notes",
        ]
        review_fieldnames = [
            "image_id", "split", "file_name", "image_path", "original_image_path", "true_color", "estimated_color",
            "color_confidence", "needs_manual_review", "confirmed_color", "notes",
        ]
        prompt_fieldnames = [
            "sample_id", "image_id", "file_name", "image_path", "original_image_path", "width", "height",
            "experiment_type", "dataset_name", "target_object", "attribute_type", "prompt_level", "prompt_code",
            "prompt_text", "true_color", "conflict_color", "color_confidence", "needs_manual_review",
            "model_name", "model_output", "label", "language_consistent", "vision_consistent", "ambiguous", "notes",
        ]

        write_csv(MANIFEST_CSV, manifest_fieldnames, all_rows, logger)
        write_csv(SAMPLES_CSV, sample_fieldnames, sample_rows, logger)
        write_csv(REVIEW_CSV, review_fieldnames, review_rows, logger)
        write_csv(PROMPTS_CSV, prompt_fieldnames, prompt_rows, logger)
        write_summary_json(len(records), all_rows, kept_rows, selected_rows, prompt_rows, logger)

        if not args.skip_preview:
            generate_contact_sheet(selected_rows, logger)
            generate_preview_markdown(selected_rows, prompt_rows, logger)
            generate_preview_html(selected_rows, logger)
            generate_setup_report(len(records), all_rows, kept_rows, selected_rows, prompt_rows, args, logger)

        summary = {
            "raw_record_count": len(records),
            "candidate_pool_count": len(all_rows),
            "clean_subset_count": len(kept_rows),
            "experiment_sample_count": len(selected_rows),
            "prompt_row_count": len(prompt_rows),
            "manifest_csv": relative_str(MANIFEST_CSV),
            "samples_csv": relative_str(SAMPLES_CSV),
            "prompts_csv": relative_str(PROMPTS_CSV),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        logger.info("Stanford Cars clean-subset generation finished successfully.")
        return 0
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
