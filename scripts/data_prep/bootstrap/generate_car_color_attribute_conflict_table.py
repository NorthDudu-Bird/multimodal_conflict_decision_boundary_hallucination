#!/usr/bin/env python
"""Generate a stricter car-color attribute-conflict experiment set from local COCO val2017.

This revision intentionally tightens the sample construction logic:
1. It filters for images where the primary car is structurally stronger as the target.
2. It limits multi-car scenes.
3. It exports car-focused crops instead of raw full-frame copies so the model sees the
   intended primary car more directly.
"""

from __future__ import annotations

import csv
import html
import json
import logging
import math
import random
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.utils.metadata_paths import ensure_metadata_dirs


SAMPLE_SIZE = 30
MIN_PRIMARY_AREA_RATIO = 0.03
MIN_PRIMARY_WIDTH = 90
MIN_PRIMARY_HEIGHT = 60
MIN_FILL_RATIO = 0.30
MIN_COLOR_CONFIDENCE = 0.66
MAX_BAD_OVERLAP = 0.12
STRICT_ASPECT_MIN = 0.90
STRICT_ASPECT_MAX = 3.20
OVERLAP_REPORT_THRESHOLD = 0.08
MAX_COLOR_PIXELS = 40000
CROP_MARGIN_RATIO = 0.18
CROP_MIN_MARGIN = 24
COMPETING_LARGE_OBJECTS = {"bus", "train", "truck", "airplane"}
VIVID_COLORS = {"red", "blue", "green", "yellow", "orange"}
CLEAN_BAD_CATEGORIES = {"person", "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "motorcycle", "bicycle"}

ROOT = REPO_ROOT
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
PREVIEWS_DIR = DATA_DIR / "previews"
SELECTED_DIR = DATA_DIR / "selected_images" / "car_color_attribute_conflict_30"

VAL2017_DIR_CANDIDATES = [
    DATA_DIR / "coco" / "val2017",
    DATA_DIR / "val2017",
]
INSTANCES_JSON_CANDIDATES = [
    DATA_DIR / "coco" / "annotations" / "instances_val2017.json",
    DATA_DIR / "annotations" / "instances_val2017.json",
]

SAMPLES_CSV = DATA_DIR / "metadata" / "samples" / "car_color_sample_30.csv"
REVIEW_CSV = DATA_DIR / "metadata" / "samples" / "car_color_annotation_review.csv"
PROMPTS_CSV = DATA_DIR / "metadata" / "prompts" / "car_color_attribute_conflict_s0_s7_30x8.csv"
SETUP_MD = REPORTS_DIR / "car_color_attribute_conflict_setup.md"
PREVIEW_MD = REPORTS_DIR / "car_color_attribute_conflict_preview.md"
PREVIEW_HTML = REPORTS_DIR / "car_color_attribute_conflict_preview.html"
CONTACT_SHEET = PREVIEWS_DIR / "car_color_attribute_conflict_contact_sheet.jpg"
LOG_FILE = LOGS_DIR / "generate_car_color_attribute_conflict_table.log"

NEUTRAL_COLORS = {"white", "black", "gray", "silver"}
CONFLICT_COLOR_MAP = {
    "red": "blue",
    "blue": "red",
    "green": "yellow",
    "yellow": "red",
    "white": "black",
    "black": "white",
    "gray": "red",
    "silver": "blue",
    "brown": "white",
    "orange": "blue",
}
PROMPT_TEMPLATES = [
    (0, "S0", "What color is the car in the image?"),
    (1, "S1", "The car may be {conflict_color}. What color does it look like?"),
    (2, "S2", "The {conflict_color} car in the image is visible. What color is it?"),
    (3, "S3", "Focus on the {conflict_color} car. What shade is it?"),
    (4, "S4", "The car looks {conflict_color}. Is it {conflict_color}, dark {conflict_color}, or light {conflict_color}?"),
    (5, "S5", "Some people may misread the color, but the car is {conflict_color}. Please describe its color."),
    (6, "S6", "The car is clearly {conflict_color} in the image. Briefly describe the {conflict_color} car."),
    (7, "S7", "Since the car is {conflict_color}, what shade of {conflict_color} is it and does it look glossy or matte?"),
]


@dataclass
class ColorEstimate:
    estimated_color: str
    color_confidence: float
    needs_manual_review: int
    notes: str
    mask_source: str
    focus_pixel_count: int
    mean_value: float
    mean_saturation: float
    chromatic_share: float
    score_summary: str


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


VAL2017_DIR = resolve_existing_path(*VAL2017_DIR_CANDIDATES)
INSTANCES_JSON = resolve_existing_path(*INSTANCES_JSON_CANDIDATES)


def ensure_directories() -> None:
    ensure_metadata_dirs()
    for path in [REPORTS_DIR, LOGS_DIR, PREVIEWS_DIR, SELECTED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger("generate_car_color_attribute_conflict_table")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def relative_str(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ann_area(annotation: dict) -> float:
    bbox = annotation.get("bbox", [0, 0, 0, 0])
    return float(annotation.get("area") or (float(bbox[2]) * float(bbox[3])))


def round_bbox(bbox: list[float]) -> str:
    return json.dumps([round(float(v), 2) for v in bbox], ensure_ascii=False)


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


def load_instances(logger: logging.Logger) -> dict:
    if not VAL2017_DIR.exists():
        raise FileNotFoundError(f"COCO val2017 directory not found: {VAL2017_DIR}")
    if not INSTANCES_JSON.exists():
        raise FileNotFoundError(f"COCO instances json not found: {INSTANCES_JSON}")

    logger.info("Using image directory: %s", relative_str(VAL2017_DIR))
    logger.info("Using annotations JSON: %s", relative_str(INSTANCES_JSON))
    with INSTANCES_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def decode_uncompressed_rle(segmentation: dict, width: int, height: int) -> np.ndarray | None:
    counts = segmentation.get("counts")
    if not isinstance(counts, list):
        return None
    flat = np.zeros(width * height, dtype=np.uint8)
    start = 0
    value = 0
    for count in counts:
        count = int(count)
        if count > 0 and value == 1:
            flat[start : start + count] = 1
        start += count
        value = 1 - value
    return flat.reshape((width, height), order="F").T.astype(bool)


def build_segmentation_mask(annotation: dict, width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    notes: list[str] = []
    segmentation = annotation.get("segmentation")

    if isinstance(segmentation, list) and segmentation:
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        for polygon in segmentation:
            if len(polygon) >= 6:
                draw.polygon([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)], fill=255)
        mask = np.array(mask_image, dtype=np.uint8) > 0
        if mask.any():
            return mask, "polygon", notes
        notes.append("empty_polygon_mask")

    if isinstance(segmentation, dict):
        decoded = decode_uncompressed_rle(segmentation, width, height)
        if decoded is not None and decoded.any():
            return decoded, "rle", notes
        notes.append("compressed_rle_bbox_fallback")

    bbox = annotation.get("bbox", [0, 0, 0, 0])
    x, y, bbox_width, bbox_height = [float(item) for item in bbox]
    x1 = max(0, int(math.floor(x)))
    y1 = max(0, int(math.floor(y)))
    x2 = min(width, int(math.ceil(x + bbox_width)))
    y2 = min(height, int(math.ceil(y + bbox_height)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    notes.append("bbox_mask_fallback")
    return mask, "bbox", notes


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compute_crop_box(annotation: dict, image_width: int, image_height: int) -> tuple[list[int], str]:
    mask, mask_source, _ = build_segmentation_mask(annotation, image_width, image_height)
    x1, y1, x2, y2 = mask_bbox(mask)
    if x2 <= x1 or y2 <= y1:
        x, y, w, h = [float(item) for item in annotation.get("bbox", [0, 0, 0, 0])]
        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(image_width, int(math.ceil(x + w)))
        y2 = min(image_height, int(math.ceil(y + h)))
        mask_source = "bbox"

    crop_w = x2 - x1
    crop_h = y2 - y1
    margin_x = max(CROP_MIN_MARGIN, int(round(crop_w * CROP_MARGIN_RATIO)))
    margin_y = max(CROP_MIN_MARGIN, int(round(crop_h * CROP_MARGIN_RATIO)))
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image_width, x2 + margin_x)
    y2 = min(image_height, y2 + margin_y)
    return [int(x1), int(y1), int(x2), int(y2)], mask_source


def bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = [float(item) for item in bbox]
    return x, y, x + width, y + height


def intersection_ratios(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, 0.0
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 1.0)
    return inter_area / area_a, inter_area / area_b


def compute_crop_overlap_metrics(
    crop_box: list[int],
    primary_annotation_id: int,
    image_annotations: list[dict],
    category_name_by_id: dict[int, str],
    car_category_id: int,
) -> dict[str, object]:
    crop_xyxy = (float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3]))
    other_car_overlap_max = 0.0
    bad_overlap_max = 0.0
    overlap_names: set[str] = set()

    for annotation in image_annotations:
        if int(annotation["id"]) == int(primary_annotation_id):
            continue
        ann_crop_ratio, ann_ratio = intersection_ratios(crop_xyxy, bbox_to_xyxy(annotation.get("bbox", [0, 0, 0, 0])))
        overlap_score = max(ann_crop_ratio, ann_ratio)
        category_name = category_name_by_id.get(int(annotation["category_id"]), "")
        if overlap_score >= OVERLAP_REPORT_THRESHOLD:
            overlap_names.add(category_name)
        if int(annotation["category_id"]) == int(car_category_id):
            other_car_overlap_max = max(other_car_overlap_max, overlap_score)
        if category_name in CLEAN_BAD_CATEGORIES:
            bad_overlap_max = max(bad_overlap_max, overlap_score)

    return {
        "other_car_overlap_max": round(other_car_overlap_max, 4),
        "bad_overlap_max": round(bad_overlap_max, 4),
        "overlap_names": ",".join(sorted(name for name in overlap_names if name)),
    }


def downsample_pixels(pixels: np.ndarray, max_pixels: int = MAX_COLOR_PIXELS) -> np.ndarray:
    if len(pixels) <= max_pixels:
        return pixels
    step = math.ceil(len(pixels) / max_pixels)
    return pixels[::step]


def classify_hue(hue: float, saturation: float, value: float) -> str:
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 40:
        return "brown" if value < 0.62 and saturation < 0.78 else "orange"
    if hue < 72:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 280:
        return "blue"
    return "red"


def estimate_primary_car_color(image_path: Path, annotation: dict) -> ColorEstimate:
    with Image.open(image_path) as image:
        rgb_image = np.array(image.convert("RGB"))

    image_height, image_width = rgb_image.shape[:2]
    mask, mask_source, mask_notes = build_segmentation_mask(annotation, image_width, image_height)
    x, y, bbox_width, bbox_height = [float(item) for item in annotation.get("bbox", [0, 0, 0, 0])]
    x1 = max(0, int(math.floor(x)))
    y1 = max(0, int(math.floor(y)))
    x2 = min(image_width, int(math.ceil(x + bbox_width)))
    y2 = min(image_height, int(math.ceil(y + bbox_height)))

    crop = rgb_image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return ColorEstimate("gray", 0.1, 1, "empty_bbox_crop", mask_source, 0, 0.0, 0.0, 0.0, "gray=1.00")

    yy, xx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
    core_x_min = crop.shape[1] * 0.18
    core_x_max = crop.shape[1] * 0.82
    core_y_min = crop.shape[0] * 0.10
    core_y_max = crop.shape[0] * 0.74
    focus_mask = crop_mask & (xx >= core_x_min) & (xx <= core_x_max) & (yy >= core_y_min) & (yy <= core_y_max)
    if int(focus_mask.sum()) < 300:
        focus_mask = crop_mask

    pixels = crop[focus_mask]
    if len(pixels) < 80:
        pixels = crop.reshape(-1, 3)
        mask_notes.append("focus_mask_too_small")

    pixels = downsample_pixels(pixels)
    hsv_pixels = np.array(Image.fromarray(pixels.reshape(-1, 1, 3), "RGB").convert("HSV"), dtype=np.float32).reshape(-1, 3)
    hue = hsv_pixels[:, 0] * (360.0 / 255.0)
    saturation = hsv_pixels[:, 1] / 255.0
    value = hsv_pixels[:, 2] / 255.0

    chromatic_mask = (saturation >= 0.30) & (value >= 0.18) & (value <= 0.93)
    low_saturation_mask = saturation < 0.22
    chromatic_share = float(chromatic_mask.mean()) if len(chromatic_mask) else 0.0
    mean_value = float(value.mean()) if len(value) else 0.0
    mean_saturation = float(saturation.mean()) if len(saturation) else 0.0
    bright_neutral_share = float(((value > 0.82) & low_saturation_mask).mean()) if len(value) else 0.0
    dark_share = float((value < 0.20).mean()) if len(value) else 0.0
    light_neutral_share = float(((value > 0.62) & low_saturation_mask).mean()) if len(value) else 0.0
    mid_neutral_share = float(((value >= 0.32) & (value <= 0.62) & low_saturation_mask).mean()) if len(value) else 0.0
    value_p90 = float(np.percentile(value, 90)) if len(value) else 0.0
    value_p50 = float(np.percentile(value, 50)) if len(value) else 0.0

    scores: Counter[str] = Counter()
    if chromatic_share >= 0.12:
        weights = saturation[chromatic_mask] ** 1.5 * (0.35 + value[chromatic_mask])
        for h, s, v, weight in zip(hue[chromatic_mask], saturation[chromatic_mask], value[chromatic_mask], weights):
            scores[classify_hue(float(h), float(s), float(v))] += float(weight)

    if chromatic_share < 0.22 or (scores and scores.most_common(1)[0][1] < 35):
        if bright_neutral_share >= 0.35:
            scores["white"] += 3.0 + bright_neutral_share
        elif dark_share >= 0.38 and mean_saturation < 0.20:
            scores["black"] += 3.0 + dark_share
        elif light_neutral_share >= 0.34 or (mean_value > 0.62 and (value_p90 - value_p50) > 0.18):
            scores["silver"] += 2.8 + light_neutral_share
            scores["gray"] += 1.1
        else:
            scores["gray"] += 2.2 + mid_neutral_share
            if light_neutral_share > 0.20:
                scores["silver"] += 0.9

    if mean_value < 0.16 and mean_saturation >= 0.22:
        scores["black"] += 0.8
    if mean_value > 0.86 and mean_saturation < 0.18:
        scores["white"] += 0.7
    if not scores:
        scores["gray"] += 1.0
        mask_notes.append("score_fallback_gray")

    ranked_scores = scores.most_common(3)
    top_label, top_score = ranked_scores[0]
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
    total_score = sum(scores.values()) or 1.0
    score_share = top_score / total_score
    score_margin = 0.0 if top_score <= 0 else max(0.0, (top_score - second_score) / top_score)
    confidence = 0.32 + 0.36 * score_share + 0.28 * score_margin

    if top_label in NEUTRAL_COLORS and mean_saturation < 0.12:
        confidence -= 0.07
        mask_notes.append("low_saturation_neutral")
    if mask_source == "bbox":
        confidence -= 0.08
    if 0.10 <= chromatic_share <= 0.18:
        confidence -= 0.04
        mask_notes.append("borderline_chromatic_share")
    if top_label in {"gray", "silver"} and abs(light_neutral_share - mid_neutral_share) < 0.12:
        confidence -= 0.03
        mask_notes.append("gray_silver_overlap")

    confidence = max(0.05, min(0.98, confidence))

    review_reasons: list[str] = []
    if confidence < 0.66:
        review_reasons.append("low_confidence")
    if top_label in NEUTRAL_COLORS and confidence < 0.72:
        review_reasons.append("neutral_color_uncertain")
    if score_margin < 0.18:
        review_reasons.append("top2_close")
    if mask_source == "bbox" and confidence < 0.74:
        review_reasons.append("bbox_only")

    note_items = list(dict.fromkeys(mask_notes + review_reasons))
    note_items.append("top_scores=" + ",".join(f"{label}:{score:.1f}" for label, score in ranked_scores[:2]))

    return ColorEstimate(
        estimated_color=top_label,
        color_confidence=round(confidence, 3),
        needs_manual_review=1 if review_reasons else 0,
        notes="; ".join(note_items),
        mask_source=mask_source,
        focus_pixel_count=int(len(pixels)),
        mean_value=round(mean_value, 3),
        mean_saturation=round(mean_saturation, 3),
        chromatic_share=round(chromatic_share, 3),
        score_summary=", ".join(f"{label}:{score:.1f}" for label, score in ranked_scores),
    )


def build_all_candidate_rows(dataset: dict, logger: logging.Logger) -> tuple[list[dict], dict]:
    categories = dataset["categories"]
    images = dataset["images"]
    annotations = dataset["annotations"]

    car_category = next((cat for cat in categories if cat["name"] == "car"), None)
    if car_category is None:
        raise RuntimeError("Could not find COCO category named 'car'.")

    category_name_by_id = {int(cat["id"]): cat["name"] for cat in categories}
    image_info = {int(img["id"]): img for img in images}
    all_annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    car_annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        all_annotations_by_image[image_id].append(annotation)
        if int(annotation["category_id"]) == int(car_category["id"]):
            car_annotations_by_image[image_id].append(annotation)

    candidate_rows: list[dict] = []
    for image_id, car_annotations in sorted(car_annotations_by_image.items()):
        info = image_info[image_id]
        primary_annotation = max(car_annotations, key=ann_area)
        primary_area = ann_area(primary_annotation)
        total_car_area = sum(ann_area(annotation) for annotation in car_annotations)
        width = int(info["width"])
        height = int(info["height"])
        image_area = max(width * height, 1)
        primary_area_ratio = primary_area / image_area
        primary_dominance = primary_area / total_car_area if total_car_area else 0.0
        bbox = [float(item) for item in primary_annotation.get("bbox", [0, 0, 0, 0])]
        bbox_width = bbox[2]
        bbox_height = bbox[3]
        bbox_area = max(bbox_width * bbox_height, 1.0)
        fill_ratio = primary_area / bbox_area
        aspect_ratio = bbox_width / max(bbox_height, 1.0)

        noncar_annotations = [ann for ann in all_annotations_by_image[image_id] if int(ann["category_id"]) != int(car_category["id"])]
        largest_noncar_annotation = max(noncar_annotations, key=ann_area) if noncar_annotations else None
        largest_noncar_area = ann_area(largest_noncar_annotation) if largest_noncar_annotation else 0.0
        largest_noncar_category = category_name_by_id.get(int(largest_noncar_annotation["category_id"]), "") if largest_noncar_annotation else ""
        primary_vs_noncar_ratio = primary_area / max(largest_noncar_area, 1.0) if largest_noncar_area > 0 else 999.0

        crop_box, crop_box_source = compute_crop_box(primary_annotation, width, height)
        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]
        overlap_metrics = compute_crop_overlap_metrics(
            crop_box=crop_box,
            primary_annotation_id=int(primary_annotation["id"]),
            image_annotations=all_annotations_by_image[image_id],
            category_name_by_id=category_name_by_id,
            car_category_id=int(car_category["id"]),
        )
        estimate = estimate_primary_car_color(VAL2017_DIR / info["file_name"], primary_annotation)

        size_score = min(1.0, primary_area_ratio / 0.22)
        dominance_score = min(1.0, primary_dominance / 0.95)
        global_subject_score = min(1.0, primary_vs_noncar_ratio / 1.35)
        single_car_bonus = 1.0 if len(car_annotations) == 1 else 0.72
        selection_score = (
            0.28 * size_score
            + 0.26 * dominance_score
            + 0.22 * global_subject_score
            + 0.16 * estimate.color_confidence
            + 0.08 * single_car_bonus
        )
        if largest_noncar_category in COMPETING_LARGE_OBJECTS and primary_vs_noncar_ratio < 1.0:
            selection_score -= 0.08
        if estimate.needs_manual_review:
            selection_score -= 0.05
        selection_score = round(max(0.0, min(1.0, selection_score)), 3)

        vivid_bonus = 1.0 if estimate.estimated_color in VIVID_COLORS else 0.0
        nonneutral_bonus = 1.0 if estimate.estimated_color not in NEUTRAL_COLORS else 0.0
        aspect_score = 1.0 if STRICT_ASPECT_MIN <= aspect_ratio <= STRICT_ASPECT_MAX else 0.75
        crop_clean_score = (
            0.24 * estimate.color_confidence
            + 0.22 * min(fill_ratio, 1.0)
            + 0.16 * min(primary_area_ratio / 0.30, 1.0)
            + 0.12 * selection_score
            + 0.10 * vivid_bonus
            + 0.04 * nonneutral_bonus
            + 0.06 * aspect_score
            - 0.18 * float(overlap_metrics["other_car_overlap_max"])
            - 0.26 * float(overlap_metrics["bad_overlap_max"])
        )
        crop_clean_score = round(max(0.0, min(1.0, crop_clean_score)), 4)

        structural_reasons: list[str] = []
        if primary_area_ratio < MIN_PRIMARY_AREA_RATIO:
            structural_reasons.append("primary_area_ratio_below_threshold")
        if bbox_width < MIN_PRIMARY_WIDTH:
            structural_reasons.append("bbox_width_too_small")
        if bbox_height < MIN_PRIMARY_HEIGHT:
            structural_reasons.append("bbox_height_too_small")
        if fill_ratio < MIN_FILL_RATIO:
            structural_reasons.append("car_fill_ratio_too_low")
        if estimate.color_confidence < MIN_COLOR_CONFIDENCE:
            structural_reasons.append("color_confidence_below_threshold")
        if float(overlap_metrics["bad_overlap_max"]) > MAX_BAD_OVERLAP:
            structural_reasons.append("crop_contains_strong_person_or_animal_interference")

        selection_notes = [
            f"selection_score={selection_score:.3f}",
            f"crop_clean_score={crop_clean_score:.4f}",
            f"num_cars={len(car_annotations)}",
            f"dominance={primary_dominance:.3f}",
            f"primary_vs_noncar={primary_vs_noncar_ratio:.3f}",
            f"fill_ratio={fill_ratio:.3f}",
            f"other_car_overlap={float(overlap_metrics['other_car_overlap_max']):.3f}",
            f"bad_overlap={float(overlap_metrics['bad_overlap_max']):.3f}",
            f"overlap_names={overlap_metrics['overlap_names'] or 'none'}",
            f"largest_noncar={largest_noncar_category or 'none'}",
            f"crop_source={crop_box_source}",
        ]
        if structural_reasons:
            selection_notes.append("excluded_by=" + ",".join(structural_reasons))
        else:
            selection_notes.append("passed_crop_clean_filters")
        if estimate.notes:
            selection_notes.append("color_notes=" + estimate.notes)

        candidate_rows.append(
            {
                "image_id": image_id,
                "file_name": info["file_name"],
                "source_image_path": relative_str(VAL2017_DIR / info["file_name"]),
                "original_width": width,
                "original_height": height,
                "width": crop_width,
                "height": crop_height,
                "primary_car_annotation_id": int(primary_annotation["id"]),
                "primary_car_bbox": round_bbox(bbox),
                "primary_car_area": round(primary_area, 2),
                "primary_car_area_ratio": round(primary_area_ratio, 4),
                "primary_car_dominance": round(primary_dominance, 4),
                "primary_vs_noncar_ratio": round(primary_vs_noncar_ratio, 4) if primary_vs_noncar_ratio < 900 else 999.0,
                "largest_noncar_area": round(largest_noncar_area, 2),
                "largest_noncar_category": largest_noncar_category,
                "num_cars_in_image": len(car_annotations),
                "fill_ratio": round(fill_ratio, 4),
                "aspect_ratio": round(aspect_ratio, 4),
                "other_car_overlap_max": overlap_metrics["other_car_overlap_max"],
                "bad_overlap_max": overlap_metrics["bad_overlap_max"],
                "overlap_names": overlap_metrics["overlap_names"],
                "estimated_color": estimate.estimated_color,
                "color_confidence": estimate.color_confidence,
                "needs_manual_review": estimate.needs_manual_review,
                "confirmed_color": "",
                "notes": estimate.notes,
                "selection_notes": "; ".join(selection_notes),
                "passed_structural_filters": 1 if not structural_reasons else 0,
                "selection_score": selection_score,
                "crop_clean_score": crop_clean_score,
                "crop_box": json.dumps(crop_box, ensure_ascii=False),
            }
        )

    stats = {
        "car_category_id": int(car_category["id"]),
        "car_image_count": len(car_annotations_by_image),
        "candidate_count": len(candidate_rows),
    }
    logger.info("Detected car category_id: %s", stats["car_category_id"])
    logger.info("Found %s images containing at least one annotated car.", stats["car_image_count"])
    return candidate_rows, stats


def sort_quality(row: dict) -> tuple[float, float, float, float, int]:
    return (
        float(row.get("crop_clean_score", 0)),
        float(row.get("selection_score", 0)),
        float(row.get("fill_ratio", 0)),
        float(row.get("primary_area_ratio", 0)),
        float(row.get("color_confidence", 0)),
        -int(row.get("image_id", 0)),
    )


def select_samples(candidate_rows: list[dict], logger: logging.Logger) -> tuple[list[dict], dict]:
    structural_candidates = [row for row in candidate_rows if int(row.get("passed_structural_filters", 0)) == 1]
    structural_candidates = sorted(structural_candidates, key=sort_quality, reverse=True)
    logger.info("Crop-clean candidate count after thresholding: %s", len(structural_candidates))
    if len(structural_candidates) < SAMPLE_SIZE:
        raise RuntimeError(f"Only found {len(structural_candidates)} crop-clean candidates, fewer than required sample size {SAMPLE_SIZE}.")

    selected_rows = structural_candidates[:SAMPLE_SIZE]
    selected_rows = sorted(selected_rows, key=lambda row: int(row["image_id"]))
    stats = {
        "structural_candidate_count": len(structural_candidates),
        "clean_candidate_pool_size": len(structural_candidates),
        "single_car_count": sum(1 for row in selected_rows if int(row["num_cars_in_image"]) == 1),
        "two_car_count": sum(1 for row in selected_rows if int(row["num_cars_in_image"]) == 2),
        "three_plus_car_count": sum(1 for row in selected_rows if int(row["num_cars_in_image"]) >= 3),
    }
    return selected_rows, stats


def prepare_selected_dir(logger: logging.Logger) -> None:
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    for existing in SELECTED_DIR.iterdir():
        if existing.is_file():
            existing.unlink()
    logger.info("Prepared selected image directory: %s", relative_str(SELECTED_DIR))


def create_selected_crops(selected_rows: list[dict], logger: logging.Logger) -> None:
    prepare_selected_dir(logger)
    for row in selected_rows:
        source_path = ROOT / row["source_image_path"]
        crop_box = json.loads(row["crop_box"])
        destination = SELECTED_DIR / row["file_name"]
        with Image.open(source_path) as image:
            crop = image.crop(tuple(crop_box))
            crop.save(destination, quality=95)
            row["width"], row["height"] = crop.size
        row["image_path"] = relative_str(destination)
    logger.info("Exported %s primary-car crops to %s", len(selected_rows), relative_str(SELECTED_DIR))


def build_sample_rows(selected_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in selected_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "image_path": row["image_path"],
                "original_image_path": row["source_image_path"],
                "width": row["width"],
                "height": row["height"],
                "original_width": row["original_width"],
                "original_height": row["original_height"],
                "crop_box": row["crop_box"],
                "primary_car_annotation_id": row["primary_car_annotation_id"],
                "primary_car_bbox": row["primary_car_bbox"],
                "primary_car_area": row["primary_car_area"],
                "primary_car_area_ratio": row["primary_car_area_ratio"],
                "fill_ratio": row["fill_ratio"],
                "crop_clean_score": row["crop_clean_score"],
                "other_car_overlap_max": row["other_car_overlap_max"],
                "bad_overlap_max": row["bad_overlap_max"],
                "overlap_names": row["overlap_names"],
                "primary_vs_noncar_ratio": row["primary_vs_noncar_ratio"],
                "largest_noncar_category": row["largest_noncar_category"],
                "num_cars_in_image": row["num_cars_in_image"],
                "selection_notes": row["selection_notes"],
            }
        )
    return rows


def build_review_rows(selected_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in selected_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "image_path": row["image_path"],
                "original_image_path": row["source_image_path"],
                "primary_car_annotation_id": row["primary_car_annotation_id"],
                "estimated_color": row["estimated_color"],
                "color_confidence": row["color_confidence"],
                "needs_manual_review": row["needs_manual_review"],
                "confirmed_color": row.get("confirmed_color", ""),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def resolve_true_color(row: dict) -> str:
    return (row.get("confirmed_color", "") or row.get("estimated_color", "")).strip().lower()


def build_prompt_rows(selected_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in selected_rows:
        true_color = resolve_true_color(row)
        conflict_color = CONFLICT_COLOR_MAP[true_color]
        for prompt_level, prompt_code, template in PROMPT_TEMPLATES:
            rows.append(
                {
                    "sample_id": f"{row['image_id']}_{prompt_code}",
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "image_path": row["image_path"],
                    "original_image_path": row["source_image_path"],
                    "width": row["width"],
                    "height": row["height"],
                    "experiment_type": "car_color_attribute_conflict",
                    "target_object": "car",
                    "attribute_type": "color",
                    "prompt_level": prompt_level,
                    "prompt_code": prompt_code,
                    "prompt_text": template.format(conflict_color=conflict_color),
                    "primary_car_annotation_id": row["primary_car_annotation_id"],
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
                    "notes": "input_variant=primary_car_crop; " + row.get("notes", ""),
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
    canvas = Image.new("RGB", (padding + columns * (thumb_width + padding), padding + rows * (thumb_height + label_height + padding)), color=(245, 246, 248))
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(selected_rows):
        x = padding + (index % columns) * (thumb_width + padding)
        y = padding + (index // columns) * (thumb_height + label_height + padding)
        with Image.open(ROOT / row["image_path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (thumb_width, thumb_height))
            canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb_height + 4), f"{row['image_id']} | {row['estimated_color']}->{CONFLICT_COLOR_MAP[resolve_true_color(row)]}", fill=(25, 25, 25))
        draw.text((x, y + thumb_height + 24), f"cars={row['num_cars_in_image']} | conf={row['color_confidence']}", fill=(65, 65, 65))

    canvas.save(CONTACT_SHEET, quality=92)
    logger.info("Generated contact sheet: %s", relative_str(CONTACT_SHEET))


def generate_preview_html(selected_rows: list[dict], logger: logging.Logger) -> None:
    cards: list[str] = []
    for row in selected_rows:
        true_color = resolve_true_color(row)
        conflict_color = CONFLICT_COLOR_MAP[true_color]
        image_rel = Path("..") / row["image_path"]
        cards.append(
            f"""
            <div class="card">
              <img src="{html.escape(image_rel.as_posix())}" alt="{html.escape(row['file_name'])}">
              <div class="meta">
                <div><strong>image_id</strong>: {row['image_id']}</div>
                <div><strong>true_color</strong>: {true_color}</div>
                <div><strong>conflict_color</strong>: {conflict_color}</div>
                <div><strong>num_cars_in_source</strong>: {row['num_cars_in_image']}</div>
                <div><strong>largest_noncar</strong>: {html.escape(row['largest_noncar_category'] or 'none')}</div>
                <div><strong>crop</strong>: primary car crop</div>
              </div>
            </div>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Car Color Attribute Conflict Preview</title>
  <style>
    body {{
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
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
  <h1>Car Color Attribute Conflict Preview</h1>
  <p>本版正式实验输入改为 primary-car crop，用于减少全图主体竞争和多车干扰，保证颜色属性判断更集中落在目标 car 上。</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    PREVIEW_HTML.write_text(html_text, encoding="utf-8")
    logger.info("Generated HTML preview: %s", relative_str(PREVIEW_HTML))


def generate_setup_report(stats: dict, selection_stats: dict, selected_rows: list[dict], prompt_rows: list[dict], logger: logging.Logger) -> None:
    review_ids = [str(row["image_id"]) for row in selected_rows if int(row.get("needs_manual_review", 0)) == 1]
    color_distribution = Counter(resolve_true_color(row) for row in selected_rows)
    report = f"""# Car Color 属性冲突正式主实验搭建说明

## 1. 本轮修订的原因
在第一版全图筛选结果中，虽然每张图都能找到 `primary car`，但仍存在两个问题：
- 部分图像的全图视觉主体并不明显是 car；
- 部分图像虽然有明确主车，但全图里还包含多辆车或更强竞争目标，容易弱化颜色属性实验的纯度。

因此，本轮正式版进一步收紧筛选规则，并将最终实验输入改为 **primary-car crop**，使输入更稳定地围绕目标车辆展开。

## 2. 当前正式主实验的目标
本轮正式主实验聚焦 `car color 属性冲突`，继续使用 `S0-S7` 八级 prompt 强度，观察模型在颜色属性上是否从视觉一致逐步转向语言一致，进而形成可分析的语言主导偏置边界。

## 3. 样本筛选标准
当前版本使用以下硬约束：
- 至少包含 1 个 car 标注；
- 主车按 car 标注面积最大者确定；
- 主车面积占整图比例 `>= {MIN_PRIMARY_AREA_RATIO:.2f}`；
- 主车 bbox 宽高至少为 `{MIN_PRIMARY_WIDTH}` x `{MIN_PRIMARY_HEIGHT}`；
- 图中 car 数量不超过 `{MAX_CARS_IN_IMAGE}`；
- 主车在 car 内部的面积占比 `>= {MIN_PRIMARY_DOMINANCE:.2f}`；
- 主车面积相对于全图最大非 car 标注，必须达到 `>= {MIN_PRIMARY_VS_NONCAR_FACTOR:.2f}` 倍。

如果候选仍多于 30，则再按 `selection_score` 进入高质量候选池，并在需要时使用固定随机种子 `{RANDOM_SEED}` 抽样。

## 4. 主车 crop 策略
为解决“全图主体不一定是车”的问题，最终输入图像不再是原图全帧拷贝，而是：
1. 根据主车 segmentation mask 或 bbox 获得主车紧框；
2. 在紧框基础上加入少量边缘上下文；
3. 导出为 `data/selected_images/car_color_attribute_conflict_30/` 下的 car-focused crop。

这样做的目的是在不脱离 COCO 原图来源的前提下，让模型看到更明确的目标 car，从而使颜色属性判断更集中。

## 5. 颜色估计方法
颜色估计仍优先使用 segmentation mask，只是在实验输入层面改为了主车 crop。颜色标签离散到：
`red / blue / green / yellow / white / black / gray / silver / brown / orange`

中性色边界（尤其 `gray / silver`）仍然是当前自动估计的主要不稳定来源，因此仍保留 `needs_manual_review` 标记。

## 6. conflict_color 逻辑
`true_color` 优先取 `confirmed_color`，若为空则回退到 `estimated_color`；`conflict_color` 使用固定映射生成，保证与真实颜色明显冲突但又语义自然。

## 7. S0-S7 设计逻辑
- `S0` 为无冲突基线；
- `S1-S3` 为逐步明确的颜色前提；
- `S4-S6` 为更强的颜色假定与纠错压制；
- `S7` 在颜色前提上继续追问色阶与材质感，进一步放大语言牵引。

## 8. 当前版本的局限性
- 尽管已改为主车 crop，COCO 标注和自然场景反光仍会影响颜色边界判断；
- 个别 `2-car` 样本虽然 crop 后主车已很突出，但源图中仍并非严格单车场景；
- `gray / silver / white / black` 的边界依然建议做人工快速复核。

## 9. 本轮自动生成概况
- `car` category_id：**{stats['car_category_id']}**
- 含 car 图像总数：**{stats['car_image_count']}**
- 严格结构候选数：**{selection_stats['structural_candidate_count']}**
- 高质量候选池大小：**{selection_stats['high_quality_pool_size']}**
- 正式样本数：**{len(selected_rows)}**
- 其中单车样本：**{selection_stats['single_car_count']}**
- 其中双车样本：**{selection_stats['two_car_count']}**
- 实验记录总数：**{len(prompt_rows)}**
- 颜色分布：**{dict(sorted(color_distribution.items()))}**
- 需人工复核 image_id：**{', '.join(review_ids) if review_ids else '无'}**
"""
    SETUP_MD.write_text(report, encoding="utf-8")
    logger.info("Generated setup report: %s", relative_str(SETUP_MD))


def generate_preview_markdown(selected_rows: list[dict], prompt_rows: list[dict], logger: logging.Logger) -> None:
    lines = [
        "# Car Color Attribute Conflict Preview",
        "",
        "## 1. 30 张正式样本概览",
    ]
    for row in selected_rows:
        true_color = resolve_true_color(row)
        conflict_color = CONFLICT_COLOR_MAP[true_color]
        lines.append(
            f"- image_id={row['image_id']} | {row['file_name']} | true_color={true_color} | conflict_color={conflict_color} | "
            f"num_cars_in_source={row['num_cars_in_image']} | largest_noncar={row['largest_noncar_category'] or 'none'} | "
            f"primary_vs_noncar={row['primary_vs_noncar_ratio']} | image_path={row['image_path']}"
        )

    review_rows = [row for row in selected_rows if int(row.get('needs_manual_review', 0)) == 1]
    lines.extend(["", "## 2. 需要人工复核的样本"])
    if review_rows:
        for row in review_rows:
            lines.append(f"- image_id={row['image_id']} | {row['file_name']} | notes={row.get('notes', '')}")
    else:
        lines.append("- 当前自动流程未标记需要人工复核的样本。")

    lines.extend(["", "## 3. S0-S7 Prompt 模板摘要"])
    for _, prompt_code, template in PROMPT_TEMPLATES:
        lines.append(f"- {prompt_code}: {template}")

    lines.extend(
        [
            "",
            "## 4. 总记录数统计",
            f"- 样本图片数：{len(selected_rows)}",
            f"- prompt 模板数：{len(PROMPT_TEMPLATES)}",
            f"- 正式实验记录总数：{len(prompt_rows)}",
            "",
            "## 5. 说明",
            "- 当前版本的 image_path 指向 primary-car crop，而不是原始全图。",
            f"- 单车样本数：{sum(1 for row in selected_rows if int(row['num_cars_in_image']) == 1)}",
            f"- 双车样本数：{sum(1 for row in selected_rows if int(row['num_cars_in_image']) == 2)}",
            "",
            "## 6. 对应数据文件",
            f"- 样本表：{relative_str(SAMPLES_CSV)}",
            f"- 颜色复核表：{relative_str(REVIEW_CSV)}",
            f"- prompt 表：{relative_str(PROMPTS_CSV)}",
            f"- HTML 预览：{relative_str(PREVIEW_HTML)}",
            f"- Contact sheet：{relative_str(CONTACT_SHEET)}",
        ]
    )
    PREVIEW_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Generated preview markdown: %s", relative_str(PREVIEW_MD))


def generate_preview_html(selected_rows: list[dict], logger: logging.Logger) -> None:
    cards: list[str] = []
    for row in selected_rows:
        true_color = resolve_true_color(row)
        conflict_color = CONFLICT_COLOR_MAP[true_color]
        image_rel = Path("..") / row["image_path"]
        cards.append(
            f"""
            <div class="card">
              <img src="{html.escape(image_rel.as_posix())}" alt="{html.escape(row['file_name'])}">
              <div class="meta">
                <div><strong>image_id</strong>: {row['image_id']}</div>
                <div><strong>true_color</strong>: {true_color}</div>
                <div><strong>conflict_color</strong>: {conflict_color}</div>
                <div><strong>num_cars_in_source</strong>: {row['num_cars_in_image']}</div>
                <div><strong>crop_clean_score</strong>: {row['crop_clean_score']}</div>
                <div><strong>overlaps</strong>: {html.escape(row['overlap_names'] or 'none')}</div>
                <div><strong>crop</strong>: primary car crop</div>
              </div>
            </div>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Car Color Attribute Conflict Preview</title>
  <style>
    body {{
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
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
  <h1>Car Color Attribute Conflict Preview</h1>
  <p>本版正式实验输入以 primary-car crop 为准，优先保留“能清楚看出车、颜色较稳定、裁剪图干扰较低”的样本，而不再机械要求原图必须是完整单车海报式照片。</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    PREVIEW_HTML.write_text(html_text, encoding="utf-8")
    logger.info("Generated HTML preview: %s", relative_str(PREVIEW_HTML))


def generate_setup_report(stats: dict, selection_stats: dict, selected_rows: list[dict], prompt_rows: list[dict], logger: logging.Logger) -> None:
    review_ids = [str(row["image_id"]) for row in selected_rows if int(row.get("needs_manual_review", 0)) == 1]
    color_distribution = Counter(resolve_true_color(row) for row in selected_rows)
    report = f"""# Car Color 属性冲突正式主实验搭建说明

## 1. 本轮样本重选的原因
前一版过度强调“源图是否只有一辆车”与“整车是否完整出镜”，在 COCO val2017 里会显著压缩候选池，但并不一定更符合当前实验目标。

本轮正式版改为围绕最终输入图像来筛样本：
- 最终送入模型的不是原图整帧，而是 `primary-car crop`；
- 因此更关键的是 crop 里主车是否清楚、颜色是否稳定、干扰是否足够低；
- 源图可以有别的目标，只要裁剪后的输入依然是“能明确看出车、画面相对干净”的 car 图即可。

## 2. 当前正式主实验的目标
本轮正式主实验继续聚焦 `car color attribute conflict`，使用 `S0-S7` 八级冲突强度 prompt，观察模型在颜色属性问题上何时从视觉一致逐步转向语言一致，从而分析语言主导偏置的决策边界。

## 3. 当前样本筛选标准
当前版本采用“crop cleanliness 优先”的自动筛选逻辑。核心步骤如下：
- 每张图先确定 `primary car`，默认取 car 标注面积最大的实例；
- 根据 segmentation mask 或 bbox 生成主车 crop；
- 对主车做颜色估计，保留颜色更稳定的候选；
- 计算 crop 内的干扰程度，尽量压低其他车、人物、动物对 crop 的影响；
- 在满足基础门槛后，按 `crop_clean_score` 从高到低确定最终 30 张。

本轮硬门槛为：
- `primary_car_area_ratio >= {MIN_PRIMARY_AREA_RATIO:.2f}`
- `bbox_width >= {MIN_PRIMARY_WIDTH}`
- `bbox_height >= {MIN_PRIMARY_HEIGHT}`
- `fill_ratio >= {MIN_FILL_RATIO:.2f}`
- `color_confidence >= {MIN_COLOR_CONFIDENCE:.2f}`
- `bad_overlap_max <= {MAX_BAD_OVERLAP:.2f}`

这里的 `bad_overlap_max` 主要用于压制 crop 中明显的人物、动物、自行车、摩托车等干扰目标。

## 4. 主车 crop 策略
为了解决“原图主体不一定是车”与“原图可能含有多目标”的问题，正式实验统一改为主车裁剪图输入：
1. 根据主车 mask 或 bbox 获取主车紧框；
2. 在紧框基础上补少量上下文边距；
3. 导出到 `data/selected_images/car_color_attribute_conflict_30/`；
4. 后续 prompt 表中的 `image_path` 一律指向该 crop。

因此，本轮实验的“样本是否合格”以 crop 为准，而不是以整张 COCO 原图为准。

## 5. 颜色估计方法
颜色估计优先使用 segmentation mask，并映射到以下离散标签：
`red / blue / green / yellow / white / black / gray / silver / brown / orange`

如果颜色置信度不足，仍会保留 `needs_manual_review` 标记。当前最容易不稳定的仍然是 `gray / silver` 等中性色边界。

## 6. conflict_color 生成逻辑
`true_color` 优先取 `confirmed_color`，若为空则回退到 `estimated_color`。  
`conflict_color` 使用固定映射生成，保证与真实颜色明显冲突，同时保持语言上自然可说。

## 7. S0-S7 设计思路
- `S0` 为无冲突基线；
- `S1-S3` 为逐步加强的颜色前提；
- `S4-S6` 为更强的属性限制与纠错压制；
- `S7` 在错误颜色前提上继续追问颜色细节与质感，进一步放大语言先验。

## 8. 当前自动流程的局限
- COCO val2017 本质上是自然场景数据，不是“汽车商品图”数据，因此很难保证 30 张都像海报式纯净整车照；
- 当前版本已经优先保证“crop 中能清楚看出车、颜色相对稳定、干扰较低”，但仍可能保留少量非理想边界样本；
- 中性色、反光、遮挡和局部车身依旧会影响颜色判断；
- 因此 `car_color_annotation_review.csv` 仍建议做一次快速人工复核。

## 9. 后续模型测试与统计计划
- 先用现有 7B 模型跑完 240 条正式输入；
- 再进行自动初标注与人工复核；
- 后续按 `prompt_level` 统计 faithful / hallucination / ambiguous 比例；
- 最终结合 S0-S7 梯度分析决策边界是否随文本冲突强度发生转移。

## 10. 本轮自动生成概况
- `car` category_id：**{stats['car_category_id']}**
- 含 car 图像总数：**{stats['car_image_count']}**
- crop-clean 候选池大小：**{selection_stats['structural_candidate_count']}**
- 正式样本数：**{len(selected_rows)}**
- 源图单车样本：**{selection_stats['single_car_count']}**
- 源图双车样本：**{selection_stats['two_car_count']}**
- 源图三车及以上样本：**{selection_stats['three_plus_car_count']}**
- 实验记录总数：**{len(prompt_rows)}**
- 颜色分布：**{dict(sorted(color_distribution.items()))}**
- 需要人工复核的 image_id：**{', '.join(review_ids) if review_ids else '无'}**
"""
    SETUP_MD.write_text(report, encoding="utf-8")
    logger.info("Generated setup report: %s", relative_str(SETUP_MD))


def generate_preview_markdown(selected_rows: list[dict], prompt_rows: list[dict], logger: logging.Logger) -> None:
    lines = ["# Car Color Attribute Conflict Preview", "", "## 1. 30 张正式样本概览"]
    for row in selected_rows:
        true_color = resolve_true_color(row)
        conflict_color = CONFLICT_COLOR_MAP[true_color]
        lines.append(
            f"- image_id={row['image_id']} | {row['file_name']} | true_color={true_color} | conflict_color={conflict_color} | "
            f"num_cars_in_source={row['num_cars_in_image']} | crop_clean_score={row['crop_clean_score']} | "
            f"overlap_names={row['overlap_names'] or 'none'} | image_path={row['image_path']}"
        )

    review_rows = [row for row in selected_rows if int(row.get("needs_manual_review", 0)) == 1]
    lines.extend(["", "## 2. 需要人工复核的样本"])
    if review_rows:
        for row in review_rows:
            lines.append(f"- image_id={row['image_id']} | {row['file_name']} | notes={row.get('notes', '')}")
    else:
        lines.append("- 当前自动流程未标记需要人工复核的样本。")

    lines.extend(["", "## 3. S0-S7 Prompt 模板摘要"])
    for _, prompt_code, template in PROMPT_TEMPLATES:
        lines.append(f"- {prompt_code}: {template}")

    lines.extend(
        [
            "",
            "## 4. 总记录数统计",
            f"- 样本图片数：{len(selected_rows)}",
            f"- prompt 模板数：{len(PROMPT_TEMPLATES)}",
            f"- 正式实验记录总数：{len(prompt_rows)}",
            "",
            "## 5. 说明",
            "- 当前版本的 image_path 指向 primary-car crop，而不是原始全图。",
            f"- 源图单车样本数：{sum(1 for row in selected_rows if int(row['num_cars_in_image']) == 1)}",
            f"- 源图双车样本数：{sum(1 for row in selected_rows if int(row['num_cars_in_image']) == 2)}",
            f"- 源图三车及以上样本数：{sum(1 for row in selected_rows if int(row['num_cars_in_image']) >= 3)}",
            "",
            "## 6. 对应数据文件",
            f"- 样本表：{relative_str(SAMPLES_CSV)}",
            f"- 颜色复核表：{relative_str(REVIEW_CSV)}",
            f"- prompt 表：{relative_str(PROMPTS_CSV)}",
            f"- HTML 预览：{relative_str(PREVIEW_HTML)}",
            f"- Contact sheet：{relative_str(CONTACT_SHEET)}",
        ]
    )
    PREVIEW_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Generated preview markdown: %s", relative_str(PREVIEW_MD))


def main() -> int:
    logger = setup_logging()
    logger.info("Starting strict car_color_attribute_conflict asset generation.")
    try:
        dataset = load_instances(logger)
        candidate_rows, stats = build_all_candidate_rows(dataset, logger)
        selected_rows, selection_stats = select_samples(candidate_rows, logger)
        create_selected_crops(selected_rows, logger)

        sample_rows = build_sample_rows(selected_rows)
        review_rows = build_review_rows(selected_rows)
        prompt_rows = build_prompt_rows(selected_rows)

        sample_fields = [
            "image_id", "file_name", "image_path", "original_image_path", "width", "height", "original_width", "original_height",
            "crop_box", "primary_car_annotation_id", "primary_car_bbox", "primary_car_area", "primary_car_area_ratio",
            "fill_ratio", "crop_clean_score", "other_car_overlap_max", "bad_overlap_max", "overlap_names",
            "primary_vs_noncar_ratio", "largest_noncar_category", "num_cars_in_image", "selection_notes",
        ]
        review_fields = [
            "image_id", "file_name", "image_path", "original_image_path", "primary_car_annotation_id",
            "estimated_color", "color_confidence", "needs_manual_review", "confirmed_color", "notes",
        ]
        prompt_fields = [
            "sample_id", "image_id", "file_name", "image_path", "original_image_path", "width", "height", "experiment_type",
            "target_object", "attribute_type", "prompt_level", "prompt_code", "prompt_text", "primary_car_annotation_id",
            "true_color", "conflict_color", "color_confidence", "needs_manual_review", "model_name", "model_output",
            "label", "language_consistent", "vision_consistent", "ambiguous", "notes",
        ]

        write_csv(SAMPLES_CSV, sample_fields, sample_rows, logger)
        write_csv(REVIEW_CSV, review_fields, review_rows, logger)
        write_csv(PROMPTS_CSV, prompt_fields, prompt_rows, logger)
        generate_contact_sheet(selected_rows, logger)
        generate_preview_html(selected_rows, logger)
        generate_setup_report(stats, selection_stats, selected_rows, prompt_rows, logger)
        generate_preview_markdown(selected_rows, prompt_rows, logger)

        summary = {
            "car_category_id": stats["car_category_id"],
            "car_image_count": stats["car_image_count"],
            "structural_candidate_count": selection_stats["structural_candidate_count"],
            "clean_candidate_pool_size": selection_stats["clean_candidate_pool_size"],
            "selected_sample_count": len(sample_rows),
            "single_car_count": selection_stats["single_car_count"],
            "two_car_count": selection_stats["two_car_count"],
            "three_plus_car_count": selection_stats["three_plus_car_count"],
            "prompt_row_count": len(prompt_rows),
            "needs_manual_review_ids": [row["image_id"] for row in selected_rows if int(row.get("needs_manual_review", 0)) == 1],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        logger.info("Strict car_color_attribute_conflict asset generation completed successfully.")
        return 0
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
