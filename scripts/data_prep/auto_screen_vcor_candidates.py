#!/usr/bin/env python
"""Auto-screen VCoR candidates with object and image-quality heuristics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import (
    build_logger,
    canonicalize_color,
    ensure_dirs,
    load_config,
    primary_main_analysis_labels,
    read_rows,
    relative_str,
    repo_path,
    write_rows,
)


COCO_CAR_LABEL = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-screen VCoR candidate images.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit-per-color", type=int, default=None)
    parser.add_argument("--top-preview-extra", type=int, default=12)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def screen_fieldnames() -> list[str]:
    return [
        "candidate_id",
        "image_id",
        "assigned_true_color",
        "source_path",
        "staged_path",
        "width",
        "height",
        "car_detection_count",
        "primary_car_score",
        "primary_car_area_ratio",
        "secondary_car_area_ratio",
        "sharpness_score",
        "highlight_ratio",
        "shadow_ratio",
        "color_match_ratio",
        "quality_score",
        "auto_recommend_keep",
        "auto_rank_within_color",
        "auto_reason",
    ]


def load_core_counts(config: dict) -> Counter[str]:
    builder_cfg = config.get("dataset_builder", {}) or {}
    base_manifest_csv = repo_path(builder_cfg.get("base_manifest_csv", "data/processed/stanford_cars/final_primary_manifest_strict_colors.csv"))
    exclude_ids = set(builder_cfg.get("core_manual_exclude_ids", []))
    rows = read_rows(base_manifest_csv)
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("include_in_primary_main_analysis") != "yes":
            continue
        if row.get("image_id") in exclude_ids:
            continue
        counts[canonicalize_color(row.get("true_color", ""))] += 1
    return counts


def color_match_ratio(rgb: np.ndarray, target_color: str) -> float:
    rgb_float = rgb.astype(np.float32) / 255.0
    maxc = rgb_float.max(axis=2)
    minc = rgb_float.min(axis=2)
    diff = maxc - minc
    v = maxc
    s = np.where(maxc > 1e-6, diff / np.maximum(maxc, 1e-6), 0.0)
    h = np.zeros_like(maxc)
    mask = diff > 1e-6
    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]
    idx = mask & (maxc == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = mask & (maxc == g)
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2
    idx = mask & (maxc == b)
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4
    h = h / 6.0

    if target_color == "red":
        hit = ((h <= 0.05) | (h >= 0.94)) & (s >= 0.25) & (v >= 0.18)
    elif target_color == "blue":
        hit = (h >= 0.52) & (h <= 0.72) & (s >= 0.18) & (v >= 0.12)
    elif target_color == "green":
        hit = (h >= 0.22) & (h <= 0.48) & (s >= 0.18) & (v >= 0.14)
    elif target_color == "yellow":
        hit = (h >= 0.10) & (h <= 0.19) & (s >= 0.22) & (v >= 0.28)
    elif target_color == "black":
        hit = (v <= 0.30) & (s <= 0.40)
    elif target_color == "white":
        hit = (v >= 0.72) & (s <= 0.18)
    else:
        hit = np.zeros_like(v, dtype=bool)
    return float(hit.mean())


def sharpness_score(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.sqrt(gx * gx + gy * gy)
    return float(np.var(magnitude))


def image_metrics(image: Image.Image, primary_box: np.ndarray | None, target_color: str) -> tuple[float, float, float, float]:
    rgb = np.asarray(image.convert("RGB"))
    if primary_box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in primary_box.tolist()]
        h, w = rgb.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        crop = rgb[y1:y2, x1:x2]
        pad_y = max(1, crop.shape[0] // 12)
        pad_x = max(1, crop.shape[1] // 12)
        if crop.shape[0] > 2 * pad_y and crop.shape[1] > 2 * pad_x:
            crop = crop[pad_y:-pad_y, pad_x:-pad_x]
    else:
        crop = rgb
    gray = crop.mean(axis=2)
    brightness = crop.max(axis=2).astype(np.float32) / 255.0
    highlight_ratio = float((brightness >= 0.97).mean())
    shadow_ratio = float((brightness <= 0.05).mean())
    return sharpness_score(gray), highlight_ratio, shadow_ratio, color_match_ratio(crop, target_color)


def load_detector(device: torch.device):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval().to(device)
    preprocess = transforms.Compose([transforms.ToTensor()])
    return model, preprocess


def car_detections(model, preprocess, image: Image.Image, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tensor = preprocess(image).to(device)
    with torch.inference_mode():
        output = model([tensor])[0]
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    mask = (labels == COCO_CAR_LABEL) & (scores >= 0.25)
    return boxes[mask], scores[mask]


def quality_score_from_metrics(primary_area_ratio: float, primary_score: float, secondary_area_ratio: float, sharpness: float, highlight_ratio: float, shadow_ratio: float, color_ratio: float) -> tuple[float, str]:
    sharpness_term = min(sharpness / 250.0, 1.0)
    score = (
        primary_area_ratio * 2.0
        + primary_score * 0.8
        + color_ratio * 2.5
        + sharpness_term * 0.8
        - secondary_area_ratio * 1.4
        - highlight_ratio * 1.2
        - shadow_ratio * 1.0
    )
    reasons = []
    if primary_area_ratio < 0.18:
        reasons.append("car_too_small")
    if secondary_area_ratio > 0.12:
        reasons.append("secondary_car_interference")
    if highlight_ratio > 0.12:
        reasons.append("highlight_overexposed")
    if shadow_ratio > 0.18:
        reasons.append("too_dark")
    if color_ratio < 0.10:
        reasons.append("weak_color_match")
    if sharpness_term < 0.10:
        reasons.append("possibly_blurry")
    if not reasons:
        reasons.append("passes_auto_screen")
    return float(score), "|".join(reasons)


def draw_contact_sheet(rows: list[dict[str, str]], output_path: Path, title: str) -> None:
    if not rows:
        return
    thumb_w = 240
    thumb_h = 160
    padding = 16
    title_h = 30
    label_h = 42
    cols = 4
    rows_n = math.ceil(len(rows) / cols)
    canvas = Image.new(
        "RGB",
        (cols * (thumb_w + padding) + padding, rows_n * (thumb_h + label_h + padding) + padding + title_h),
        color=(250, 250, 250),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((padding, 8), title, fill=(15, 15, 15), font=font)
    for idx, row in enumerate(rows):
        x = padding + (idx % cols) * (thumb_w + padding)
        y = padding + title_h + (idx // cols) * (thumb_h + label_h + padding)
        image = Image.open(repo_path(row["staged_path"])).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        canvas.paste(image, (x + (thumb_w - image.width) // 2, y))
        label = f"{row['image_id']} q={float(row['quality_score']):.2f}"
        draw.text((x, y + thumb_h + 6), label, fill=(20, 20, 20), font=font)
        draw.text((x, y + thumb_h + 20), row["auto_reason"][:36], fill=(70, 70, 70), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    vcor_cfg = config.get("vcor", {}) or {}
    screen_csv = repo_path(vcor_cfg.get("screen_csv", "data_external/vcor_selected/candidate_auto_screen.csv"))
    summary_csv = repo_path(vcor_cfg.get("screen_summary_csv", "data_external/vcor_selected/candidate_auto_screen_summary.csv"))
    preview_dir = repo_path(vcor_cfg.get("screen_preview_dir", "reports/current/vcor_auto_screen"))
    candidate_review_csv = repo_path(vcor_cfg.get("candidate_review_csv", "data_external/vcor_selected/candidate_review.csv"))
    log_path = args.log_path or repo_path(vcor_cfg.get("screen_log_path", "logs/auto_screen_vcor_candidates.log"))
    ensure_dirs([screen_csv.parent, summary_csv.parent, preview_dir, log_path.parent])
    logger = build_logger("auto_screen_vcor_candidates", log_path)

    review_rows = read_rows(candidate_review_csv)
    allowed_colors = primary_main_analysis_labels(config)
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in review_rows:
        color = canonicalize_color(row.get("assigned_true_color", ""))
        if color in set(allowed_colors):
            grouped_rows[color].append(row)

    if args.limit_per_color is not None:
        for color in allowed_colors:
            grouped_rows[color] = grouped_rows[color][: args.limit_per_color]

    # This environment uses a CPU-only torchvision build, so detection must stay on CPU
    # even though CUDA is available for the language models.
    device = torch.device("cpu")
    logger.info("Loading detector on device=%s", device)
    model, preprocess = load_detector(device)

    screen_rows: list[dict[str, str]] = []
    for color in allowed_colors:
        color_rows = grouped_rows[color]
        logger.info("Screening color=%s rows=%s", color, len(color_rows))
        for idx, row in enumerate(color_rows, start=1):
            image = Image.open(repo_path(row["staged_path"])).convert("RGB")
            width, height = image.size
            boxes, scores = car_detections(model, preprocess, image=image, device=device)
            image_area = float(width * height)
            if len(boxes) > 0:
                areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=np.float32)
                top_idx = int(np.argmax(areas))
                primary_box = boxes[top_idx]
                primary_area_ratio = float(areas[top_idx] / image_area)
                primary_score = float(scores[top_idx])
                secondary_area_ratio = float(max([areas[i] / image_area for i in range(len(areas)) if i != top_idx] + [0.0]))
            else:
                primary_box = None
                primary_area_ratio = 0.0
                primary_score = 0.0
                secondary_area_ratio = 0.0

            sharpness, highlight_ratio, shadow_ratio, match_ratio = image_metrics(image=image, primary_box=primary_box, target_color=color)
            quality_score, auto_reason = quality_score_from_metrics(
                primary_area_ratio=primary_area_ratio,
                primary_score=primary_score,
                secondary_area_ratio=secondary_area_ratio,
                sharpness=sharpness,
                highlight_ratio=highlight_ratio,
                shadow_ratio=shadow_ratio,
                color_ratio=match_ratio,
            )
            screen_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "image_id": row["image_id"],
                    "assigned_true_color": color,
                    "source_path": row["source_path"],
                    "staged_path": row["staged_path"],
                    "width": str(width),
                    "height": str(height),
                    "car_detection_count": str(len(boxes)),
                    "primary_car_score": f"{primary_score:.4f}",
                    "primary_car_area_ratio": f"{primary_area_ratio:.4f}",
                    "secondary_car_area_ratio": f"{secondary_area_ratio:.4f}",
                    "sharpness_score": f"{sharpness:.4f}",
                    "highlight_ratio": f"{highlight_ratio:.4f}",
                    "shadow_ratio": f"{shadow_ratio:.4f}",
                    "color_match_ratio": f"{match_ratio:.4f}",
                    "quality_score": f"{quality_score:.4f}",
                    "auto_recommend_keep": "0",
                    "auto_rank_within_color": "",
                    "auto_reason": auto_reason,
                }
            )
            if idx % 25 == 0:
                logger.info("... color=%s processed=%s/%s", color, idx, len(color_rows))

    core_counts = load_core_counts(config)
    target_per_color = int((config.get("dataset_builder") or {}).get("target_per_color", 50))
    summary_rows: list[dict[str, str]] = []
    by_color_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in screen_rows:
        by_color_rows[row["assigned_true_color"]].append(row)

    for color in allowed_colors:
        rows = sorted(by_color_rows[color], key=lambda item: float(item["quality_score"]), reverse=True)
        keep_n = max(0, target_per_color - int(core_counts.get(color, 0)))
        for rank, row in enumerate(rows, start=1):
            row["auto_rank_within_color"] = str(rank)
            if rank <= keep_n:
                row["auto_recommend_keep"] = "1"
        summary_rows.append(
            {
                "color": color,
                "core_count": str(core_counts.get(color, 0)),
                "target_per_color": str(target_per_color),
                "needed_vcor": str(keep_n),
                "screened_candidates": str(len(rows)),
                "auto_recommended": str(min(keep_n, len(rows))),
            }
        )
        preview_n = min(len(rows), keep_n + args.top_preview_extra)
        draw_contact_sheet(
            rows[:preview_n],
            output_path=preview_dir / f"{color}_top_preview.jpg",
            title=f"{color} | target_keep={keep_n} | preview={preview_n}",
        )

    write_rows(screen_csv, screen_fieldnames(), screen_rows)
    write_rows(summary_csv, ["color", "core_count", "target_per_color", "needed_vcor", "screened_candidates", "auto_recommended"], summary_rows)

    payload = {
        "screen_csv": relative_str(screen_csv),
        "summary_csv": relative_str(summary_csv),
        "preview_dir": relative_str(preview_dir),
        "target_per_color": target_per_color,
        "screened_rows": len(screen_rows),
    }
    logger.info("Auto screening complete: %s", payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
