#!/usr/bin/env python
"""Prepare a COCO val2017 no-dog subset for existence-conflict experiments.

This script is intentionally written to be rerunnable:
- It reuses existing downloads and extracted folders when available.
- It records detailed progress to logs/prepare_dataset.log.
- It generates metadata, copied sample images, HTML preview, contact sheet,
  and a short markdown report for the experiment workflow.
"""

from __future__ import annotations

import csv
import html
import json
import logging
import random
import shutil
import sys
import traceback
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from PIL import Image, ImageOps, ImageDraw
except ImportError:  # pragma: no cover - graceful degradation
    Image = None
    ImageOps = None
    ImageDraw = None


VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
RANDOM_SEED = 42
SAMPLE_SIZE = 50
MIN_DIMENSION = 300
MAX_ANNOTATIONS = 15


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
COCO_DIR = DATA_DIR / "coco"
VAL2017_DIR = COCO_DIR / "val2017"
ANNOTATIONS_DIR = COCO_DIR / "annotations"
SELECTED_DIR = DATA_DIR / "selected_images" / "no_dog_sample_50"
PREVIEWS_DIR = DATA_DIR / "previews"
METADATA_DIR = DATA_DIR / "metadata"
SAMPLES_DIR = METADATA_DIR / "samples"
PROMPTS_DIR = METADATA_DIR / "prompts"
OUTPUTS_RAW_DIR = METADATA_DIR / "outputs_raw"
OUTPUTS_LABELED_DIR = METADATA_DIR / "outputs_labeled"
ANALYSIS_DIR = METADATA_DIR / "analysis"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

VAL2017_ZIP = RAW_DIR / "val2017.zip"
ANNOTATIONS_ZIP = RAW_DIR / "annotations_trainval2017.zip"
INSTANCES_JSON = ANNOTATIONS_DIR / "instances_val2017.json"

NO_DOG_ALL_CSV = SAMPLES_DIR / "no_dog_all.csv"
NO_DOG_FILTERED_CSV = SAMPLES_DIR / "no_dog_filtered_candidates.csv"
NO_DOG_SAMPLE_CSV = SAMPLES_DIR / "no_dog_sample_50.csv"
PREVIEW_HTML = REPORTS_DIR / "no_dog_sample_50_preview.html"
CONTACT_SHEET = PREVIEWS_DIR / "no_dog_sample_50_contact_sheet.jpg"
REPORT_MD = REPORTS_DIR / "dataset_preparation_report.md"
LOG_FILE = LOGS_DIR / "prepare_dataset.log"


def ensure_directories() -> None:
    for path in [
        RAW_DIR,
        VAL2017_DIR.parent,
        SELECTED_DIR,
        PREVIEWS_DIR,
        METADATA_DIR,
        SAMPLES_DIR,
        PROMPTS_DIR,
        OUTPUTS_RAW_DIR,
        OUTPUTS_LABELED_DIR,
        ANALYSIS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger("prepare_no_dog_subset")
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
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def download_file(url: str, destination: Path, logger: logging.Logger) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Reusing existing download: %s", relative_str(destination))
        return

    logger.info("Downloading %s -> %s", url, relative_str(destination))

    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(
            f"\rDownloading {destination.name}: {percent:6.2f}% "
            f"({downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB)"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, destination, reporthook=reporthook)
        sys.stdout.write("\n")
    except Exception:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise

    logger.info("Downloaded file: %s (%.2f MB)", relative_str(destination), destination.stat().st_size / (1024 * 1024))


def extract_zip(zip_path: Path, expected_path: Path, logger: logging.Logger) -> None:
    if expected_path.exists():
        logger.info("Reusing existing extracted content: %s", relative_str(expected_path))
        return

    logger.info("Extracting %s", relative_str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(COCO_DIR if zip_path == VAL2017_ZIP else DATA_DIR / "coco")
    logger.info("Extraction complete: %s", relative_str(expected_path))


def load_instances(logger: logging.Logger) -> Dict:
    logger.info("Loading annotations JSON: %s", relative_str(INSTANCES_JSON))
    with INSTANCES_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict], logger: logging.Logger) -> int:
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    logger.info("Wrote CSV with %s rows: %s", count, relative_str(path))
    return count


def build_metadata(dataset: Dict, logger: logging.Logger) -> Dict[str, object]:
    categories = dataset["categories"]
    images = dataset["images"]
    annotations = dataset["annotations"]

    dog_category = next((cat for cat in categories if cat["name"] == "dog"), None)
    if not dog_category:
        raise RuntimeError("Could not find category named 'dog' in COCO categories.")
    dog_category_id = dog_category["id"]
    logger.info("Detected dog category_id: %s", dog_category_id)

    image_info = {img["id"]: img for img in images}
    annotation_counts = Counter()
    dog_image_ids = set()
    for ann in annotations:
        image_id = ann["image_id"]
        annotation_counts[image_id] += 1
        if ann["category_id"] == dog_category_id:
            dog_image_ids.add(image_id)

    all_image_ids = set(image_info.keys())
    no_dog_image_ids = sorted(all_image_ids - dog_image_ids)
    filtered_candidates = []
    no_dog_rows = []

    for image_id in no_dog_image_ids:
        info = image_info[image_id]
        row = {
            "image_id": image_id,
            "file_name": info["file_name"],
            "image_path": relative_str(VAL2017_DIR / info["file_name"]),
            "width": info["width"],
            "height": info["height"],
            "contains_dog": 0,
            "num_annotations": annotation_counts[image_id],
        }
        no_dog_rows.append(row)

        if (
            info["width"] >= MIN_DIMENSION
            and info["height"] >= MIN_DIMENSION
            and annotation_counts[image_id] <= MAX_ANNOTATIONS
        ):
            filtered_candidates.append(row)

    if len(filtered_candidates) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only found {len(filtered_candidates)} filtered candidates, fewer than required sample size {SAMPLE_SIZE}."
        )

    rng = random.Random(RANDOM_SEED)
    sampled_rows = sorted(rng.sample(filtered_candidates, SAMPLE_SIZE), key=lambda item: item["image_id"])

    return {
        "dog_category_id": dog_category_id,
        "dog_image_count": len(dog_image_ids),
        "no_dog_rows": no_dog_rows,
        "no_dog_count": len(no_dog_rows),
        "filtered_candidates": filtered_candidates,
        "filtered_count": len(filtered_candidates),
        "sampled_rows": sampled_rows,
    }


def clean_selected_directory(logger: logging.Logger) -> None:
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    for existing in SELECTED_DIR.iterdir():
        if existing.is_file():
            existing.unlink()
    logger.info("Prepared selected image directory: %s", relative_str(SELECTED_DIR))


def copy_sample_images(sample_rows: List[Dict], logger: logging.Logger) -> None:
    clean_selected_directory(logger)
    for row in sample_rows:
        source = ROOT / row["image_path"]
        destination = SELECTED_DIR / row["file_name"]
        shutil.copy2(source, destination)
    logger.info("Copied %s sample images into %s", len(sample_rows), relative_str(SELECTED_DIR))


def build_sample_rows(sampled_rows: List[Dict]) -> List[Dict]:
    result = []
    for row in sampled_rows:
        item = dict(row)
        item["image_path"] = relative_str(SELECTED_DIR / row["file_name"])
        item.update(
            {
                "prompt_level": "",
                "prompt_text": "",
                "model_name": "",
                "model_output": "",
                "label": "",
                "notes": "",
            }
        )
        result.append(item)
    return result


def generate_preview_html(sample_rows: List[Dict], logger: logging.Logger) -> None:
    cards = []
    for row in sample_rows:
        img_rel = Path("..") / row["image_path"]
        cards.append(
            f"""
            <div class="card">
              <img src="{html.escape(img_rel.as_posix())}" alt="{html.escape(row['file_name'])}">
              <div class="meta">
                <div><strong>image_id</strong>: {row['image_id']}</div>
                <div><strong>file_name</strong>: {html.escape(row['file_name'])}</div>
                <div><strong>size</strong>: {row['width']} x {row['height']}</div>
                <div><strong>num_annotations</strong>: {row['num_annotations']}</div>
                <div><strong>人工建议</strong>: [ ] 保留  [ ] 剔除</div>
                <div><strong>备注</strong>: ____________________</div>
              </div>
            </div>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>No-Dog Sample 50 Preview</title>
  <style>
    body {{
      font-family: "Microsoft YaHei", sans-serif;
      margin: 24px;
      background: #f6f7f9;
      color: #222;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: white;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
      overflow: hidden;
      border: 1px solid #e5e7eb;
    }}
    .card img {{
      width: 100%;
      height: 220px;
      object-fit: cover;
      background: #ddd;
      display: block;
    }}
    .meta {{
      padding: 12px;
      font-size: 14px;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <h1>COCO val2017 No-Dog Sample 50 Preview</h1>
  <p>用于存在性冲突实验的第一版候选图像。请结合缩略图进行人工复核。</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    PREVIEW_HTML.write_text(html_text, encoding="utf-8")
    logger.info("Generated HTML preview: %s", relative_str(PREVIEW_HTML))


def generate_contact_sheet(sample_rows: List[Dict], logger: logging.Logger) -> Optional[str]:
    if Image is None or ImageOps is None or ImageDraw is None:
        logger.warning("Pillow is not available; skipping contact sheet generation.")
        return "Pillow unavailable, contact sheet not generated."

    thumb_w = 220
    thumb_h = 220
    columns = 5
    rows = (len(sample_rows) + columns - 1) // columns
    padding = 18
    label_h = 36
    canvas_w = padding + columns * (thumb_w + padding)
    canvas_h = padding + rows * (thumb_h + label_h + padding)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 246, 248))
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(sample_rows):
        x = padding + (index % columns) * (thumb_w + padding)
        y = padding + (index // columns) * (thumb_h + label_h + padding)
        src = ROOT / row["image_path"]
        with Image.open(src) as img:
            thumb = ImageOps.fit(img.convert("RGB"), (thumb_w, thumb_h))
            canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb_h + 8), f"{row['image_id']} | ann={row['num_annotations']}", fill=(32, 32, 32))

    canvas.save(CONTACT_SHEET, quality=90)
    logger.info("Generated contact sheet: %s", relative_str(CONTACT_SHEET))
    return None


def generate_report(stats: Dict[str, object], download_notes: Dict[str, str], contact_sheet_note: Optional[str], logger: logging.Logger) -> None:
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = f"""# Dataset Preparation Report

## 1. 数据来源
- 数据集：COCO 2017 `val2017`
- 图像下载地址：`{VAL2017_URL}`
- 标注下载地址：`{ANNOTATIONS_URL}`
- 处理时间：`{utc_now}`

## 2. 下载文件信息
- `data/raw/val2017.zip`：{download_notes['val_zip']}
- `data/raw/annotations_trainval2017.zip`：{download_notes['ann_zip']}
- 解压图像目录：`data/coco/val2017/`
- 解压标注目录：`data/coco/annotations/`

## 3. dog 的 category_id
- `dog` 的 category_id = **{stats['dog_category_id']}**

## 4. 含 dog 图片数量
- 含 `dog` 标注的图片数量：**{stats['dog_image_count']}**

## 5. 不含 dog 图片数量
- 不含 `dog` 标注的图片数量：**{stats['no_dog_count']}**
- 完整清单：`data/metadata/samples/no_dog_all.csv`

## 6. 经过轻量筛选后的候选数量
- 轻量筛选条件：
  - 宽度 >= {MIN_DIMENSION}
  - 高度 >= {MIN_DIMENSION}
  - `num_annotations` <= {MAX_ANNOTATIONS}
- 通过轻量筛选的候选数量：**{stats['filtered_count']}**
- 候选清单：`data/metadata/samples/no_dog_filtered_candidates.csv`

## 7. 最终抽样 50 张的说明
- 抽样来源：轻量筛选后的候选池
- 抽样数量：**{SAMPLE_SIZE}**
- 随机种子：**{RANDOM_SEED}**
- 样本元数据：`data/metadata/samples/no_dog_sample_50.csv`
- 样本图片目录：`data/selected_images/no_dog_sample_50/`
- HTML 预览页：`reports/no_dog_sample_50_preview.html`
- Contact sheet：`data/previews/no_dog_sample_50_contact_sheet.jpg`

## 8. 当前自动筛选的局限性
- 当前仅做了基础存在性过滤：通过 COCO 标注判断图片中是否存在 `dog`。
- 当前仅做了轻量质量控制：尺寸阈值和标注数量阈值。
- 未自动检测模糊、遮挡、极端裁切、强反光、文字干扰、复杂拥挤背景等情况。
- COCO 标注并不保证“视觉上绝对不存在狗”，只保证“无 `dog` 类别标注”，因此仍需人工二次复核。
- {contact_sheet_note or "已生成 contact sheet，便于人工快速目检。"}

## 9. 建议的人工复核步骤
1. 打开 `reports/no_dog_sample_50_preview.html`，逐张确认图像中确实不存在狗。
2. 优先剔除背景过于复杂、主体不清晰、难以构造稳定冲突提示的图像。
3. 在 `data/metadata/samples/no_dog_sample_50.csv` 的 `notes` 列中记录保留/剔除理由。
4. 若发现隐藏狗、疑似狗、玩具狗、卡通狗或局部狗元素，建议直接剔除。
5. 完成人工复核后，冻结一版“最终实验样本清单”，避免后续实验中样本漂移。
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    logger.info("Generated markdown report: %s", relative_str(REPORT_MD))


def main() -> int:
    logger = setup_logging()
    logger.info("Starting COCO no-dog subset preparation.")
    ensure_directories()

    try:
        download_file(VAL2017_URL, VAL2017_ZIP, logger)
        download_file(ANNOTATIONS_URL, ANNOTATIONS_ZIP, logger)

        extract_zip(VAL2017_ZIP, VAL2017_DIR, logger)
        extract_zip(ANNOTATIONS_ZIP, ANNOTATIONS_DIR, logger)

        dataset = load_instances(logger)
        stats = build_metadata(dataset, logger)

        base_fields = [
            "image_id",
            "file_name",
            "image_path",
            "width",
            "height",
            "contains_dog",
            "num_annotations",
        ]
        sample_fields = base_fields + [
            "prompt_level",
            "prompt_text",
            "model_name",
            "model_output",
            "label",
            "notes",
        ]

        write_csv(NO_DOG_ALL_CSV, base_fields, stats["no_dog_rows"], logger)
        write_csv(NO_DOG_FILTERED_CSV, base_fields, stats["filtered_candidates"], logger)

        sample_rows = build_sample_rows(stats["sampled_rows"])
        write_csv(NO_DOG_SAMPLE_CSV, sample_fields, sample_rows, logger)

        copy_sample_images(stats["sampled_rows"], logger)
        generate_preview_html(sample_rows, logger)
        contact_sheet_note = generate_contact_sheet(sample_rows, logger)

        download_notes = {
            "val_zip": f"size={VAL2017_ZIP.stat().st_size / (1024 * 1024):.2f} MB",
            "ann_zip": f"size={ANNOTATIONS_ZIP.stat().st_size / (1024 * 1024):.2f} MB",
        }
        generate_report(stats, download_notes, contact_sheet_note, logger)

        logger.info("Preparation finished successfully.")
        return 0
    except Exception as exc:
        logger.error("Preparation failed: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
