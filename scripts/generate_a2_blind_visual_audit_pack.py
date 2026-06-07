#!/usr/bin/env python
"""Generate a blinded manual-review pack for the A2 visual-validity audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "results" / "audit" / "visual_clarity_audit_manifest_completed.csv"
DEFAULT_OUTPUT = REPO_ROOT / "deliverables" / "pending review" / "A2_human_visual_audit_pack"
SEED = 20260601

SENSITIVE_COLUMNS = [
    "review_id",
    "image_file",
    "audit_row_id",
    "target_row_id",
    "sample_id",
    "image_id",
    "image_path",
    "model",
    "condition",
    "robustness_variant",
    "audit_source_module",
    "true_color",
    "false_prompt_color",
    "model_output",
    "source_dataset",
    "is_conflict_aligned",
    "is_faithful",
    "audit_group",
    "parsed_label",
    "match_strategy",
]

PUBLIC_COLUMNS = [
    "review_id",
    "image_file",
    "review_status",
    "body_color_visible",
    "audit_visual_clarity",
    "audit_body_color_salience",
    "audit_specular_reflection",
    "audit_shadow_or_night_effect",
    "audit_background_color_bias",
    "audit_multi_car_interference",
    "audit_occlusion",
    "include_in_validity_analysis",
    "audit_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_blinded_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        converted = image.convert("RGB")
        converted.save(destination, format="JPEG", quality=94, optimize=True)


def make_contact_sheets(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[Path] = []
    per_page = 12
    thumb_w, thumb_h = 280, 190
    label_h = 38
    cols = 3
    cell_w, cell_h = thumb_w, thumb_h + label_h
    for page_idx in range(0, len(rows), per_page):
        page_rows = rows[page_idx : page_idx + per_page]
        page = Image.new("RGB", (cols * cell_w, 4 * cell_h), "white")
        draw = ImageDraw.Draw(page)
        for idx, row in enumerate(page_rows):
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            image_path = output_dir.parent / row["image_file"]
            with Image.open(image_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((thumb_w - 4, thumb_h - 4))
                canvas = Image.new("RGB", (thumb_w, thumb_h), "#f7f7f7")
                canvas.paste(thumb, ((thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))
            page.paste(ImageOps.expand(canvas, border=1, fill="#cccccc"), (x, y))
            draw.text((x + 8, y + thumb_h + 8), row["review_id"], fill="#111111")
        page_path = output_dir / f"contact_sheet_{page_idx // per_page + 1:02d}.jpg"
        page.save(page_path, format="JPEG", quality=92, optimize=True)
        page_paths.append(page_path)
    return page_paths


def write_gallery(rows: list[dict[str, str]], output_path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(row['image_file'])}" alt="{html.escape(row['review_id'])}">
              <h2>{html.escape(row['review_id'])}</h2>
            </article>
            """
        )
    output_path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A2 blinded visual audit gallery</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #172033; }
    h1 { margin-bottom: 4px; }
    p { color: #4b5563; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 18px; }
    .card { border: 1px solid #d7dce5; border-radius: 8px; padding: 10px; background: #fff; }
    img { width: 100%; height: 185px; object-fit: contain; background: #f5f6f8; }
    h2 { margin: 8px 2px 0; font-size: 16px; }
  </style>
</head>
<body>
  <h1>A2 blinded visual audit gallery</h1>
  <p>Use the matching review ID in the workbook. This gallery intentionally hides experimental group, prompt, model output, dataset source, and previous annotations.</p>
  <section class="grid">
"""
        + "\n".join(cards)
        + """
  </section>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_readme(output_path: Path, row_count: int, unique_image_count: int, contact_sheets: list[Path]) -> None:
    output_path.write_text(
        f"""# A2 Independent Human Visual Audit Pack

## Purpose

This package supports an independent manual review of the visual validity of the sampled car-colour images. Do not use previous annotations while completing the review.

## Reviewer Workflow

1. Open `A2_visual_audit_manual_review.xlsx`.
2. Fill reviewer metadata in the `Protocol` sheet.
3. Review each blinded image using `blind_gallery.html` or the files in `images/`.
4. Complete every row in `Manual_Audit`.
5. Save the completed workbook under a new filename and return it for unblinding and summary regeneration.

## Blinding

- Public review rows: {row_count}
- Unique underlying source images: {unique_image_count}
- Contact sheets: {len(contact_sheets)}
- Fixed shuffle seed: `{SEED}`

The review workbook intentionally hides target/control status, pairing, original filenames, prompts, model outputs, dataset sources, and old annotations.

## Independence Language

- If the reviewer did not participate in the original auto-filled audit or manuscript analysis, describe the result as an `independent human visual audit`.
- If an author completes the sheet, describe it as an `author-conducted manual visual audit`.

## Important

Do not open `DO_NOT_OPEN_UNTIL_REVIEW_COMPLETE/` until the workbook has been completed and frozen. That folder contains the unblinding map.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source_rows = read_csv(args.source)
    if len(source_rows) != 84:
        raise RuntimeError(f"Expected 84 source rows, found {len(source_rows)}")

    rng = random.Random(args.seed)
    indexed_rows = list(enumerate(source_rows))
    rng.shuffle(indexed_rows)

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    private_dir = output_dir / "DO_NOT_OPEN_UNTIL_REVIEW_COMPLETE"
    contact_dir = output_dir / "blind_contact_sheets"
    build_dir = output_dir / "_build"
    for path in [output_dir, images_dir, private_dir, contact_dir, build_dir]:
        path.mkdir(parents=True, exist_ok=True)

    public_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []
    image_hash_rows: list[dict[str, str]] = []
    for position, (_, source_row) in enumerate(indexed_rows, start=1):
        review_id = f"VC-{position:03d}"
        image_file = f"images/{review_id}.jpg"
        source_image = REPO_ROOT / source_row["image_path"]
        if not source_image.exists():
            raise FileNotFoundError(source_image)
        blinded_image = output_dir / image_file
        save_blinded_image(source_image, blinded_image)

        public_row = {column: "" for column in PUBLIC_COLUMNS}
        public_row["review_id"] = review_id
        public_row["image_file"] = image_file
        public_rows.append(public_row)

        mapping_row = {"review_id": review_id, "image_file": image_file}
        for column in SENSITIVE_COLUMNS:
            if column not in mapping_row:
                mapping_row[column] = source_row.get(column, "")
        mapping_rows.append(mapping_row)
        image_hash_rows.append(
            {
                "review_id": review_id,
                "image_file": image_file,
                "sha256": sha256(blinded_image),
            }
        )

    write_csv(output_dir / "A2_visual_audit_manual_review.csv", public_rows, PUBLIC_COLUMNS)
    write_csv(private_dir / "blind_mapping.csv", mapping_rows, SENSITIVE_COLUMNS)
    write_csv(private_dir / "blinded_image_sha256.csv", image_hash_rows, ["review_id", "image_file", "sha256"])
    (build_dir / "audit_rows.json").write_text(json.dumps(public_rows, indent=2), encoding="utf-8")
    contact_sheets = make_contact_sheets(public_rows, contact_dir)
    write_gallery(public_rows, output_dir / "blind_gallery.html")
    write_readme(
        output_dir / "README.md",
        row_count=len(public_rows),
        unique_image_count=len({row["image_id"] for row in source_rows}),
        contact_sheets=contact_sheets,
    )
    payload = {
        "output_dir": str(output_dir),
        "rows": len(public_rows),
        "unique_source_images": len({row["image_id"] for row in source_rows}),
        "contact_sheets": [str(path) for path in contact_sheets],
        "manual_csv": str(output_dir / "A2_visual_audit_manual_review.csv"),
        "mapping_csv": str(private_dir / "blind_mapping.csv"),
        "missing_images": 0,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

