#!/usr/bin/env python
"""Generate an image-grouped HTML preview for the 200-row Qwen2-VL round-1 results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from metadata_paths import BASELINE_FINAL_LABELED_CSV, LEGACY_BASELINE_FINAL_LABELED_CSV, resolve_existing_path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = resolve_existing_path(BASELINE_FINAL_LABELED_CSV, LEGACY_BASELINE_FINAL_LABELED_CSV)
DEFAULT_OUTPUT_HTML = ROOT / "reports" / "qwen2vl_main_experiment_round1_preview.html"
LEVEL_ORDER = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
LABEL_CLASS = {
    "faithful": "faithful",
    "hallucination": "hallucination",
    "ambiguous": "ambiguous",
    "needs_manual_review": "manual",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def to_html_rel_path(output_html: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), output_html.parent.resolve())).as_posix()


def label_badge(label: str) -> str:
    css = LABEL_CLASS.get(label, "unknown")
    text = html.escape(label or "unlabeled")
    return f'<span class="badge {css}">{text}</span>'


def build_grouped_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("image_id", "")].append(row)

    grouped: list[dict[str, object]] = []
    for image_id, image_rows in groups.items():
        ordered_rows = sorted(
            image_rows,
            key=lambda row: (LEVEL_ORDER.get(row.get("prompt_level", ""), 99), row.get("sample_id", "")),
        )
        grouped.append(
            {
                "image_id": image_id,
                "file_name": ordered_rows[0].get("file_name", ""),
                "image_path": ordered_rows[0].get("image_path", ""),
                "width": ordered_rows[0].get("width", ""),
                "height": ordered_rows[0].get("height", ""),
                "records": ordered_rows,
                "levels": [row.get("prompt_level", "") for row in ordered_rows],
            }
        )

    def sort_key(group: dict[str, object]) -> tuple[int, str]:
        image_id = str(group.get("image_id", ""))
        return (int(image_id) if image_id.isdigit() else 10**18, image_id)

    return sorted(grouped, key=sort_key)


def generate_html(rows: list[dict[str, str]], output_html: Path) -> str:
    grouped_rows = build_grouped_rows(rows)
    label_counts = Counter(row.get("label", "") or "unlabeled" for row in rows)
    level_counts = Counter(row.get("prompt_level", "") or "unknown" for row in rows)
    complete_groups = sum(1 for group in grouped_rows if set(group["levels"]) == {"S0", "S1", "S2", "S3"})

    group_cards: list[str] = []
    for group in grouped_rows:
        image_path = ROOT / str(group["image_path"])
        image_src = to_html_rel_path(output_html, image_path)
        group_label_counts = Counter((row.get("label", "") or "unlabeled") for row in group["records"])
        group_summary = " / ".join(f"{k}: {v}" for k, v in sorted(group_label_counts.items())) or "-"
        rows_html: list[str] = []
        for row in group["records"]:
            prompt_level = html.escape(row.get("prompt_level", ""))
            label = row.get("label", "") or "unlabeled"
            label_html = label_badge(label)
            sample_id = html.escape(row.get("sample_id", ""))
            prompt_text = html.escape(row.get("prompt_text", ""))
            raw_output = html.escape(row.get("raw_output", ""))
            notes = html.escape(row.get("notes", "")) or "-"
            status = html.escape(row.get("status", "")) or "ok"
            confidence = html.escape(row.get("auto_label_confidence", "")) or "-"
            rows_html.append(
                f"""
                <section class=\"record\" data-level=\"{prompt_level}\" data-label=\"{html.escape(label)}\">
                  <div class=\"record-head\">
                    <div class=\"record-left\">
                      <span class=\"badge level\">{prompt_level}</span>
                      {label_html}
                      <span class=\"sample-id\">{sample_id}</span>
                    </div>
                    <div class=\"record-right\">status: {status} | confidence: {confidence}</div>
                  </div>
                  <div class=\"record-grid\">
                    <div>
                      <div class=\"section-title\">Prompt</div>
                      <div class=\"block prompt\">{prompt_text}</div>
                    </div>
                    <div>
                      <div class=\"section-title\">Raw Output</div>
                      <div class=\"block output\">{raw_output}</div>
                    </div>
                  </div>
                  <div>
                    <div class=\"section-title\">Notes</div>
                    <div class=\"block notes\">{notes}</div>
                  </div>
                </section>
                """
            )

        group_cards.append(
            f"""
            <article class=\"group-card\">
              <div class=\"group-top\">
                <div class=\"thumb-wrap\">
                  <img class=\"thumb\" src=\"{image_src}\" alt=\"{html.escape(str(group.get('file_name', 'image')))}\" loading=\"lazy\">
                </div>
                <div class=\"group-meta\">
                  <div class=\"group-title\">image_id: {html.escape(str(group.get('image_id', '')))}</div>
                  <div class=\"meta-line\">file_name: {html.escape(str(group.get('file_name', '')))}</div>
                  <div class=\"meta-line\">size: {html.escape(str(group.get('width', '')))} x {html.escape(str(group.get('height', '')))}</div>
                  <div class=\"meta-line\">records: {len(group['records'])} | levels: {html.escape(', '.join(group['levels']))}</div>
                  <div class=\"meta-line\">group labels: {html.escape(group_summary)}</div>
                </div>
              </div>
              <div class=\"records-wrap\">
                {''.join(rows_html)}
              </div>
            </article>
            """
        )

    summary_label = " / ".join(f"{k}: {v}" for k, v in sorted(label_counts.items()))
    summary_level = " / ".join(f"{k}: {v}" for k, v in sorted(level_counts.items()))

    return f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Qwen2-VL Round 1 Preview</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #6b7280;
      --line: #ddd6c8;
      --faithful: #0f766e;
      --hallucination: #b42318;
      --ambiguous: #a16207;
      --manual: #475467;
      --level: #1d4ed8;
      --shadow: 0 12px 30px rgba(31, 41, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: \"Microsoft YaHei\", \"PingFang SC\", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(180, 35, 24, 0.08), transparent 24%),
        var(--bg);
      color: var(--ink);
    }}
    .page {{ max-width: 1580px; margin: 0 auto; padding: 28px 24px 48px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(250,245,236,0.95));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 26px 28px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    .subtitle {{ margin: 0; color: var(--muted); line-height: 1.7; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 18px; }}
    .stat {{ background: rgba(255,255,255,0.82); border: 1px solid var(--line); border-radius: 18px; padding: 14px 16px; }}
    .stat .k {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .stat .v {{ font-size: 15px; line-height: 1.6; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 14px 16px;
      margin-bottom: 18px;
      background: rgba(244, 241, 234, 0.92);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    .toolbar label {{ font-size: 14px; color: var(--muted); }}
    .toolbar select {{ min-width: 140px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 10px; background: white; font-size: 14px; }}
    .toolbar .count {{ margin-left: auto; font-size: 14px; color: var(--muted); }}
    .groups {{ display: flex; flex-direction: column; gap: 18px; }}
    .group-card {{ background: var(--panel); border-radius: 22px; border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }}
    .group-top {{ display: grid; grid-template-columns: 320px 1fr; gap: 18px; padding: 18px; border-bottom: 1px solid var(--line); }}
    .thumb-wrap {{ background: #e5e7eb; border-radius: 18px; overflow: hidden; }}
    .thumb {{ width: 100%; height: 260px; object-fit: cover; display: block; }}
    .group-meta {{ display: flex; flex-direction: column; gap: 8px; justify-content: center; }}
    .group-title {{ font-size: 22px; font-weight: 700; }}
    .meta-line {{ color: var(--muted); line-height: 1.7; }}
    .records-wrap {{ display: flex; flex-direction: column; gap: 14px; padding: 18px; }}
    .record {{ border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.82); padding: 14px; }}
    .record-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
    .record-left {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .record-right {{ color: var(--muted); font-size: 13px; }}
    .sample-id {{ font-size: 13px; color: var(--muted); }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
      background: #f3f4f6;
      color: #111827;
    }}
    .badge.faithful {{ background: rgba(15, 118, 110, 0.14); color: var(--faithful); }}
    .badge.hallucination {{ background: rgba(180, 35, 24, 0.12); color: var(--hallucination); }}
    .badge.ambiguous {{ background: rgba(161, 98, 7, 0.14); color: var(--ambiguous); }}
    .badge.manual {{ background: rgba(71, 84, 103, 0.14); color: var(--manual); }}
    .badge.level {{ background: rgba(29, 78, 216, 0.12); color: var(--level); }}
    .record-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
    .section-title {{ font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
    .block {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: rgba(255,255,255,0.86); line-height: 1.7; font-size: 14px; white-space: pre-wrap; word-break: break-word; }}
    .prompt {{ background: rgba(29, 78, 216, 0.05); }}
    .output {{ background: rgba(15, 118, 110, 0.05); }}
    .notes {{ background: rgba(71, 84, 103, 0.05); }}
    .hidden {{ display: none !important; }}
    @media (max-width: 920px) {{
      .group-top {{ grid-template-columns: 1fr; }}
      .record-grid {{ grid-template-columns: 1fr; }}
      .thumb {{ height: 220px; }}
    }}
    @media (max-width: 720px) {{
      .page {{ padding: 18px 12px 28px; }}
      .hero {{ padding: 20px; border-radius: 18px; }}
      h1 {{ font-size: 26px; }}
      .toolbar {{ gap: 8px; }}
      .toolbar .count {{ width: 100%; margin-left: 0; }}
      .record-head {{ flex-direction: column; align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <h1>Qwen2-VL Round 1 Preview</h1>
      <p class=\"subtitle\">当前预览页按 <code>image_id</code> 分组展示。每张图片只出现一次，下面固定列出它对应的 <code>S0-S3</code> 四条实验记录，避免把 50 张唯一图片 x 4 个 prompt 误看成 200 张彼此独立的图片。</p>
      <div class=\"stats\">
        <div class=\"stat\"><div class=\"k\">总记录数</div><div class=\"v\">{len(rows)} 条</div></div>
        <div class=\"stat\"><div class=\"k\">唯一图片数</div><div class=\"v\">{len(grouped_rows)} 张</div></div>
        <div class=\"stat\"><div class=\"k\">完整四级图片组</div><div class=\"v\">{complete_groups} / {len(grouped_rows)}</div></div>
        <div class=\"stat\"><div class=\"k\">Prompt Level 分布</div><div class=\"v\">{html.escape(summary_level)}</div></div>
        <div class=\"stat\"><div class=\"k\">当前标签分布</div><div class=\"v\">{html.escape(summary_label)}</div></div>
      </div>
    </section>

    <section class=\"toolbar\">
      <label>Prompt Level
        <select id=\"levelFilter\">
          <option value=\"all\">全部</option>
          <option value=\"S0\">S0</option>
          <option value=\"S1\">S1</option>
          <option value=\"S2\">S2</option>
          <option value=\"S3\">S3</option>
        </select>
      </label>
      <label>Label
        <select id=\"labelFilter\">
          <option value=\"all\">全部</option>
          <option value=\"faithful\">faithful</option>
          <option value=\"hallucination\">hallucination</option>
          <option value=\"ambiguous\">ambiguous</option>
          <option value=\"needs_manual_review\">needs_manual_review</option>
        </select>
      </label>
      <div class=\"count\" id=\"visibleCount\"></div>
    </section>

    <section class=\"groups\" id=\"groups\">
      {''.join(group_cards)}
    </section>
  </div>

  <script>
    const levelFilter = document.getElementById('levelFilter');
    const labelFilter = document.getElementById('labelFilter');
    const groups = Array.from(document.querySelectorAll('.group-card'));
    const visibleCount = document.getElementById('visibleCount');

    function applyFilters() {{
      const selectedLevel = levelFilter.value;
      const selectedLabel = labelFilter.value;
      let visibleGroups = 0;
      let visibleRecords = 0;

      groups.forEach((group) => {{
        const records = Array.from(group.querySelectorAll('.record'));
        let groupHasVisible = false;
        records.forEach((record) => {{
          const levelOk = selectedLevel === 'all' || record.dataset.level === selectedLevel;
          const labelOk = selectedLabel === 'all' || record.dataset.label === selectedLabel;
          const show = levelOk && labelOk;
          record.classList.toggle('hidden', !show);
          if (show) {{
            groupHasVisible = true;
            visibleRecords += 1;
          }}
        }});
        group.classList.toggle('hidden', !groupHasVisible);
        if (groupHasVisible) {{
          visibleGroups += 1;
        }}
      }});

      visibleCount.textContent = `当前显示 ${'{'}visibleGroups{'}'} 个图片组 / ${'{'}visibleRecords{'}'} 条记录`;
    }}

    levelFilter.addEventListener('change', applyFilters);
    labelFilter.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a grouped HTML preview for the 200-row Qwen2-VL round-1 outputs.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input_csv)
    html_text = generate_html(rows, args.output_html)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_text, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "grouped_images": len(build_grouped_rows(rows)), "output_html": str(args.output_html)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
