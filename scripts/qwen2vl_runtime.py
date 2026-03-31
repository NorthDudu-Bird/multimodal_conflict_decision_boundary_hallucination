#!/usr/bin/env python
"""Shared Qwen2-VL runtime helpers for smoke tests and batch experiments."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

from _local_deps import ensure_local_deps

ensure_local_deps()

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen2_vl_7b"
DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
RESULT_FIELDS = [
    "sample_id",
    "image_id",
    "file_name",
    "image_path",
    "prompt_level",
    "prompt_text",
    "model_name",
    "raw_output",
    "label",
    "language_consistent",
    "vision_consistent",
    "ambiguous",
    "notes",
    "status",
    "error",
]


def build_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def read_rows(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit is not None:
        rows = rows[:limit]
    return rows


def read_completed_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row.get("sample_id", "") for row in csv.DictReader(fh) if row.get("sample_id")}


def append_result(output_csv: Path, row: dict[str, str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    with output_csv.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def chunked(rows: list[dict[str, str]], batch_size: int) -> Iterable[list[dict[str, str]]]:
    for idx in range(0, len(rows), batch_size):
        yield rows[idx : idx + batch_size]


class Qwen2VLRunner:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        use_4bit: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.processor = None
        self.model = None
        self.load_mode = ""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        return resolved

    def _build_generate_kwargs(self, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["temperature"] = None
            generate_kwargs["top_p"] = None
            generate_kwargs["top_k"] = None
        return generate_kwargs

    def load(self, logger: logging.Logger | None = None) -> None:
        if self.model is not None and self.processor is not None:
            return

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        quantization_config = None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            self.load_mode = "4bit_nf4"
        else:
            self.load_mode = "bf16_or_fp16"

        if logger:
            logger.info("Loading model from %s with mode=%s", self.model_dir, self.load_mode)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_dir, use_fast=False)
        if logger:
            logger.info("Model and processor loaded successfully.")

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model has not been loaded.")

        chat_messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        image_handles: list[Image.Image] = []
        try:
            for message in messages:
                raw_content = message.get("content", [])
                if isinstance(raw_content, str):
                    raw_content = [{"type": "text", "text": raw_content}]

                content_items: list[dict[str, str]] = []
                for item in raw_content:
                    item_type = item.get("type")
                    if item_type == "text":
                        content_items.append({"type": "text", "text": item["text"]})
                        continue
                    if item_type in {"image", "image_path"}:
                        image_path = item.get("image_path") or item.get("path") or item.get("image")
                        if not image_path:
                            raise ValueError("Image content item is missing an image path.")
                        resolved_path = self._resolve_image_path(str(image_path))
                        image = Image.open(resolved_path).convert("RGB")
                        image_handles.append(image)
                        images.append(image)
                        content_items.append({"type": "image"})
                        continue
                    raise ValueError(f"Unsupported content type: {item_type}")

                chat_messages.append(
                    {
                        "role": message["role"],
                        "content": content_items,
                    }
                )

            text = self.processor.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            processor_kwargs: dict[str, Any] = {
                "text": [text],
                "return_tensors": "pt",
                "padding": True,
            }
            if images:
                processor_kwargs["images"] = images

            inputs = self.processor(**processor_kwargs)
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(
                **inputs,
                **self._build_generate_kwargs(max_new_tokens=max_new_tokens, temperature=temperature),
            )
            trimmed_ids = generated_ids[:, inputs.input_ids.shape[1] :]
            outputs = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return outputs[0]
        finally:
            for image in image_handles:
                image.close()

    def generate_batch(
        self,
        batch_rows: list[dict[str, str]],
        max_new_tokens: int = 96,
        temperature: float = 0.0,
    ) -> list[str]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model has not been loaded.")

        texts: list[str] = []
        images: list[Image.Image] = []
        image_handles: list[Image.Image] = []
        try:
            for row in batch_rows:
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = ROOT / image_path
                image = Image.open(image_path).convert("RGB")
                image_handles.append(image)
                images.append(image)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": row["prompt_text"]},
                        ],
                    }
                ]
                texts.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(
                **inputs,
                **self._build_generate_kwargs(max_new_tokens=max_new_tokens, temperature=temperature),
            )
            trimmed_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            return self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        finally:
            for image in image_handles:
                image.close()


def make_result_row(source_row: dict[str, str], model_name: str, raw_output: str, status: str = "ok", error: str = "") -> dict[str, str]:
    return {
        "sample_id": source_row.get("sample_id", ""),
        "image_id": source_row.get("image_id", ""),
        "file_name": source_row.get("file_name", ""),
        "image_path": source_row.get("image_path", ""),
        "prompt_level": source_row.get("prompt_level", ""),
        "prompt_text": source_row.get("prompt_text", ""),
        "model_name": model_name,
        "raw_output": raw_output,
        "label": "",
        "language_consistent": "",
        "vision_consistent": "",
        "ambiguous": "",
        "notes": source_row.get("notes", ""),
        "status": status,
        "error": error,
    }
