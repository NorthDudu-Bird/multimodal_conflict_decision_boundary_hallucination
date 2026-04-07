#!/usr/bin/env python
"""Shared full-precision runtime helpers for the multimodel v2 experiment."""

from __future__ import annotations

import csv
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils._local_deps import ensure_local_deps

ensure_local_deps()

import torch
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)


ROOT = REPO_ROOT
RESULT_FIELDS = [
    "sample_id",
    "image_id",
    "file_name",
    "image_path",
    "condition_name",
    "condition_family",
    "prompt_template_version",
    "prompt_text",
    "model_key",
    "model_name",
    "checkpoint_name",
    "precision",
    "device",
    "device_map",
    "batch_size",
    "elapsed_seconds",
    "raw_output",
    "status",
    "error",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def choose_torch_dtype(precision: str) -> torch.dtype:
    precision = str(precision or "auto").strip().lower()
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "float16":
        return torch.float16
    if precision == "float32":
        return torch.float32
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    return mapping.get(dtype, str(dtype))


def resolve_output_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_model_dir(local_dir: str | Path) -> Path:
    path = Path(local_dir)
    if path.is_absolute():
        return path
    return ROOT / path


def ensure_model_downloaded(checkpoint_name: str, model_dir: Path, logger: logging.Logger) -> Path:
    has_config = (model_dir / "config.json").exists()
    has_weights = bool(list(model_dir.glob("*.safetensors"))) or bool(list(model_dir.glob("*.bin")))
    if model_dir.exists() and has_config and has_weights:
        logger.info("Reusing existing model directory: %s", model_dir)
        return model_dir

    logger.info("Downloading checkpoint %s into %s", checkpoint_name, model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=checkpoint_name,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return model_dir


@dataclass
class ModelSpec:
    model_key: str
    model_type: str
    model_name: str
    checkpoint_name: str
    local_dir: Path
    precision: str = "auto"
    batch_size: int = 1
    max_new_tokens: int = 8
    extra: dict[str, Any] = field(default_factory=dict)


def model_spec_from_config(config: dict, model_key: str) -> ModelSpec:
    for model_cfg in config.get("models", []):
        if model_cfg.get("model_key") != model_key:
            continue
        extra = dict(model_cfg)
        for key in ["model_key", "model_type", "model_name", "checkpoint_name", "local_dir", "precision", "batch_size", "max_new_tokens"]:
            extra.pop(key, None)
        return ModelSpec(
            model_key=model_cfg["model_key"],
            model_type=model_cfg["model_type"],
            model_name=model_cfg["model_name"],
            checkpoint_name=model_cfg["checkpoint_name"],
            local_dir=resolve_model_dir(model_cfg["local_dir"]),
            precision=model_cfg.get("precision", "auto"),
            batch_size=int(model_cfg.get("batch_size", 1)),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 8)),
            extra=extra,
        )
    raise KeyError(f"Unknown model_key in config: {model_key}")


class BaseVLMRunner:
    def __init__(self, spec: ModelSpec, offload_dir: Path) -> None:
        self.spec = spec
        self.offload_dir = offload_dir
        self.output_device = resolve_output_device()
        self.torch_dtype = choose_torch_dtype(spec.precision)
        self.device_map = "auto" if torch.cuda.is_available() else "cpu"
        self.model = None

    @property
    def batch_size(self) -> int:
        return int(self.spec.batch_size)

    @property
    def device_description(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        return "cuda:0+cpu_offload" if self.device_map == "auto" else "cuda:0"

    def generation_config(self, max_new_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        config: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            config["temperature"] = temperature
            config["top_p"] = top_p
        return config

    def load(self, logger: logging.Logger) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        for attr in ["processor", "tokenizer", "model"]:
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def metadata(self) -> dict[str, str]:
        return {
            "model_key": self.spec.model_key,
            "model_name": self.spec.model_name,
            "checkpoint_name": self.spec.checkpoint_name,
            "precision": dtype_name(self.torch_dtype),
            "device": self.device_description,
            "device_map": str(self.device_map),
            "batch_size": str(self.batch_size),
        }

    def generate_one(self, row: dict[str, str], max_new_tokens: int, temperature: float, top_p: float) -> str:
        raise NotImplementedError


class Qwen2VLRunner(BaseVLMRunner):
    def load(self, logger: logging.Logger) -> None:
        if self.model is not None:
            return

        ensure_model_downloaded(self.spec.checkpoint_name, self.spec.local_dir, logger=logger)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading Qwen2-VL from %s with dtype=%s", self.spec.local_dir, dtype_name(self.torch_dtype))
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.spec.local_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            offload_folder=str(self.offload_dir),
        )
        self.processor = AutoProcessor.from_pretrained(self.spec.local_dir, use_fast=False)

    def generate_one(self, row: dict[str, str], max_new_tokens: int, temperature: float, top_p: float) -> str:
        image_path = resolve_model_dir(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["prompt_text"]},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            inputs = inputs.to(self.output_device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_config(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p),
            )
            trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
            outputs = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return outputs[0].strip()
        finally:
            image.close()


class LlavaRunner(BaseVLMRunner):
    def load(self, logger: logging.Logger) -> None:
        if self.model is not None:
            return

        ensure_model_downloaded(self.spec.checkpoint_name, self.spec.local_dir, logger=logger)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading LLaVA-1.5 from %s with dtype=%s", self.spec.local_dir, dtype_name(self.torch_dtype))
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.spec.local_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            offload_folder=str(self.offload_dir),
        )
        self.processor = AutoProcessor.from_pretrained(self.spec.local_dir)

    def generate_one(self, row: dict[str, str], max_new_tokens: int, temperature: float, top_p: float) -> str:
        image_path = resolve_model_dir(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["prompt_text"]},
                        {"type": "image"},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=[image], text=[prompt], return_tensors="pt", padding=True)
            inputs = inputs.to(self.output_device, self.torch_dtype)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_config(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p),
            )
            trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
            outputs = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return outputs[0].strip()
        finally:
            image.close()


class InternVL2Runner(BaseVLMRunner):
    def __init__(self, spec: ModelSpec, offload_dir: Path) -> None:
        super().__init__(spec=spec, offload_dir=offload_dir)
        self.max_num_tiles = int(spec.extra.get("max_num_tiles", 1))
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def load(self, logger: logging.Logger) -> None:
        if self.model is not None:
            return

        ensure_model_downloaded(self.spec.checkpoint_name, self.spec.local_dir, logger=logger)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading InternVL2 from %s with dtype=%s", self.spec.local_dir, dtype_name(self.torch_dtype))
        self.model = AutoModel.from_pretrained(
            self.spec.local_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
            device_map=self.device_map,
            offload_folder=str(self.offload_dir),
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.local_dir, trust_remote_code=True, use_fast=False)

    def load_image_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        try:
            pixel_values = [self.transform(image)]
            return torch.stack(pixel_values).to(self.output_device, dtype=self.torch_dtype)
        finally:
            image.close()

    def generate_one(self, row: dict[str, str], max_new_tokens: int, temperature: float, top_p: float) -> str:
        image_path = resolve_model_dir(row["image_path"])
        pixel_values = self.load_image_tensor(image_path)
        question = f"<image>\n{row['prompt_text']}"
        generation_config = self.generation_config(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=False,
        )
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()


def create_runner(spec: ModelSpec, offload_dir: Path) -> BaseVLMRunner:
    if spec.model_type == "qwen2_vl":
        return Qwen2VLRunner(spec=spec, offload_dir=offload_dir)
    if spec.model_type == "llava":
        return LlavaRunner(spec=spec, offload_dir=offload_dir)
    if spec.model_type == "internvl2":
        return InternVL2Runner(spec=spec, offload_dir=offload_dir)
    raise KeyError(f"Unsupported model_type: {spec.model_type}")


def make_result_row(
    source_row: dict[str, str],
    runner: BaseVLMRunner,
    raw_output: str,
    elapsed_seconds: float,
    status: str = "ok",
    error: str = "",
) -> dict[str, str]:
    metadata = runner.metadata()
    return {
        "sample_id": source_row.get("sample_id", ""),
        "image_id": source_row.get("image_id", ""),
        "file_name": source_row.get("file_name", ""),
        "image_path": source_row.get("image_path", ""),
        "condition_name": source_row.get("condition_name", ""),
        "condition_family": source_row.get("condition_family", ""),
        "prompt_template_version": source_row.get("prompt_template_version", ""),
        "prompt_text": source_row.get("prompt_text", ""),
        "model_key": metadata["model_key"],
        "model_name": metadata["model_name"],
        "checkpoint_name": metadata["checkpoint_name"],
        "precision": metadata["precision"],
        "device": metadata["device"],
        "device_map": metadata["device_map"],
        "batch_size": metadata["batch_size"],
        "elapsed_seconds": f"{elapsed_seconds:.4f}",
        "raw_output": raw_output,
        "status": status,
        "error": error,
    }


def timed_generation(runner: BaseVLMRunner, row: dict[str, str], max_new_tokens: int, temperature: float, top_p: float) -> tuple[str, float]:
    started = time.perf_counter()
    output = runner.generate_one(row=row, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    elapsed = time.perf_counter() - started
    return output, elapsed
