#!/usr/bin/env python
"""Check whether the local cv_proj environment can run Qwen2-VL models."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _local_deps import LOCAL_DEPS, ensure_local_deps


ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "env_check_cv_proj.txt"
REPORT_PATH = ROOT / "reports" / "vlm_env_check_report.md"


PACKAGE_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "pillow": "PIL",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "bitsandbytes": "bitsandbytes",
    "sentencepiece": "sentencepiece",
    "qwen_vl_utils": "qwen_vl_utils",
    "huggingface_hub": "huggingface_hub",
}


@dataclass
class PackageStatus:
    name: str
    import_name: str
    ok: bool
    version: str
    detail: str


def collect_package_status() -> list[PackageStatus]:
    results: list[PackageStatus] = []
    for name, import_name in PACKAGE_IMPORTS.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            module_path = getattr(module, "__file__", "built-in")
            results.append(PackageStatus(name, import_name, True, str(version), str(module_path)))
        except Exception as exc:
            results.append(PackageStatus(name, import_name, False, "", f"{type(exc).__name__}: {exc}"))
    return results


def collect_torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_import_ok": False,
        "torch_version": "",
        "cuda_available": False,
        "cuda_version": "",
        "device_count": 0,
        "gpu_name": "",
        "gpu_total_memory_gb": "",
        "bf16_supported": False,
    }
    try:
        import torch

        info["torch_import_ok"] = True
        info["torch_version"] = torch.__version__
        info["torch_module_path"] = getattr(torch, "__file__", "")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = str(torch.version.cuda)
        info["device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
            if hasattr(torch.cuda, "is_bf16_supported"):
                info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def run_nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "nvidia-smi not found in PATH."
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return output.strip() or f"nvidia-smi returned code {result.returncode} with no output."


def estimate_7b_feasibility(packages: list[PackageStatus], torch_info: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    pkg_map = {pkg.name: pkg for pkg in packages}
    mem = float(torch_info.get("gpu_total_memory_gb") or 0)
    cuda_available = bool(torch_info.get("cuda_available"))

    if not cuda_available:
        return "不适合直接运行 7B", ["CUDA 不可用，无法进行高效 GPU 推理。"]

    if mem >= 20:
        reasons.append(f"显存约 {mem} GB，7B 多模态模型以 BF16/FP16 运行更有把握。")
    elif mem >= 12:
        reasons.append(f"显存约 {mem} GB，7B 更适合采用 4-bit 量化或 CPU offload。")
    else:
        reasons.append(f"显存约 {mem} GB，7B 在本机上风险较高。")

    missing_runtime = [
        name
        for name in ("transformers", "accelerate", "safetensors")
        if not pkg_map.get(name, PackageStatus(name, name, False, "", "missing")).ok
    ]
    if missing_runtime:
        reasons.append("核心推理依赖缺失：" + ", ".join(missing_runtime))

    bnb_pkg = pkg_map.get("bitsandbytes")
    if bnb_pkg and bnb_pkg.ok:
        reasons.append("bitsandbytes 可导入，具备尝试 4-bit/8-bit 量化的基础条件。")
    else:
        reasons.append("bitsandbytes 当前不可用，12GB 显存下直接跑 7B 半精度成功率较低。")

    if missing_runtime:
        return "需要先修复依赖后再评估 7B", reasons
    if mem >= 20:
        return "适合直接尝试 7B", reasons
    if mem >= 12 and bnb_pkg and bnb_pkg.ok:
        return "适合优先尝试 7B（建议 4-bit）", reasons
    if mem >= 12:
        return "可尝试 7B，但需量化或 offload 支持", reasons
    return "更适合先降级到 2B 或增加量化/卸载方案", reasons


def format_log(packages: list[PackageStatus], torch_info: dict[str, Any], feasibility: str, reasons: list[str], nvidia_smi: str, overlay_path: str | None) -> str:
    lines: list[str] = []
    lines.append("=== cv_proj Environment Check ===")
    lines.append(f"Python executable: {sys.executable}")
    lines.append(f"Python version: {sys.version}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Working directory: {ROOT}")
    lines.append(f"Local dependency overlay: {overlay_path or 'not enabled'}")
    lines.append("")
    lines.append("=== Torch / CUDA ===")
    for key, value in torch_info.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("=== Packages ===")
    for pkg in packages:
        status = "OK" if pkg.ok else "MISSING"
        detail = pkg.version if pkg.ok else pkg.detail
        extra = pkg.detail if pkg.ok else ""
        suffix = f" | source={extra}" if extra else ""
        lines.append(f"{pkg.name}: {status} | {detail}{suffix}")
    lines.append("")
    lines.append("=== 7B Feasibility ===")
    lines.append(f"summary: {feasibility}")
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("=== nvidia-smi ===")
    lines.append(nvidia_smi)
    return "\n".join(lines) + "\n"


def format_report(packages: list[PackageStatus], torch_info: dict[str, Any], feasibility: str, reasons: list[str], overlay_path: str | None) -> str:
    ok_packages = [pkg for pkg in packages if pkg.ok]
    missing_packages = [pkg for pkg in packages if not pkg.ok]
    lines: list[str] = []
    lines.append("# VLM Environment Check Report")
    lines.append("")
    lines.append("## 基础环境")
    lines.append(f"- Python: `{sys.version.split()[0]}`")
    lines.append(f"- 平台: `{platform.platform()}`")
    lines.append(f"- Torch: `{torch_info.get('torch_version', 'N/A')}`")
    lines.append(f"- Torch 来源: `{torch_info.get('torch_module_path', 'N/A')}`")
    lines.append(f"- CUDA 可用: `{torch_info.get('cuda_available', False)}`")
    lines.append(f"- CUDA 版本: `{torch_info.get('cuda_version', 'N/A')}`")
    lines.append(f"- GPU: `{torch_info.get('gpu_name', 'N/A')}`")
    lines.append(f"- 显存(GB): `{torch_info.get('gpu_total_memory_gb', 'N/A')}`")
    lines.append(f"- BF16 支持: `{torch_info.get('bf16_supported', False)}`")
    lines.append(f"- 本地依赖覆盖层: `{overlay_path or '未启用'}`")
    lines.append("")
    lines.append("## 依赖检查")
    for pkg in ok_packages:
        lines.append(f"- 已安装 `{pkg.name}`: `{pkg.version}`，来源 `{pkg.detail}`")
    for pkg in missing_packages:
        lines.append(f"- 缺失 `{pkg.name}`: `{pkg.detail}`")
    lines.append("")
    lines.append("## 7B 可运行性评估")
    lines.append(f"- 结论: `{feasibility}`")
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## 建议")
    lines.append("- 若继续部署 Qwen2-VL-7B-Instruct，优先尝试 4-bit 量化。")
    lines.append("- 若 Windows 下 bitsandbytes 无法稳定工作，可改为 CPU offload 或退到 Qwen2-VL-2B-Instruct。")
    return "\n".join(lines) + "\n"


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    overlay_path = ensure_local_deps()
    packages = collect_package_status()
    torch_info = collect_torch_info()
    feasibility, reasons = estimate_7b_feasibility(packages, torch_info)
    nvidia_smi = run_nvidia_smi()

    LOG_PATH.write_text(
        format_log(packages, torch_info, feasibility, reasons, nvidia_smi, overlay_path),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        format_report(packages, torch_info, feasibility, reasons, overlay_path),
        encoding="utf-8",
    )
    print(f"Environment log written to: {LOG_PATH}")
    print(f"Environment report written to: {REPORT_PATH}")
    print(f"7B assessment: {feasibility}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
