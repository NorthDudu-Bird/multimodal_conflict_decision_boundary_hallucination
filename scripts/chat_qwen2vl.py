#!/usr/bin/env python
"""Small interactive chat script for the local Qwen2-VL model."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen2vl_runtime import DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME, Qwen2VLRunner


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM_PROMPT = "你是一个严谨的多模态助手。请基于图像和文本真实作答，不要编造图像中不存在的内容。"
HELP_TEXT = """
Commands:
  /image <path>   Attach an image to the next user message.
  /image clear    Clear the pending image.
  /clear          Clear the conversation history.
  /status         Show current chat status.
  /help           Show this help message.
  /quit           Exit the chat.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the local Qwen2-VL-7B-Instruct model.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", help="Run a single prompt and exit instead of starting interactive mode.")
    parser.add_argument("--image", help="Optional image path for one-shot mode or as the initial pending image.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading. Not recommended on 12 GB VRAM.")
    return parser.parse_args()


def resolve_image_arg(image_arg: str) -> Path:
    candidate = Path(image_arg).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    root_candidate = ROOT / candidate
    if root_candidate.exists():
        return root_candidate.resolve()

    raise FileNotFoundError(f"Image not found: {image_arg}")


def build_initial_history(system_prompt: str) -> list[dict]:
    history: list[dict] = []
    if system_prompt:
        history.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    return history


def build_user_message(text: str, image_path: Path | None = None) -> dict:
    content: list[dict[str, str]] = []
    if image_path is not None:
        content.append({"type": "image_path", "image_path": str(image_path)})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def build_assistant_message(text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }


def print_status(runner: Qwen2VLRunner, history: list[dict], pending_image: Path | None) -> None:
    print(f"model_dir: {runner.model_dir}")
    print(f"model_name: {runner.model_name}")
    print(f"load_mode: {runner.load_mode}")
    print(f"history_messages: {len(history)}")
    print(f"pending_image: {pending_image if pending_image else 'None'}")


def main() -> int:
    args = parse_args()
    runner = Qwen2VLRunner(
        model_dir=args.model_dir,
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
    )
    print(f"Loading {args.model_name} from {args.model_dir} ...")
    runner.load()
    print(f"Model loaded. mode={runner.load_mode}")

    initial_pending_image = resolve_image_arg(args.image) if args.image else None
    history = build_initial_history(args.system_prompt)

    if args.prompt:
        history.append(build_user_message(args.prompt, initial_pending_image))
        reply = runner.generate_from_messages(
            history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(reply)
        return 0

    pending_image = initial_pending_image
    print("Interactive chat started.")
    print(HELP_TEXT)
    if pending_image:
        print(f"Initial pending image: {pending_image}")

    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print("\nExiting chat.")
            return 0
        except KeyboardInterrupt:
            print("\nExiting chat.")
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit"}:
            print("Bye.")
            return 0
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/clear":
            history = build_initial_history(args.system_prompt)
            pending_image = None
            print("Conversation history cleared.")
            continue
        if user_input == "/status":
            print_status(runner, history, pending_image)
            continue
        if user_input.startswith("/image "):
            image_arg = user_input[len("/image ") :].strip()
            if image_arg.lower() in {"clear", "none"}:
                pending_image = None
                print("Pending image cleared.")
                continue
            pending_image = resolve_image_arg(image_arg)
            print(f"Pending image set to: {pending_image}")
            continue

        history.append(build_user_message(user_input, pending_image))
        try:
            reply = runner.generate_from_messages(
                history,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:
            history.pop()
            print(f"[error] {type(exc).__name__}: {exc}")
            continue

        print(f"Assistant> {reply}\n")
        history.append(build_assistant_message(reply))
        pending_image = None


if __name__ == "__main__":
    raise SystemExit(main())
