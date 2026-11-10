#!/usr/bin/env python3
"""
server/webchat.py
-----------------
Lightweight browser chat UI (Gradio) for the local vLLM server.
The model's <think> reasoning is shown in a collapsible block above
each answer and is left out of the history sent back to the model.

Run from project root (inside WSL, conda env activated), with
server/serve.sh already running:
    python server/webchat.py [--base-url http://localhost:8000/v1] [--port 7860]

Then open http://localhost:7860 in a Windows browser.
"""

import argparse
import os

import gradio as gr
from gradio import ChatMessage
from openai import OpenAI

MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen3.5-9B")
THINK_TITLE = "Thinking"


def text_of(content) -> str:
    """Gradio message content may be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def to_api_history(history: list) -> list[dict]:
    """Convert Gradio history to OpenAI messages, dropping thinking blocks."""
    messages = []
    for msg in history:
        meta = msg.get("metadata") or {}
        if meta.get("title") == THINK_TITLE:
            continue
        messages.append({"role": msg["role"], "content": text_of(msg["content"])})
    return messages


def split_reasoning(raw: str) -> tuple[str, str, bool]:
    """Return (thinking, answer, thinking_done) from raw streamed text."""
    raw = raw.replace("<think>", "", 1).lstrip()
    if "</think>" in raw:
        thinking, answer = raw.split("</think>", 1)
        return thinking.strip(), answer.strip(), True
    return raw.strip(), "", False


def make_chat_fn(client: OpenAI):
    def chat(message: str, history: list, temperature: float, max_tokens: int):
        messages = to_api_history(history) + [{"role": "user", "content": message}]
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
            stream=True,
        )

        raw = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            raw += chunk.choices[0].delta.content or ""
            thinking, answer, done = split_reasoning(raw)
            out = [
                ChatMessage(
                    role="assistant",
                    content=thinking or "...",
                    metadata={"title": THINK_TITLE, "status": "done" if done else "pending"},
                )
            ]
            if answer:
                out.append(ChatMessage(role="assistant", content=answer))
            yield out

    return chat


def main():
    parser = argparse.ArgumentParser(description="Web chat UI for the local DeepSeek server")
    parser.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="unused")

    demo = gr.ChatInterface(
        fn=make_chat_fn(client),
        title=MODEL,
        description=f"Local vLLM server at {args.base_url}",
        additional_inputs=[
            gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="Temperature"),
            gr.Slider(256, 7168, value=2048, step=256, label="Max tokens"),
        ],
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
