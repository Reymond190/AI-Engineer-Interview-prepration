#!/usr/bin/env python3
"""
server/smoke_test.py
--------------------
Verify the local vLLM server is alive and producing coherent output.

Run from project root (inside WSL, conda env activated):
    python server/smoke_test.py [--base-url http://localhost:8000/v1]

Exit code 0 = all checks passed.
"""

import argparse
import json
import os
import sys
import time

import httpx

MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen3.5-9B")

PROMPTS = [
    {
        "name": "health",
        "type": "models_list",
        "expect_model": MODEL,
    },
    {
        "name": "arithmetic",
        "type": "chat",
        "messages": [
            {"role": "user", "content": "What is 7 multiplied by 8? Reply with just the number."},
        ],
        "expect_contains": "56",
    },
    {
        "name": "reasoning",
        "type": "chat",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A train leaves City A at 60 mph. Another leaves City B at 40 mph towards City A. "
                    "The cities are 200 miles apart. When do they meet? Give a numeric answer in hours."
                ),
            },
        ],
        "expect_contains": "2",
    },
]


def check_models(base_url: str, client: httpx.Client) -> dict:
    resp = client.get(f"{base_url}/models")
    resp.raise_for_status()
    data = resp.json()
    ids = [m["id"] for m in data.get("data", [])]
    return {"ok": MODEL in ids, "models": ids}


def check_chat(base_url: str, client: httpx.Client, messages: list, expect: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    resp = client.post(f"{base_url}/chat/completions", json=payload, timeout=120.0)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {
        "ok": expect.lower() in content.lower(),
        "content": content[:300],
        "latency_s": round(elapsed, 2),
        "usage": usage,
    }


def vram_snapshot() -> dict:
    """Read current VRAM use via nvidia-smi (available in WSL)."""
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, free, total = [int(x.strip()) for x in out.split(",")]
        return {"used_mib": used, "free_mib": free, "total_mib": total}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run(base_url: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  DeepSeek Smoke Test  |  {base_url}")
    print(f"{'='*60}\n")

    all_ok = True
    client = httpx.Client()

    for spec in PROMPTS:
        name = spec["name"]
        print(f"[{name}]", end=" ")

        try:
            if spec["type"] == "models_list":
                result = check_models(base_url, client)
            else:
                result = check_chat(base_url, client, spec["messages"], spec["expect_contains"])

            status = "PASS" if result["ok"] else "FAIL"
            if not result["ok"]:
                all_ok = False

            print(status)
            for k, v in result.items():
                if k == "ok":
                    continue
                val_str = json.dumps(v)[:120] if not isinstance(v, str) else v[:120]
                print(f"    {k}: {val_str}")

        except Exception as exc:  # noqa: BLE001
            print("ERROR")
            print(f"    {exc}")
            all_ok = False

        print()

    # VRAM snapshot
    print("[vram]")
    vram = vram_snapshot()
    for k, v in vram.items():
        print(f"    {k}: {v}")
    print()

    print("="*60)
    print("Result:", "ALL PASSED" if all_ok else "SOME FAILED")
    print("="*60)
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the local vLLM server")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Server base URL")
    args = parser.parse_args()
    success = run(args.base_url)
    sys.exit(0 if success else 1)
