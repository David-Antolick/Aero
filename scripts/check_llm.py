#!/usr/bin/env python3
"""
Minimal readiness check for an OpenAI-compatible LLM endpoint.

- Tries a /v1/models list to confirm the server responds.
- Sends a tiny chat.completions request to verify inference works.
- Exits non-zero on failure so you can wire it into CI.
"""

import os
import sys
import time
from typing import Any, Optional, Sequence

from openai import OpenAI
from openai._exceptions import OpenAIError

from dotenv import load_dotenv
load_dotenv()  # Load .env file if present


def _normalize_content(raw: Any) -> str:
    """Best-effort conversion of OpenAI response content to plain text."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        parts: list[str] = []
        for item in raw:
            text: str | None = None
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)
    return str(raw)


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        print(f"[ERR] Missing required env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def main() -> None:
    # Config (env-first, with sensible defaults for your setup)
    base_url = get_env("AERO_LLM_BASE_URL")
    api_key = get_env("AERO_LLM_API_KEY")  # most local servers accept any token
    model = get_env("AERO_LLM_MODEL")
    timeout_s = float(get_env("AERO_LLM_TIMEOUT_S", "20"))
    retries = int(get_env("AERO_LLM_RETRIES", "3"))
    retry_delay_s = float(get_env("AERO_LLM_RETRY_DELAY_S", "1.5"))

    print(f"[INFO] Base URL: {base_url}")
    print(f"[INFO] Model: {model}")
    print("[INFO] Starting endpoint checks...")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 1) Model listing check
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Checking /v1/models (attempt {attempt}/{retries})...")
            models = client.models.list()
            ids = [m.id for m in models.data]
            print(f"[OK] Models available: {ids}")
            if model not in ids:
                print(f"[WARN] Requested model '{model}' not found in list. "
                      f"Proceeding to inference anyway.")
            break
        except OpenAIError as e:
            print(f"[WARN] Models list failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Unexpected error listing models: {e}", file=sys.stderr)
        if attempt == retries:
            print("[ERR] Could not reach /v1/models successfully.", file=sys.stderr)
            sys.exit(3)
        time.sleep(retry_delay_s)

    # 2) Inference check (minimal chat completion)
    prompt = "Hello! respoind with 'OK' to confirm endpoint works."
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Sending test completion (attempt {attempt}/{retries})...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
                timeout=timeout_s,  # propagated by SDK
            )
            choice = resp.choices[0]
            message = choice.message if choice else None
            content = _normalize_content(getattr(message, "content", None)).strip()
            if not content:
                print("[WARN] Model response had no text content; raw message:"
                      f" {message!r}", file=sys.stderr)
            print(f"[OK] Model response: {content!r}")
            if content.upper() != "OK":
                print("[WARN] Model did not echo 'OK'. Endpoint works but content differs.")
            print("[SUCCESS] Endpoint is responsive and serving completions.")
            sys.exit(0)
        except OpenAIError as e:
            print(f"[WARN] Chat completion failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Unexpected completion error: {e}", file=sys.stderr)

        if attempt < retries:
            time.sleep(retry_delay_s)

    print("[ERR] Endpoint did not complete successfully after retries.", file=sys.stderr)
    sys.exit(4)


if __name__ == "__main__":
    main()
