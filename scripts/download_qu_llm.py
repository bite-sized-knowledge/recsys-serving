"""Query Understanding LLM(GGUF) 다운로드.

Qwen2.5-0.5B-Instruct Q4_0 (~400MB)을 HuggingFace에서 다운로드한다.
산출물: models/qwen2.5-0.5b-instruct-gguf/qwen2.5-0.5b-instruct-q4_0.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_0.gguf"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b-instruct-gguf"


def main() -> int:
    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    target = DEFAULT_DIR / FILENAME
    if target.exists():
        size_mb = target.stat().st_size / 1024 / 1024
        print(f"Already downloaded: {target} ({size_mb:.1f} MB)")
        return 0

    print(f"Downloading {REPO_ID}/{FILENAME} → {DEFAULT_DIR}")
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(DEFAULT_DIR))
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
