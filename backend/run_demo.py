"""Quick smoke test — runs one fixed ARC-RAIDERS question and prints the result.

Usage:
    VERBOSE_TOOL_CALLS=1 python run_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

os.environ.setdefault("VERBOSE_TOOL_CALLS", "1")

from arc_raider_bot.agent import ask  # noqa: E402

DEMO_QUESTION = "Is the Ferro better to use on Stella Montis or Blue Gate?"


def main() -> None:
    print(f"Demo question: {DEMO_QUESTION!r}\n")
    resp, session_id = ask(DEMO_QUESTION)

    print("=" * 60)
    print(f"Session: {session_id}")
    print("Structured output (ArcRaidersResponse):")
    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
    print("=" * 60)

    print(f"\nAnswer:\n{resp.answer}\n")
    print(f"Sources ({len(resp.sources)}):")
    for url in resp.sources:
        print(f"  - {url}")


if __name__ == "__main__":
    main()
