"""Interactive CLI for the ARC-RAIDERS WikiBot.

Usage:
    python -m arc_raider_bot.cli
"""

from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from arc_raider_bot.agent import ask


def main() -> None:
    print("=== ARC-RAIDERS WikiBot (type 'quit' to exit) ===\n")
    session_id: str | None = None

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", ":q"):
            print("Bye!")
            break

        print("Thinking...\n")
        try:
            resp, session_id = ask(question, session_id)
            print(f"Answer:\n{resp.answer}\n")
            if resp.sources:
                print("Sources:")
                for url in resp.sources:
                    print(f"  - {url}")
            else:
                print("(no sources returned)")
            print()
        except Exception as exc:
            print(f"Error: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
