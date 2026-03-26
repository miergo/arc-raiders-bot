"""Evaluation harness — run the golden dataset through the RAG pipeline and
measure quality.

=============================================================================
WHAT THIS SCRIPT DOES:
    1. Loads test questions from eval_dataset.json
    2. Sends each question through the full ask() pipeline
    3. Measures quality metrics for each answer:
       - keyword_recall: what fraction of expected keywords appear in the answer
       - latency: how long the full pipeline took (seconds)
       - source_count: how many source URLs were returned
       - answer_length: character count of the answer
    4. Logs everything to MLflow as a named "evaluation run" so you can compare
       different configurations (models, prompts, thresholds) side by side
    5. Prints a summary table to the terminal

WHY THIS MATTERS:
    Without systematic evaluation, every change to the bot (new prompt, different
    model, adjusted threshold) is a guess. This script gives you numbers to
    compare. For example:
      - Run eval with CACHE_SIMILARITY_THRESHOLD=0.75 → avg recall 0.72
      - Change to 0.60 → run eval again → avg recall 0.68
      - Now you KNOW 0.75 was better, instead of guessing.

    In MLOps this is called an "offline evaluation" — testing against a fixed
    dataset before deploying changes to production.

USAGE:
    cd backend
    python run_eval.py                         # run all questions
    python run_eval.py --max-questions 5       # quick test with 5 questions
    VERBOSE_TOOL_CALLS=1 python run_eval.py    # with debug output

VIEWING RESULTS:
    Open http://localhost:5000 (MLflow UI) → click the arc-raiders-rag
    experiment → you'll see evaluation runs with all metrics logged.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import argparse
from pathlib import Path

# Ensure the backend directory is on the Python path so imports work
# when running this script directly with `python run_eval.py`.
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import mlflow

from arc_raider_bot.agent import ask
from arc_raider_bot.config import (
    OLLAMA_MODEL,
    CACHE_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
)
from arc_raider_bot.prompts import MAIN_AGENT_PROMPT


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    """What fraction of expected keywords appear in the answer?

    WHY THIS METRIC:
        A good RAG answer should mention the key concepts from the question.
        If we ask about "escalation levels" and expect ["Alert", "Lockdown",
        "Annihilation"], an answer that mentions all 3 gets recall=1.0.
        An answer mentioning only "Alert" gets recall=0.33.

        This is a simple but effective proxy for answer relevance without
        needing a judge LLM (which would add cost and latency).
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def prompt_version_hash(prompt_text: str) -> str:
    """Short hash of the prompt text to track which prompt version was used.

    WHY: When you change prompts.py and re-run eval, this hash changes
    automatically, making it easy to see which prompt produced which results
    in MLflow without manually versioning prompts.
    """
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


def run_evaluation(questions: list[dict]) -> list[dict]:
    """Run each question through the pipeline and collect per-question metrics."""
    results = []

    for i, entry in enumerate(questions, 1):
        qid = entry["id"]
        question = entry["question"]
        expected_kw = entry.get("expected_keywords", [])
        category = entry.get("category", "unknown")

        print(f"\n[{i}/{len(questions)}] {qid}: {question}")

        t0 = time.perf_counter()
        try:
            response, _sid = ask(question)
            latency = time.perf_counter() - t0
            recall = compute_keyword_recall(response.answer, expected_kw)
            error = None
        except Exception as exc:
            latency = time.perf_counter() - t0
            recall = 0.0
            response = None
            error = str(exc)
            print(f"  ERROR: {error}")

        result = {
            "id": qid,
            "question": question,
            "category": category,
            "answer": response.answer if response else "",
            "sources": response.sources if response else [],
            "keyword_recall": recall,
            "latency_s": round(latency, 2),
            "source_count": len(response.sources) if response else 0,
            "answer_length": len(response.answer) if response else 0,
            "error": error,
        }
        results.append(result)

        # Print a quick per-question summary.
        print(f"  recall={recall:.0%}  latency={latency:.1f}s  "
              f"sources={result['source_count']}  len={result['answer_length']}")

    return results


def log_to_mlflow(results: list[dict]) -> None:
    """Log the evaluation run to MLflow for comparison in the UI.

    WHY LOG TO MLFLOW:
        Each eval run becomes a "run" in MLflow with parameters (model,
        threshold, prompt hash) and aggregate metrics (avg recall, avg latency).
        When you change a config and re-run, you get a new row in MLflow and
        can compare side by side. This is the core of MLOps experimentation.
    """
    with mlflow.start_run(run_name=f"eval-{OLLAMA_MODEL}"):
        # Log configuration as parameters — these identify WHAT was tested.
        mlflow.log_params({
            "model": OLLAMA_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "similarity_threshold": CACHE_SIMILARITY_THRESHOLD,
            "prompt_version": prompt_version_hash(MAIN_AGENT_PROMPT),
            "dataset_size": len(results),
        })

        # Log aggregate metrics — these measure HOW WELL it performed.
        recalls = [r["keyword_recall"] for r in results]
        latencies = [r["latency_s"] for r in results]
        source_counts = [r["source_count"] for r in results]
        error_count = sum(1 for r in results if r["error"])

        mlflow.log_metrics({
            "avg_keyword_recall": round(sum(recalls) / len(recalls), 3),
            "min_keyword_recall": round(min(recalls), 3),
            "avg_latency_s": round(sum(latencies) / len(latencies), 2),
            "max_latency_s": round(max(latencies), 2),
            "avg_source_count": round(sum(source_counts) / len(source_counts), 1),
            "error_count": error_count,
        })

        # Log the full results as a JSON artifact for detailed inspection.
        results_path = Path("eval_results.json")
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        mlflow.log_artifact(str(results_path))
        results_path.unlink()

    print("\nResults logged to MLflow. View at http://localhost:5000")


def print_summary(results: list[dict]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 72)
    print(f"{'ID':<25} {'Recall':>7} {'Latency':>8} {'Sources':>8} {'Len':>6}")
    print("-" * 72)

    for r in results:
        status = "ERROR" if r["error"] else f"{r['keyword_recall']:.0%}"
        print(f"{r['id']:<25} {status:>7} {r['latency_s']:>7.1f}s "
              f"{r['source_count']:>8} {r['answer_length']:>6}")

    print("-" * 72)
    ok = [r for r in results if not r["error"]]
    if ok:
        avg_recall = sum(r["keyword_recall"] for r in ok) / len(ok)
        avg_latency = sum(r["latency_s"] for r in ok) / len(ok)
        print(f"{'AVERAGE':<25} {avg_recall:>6.0%} {avg_latency:>7.1f}s")
    errors = len(results) - len(ok)
    if errors:
        print(f"\n{errors} question(s) failed with errors.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ARC Raiders WikiBot")
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Limit number of questions to run (useful for quick tests)",
    )
    parser.add_argument(
        "--dataset", type=str,
        default=str(Path(__file__).resolve().parent / "eval_dataset.json"),
        help="Path to evaluation dataset JSON file",
    )
    args = parser.parse_args()

    print("Loading evaluation dataset...")
    questions = load_dataset(args.dataset)
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"Running {len(questions)} questions through the pipeline...")
    results = run_evaluation(questions)

    print_summary(results)
    log_to_mlflow(results)


if __name__ == "__main__":
    main()
