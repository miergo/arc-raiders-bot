"""MLflow tracing setup and helpers for the RAG pipeline.

WHY THIS MODULE EXISTS:
    MLflow is our observability layer — it records what happens inside the RAG
    pipeline (cache lookups, LLM calls, web searches, validation) so we can
    view traces and metrics in the MLflow UI at http://localhost:5000.

    This module is imported once at startup (from agent.py) and configures
    the MLflow tracking URI and experiment name. All actual tracing calls
    (mlflow.start_span, mlflow.log_metrics) happen in agent.py.

WHY THE CONNECTIVITY CHECK:
    mlflow.set_experiment() makes a blocking HTTP call to the MLflow tracking
    server. If the server isn't running, this call hangs *forever* — which
    would prevent the bot from starting at all. The 2-second timeout check
    below ensures the app always starts, even if someone forgets to run
    ./run_mlflow_ui.sh first. Tracing is best-effort: nice to have, but
    never a blocker.
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error

import mlflow

from arc_raider_bot.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

log = logging.getLogger(__name__)

_initialized = False


def _server_is_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Quick check: can we reach the MLflow server within *timeout* seconds?

    Only needed when the tracking URI is an HTTP URL (i.e. a remote server).
    Local file-based URIs (like ./mlruns) don't need a server.
    """
    if not uri.startswith("http"):
        return True
    try:
        urllib.request.urlopen(f"{uri.rstrip('/')}/version", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError):
        return False


def init_tracing() -> None:
    """Configure MLflow tracking URI and experiment. Safe to call multiple times.

    If the MLflow server is unreachable, tracing is silently disabled so the
    bot can still run normally. Start the MLflow server first for full tracing.
    """
    global _initialized
    if _initialized:
        return

    if not _server_is_reachable(MLFLOW_TRACKING_URI):
        log.warning(
            "MLflow server at %s is not reachable — tracing disabled. "
            "Start it with: cd backend && ./run_mlflow_ui.sh",
            MLFLOW_TRACKING_URI,
        )
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        _initialized = True
        log.info(
            "MLflow tracing enabled (uri=%s, experiment=%s)",
            MLFLOW_TRACKING_URI,
            MLFLOW_EXPERIMENT_NAME,
        )
    except Exception as exc:
        log.warning("MLflow tracing setup failed, tracing disabled: %s", exc)
