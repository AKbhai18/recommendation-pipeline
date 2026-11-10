"""Pipeline Orchestration - structured, reproducible Python workflow.

This is the single entry point that runs the whole pipeline end-to-end with
correct task dependencies:

    generate_data -> ingest -> validate -> prepare
                     -> build_features -> train_and_evaluate

Each stage is a node with declared upstream dependencies. The orchestrator
performs a topological sort, runs the stages in dependency order, times each
stage, and prints a run summary (the "DAG run" the screenshot captures).

The same dependency graph is mirrored in ``dags/recommendation_pipeline_dag.py``
as an Apache Airflow DAG for schedulers that use Airflow; this module lets the
whole thing run with nothing but Python installed, which keeps the project
reproducible on any machine.
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List

from src.features.build_features import build_features
from src.ingestion.generate_data import generate
from src.ingestion.ingest import ingest
from src.preparation.prepare import prepare
from src.training.train import train
from src.utils.common import ensure_dirs, get_logger, load_config, set_seed
from src.validation.validate import validate

LOG = get_logger("orchestrator", log_file="reports/pipeline.log")


class Task:
    """A single pipeline node: a callable with named upstream dependencies."""

    def __init__(self, name: str, fn: Callable, upstream: List[str]):
        self.name = name
        self.fn = fn
        self.upstream = upstream


def _build_dag(cfg: dict) -> Dict[str, Task]:
    """Declare the DAG. Each task returns a dict that downstream tasks read."""
    state: Dict[str, dict] = {}

    def t_generate():
        return generate(cfg)

    def t_ingest():
        state["ingest"] = ingest(cfg)
        return state["ingest"]

    def t_validate():
        state["validate"] = validate(state.get("ingest"), cfg)
        return state["validate"]

    def t_prepare():
        state["prepare"] = prepare(state.get("validate"), cfg)
        return state["prepare"]

    def t_features():
        return build_features(state.get("prepare"), cfg)

    def t_train():
        return train(cfg)

    return {
        "generate_data": Task("generate_data", t_generate, []),
        "ingest": Task("ingest", t_ingest, ["generate_data"]),
        "validate": Task("validate", t_validate, ["ingest"]),
        "prepare": Task("prepare", t_prepare, ["validate"]),
        "build_features": Task("build_features", t_features, ["prepare"]),
        "train_and_evaluate": Task("train_and_evaluate", t_train, ["build_features"]),
    }


def _topological_order(dag: Dict[str, Task]) -> List[str]:
    """Kahn's algorithm -> a valid execution order respecting dependencies."""
    indeg = {name: len(task.upstream) for name, task in dag.items()}
    # map each node to the nodes that depend on it
    downstream: Dict[str, List[str]] = {name: [] for name in dag}
    for name, task in dag.items():
        for up in task.upstream:
            downstream[up].append(name)

    queue = deque(sorted(n for n, d in indeg.items() if d == 0))
    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in downstream[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(dag):
        raise RuntimeError("Cycle detected in pipeline DAG")
    return order


def run_pipeline(config: dict | None = None) -> dict:
    cfg = config or load_config()
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg["paths"]["raw"], cfg["paths"]["processed"],
                cfg["paths"]["features"], cfg["paths"]["errors"],
                cfg["paths"]["models"], cfg["paths"]["reports"])

    dag = _build_dag(cfg)
    order = _topological_order(dag)

    LOG.info("=" * 68)
    LOG.info("PIPELINE RUN STARTED  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    LOG.info("Execution order: %s", " -> ".join(order))
    LOG.info("=" * 68)

    timings = []
    for name in order:
        LOG.info(">>> STAGE START: %s", name)
        t0 = time.time()
        dag[name].fn()
        dt = time.time() - t0
        timings.append((name, dt))
        LOG.info("<<< STAGE DONE : %s  (%.2fs)", name, dt)

    total = sum(dt for _, dt in timings)
    LOG.info("=" * 68)
    LOG.info("PIPELINE RUN SUCCEEDED  (total %.2fs)", total)
    for name, dt in timings:
        LOG.info("   %-20s %6.2fs   [SUCCESS]", name, dt)
    LOG.info("=" * 68)

    return {"order": order, "timings": timings, "total_seconds": total}


if __name__ == "__main__":
    run_pipeline()
