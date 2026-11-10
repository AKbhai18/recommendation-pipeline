"""Apache Airflow DAG for the recommendation data-management pipeline.

Drop this file into your Airflow ``dags/`` folder (or point ``dags_folder`` at
this directory) to schedule the pipeline. It mirrors exactly the dependency
graph implemented in ``src/orchestrate.py``:

    generate_data >> ingest >> validate >> prepare
                  >> build_features >> train_and_evaluate

Each Airflow task is a thin ``PythonOperator`` wrapper around the corresponding
pipeline stage function, so Airflow and the plain-Python orchestrator run the
identical code. The project root is added to ``sys.path`` so the ``src``
package imports resolve regardless of where Airflow launches the task.

This file is import-safe without Airflow installed: if Airflow is absent it
simply does not register a DAG (the plain-Python orchestrator still runs the
pipeline), which keeps the repository runnable on a machine without Airflow.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# Make the project's ``src`` package importable inside Airflow workers.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except Exception:  # Airflow not installed -> skip DAG registration.
    _AIRFLOW_AVAILABLE = False


def _stage_generate(**_):
    from src.ingestion.generate_data import generate
    return generate()


def _stage_ingest(**_):
    from src.ingestion.ingest import ingest
    ingest()


def _stage_validate(**_):
    from src.validation.validate import validate
    validate()


def _stage_prepare(**_):
    from src.preparation.prepare import prepare
    prepare()


def _stage_features(**_):
    from src.features.build_features import build_features
    build_features()


def _stage_train(**_):
    from src.training.train import train
    train()


if _AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "data-eng",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    }

    with DAG(
        dag_id="recommendation_pipeline",
        description="End-to-end recommendation data-management pipeline",
        default_args=default_args,
        start_date=datetime(2024, 1, 1),
        schedule_interval="@daily",   # run daily; also triggerable manually
        catchup=False,
        tags=["recsys", "dm4ml"],
    ) as dag:

        generate_data = PythonOperator(
            task_id="generate_data", python_callable=_stage_generate)
        ingest = PythonOperator(
            task_id="ingest", python_callable=_stage_ingest)
        validate = PythonOperator(
            task_id="validate", python_callable=_stage_validate)
        prepare = PythonOperator(
            task_id="prepare", python_callable=_stage_prepare)
        build_features = PythonOperator(
            task_id="build_features", python_callable=_stage_features)
        train_and_evaluate = PythonOperator(
            task_id="train_and_evaluate", python_callable=_stage_train)

        # Task dependencies (the DAG edges).
        generate_data >> ingest >> validate >> prepare
        prepare >> build_features >> train_and_evaluate
