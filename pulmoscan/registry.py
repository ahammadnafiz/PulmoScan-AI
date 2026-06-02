"""MLflow Model Registry helpers — the lifecycle layer over experiment tracking.

Training logs *runs*; the registry turns a chosen run's checkpoint into a
versioned, promotable model. Promotion uses **aliases** (``champion`` /
``challenger``) rather than the deprecated stage API, which is the current MLflow
recommendation and lets serving resolve ``models:/<name>@champion`` to whatever
version is live without touching config.

A registered version's source is the run artifact ``runs:/<run_id>/model``, which
contains the self-describing ``model.pt`` — so neither the registry nor the
serving loader needs an MLflow model *flavor*; they download the artifact and the
existing ``torch.load`` path handles the rest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
from mlflow.tracking import MlflowClient

from pulmoscan import logger

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion

# Tag keys we attach to each version so promotion can compare without re-evaluating.
METRIC_TAG_PREFIX = "metric."


def get_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create the registered model if it doesn't exist yet (idempotent)."""
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)
        logger.info(f"Created registered model '{name}'")


def register_version(
    client: MlflowClient,
    *,
    name: str,
    run_id: str,
    artifact_path: str = "model",
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> ModelVersion:
    """Register a run's checkpoint artifact as a new model version.

    Metrics are stored as ``metric.<key>`` version tags so a later promotion run
    can read a candidate's test accuracy directly off the version, with no need to
    locate and re-run its evaluation.
    """
    ensure_registered_model(client, name)
    source = f"runs:/{run_id}/{artifact_path}"
    version = client.create_model_version(name=name, source=source, run_id=run_id)

    for key, value in (metrics or {}).items():
        client.set_model_version_tag(name, version.version, f"{METRIC_TAG_PREFIX}{key}", str(value))
    for key, value in (tags or {}).items():
        client.set_model_version_tag(name, version.version, key, value)

    logger.info(f"Registered '{name}' version {version.version} from run {run_id}")
    return version


def get_alias_version(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    """Return the version an alias points at, or None if the alias is unset."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception:
        return None


def set_alias(client: MlflowClient, name: str, alias: str, version: str | int) -> None:
    client.set_registered_model_alias(name, alias, str(version))
    logger.info(f"Set alias '{alias}' -> '{name}' v{version}")


def get_version_metric(
    client: MlflowClient, name: str, version: str | int, metric_key: str
) -> float | None:
    """Read a metric stored as a ``metric.<key>`` tag on a version."""
    mv = client.get_model_version(name, str(version))
    raw = mv.tags.get(f"{METRIC_TAG_PREFIX}{metric_key}")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def latest_version(client: MlflowClient, name: str) -> ModelVersion | None:
    """The highest version number registered for ``name`` (the newest challenger)."""
    versions = client.search_model_versions(f"name = '{name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))
