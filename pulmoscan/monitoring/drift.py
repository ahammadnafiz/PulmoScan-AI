"""Evidently drift detection over the monitoring feature space.

Evidently's public API changed substantially between the 0.4.x line
(``evidently.report.Report`` + ``evidently.metric_preset``) and 0.6+
(``evidently.Report`` + ``evidently.presets``). This module hides that behind a
single ``run_drift_report`` so the rest of the codebase — and the tests — don't
care which is installed. It also keeps the summary extraction tolerant of the
two slightly different result schemas.

Drift is measured on tabular columns: the image-quality features
(``FEATURE_COLUMNS``) plus the model's own ``confidence`` / ``entropy`` and the
categorical ``predicted_label`` (prediction-distribution drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pulmoscan.monitoring.features import FEATURE_COLUMNS

if TYPE_CHECKING:
    import pandas as pd

# Numeric columns Evidently should treat as continuous distributions.
NUMERIC_COLUMNS: list[str] = [*FEATURE_COLUMNS, "confidence", "entropy"]
# Categorical column for prediction-distribution drift.
CATEGORICAL_COLUMNS: list[str] = ["predicted_label"]


@dataclass(frozen=True)
class DriftSummary:
    dataset_drift: bool
    drift_share: float
    n_drifted_columns: int
    n_columns: int
    drifted_columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_drift": self.dataset_drift,
            "drift_share": self.drift_share,
            "n_drifted_columns": self.n_drifted_columns,
            "n_columns": self.n_columns,
            "drifted_columns": self.drifted_columns,
        }


def _align_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    """The columns present in both frames, in our canonical order."""
    shared = set(reference.columns) & set(current.columns)
    ordered = [c for c in (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS) if c in shared]
    if not ordered:
        raise ValueError(
            "Reference and current data share no known monitoring columns. "
            f"Expected some of {NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}."
        )
    return ordered


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    html_path: str | Path | None = None,
) -> tuple[DriftSummary, dict[str, Any]]:
    """Run an Evidently data-drift report and return (summary, raw_result_dict).

    Tries the modern Evidently API first, falls back to the legacy one. Optionally
    writes a shareable HTML report to ``html_path``.
    """
    columns = _align_columns(reference, current)
    ref = reference[columns]
    cur = current[columns]

    try:
        summary, raw = _run_modern(ref, cur, html_path)
    except (ImportError, Exception):
        # New API not present or shaped differently — fall back to legacy.
        summary, raw = _run_legacy(ref, cur, html_path)
    return summary, raw


def _run_modern(
    ref: pd.DataFrame, cur: pd.DataFrame, html_path: str | Path | None
) -> tuple[DriftSummary, dict[str, Any]]:
    """Evidently >= 0.6 API."""
    from evidently import Report  # type: ignore
    from evidently.presets import DataDriftPreset  # type: ignore

    report = Report([DataDriftPreset()])
    run = report.run(current_data=cur, reference_data=ref)
    if html_path is not None:
        run.save_html(str(html_path))
    raw = run.dict() if hasattr(run, "dict") else {}
    return _summarize(raw), raw


def _run_legacy(
    ref: pd.DataFrame, cur: pd.DataFrame, html_path: str | Path | None
) -> tuple[DriftSummary, dict[str, Any]]:
    """Evidently 0.4.x API."""
    from evidently.metric_preset import DataDriftPreset  # type: ignore
    from evidently.report import Report  # type: ignore

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    if html_path is not None:
        report.save_html(str(html_path))
    raw = report.as_dict()
    return _summarize(raw), raw


def _summarize(raw: dict[str, Any]) -> DriftSummary:
    """Pull the dataset-level drift verdict out of either result schema.

    Both API versions surface a DatasetDrift result with the keys we need
    (``dataset_drift``, ``share_of_drifted_columns`` / ``drift_share``,
    ``number_of_drifted_columns``); we scan defensively rather than hard-coding a
    path that a minor version might move.
    """
    result = _find_drift_block(raw)
    dataset_drift = bool(result.get("dataset_drift", False))
    drift_share = float(
        result.get("share_of_drifted_columns", result.get("drift_share", 0.0)) or 0.0
    )
    n_drifted = int(
        result.get("number_of_drifted_columns", result.get("n_drifted_columns", 0)) or 0
    )
    n_columns = int(
        result.get("number_of_columns", result.get("n_columns", 0)) or 0
    )
    drifted_columns = _extract_drifted_columns(result)
    return DriftSummary(
        dataset_drift=dataset_drift,
        drift_share=drift_share,
        n_drifted_columns=n_drifted,
        n_columns=n_columns,
        drifted_columns=drifted_columns,
    )


def _find_drift_block(raw: dict[str, Any]) -> dict[str, Any]:
    """Locate the dict that carries the ``dataset_drift`` verdict, wherever it sits."""
    # Prioritize metrics that contain drift_by_columns to get full column drift details
    for metric in raw.get("metrics", []):
        result = metric.get("result", {}) if isinstance(metric, dict) else {}
        if "drift_by_columns" in result:
            return result

    for metric in raw.get("metrics", []):
        result = metric.get("result", {}) if isinstance(metric, dict) else {}
        if "dataset_drift" in result or "share_of_drifted_columns" in result:
            return result
    # Modern API may put it under top-level "metrics" with a different shape, or
    # directly on the dict.
    if "dataset_drift" in raw or "share_of_drifted_columns" in raw:
        return raw
    return {}


def _extract_drifted_columns(result: dict[str, Any]) -> list[str]:
    drift_by_columns = result.get("drift_by_columns", {})
    if not isinstance(drift_by_columns, dict):
        return []
    return [
        name
        for name, info in drift_by_columns.items()
        if isinstance(info, dict) and info.get("drift_detected")
    ]
