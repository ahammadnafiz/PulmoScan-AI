"""Pure decision logic for the MLOps lifecycle.

Kept free of MLflow / Evidently / I/O so the *policy* — when to promote a model,
when to retrain — is trivially unit-testable and reviewable in one place. The
scripts wire these functions to real data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def should_promote(
    challenger: float | None,
    champion: float | None,
    *,
    min_margin: float = 0.0,
) -> PromotionDecision:
    """Decide whether a challenger model should become the new champion.

    - No challenger metric -> never promote (nothing to judge).
    - No incumbent champion -> promote (first model wins by default).
    - Otherwise promote only if the challenger beats the champion by at least
      ``min_margin`` (absolute, on the same metric). The margin guards against
      promoting on noise.
    """
    if challenger is None:
        return PromotionDecision(False, "no challenger metric provided")
    if champion is None:
        return PromotionDecision(True, "no existing champion — promoting first model")
    if challenger >= champion + min_margin:
        return PromotionDecision(
            True,
            f"challenger {challenger:.4f} >= champion {champion:.4f} + margin {min_margin:.4f}",
        )
    return PromotionDecision(
        False,
        f"challenger {challenger:.4f} does not beat champion {champion:.4f} by {min_margin:.4f}",
    )


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reasons: list[str]


def decide_retrain(
    *,
    dataset_drift: bool,
    drift_share: float,
    low_confidence_rate: float,
    n_current_samples: int,
    drift_share_threshold: float = 0.5,
    low_confidence_rate_threshold: float = 0.25,
    min_samples: int = 50,
) -> RetrainDecision:
    """Decide whether to trigger a retraining run from monitoring signals.

    Two independent triggers, either is sufficient:
      1. **Input/prediction drift** — Evidently flags the dataset as drifted, or
         the share of drifted columns crosses ``drift_share_threshold``.
      2. **Confidence collapse** — the fraction of low-confidence (OOD-flagged)
         predictions crosses ``low_confidence_rate_threshold``.

    Below ``min_samples`` of production traffic we abstain: the signals are too
    noisy to act on, so we never retrain on a handful of requests.
    """
    if n_current_samples < min_samples:
        return RetrainDecision(
            False,
            [f"insufficient data: {n_current_samples} < {min_samples} samples"],
        )

    reasons: list[str] = []
    if dataset_drift:
        reasons.append("Evidently reported dataset drift")
    if drift_share >= drift_share_threshold:
        reasons.append(
            f"drift share {drift_share:.2f} >= threshold {drift_share_threshold:.2f}"
        )
    if low_confidence_rate >= low_confidence_rate_threshold:
        reasons.append(
            f"low-confidence rate {low_confidence_rate:.2f} >= "
            f"threshold {low_confidence_rate_threshold:.2f}"
        )

    return RetrainDecision(bool(reasons), reasons or ["no drift or confidence triggers fired"])
