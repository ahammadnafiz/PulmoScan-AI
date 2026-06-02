"""Monitoring primitives shared by the serving API and the offline tooling.

Everything here is intentionally dependency-light (PIL + numpy + stdlib) so the
exact same feature extraction runs inside the slim serving container and in the
offline reference/drift scripts — the same train/serve-parity principle the
model code follows, applied to monitoring.
"""

from pulmoscan.monitoring.features import (
    FEATURE_COLUMNS,
    extract_image_features,
    normalized_entropy,
    softmax_entropy,
)

__all__ = [
    "FEATURE_COLUMNS",
    "extract_image_features",
    "normalized_entropy",
    "softmax_entropy",
]
