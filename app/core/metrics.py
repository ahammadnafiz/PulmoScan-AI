from prometheus_client import Counter, Histogram

# prediction counters
PREDICTION_COUNT = Counter(
    "prediction_count_total",
    "Total number of predictions made by the model",
    ["predicted_class"]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Model confidence on predictions",
    ["predicted_class"],
    buckets=[0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)

PREDICTION_ENTROPY = Histogram(
    "prediction_entropy",
    "Model prediction normalized entropy (confidence uncertainty)",
    ["predicted_class"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Image feature metrics for input distribution drift tracking
IMAGE_BRIGHTNESS = Histogram(
    "image_brightness",
    "Mean brightness of incoming CT scans",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

IMAGE_CONTRAST = Histogram(
    "image_contrast",
    "Intensity standard deviation (contrast) of incoming CT scans",
    buckets=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
)

IMAGE_SHARPNESS = Histogram(
    "image_sharpness",
    "Sharpness (Laplacian variance) of incoming CT scans",
    buckets=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)
