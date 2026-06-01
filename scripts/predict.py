"""Standalone CLI prediction — sanity-check the trained model without the API.

Usage:
    python scripts/predict.py path/to/image.png
    python scripts/predict.py path/to/image.png --model artifacts/training/model.pt
"""

import argparse
import json

from app.services.inference import inference_service


def main() -> None:
    parser = argparse.ArgumentParser(description="PulmoScan AI — CLI prediction")
    parser.add_argument("image", help="Path to the CT-scan image")
    parser.add_argument(
        "--model", default="artifacts/training/model.pt", help="Path to the checkpoint"
    )
    args = parser.parse_args()

    inference_service.load(args.model)
    with open(args.image, "rb") as f:
        result = inference_service.predict(f.read())

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
