"""
Stage 2 NSFW Detector.
Executes detailed object detection using the NudeNet YOLOv8 ONNX model.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

NUDENET_PATH = os.path.join(os.path.dirname(__file__), "../../references/NudeNet")
sys.path.insert(0, NUDENET_PATH)

NSFW_CLASSES = {
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
    "FEET_EXPOSED",
}

BORDERLINE_CLASSES = {
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
    "ANUS_COVERED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
}


class NudeNetDetector:
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self._detector = None
        self._model_path = model_path

    def _ensure_loaded(self):
        if self._detector is not None:
            return
        try:
            from nudenet import NudeDetector as _NudeDetector
            self._detector = _NudeDetector(model_path=self._model_path)
        except ImportError:
            raise ImportError(
                "NudeNet not found."
            )

    def predict(self, image_path: str = None, image_bytes: bytes = None) -> dict:
        self._ensure_loaded()

        target = image_path or image_bytes
        if target is None:
            raise ValueError("Provide image_path or image_bytes")

        detections = self._detector.detect(target)
        detections = [d for d in detections if d["score"] >= self.confidence_threshold]

        nsfw_found = set()
        max_score = 0.0
        nsfw_scores = []

        for det in detections:
            cls = det["class"]
            score = det["score"]
            if cls in NSFW_CLASSES:
                nsfw_found.add(cls)
                max_score = max(max_score, score)
                nsfw_scores.append(score)

        nsfw_score = max(nsfw_scores) if nsfw_scores else 0.0

        return {
            "is_nsfw": len(nsfw_found) > 0,
            "confidence": max_score,
            "nsfw_score": nsfw_score,
            "detections": detections,
            "nsfw_classes_found": nsfw_found,
        }

    def predict_and_censor(self, image_path: str, output_path: str = None) -> dict:
        self._ensure_loaded()
        result = self.predict(image_path)

        if result["is_nsfw"] and output_path:
            censored_path = self._detector.censor(
                image_path,
                classes=list(NSFW_CLASSES),
                output_path=output_path,
            )
            result["censored_path"] = censored_path

        return result


def load_model():
    detector = NudeNetDetector()
    detector._ensure_loaded()
    print("[Stage2-NSFW] NudeNet model loaded successfully.")
    return detector


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = NudeNetDetector()
        result = detector.predict(sys.argv[1])
        print(f"Image: {sys.argv[1]}")
        print(f"  Is NSFW: {result['is_nsfw']}")
        print(f"  Score: {result['nsfw_score']:.4f}")
        print(f"  Classes: {result['nsfw_classes_found']}")
        for d in result['detections']:
            print(f"  - {d['class']}: {d['score']:.3f}")
    else:
        print("Usage: python nudenet_nsfw.py <image_path>")
