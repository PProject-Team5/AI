"""
Stage 1 NSFW Detector.
Performs fast image classification using CLIP embeddings and an MLP head.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple


class CLIPNSFWHead(nn.Module):
    def __init__(self, input_size: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CLIPNSFWHeadSlim(nn.Module):
    def __init__(self, input_size: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CLIPNSFWDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        clip_model: str = "ViT-B/32",
        device: str = "cpu",
        slim: bool = False,
        suspicious_range: Tuple[float, float] = (0.3, 0.7),
    ):
        self.device = torch.device(device)
        self.clip_model_name = clip_model
        self.suspicious_range = suspicious_range

        import clip
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device)
        self.clip_model.eval()

        embed_dim = 512 if "B/32" in clip_model or "B/16" in clip_model else 1024

        head_cls = CLIPNSFWHeadSlim if slim else CLIPNSFWHead
        self.head = head_cls(input_size=embed_dim).to(self.device)

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.head.load_state_dict(state)
            print(f"[Stage1-NSFW] Loaded weights from {model_path}")
        else:
            print("[Stage1-NSFW] Using untrained head")

        self.head.eval()

    @torch.no_grad()
    def predict(self, image) -> dict:
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.clip_model.encode_image(img_tensor)
        embedding = embedding.float()

        logit = self.head(embedding)
        prob = torch.sigmoid(logit).item()

        lo, hi = self.suspicious_range
        if prob < lo:
            label = "safe"
        elif prob > hi:
            label = "nsfw"
        else:
            label = "suspicious"

        return {
            "nsfw_prob": prob,
            "label": label,
            "needs_stage2": label == "suspicious",
            "embedding": embedding.cpu().numpy().flatten(),
        }

    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        from PIL import Image

        tensors = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            tensors.append(self.preprocess(img))

        batch = torch.stack(tensors).to(self.device)
        embeddings = self.clip_model.encode_image(batch).float()
        logits = self.head(embeddings)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        results = []
        for i, prob in enumerate(probs):
            lo, hi = self.suspicious_range
            if prob < lo:
                label = "safe"
            elif prob > hi:
                label = "nsfw"
            else:
                label = "suspicious"
            results.append({
                "nsfw_prob": float(prob),
                "label": label,
                "needs_stage2": label == "suspicious",
                "embedding": embeddings[i].cpu().numpy(),
            })
        return results


if __name__ == "__main__":
    import sys
    detector = CLIPNSFWDetector()
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    result = detector.predict(img_path)
    print(f"Image: {img_path}")
    print(f"  NSFW prob: {result['nsfw_prob']:.4f}")
    print(f"  Label: {result['label']}")
    print(f"  Needs Stage 2: {result['needs_stage2']}")
