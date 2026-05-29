"""
NSFW Knowledge Distillation.
Collects image verification samples and retrains the Stage 1 head using soft labels.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class DistillationSample:
    embedding: List[float]
    stage1_prob: float
    stage2_prob: float
    label: int
    timestamp: float
    source: str


class DistillationBuffer:
    def __init__(self, buffer_path: str = "data/nsfw_distill_buffer.jsonl"):
        self.buffer_path = buffer_path
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)

    def add(self, sample: DistillationSample):
        with open(self.buffer_path, 'a') as f:
            f.write(json.dumps(asdict(sample)) + '\n')

    def load(self, max_age_days: float = 30) -> List[DistillationSample]:
        if not os.path.exists(self.buffer_path):
            return []

        cutoff = time.time() - (max_age_days * 86400)
        samples = []
        with open(self.buffer_path) as f:
            for line in f:
                data = json.loads(line)
                if data['timestamp'] >= cutoff:
                    samples.append(DistillationSample(**data))
        return samples

    def clear(self):
        if os.path.exists(self.buffer_path):
            os.remove(self.buffer_path)

    @property
    def count(self) -> int:
        if not os.path.exists(self.buffer_path):
            return 0
        with open(self.buffer_path) as f:
            return sum(1 for _ in f)


class NSFWDistiller:
    def __init__(
        self,
        buffer_path: str = "data/nsfw_distill_buffer.jsonl",
        temperature: float = 3.0,
        alpha: float = 0.5,
        learning_rate: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.buffer = DistillationBuffer(buffer_path)
        self.temperature = temperature
        self.alpha = alpha
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

    def collect_sample(
        self,
        embedding: np.ndarray,
        stage1_prob: float,
        stage2_result: dict,
    ):
        is_nsfw = stage2_result.get("is_nsfw", False)
        hard_label = 1 if is_nsfw else 0
        soft_prob = stage2_result.get("nsfw_score", float(is_nsfw))

        sample = DistillationSample(
            embedding=embedding.tolist(),
            stage1_prob=stage1_prob,
            stage2_prob=soft_prob,
            label=hard_label,
            timestamp=time.time(),
            source="distillation",
        )
        self.buffer.add(sample)

    def distill(
        self,
        student_model: nn.Module,
        original_data_path: Optional[str] = None,
        output_path: str = "models/clip_nsfw_head.pth",
    ) -> dict:
        distilled = self.buffer.load()
        if not distilled:
            print("[Distill-NSFW] No distilled samples. Skipping.")
            return {"status": "no_data"}

        print(f"[Distill-NSFW] {len(distilled)} distilled samples loaded")

        X_distill = np.array([s.embedding for s in distilled], dtype=np.float32)
        y_soft = np.array([s.stage2_prob for s in distilled], dtype=np.float32)

        X_orig, y_orig = np.array([]), np.array([])
        if original_data_path and os.path.exists(original_data_path):
            data = np.load(original_data_path)
            X_orig = data['embeddings'].astype(np.float32)
            y_orig = data['labels'].astype(np.float32)
            print(f"[Distill-NSFW] {len(y_orig)} original samples loaded")

        if len(X_orig) > 0:
            X_all = np.concatenate([X_distill, X_orig])
            y_all = np.concatenate([y_soft, y_orig])
            weights = np.concatenate([
                np.full(len(X_distill), self.alpha),
                np.full(len(X_orig), 1.0 - self.alpha),
            ])
        else:
            X_all = X_distill
            y_all = y_soft
            weights = np.ones(len(X_all))

        X_tensor = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_all, dtype=torch.float32).unsqueeze(1).to(self.device)
        w_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        student_model.train()
        student_model.to(self.device)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        losses = []
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_tensor))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_tensor), self.batch_size):
                idx = perm[i:i + self.batch_size]
                x_batch = X_tensor[idx]
                y_batch = y_tensor[idx]
                w_batch = w_tensor[idx]

                optimizer.zero_grad()
                logits = student_model(x_batch)
                loss_per_sample = criterion(logits, y_batch)
                weighted_loss = (loss_per_sample * w_batch).mean()
                weighted_loss.backward()
                optimizer.step()

                epoch_loss += weighted_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(student_model.state_dict(), output_path)
        print(f"[Distill-NSFW] Updated model saved to {output_path}")

        self.buffer.clear()

        stats = {
            "status": "success",
            "distilled_samples": len(distilled),
            "original_samples": len(y_orig),
            "epochs": self.epochs,
            "final_loss": losses[-1] if losses else 0,
            "model_path": output_path,
        }
        return stats


def run_nsfw_distillation(
    student_model_path: str = "models/clip_nsfw_head.pth",
    buffer_path: str = "data/nsfw_distill_buffer.jsonl",
    original_data_path: str = "data/nsfw_original_embeddings.npz",
):
    from src.stage1.clip_nsfw import CLIPNSFWHead

    model = CLIPNSFWHead(input_size=512)
    if os.path.exists(student_model_path):
        model.load_state_dict(torch.load(student_model_path, map_location="cpu"))

    distiller = NSFWDistiller(buffer_path=buffer_path)
    stats = distiller.distill(
        student_model=model,
        original_data_path=original_data_path,
        output_path=student_model_path,
    )
    return stats


if __name__ == "__main__":
    stats = run_nsfw_distillation()
    print(json.dumps(stats, indent=2))
