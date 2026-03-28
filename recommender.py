"""
recommender.py — Image-based fashion recommendation engine

Loads the prebuilt feature store and finds visually similar items
using cosine similarity on ResNet-50 embeddings.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── feature extractor (same as indexing) ──────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class Recommender:
    """
    Usage:
        rec = Recommender("./feature_store")
        results = rec.recommend(pil_image, top_k=8)
        # returns list of {"path": str, "score": float, "meta": dict}
    """

    def __init__(self, feature_store_dir: str):
        self.store_dir = Path(feature_store_dir)
        self.device    = get_device()
        self._load_store()
        self._load_model()

    def _load_store(self):
        self.features = np.load(self.store_dir / "features.npy")   # (N, 2048)
        self.paths    = np.load(self.store_dir / "paths.npy",
                                allow_pickle=True)
        with open(self.store_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        print(f"Feature store loaded: {len(self.paths)} items")

    def _load_model(self):
        base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(base.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def _embed(self, image: Image.Image) -> np.ndarray:
        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)
        feat   = self.model(tensor).squeeze(-1).squeeze(-1)  # (1, 2048)
        feat   = feat / feat.norm(dim=1, keepdim=True)
        return feat.cpu().numpy()[0]                          # (2048,)

    def recommend(self, image: Image.Image, top_k: int = 8,
                  exclude_same_path: str = None) -> list[dict]:
        """
        Args:
            image:            PIL.Image (the input / query image)
            top_k:            number of recommendations to return
            exclude_same_path: path string to exclude (avoid returning query itself)

        Returns:
            list of dicts: {path, score, meta}
        """
        query = self._embed(image)                       # (2048,)
        sims  = self.features @ query                   # (N,) cosine similarity

        # Sort descending
        ranked = np.argsort(sims)[::-1]

        results = []
        for idx in ranked:
            path = str(self.paths[idx])
            if exclude_same_path and path == exclude_same_path:
                continue
            img_id = Path(path).stem
            meta   = self.metadata.get(img_id, {})
            results.append({
                "path":  path,
                "score": float(sims[idx]),
                "meta":  meta,
            })
            if len(results) >= top_k:
                break

        return results
