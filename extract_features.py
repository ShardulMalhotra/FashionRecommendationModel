"""
extract_features.py — Build a feature index from the fashion dataset

How it works:
  1. Strips the ResNet-50 classification head
  2. Passes every image through → 2048-dim embedding
  3. Saves embeddings + image paths to disk
  → At recommendation time we do cosine similarity against this index

Usage:
  python extract_features.py --data_dir ./fashion-dataset --output_dir ./feature_store

Dataset: fashion-product-images-small (Kaggle)
  kaggle datasets download -d paramaggarwal/fashion-product-images-small
"""

import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import pandas as pd


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── model (ResNet50 backbone, no head) ────────────────────────────────────────

def build_feature_extractor(device):
    base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final FC layer → outputs (B, 2048)
    extractor = nn.Sequential(*list(base.children())[:-1])
    extractor.eval()
    return extractor.to(device)


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_batch(model, paths, device):
    imgs = []
    valid_paths = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(TRANSFORM(img))
            valid_paths.append(str(p))
        except Exception:
            continue
    if not imgs:
        return [], []
    batch = torch.stack(imgs).to(device)
    feats = model(batch).squeeze(-1).squeeze(-1)  # (B, 2048)
    feats = feats / feats.norm(dim=1, keepdim=True)  # L2 normalise
    return feats.cpu().numpy(), valid_paths


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./fashion-dataset")
    parser.add_argument("--output_dir", default="./feature_store")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_images", type=int, default=10000,
                        help="Cap images for speed (set 0 for all)")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir  = data_dir / "images"
    csv_path = data_dir / "styles.csv"

    # ── read metadata ─────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    needed = ["id", "articleType", "baseColour", "gender", "season", "productDisplayName"]
    df = df[[c for c in needed if c in df.columns]].dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    # Only keep images that exist
    df["img_path"] = df["id"].apply(lambda x: img_dir / f"{x}.jpg")
    df = df[df["img_path"].apply(lambda p: p.exists())].copy()

    if args.max_images and len(df) > args.max_images:
        df = df.sample(args.max_images, random_state=42)

    print(f"Images to index: {len(df)}")

    # Save metadata lookup: id → {articleType, colour, ...}
    meta = df.drop(columns=["img_path"]).set_index("id").to_dict(orient="index")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({str(k): v for k, v in meta.items()}, f)

    # ── extract features ──────────────────────────────────────────────────────
    device    = get_device()
    model     = build_feature_extractor(device)

    all_feats = []
    all_paths = []

    paths = df["img_path"].tolist()
    bs    = args.batch_size

    for i in tqdm(range(0, len(paths), bs), desc="Extracting"):
        batch_paths = paths[i:i+bs]
        feats, valid = extract_batch(model, batch_paths, device)
        if len(feats):
            all_feats.append(feats)
            all_paths.extend(valid)

    features = np.vstack(all_feats).astype(np.float32)  # (N, 2048)
    paths_arr = np.array(all_paths)

    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "paths.npy",    paths_arr)

    print(f"\nFeature store saved to {out_dir}")
    print(f"  features.npy : {features.shape}")
    print(f"  paths.npy    : {len(paths_arr)} images")
    print(f"  metadata.json: {len(meta)} records")


if __name__ == "__main__":
    main()
