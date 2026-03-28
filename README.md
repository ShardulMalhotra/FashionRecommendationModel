# DRAPE — Visual Fashion Recommendation

> Upload any garment image → instantly surface visually similar items from the catalogue.
> No text queries, no category filters — pure visual similarity via deep learning embeddings.

---

## What is DRAPE?

DRAPE is an AI-powered visual fashion discovery engine. It uses a pretrained **ResNet-50** backbone to convert every catalogue image into a compact 2048-dimensional vector (embedding). When you upload a query image, it gets the same treatment and the engine ranks the entire catalogue by **cosine similarity** — returning the most visually similar items in milliseconds.

No model training required. The ImageNet-pretrained backbone already understands textures, colours, shapes, and garment structure out of the box.

---

## Demo

| Upload a piece | Get instant matches |
|---|---|
| Any garment photo, screenshot, or inspiration image | Top-12 visually similar items with match % scores |

The UI is built with **Streamlit** and styled as a luxury editorial fashion interface — dark theme, animated product grid, hover overlays, and filter pills.

---

## Project Structure

```
FashionRecommendationModel/
├── extract_features.py        ← Step 1: builds the image embedding index
├── recommender.py             ← Core engine: cosine similarity search
├── app.py                     ← Streamlit web app (UI + recommendation flow)
├── build_features_colab.ipynb ← Google Colab notebook (GPU-accelerated indexing)
├── requirements.txt           ← Python dependencies
└── model_output/
    ├── best_model.pth         ← Saved model weights
    ├── class_names.json       ← Class label mapping
    └── history.json           ← Training history
```

---

## How It Works

### 1. Feature Extraction (`extract_features.py`)
- Loads ResNet-50 with pretrained ImageNet weights (`IMAGENET1K_V2`)
- Removes the final classification head → outputs raw `(B, 2048)` feature vectors
- Every catalogue image is passed through in batches → embeddings are **L2-normalised**
- Saves three artefacts to `./feature_store/`:
  - `features.npy` — float32 matrix of shape `(N, 2048)`
  - `paths.npy` — absolute paths to each indexed image
  - `metadata.json` — product info (name, type, colour, gender, season) keyed by image ID

### 2. Recommendation Engine (`recommender.py`)
- Loads the feature store into memory at startup
- For a query image: runs the same ResNet-50 backbone → produces a normalised 2048-dim vector
- Computes **cosine similarity** via a single matrix-vector dot product: `features @ query`
- Returns top-K results sorted by similarity score, each with `path`, `score`, and `meta`

### 3. Web App (`app.py`)
- Built with Streamlit, fully custom CSS (dark luxury editorial theme)
- Sticky nav bar, hero section, drag-and-drop upload zone
- Displays query image preview alongside a 4-column animated product grid
- Each card shows: product image, article type, display name, and match % on hover
- Decorative filter pills for category filtering
- Falls back to a random trending grid when no image is uploaded
- Shows setup instructions inline if the feature store hasn't been built yet

---

## Tech Stack

| Component | Technology |
|---|---|
| Feature extraction | ResNet-50 (torchvision, ImageNet pretrained) |
| Similarity search | NumPy dot product (cosine similarity) |
| Web framework | Streamlit |
| Image processing | Pillow |
| Data handling | Pandas, NumPy |
| Hardware acceleration | CUDA / Apple MPS (auto-detected) |

---

## Setup

### Prerequisites

- Python 3.9+
- pip

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `streamlit`, `pillow`, `pandas`, `numpy`, `tqdm`, `scikit-learn`

---

## Step 1 — Get the Dataset

This project uses the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle (~600 MB, ~44k images).

```bash
pip install kaggle

# Place kaggle.json in ~/.kaggle/
# Download from: kaggle.com → Account → Settings → API → Create New Token
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip -d fashion-dataset
```

Expected structure after unzip:
```
fashion-dataset/
├── images/          ← JPEG product images (named by product ID)
└── styles.csv       ← Product metadata (name, type, colour, gender, season)
```

---

## Step 2 — Build the Feature Index

### Option A: Local (Mac M1/M2 — MPS accelerated)

```bash
python extract_features.py \
    --data_dir ./fashion-dataset \
    --output_dir ./feature_store \
    --batch_size 32 \
    --max_images 10000
```

MPS is auto-detected on Apple Silicon. For CPU-only machines, reduce `--batch_size` to avoid memory pressure.

### Option B: Google Colab (T4 GPU — recommended for speed)

1. Open `build_features_colab.ipynb` in [Google Colab](https://colab.research.google.com/) with a **GPU runtime** (T4 or better)
2. Run all cells
3. Download the generated `feature_store.zip`
4. Unzip locally: `unzip feature_store.zip -d ./feature_store`

### Scale vs. Speed

| `--max_images` | Quality | M1 time | Colab T4 time |
|---|---|---|---|
| 5,000 | Good | ~10 min | ~3 min |
| 10,000 | Better | ~20 min | ~5 min |
| 44,000 | Best (full dataset) | ~90 min | ~20 min |

After this step, `./feature_store/` will contain:
```
feature_store/
├── features.npy    ← (N, 2048) float32 embedding matrix
├── paths.npy       ← image paths array
└── metadata.json   ← product metadata lookup
```

---

## Step 3 — Launch the App

```bash
streamlit run app.py
```

With a custom feature store path:

```bash
FEATURE_STORE=./feature_store streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. Open the app in your browser
2. Drag and drop (or click to upload) any garment image — JPG, PNG, or WebP
3. The engine embeds your image and ranks the catalogue by visual similarity
4. Browse the top-12 matches in the animated product grid
5. Hover over any card to see the product name, type, and match score

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `FEATURE_STORE` | `./feature_store` | Path to the prebuilt feature index directory |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `M1 out of memory` | Reduce `--batch_size` to 16 or lower |
| `Slow on CPU` | Build the index on Colab, run the app locally |
| `Feature store not found` | Set `FEATURE_STORE=./your/path` or re-run `extract_features.py` |
| `Images not loading in app` | Ensure `feature_store/paths.npy` points to where images actually live — re-run extraction if you moved the dataset |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |

---

## Why ResNet-50?

- **No fine-tuning needed** — ImageNet pretraining gives strong visual representations for clothing (textures, colours, patterns, shapes)
- **Fast inference** — 2048-dim embeddings are compact enough for real-time cosine search over 44k items
- **Portable** — the feature store is just `.npy` files; no database or vector store required
- **Extensible** — swap in any backbone (EfficientNet, ViT, CLIP) by changing the extractor in `extract_features.py` and `recommender.py`

---

## License

MIT License — free to use, modify, and distribute.
