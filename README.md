# DRAPE — Visual Fashion Recommendation

Upload any garment image → instantly surface visually similar items from the catalogue.
No text, no categories — pure visual similarity via ResNet-50 embeddings.

```
fashion_rec_v2/
├── extract_features.py       ← builds the image embedding index
├── recommender.py            ← cosine similarity search engine
├── app.py                    ← Streamlit website
├── build_features_colab.ipynb← Colab notebook (GPU, faster)
└── requirements.txt
```

---

## How it works

1. Every catalogue image is passed through ResNet-50 (pretrained, no fine-tuning needed) → 2048-dim embedding, L2-normalised
2. Your uploaded image gets the same treatment → 2048-dim query vector
3. Cosine similarity ranks the catalogue → top matches returned as images

No training required. The pretrained ImageNet backbone already knows what garments look like.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Get the dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Place kaggle.json in ~/.kaggle/  (download from kaggle.com/settings → API)
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip -d fashion-dataset
```

---

## Step 2 — Build the feature index

### Option A: Mac M1 (local, ~15–25 min for 10k images)

```bash
python extract_features.py \
    --data_dir ./fashion-dataset \
    --output_dir ./feature_store \
    --batch_size 32 \
    --max_images 10000
```

MPS is auto-detected on M1.

### Option B: Google Colab T4 (~5 min for 10k images) ← recommended for speed

Open `build_features_colab.ipynb` in Colab with GPU runtime.  
Run all cells → download `feature_store.zip` → unzip to `./feature_store/`.

### max_images guidance

| Images | Quality  | M1 time | Colab time |
|--------|----------|---------|------------|
| 5,000  | Good     | ~10 min | ~3 min     |
| 10,000 | Better   | ~20 min | ~5 min     |
| 44,000 | Best     | ~90 min | ~20 min    |

---

## Step 3 — Launch the website

```bash
streamlit run app.py
```

Or with a custom feature store path:

```bash
FEATURE_STORE=./feature_store streamlit run app.py
```

Open http://localhost:8501

---

## Troubleshooting

**M1 out of memory**: reduce `--batch_size` to 16  
**Slow on CPU**: use Colab to build the index, then run the app locally  
**Wrong feature store path**: set `FEATURE_STORE=./your/path` env var  
**Images not loading in app**: make sure feature_store paths point to where images actually are — if you unzipped to a different folder, re-run `extract_features.py`
