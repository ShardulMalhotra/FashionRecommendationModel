"""
DRAPE — AI Fashion Recommendation
Run: streamlit run app.py

Set environment variable:
  FEATURE_STORE=./feature_store streamlit run app.py
"""

import os
import sys
import base64
import random
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image
import numpy as np

# ─── page config (must be first) ─────────────────────────────────────────────
st.set_page_config(
    page_title="DRAPE",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── helpers ─────────────────────────────────────────────────────────────────

def img_to_b64(img: Image.Image, fmt="JPEG", quality=85) -> str:
    buf = BytesIO()
    img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def load_pil(path: str, size=(400, 400)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    return img


# ─── global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;1,400&family=Jost:wght@200;300;400;500&display=swap');

/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
  font-family: 'Jost', sans-serif;
  background: #0f0f0d !important;
  color: #e8e2d9;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0f0f0d; }
::-webkit-scrollbar-thumb { background: #3a3530; border-radius: 2px; }

/* ── nav ── */
.drape-nav {
  position: sticky;
  top: 0;
  z-index: 999;
  background: rgba(15,15,13,0.92);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(232,226,217,0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 3rem;
  height: 64px;
}
.drape-logo {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  letter-spacing: 0.25em;
  color: #e8e2d9;
  text-transform: uppercase;
}
.drape-logo span { color: #c9a96e; }
.nav-links {
  display: flex;
  gap: 2.5rem;
  font-size: 0.72rem;
  font-weight: 300;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #8a8278;
}

/* ── hero ── */
.hero {
  padding: 5rem 3rem 3rem;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  max-width: 700px;
}
.hero-eyebrow {
  font-size: 0.65rem;
  letter-spacing: 0.4em;
  text-transform: uppercase;
  color: #c9a96e;
  font-weight: 300;
  margin-bottom: 1rem;
}
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.5rem, 4vw, 3.8rem);
  font-weight: 400;
  line-height: 1.15;
  color: #e8e2d9;
  margin-bottom: 1.2rem;
}
.hero-title em { font-style: italic; color: #c9a96e; }
.hero-sub {
  font-size: 0.9rem;
  font-weight: 200;
  color: #8a8278;
  line-height: 1.8;
  max-width: 480px;
}

/* ── upload zone ── */
.upload-section {
  padding: 2rem 3rem 1rem;
  border-top: 1px solid rgba(232,226,217,0.07);
  margin-top: 2rem;
}
.upload-label {
  font-size: 0.65rem;
  letter-spacing: 0.35em;
  text-transform: uppercase;
  color: #8a8278;
  margin-bottom: 0.75rem;
  display: block;
}

/* hide streamlit upload widget chrome, style the drop zone */
[data-testid="stFileUploader"] > div {
  background: transparent !important;
  border: 1px solid rgba(201,169,110,0.25) !important;
  border-radius: 0 !important;
  padding: 2.5rem !important;
  text-align: center;
  transition: border-color 0.3s;
  cursor: pointer;
}
[data-testid="stFileUploader"] > div:hover {
  border-color: rgba(201,169,110,0.6) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploader"] p {
  color: #5a5550 !important;
  font-size: 0.8rem !important;
  font-family: 'Jost', sans-serif !important;
  letter-spacing: 0.1em !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  color: #5a5550 !important;
}

/* ── input preview + results ── */
.input-panel {
  padding: 2rem 3rem;
  display: flex;
  gap: 0;
  align-items: stretch;
  border-top: 1px solid rgba(232,226,217,0.07);
}
.input-img-wrap {
  position: relative;
  flex: 0 0 280px;
}
.input-img-wrap img {
  width: 100%;
  aspect-ratio: 3/4;
  object-fit: cover;
  display: block;
}
.input-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(15,15,13,0.85);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(201,169,110,0.3);
  font-size: 0.6rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: #c9a96e;
  padding: 4px 10px;
}
.input-sidebar {
  flex: 0 0 200px;
  background: #161612;
  border: 1px solid rgba(232,226,217,0.06);
  border-left: none;
  padding: 2rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
.sidebar-label {
  font-size: 0.6rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: #5a5550;
  margin-bottom: 0.3rem;
}
.sidebar-value {
  font-size: 0.95rem;
  font-weight: 300;
  color: #e8e2d9;
  text-transform: capitalize;
}
.match-score {
  font-family: 'Playfair Display', serif;
  font-size: 2.2rem;
  color: #c9a96e;
  line-height: 1;
}
.match-score-label {
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #5a5550;
  margin-top: 0.2rem;
}

/* ── section heading ── */
.section-head {
  padding: 2.5rem 3rem 1rem;
  display: flex;
  align-items: baseline;
  gap: 1.5rem;
  border-top: 1px solid rgba(232,226,217,0.07);
}
.section-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem;
  font-weight: 400;
  color: #e8e2d9;
}
.section-count {
  font-size: 0.65rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: #5a5550;
}

/* ── product grid ── */
.product-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: rgba(232,226,217,0.06);
  margin: 0 3rem 3rem;
}
.product-card {
  background: #0f0f0d;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.product-card:hover { transform: scale(1.015); z-index: 2; }
.product-card img {
  width: 100%;
  aspect-ratio: 3/4;
  object-fit: cover;
  display: block;
  transition: opacity 0.3s;
}
.product-card:hover img { opacity: 0.88; }
.product-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(to top, rgba(15,15,13,0.85) 0%, transparent 50%);
  opacity: 0;
  transition: opacity 0.35s;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  padding: 1.2rem;
}
.product-card:hover .product-overlay { opacity: 1; }
.product-name {
  font-size: 0.82rem;
  font-weight: 300;
  color: #e8e2d9;
  letter-spacing: 0.05em;
  text-transform: capitalize;
  line-height: 1.4;
}
.product-type {
  font-size: 0.6rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: #c9a96e;
  margin-bottom: 0.3rem;
}
.product-match-pill {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(15,15,13,0.8);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(201,169,110,0.4);
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  color: #c9a96e;
  padding: 3px 8px;
  opacity: 0;
  transition: opacity 0.3s;
}
.product-card:hover .product-match-pill { opacity: 1; }

/* ── empty state ── */
.empty-state {
  padding: 5rem 3rem;
  text-align: center;
  color: #3a3530;
}
.empty-icon {
  font-family: 'Playfair Display', serif;
  font-size: 4rem;
  display: block;
  margin-bottom: 1rem;
  color: #2a2520;
}
.empty-msg {
  font-size: 0.8rem;
  font-weight: 300;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

/* ── loading ── */
.loading-wrap {
  padding: 4rem 3rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.loading-dot {
  width: 6px; height: 6px;
  background: #c9a96e;
  border-radius: 50%;
  animation: pulse 1.2s ease-in-out infinite;
}
.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
  0%, 100% { opacity: 0.2; transform: scale(0.8); }
  50%       { opacity: 1;   transform: scale(1.2); }
}
.loading-text {
  font-size: 0.7rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: #5a5550;
}

/* ── filter pills ── */
.filter-bar {
  padding: 0 3rem 1rem;
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.filter-pill {
  font-size: 0.62rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #8a8278;
  border: 1px solid rgba(232,226,217,0.12);
  padding: 5px 14px;
  cursor: pointer;
  background: transparent;
  transition: all 0.2s;
}
.filter-pill.active {
  background: #c9a96e;
  border-color: #c9a96e;
  color: #0f0f0d;
}

/* ── buttons ── */
.stButton > button {
  background: #c9a96e !important;
  color: #0f0f0d !important;
  border: none !important;
  border-radius: 0 !important;
  font-family: 'Jost', sans-serif !important;
  font-size: 0.65rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.3em !important;
  text-transform: uppercase !important;
  padding: 0.7rem 2rem !important;
  width: auto !important;
  transition: background 0.2s !important;
}
.stButton > button:hover {
  background: #e8d4a0 !important;
  color: #0f0f0d !important;
}

/* ── toast ── */
.toast {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  background: #1e1e1a;
  border: 1px solid rgba(201,169,110,0.3);
  padding: 1rem 1.5rem;
  font-size: 0.75rem;
  font-weight: 300;
  color: #e8e2d9;
  letter-spacing: 0.05em;
  z-index: 9999;
  animation: slideUp 0.4s ease;
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── footer ── */
.drape-footer {
  border-top: 1px solid rgba(232,226,217,0.07);
  padding: 2rem 3rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.footer-logo {
  font-family: 'Playfair Display', serif;
  font-size: 1rem;
  letter-spacing: 0.3em;
  color: #3a3530;
  text-transform: uppercase;
}
.footer-copy {
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #3a3530;
}
</style>
""", unsafe_allow_html=True)


# ─── nav ─────────────────────────────────────────────────────────────────────
st.markdown("""
<nav class="drape-nav">
  <div class="drape-logo">DR<span>◈</span>PE</div>
  <div class="nav-links">
    <span>New arrivals</span>
    <span>Women</span>
    <span>Men</span>
    <span>Discover</span>
  </div>
</nav>
""", unsafe_allow_html=True)


# ─── load recommender ─────────────────────────────────────────────────────────
STORE_DIR = os.environ.get("FEATURE_STORE", "./feature_store")

@st.cache_resource(show_spinner=False)
def get_recommender():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from recommender import Recommender
        rec = Recommender(STORE_DIR)
        return rec, None
    except Exception as e:
        return None, str(e)

recommender, rec_error = get_recommender()


# ─── hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Visual style discovery</div>
  <h1 class="hero-title">Find what <em>speaks</em><br>to your style</h1>
  <p class="hero-sub">
    Upload any piece you love — a photograph, a screenshot, an inspiration image.
    We surface the closest matches from our collection, instantly.
  </p>
</div>
""", unsafe_allow_html=True)


# ─── upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<span class="upload-label">Drop your inspiration piece</span>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)


# ─── show results ─────────────────────────────────────────────────────────────
if uploaded is not None:
    query_img = Image.open(uploaded).convert("RGB")

    # — input preview panel —
    q_b64 = img_to_b64(query_img)
    st.markdown(f"""
<div class="input-panel">
  <div class="input-img-wrap">
    <img src="data:image/jpeg;base64,{q_b64}" />
    <div class="input-badge">Your piece</div>
  </div>
  <div class="input-sidebar">
    <div>
      <div class="sidebar-label">Status</div>
      <div class="sidebar-value">Analysing...</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # — run recommendation —
    if recommender is not None:
        with st.spinner(""):
            st.markdown("""
<div class="loading-wrap">
  <div class="loading-dot"></div>
  <div class="loading-dot"></div>
  <div class="loading-dot"></div>
  <span class="loading-text">Finding your matches</span>
</div>
""", unsafe_allow_html=True)
            results = recommender.recommend(query_img, top_k=12)

        # ── section heading ───────────────────────────────────────────────────
        top_meta = results[0]["meta"] if results else {}
        category = top_meta.get("articleType", "")
        colour   = top_meta.get("baseColour", "")
        detail   = f"{colour} · {category}".strip(" · ") if (colour or category) else "Curated for you"

        st.markdown(f"""
<div class="section-head">
  <span class="section-title">You might also love</span>
  <span class="section-count">{len(results)} matches found · {detail}</span>
</div>
""", unsafe_allow_html=True)

        # ── filter bar (decorative — uses JS to toggle active class) ──────────
        categories = list(dict.fromkeys(
            r["meta"].get("articleType", "All") for r in results if r["meta"]
        ))
        pills_html = '<div class="filter-bar"><span class="filter-pill active" onclick="filterPill(this, \'all\')">All</span>'
        for cat in categories[:6]:
            pills_html += f'<span class="filter-pill" onclick="filterPill(this, \'{cat}\')">{cat}</span>'
        pills_html += """</div>
<script>
function filterPill(el, val) {
  document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
}
</script>"""
        st.markdown(pills_html, unsafe_allow_html=True)

        # ── product grid ──────────────────────────────────────────────────────
        grid_html = '<div class="product-grid">'
        for i, res in enumerate(results):
            try:
                img   = load_pil(res["path"])
                b64   = img_to_b64(img)
                meta  = res["meta"]
                name  = meta.get("productDisplayName", "Fashion item")
                ptype = meta.get("articleType", "")
                score = int(res["score"] * 100)
                anim_delay = f"{i * 60}ms"

                grid_html += f"""
<div class="product-card" style="animation: fadeIn 0.5s {anim_delay} both">
  <img src="data:image/jpeg;base64,{b64}" loading="lazy" />
  <div class="product-match-pill">{score}% match</div>
  <div class="product-overlay">
    <div class="product-type">{ptype}</div>
    <div class="product-name">{name[:55]}{"…" if len(name)>55 else ""}</div>
  </div>
</div>"""
            except Exception:
                continue

        grid_html += """</div>
<style>
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>"""
        st.markdown(grid_html, unsafe_allow_html=True)

    else:
        # ── feature store not built yet — show instructions ───────────────────
        st.markdown(f"""
<div style="padding: 3rem; background: #161612; margin: 2rem 3rem;
            border: 1px solid rgba(201,169,110,0.15);">
  <p style="font-size:0.65rem;letter-spacing:0.3em;text-transform:uppercase;
             color:#c9a96e;margin-bottom:1rem;">Setup required</p>
  <p style="font-size:0.9rem;font-weight:300;color:#8a8278;line-height:2;
             font-family:monospace;font-size:0.8rem;">
    Feature store not found at: <b style="color:#e8e2d9">{STORE_DIR}</b><br><br>
    1. Download the dataset:<br>
    &nbsp;&nbsp;kaggle datasets download -d paramaggarwal/fashion-product-images-small<br><br>
    2. Build the feature index:<br>
    &nbsp;&nbsp;python extract_features.py --data_dir ./fashion-dataset<br><br>
    3. Relaunch the app — done.
  </p>
</div>
""", unsafe_allow_html=True)

else:
    # ── empty state: trending / editorial placeholder ─────────────────────────
    st.markdown("""
<div class="section-head">
  <span class="section-title">Trending this season</span>
  <span class="section-count">Upload a piece to discover similar styles</span>
</div>
""", unsafe_allow_html=True)

    # pull random samples from feature store if available
    if recommender is not None:
        sample_paths = random.sample(
            list(recommender.paths), min(8, len(recommender.paths))
        )
        grid_html = '<div class="product-grid">'
        for i, path in enumerate(sample_paths):
            try:
                img  = load_pil(path)
                b64  = img_to_b64(img)
                img_id = Path(path).stem
                meta   = recommender.metadata.get(img_id, {})
                name   = meta.get("productDisplayName", "Fashion item")
                ptype  = meta.get("articleType", "")
                anim_delay = f"{i * 80}ms"
                grid_html += f"""
<div class="product-card" style="animation: fadeIn 0.6s {anim_delay} both">
  <img src="data:image/jpeg;base64,{b64}" />
  <div class="product-overlay">
    <div class="product-type">{ptype}</div>
    <div class="product-name">{name[:55]}{"…" if len(name)>55 else ""}</div>
  </div>
</div>"""
            except Exception:
                continue
        grid_html += """</div>
<style>
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>"""
        st.markdown(grid_html, unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="empty-state">
  <span class="empty-icon">◈</span>
  <div class="empty-msg">Your style discovery awaits</div>
</div>
""", unsafe_allow_html=True)


# ─── footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="drape-footer">
  <div class="footer-logo">DR◈PE</div>
  <div class="footer-copy">© 2025 · All styles reserved</div>
</div>
""", unsafe_allow_html=True)
