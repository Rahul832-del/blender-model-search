"""
3D Model Search API — Hybrid Search
======================================
Supports 3 search modes:
  POST /search/hybrid  — text + image combined (best accuracy)
  GET  /search/text    — text only
  POST /search/image   — image only
  GET  /models         — list/browse models
  GET  /categories     — list categories
  GET  /model/{name}   — model detail with all views

SETUP:
  pip install fastapi uvicorn python-multipart torch transformers pillow numpy faiss-cpu

USAGE:
  python search_api.py
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from typing import Optional

from transformers import CLIPModel, CLIPProcessor
import faiss

from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMBEDDINGS_DIR = "./embeddings/master"
CATALOG_PATH = "./embeddings/master/master_catalog.json"
RENDERS_ROOT = "./renders"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
HOST = "0.0.0.0"
PORT = 8000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD AT STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Loading search engine...\n")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}")

print(f"  Loading CLIP...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()
print(f"  CLIP loaded!")

print(f"  Loading FAISS indexes...")
image_index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "image_index.faiss"))
text_index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "text_index.faiss"))
print(f"  Image index: {image_index.ntotal} vectors")
print(f"  Text index:  {text_index.ntotal} vectors")

with open(os.path.join(EMBEDDINGS_DIR, "image_metadata.json")) as f:
    image_metadata = json.load(f)
with open(os.path.join(EMBEDDINGS_DIR, "text_metadata.json")) as f:
    text_metadata = json.load(f)
with open(CATALOG_PATH) as f:
    catalog = json.load(f)

print(f"  Catalog: {len(catalog)} models")
print(f"\n  Search engine ready!\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLIP ENCODING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def encode_image(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output if hasattr(emb, 'pooler_output') else emb.last_hidden_state[:, 0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def encode_text(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output if hasattr(emb, 'pooler_output') else emb.last_hidden_state[:, 0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Get image scores per model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_image_scores(image_vec):
    """Get best image match score for each model (aggregated across 16 views)."""
    k = image_index.ntotal
    distances, indices = image_index.search(image_vec, k)

    model_scores = {}
    model_best_view = {}

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(image_metadata):
            continue
        entry = image_metadata[idx]
        name = entry["model_name"]

        if name not in model_scores or dist > model_scores[name]:
            model_scores[name] = float(dist)
            model_best_view[name] = entry["image_file"]

    return model_scores, model_best_view


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Get text scores per model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_text_scores(text_vec):
    """Get CLIP text similarity score for each model."""
    k = text_index.ntotal
    distances, indices = text_index.search(text_vec, k)

    model_scores = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(text_metadata):
            continue
        name = text_metadata[idx]["model_name"]
        model_scores[name] = float(dist)

    return model_scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Keyword matching score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def expand_word_variants(word):
    """Generate singular/plural variants of a word."""
    variants = {word}
    # Plural → singular
    if word.endswith("ies"):
        variants.add(word[:-3] + "y")       # "candies" → "candy"
    elif word.endswith("ves"):
        variants.add(word[:-3] + "f")        # "shelves" → "shelf"
        variants.add(word[:-3] + "fe")       # "knives" → "knife"
    elif word.endswith("ses") or word.endswith("xes") or word.endswith("ches") or word.endswith("shes"):
        variants.add(word[:-2])              # "glasses" → "glass"
    elif word.endswith("s") and not word.endswith("ss"):
        variants.add(word[:-1])              # "bags" → "bag"
    # Singular → plural
    if word.endswith("y"):
        variants.add(word[:-1] + "ies")      # "candy" → "candies"
    elif word.endswith("f"):
        variants.add(word[:-1] + "ves")      # "shelf" → "shelves"
    elif word.endswith("fe"):
        variants.add(word[:-2] + "ves")      # "knife" → "knives"
    elif word.endswith(("s", "x", "ch", "sh")):
        variants.add(word + "es")            # "glass" → "glasses"
    else:
        variants.add(word + "s")             # "bag" → "bags"
    return variants


def get_keyword_scores(query_text):
    """Score each model by keyword match against query text (with plural/singular support)."""
    query_words = set(query_text.lower().replace("_", " ").split())
    query_words = {w for w in query_words if len(w) > 2}

    if not query_words:
        return {}, {}

    # Expand query words with plural/singular variants
    expanded_query = set()
    for w in query_words:
        expanded_query.update(expand_word_variants(w))

    model_keyword_scores = {}
    model_matched_words = {}

    for model_name, data in catalog.items():
        name_words = set(model_name.lower().replace("_", " ").split())
        tag_words = set(t.lower() for t in data.get("tags", []))
        all_words = name_words | tag_words

        # Also expand model words for matching
        expanded_model = set()
        for w in all_words:
            expanded_model.update(expand_word_variants(w))

        # Match expanded query against expanded model words
        matched = expanded_query & expanded_model
        # Calculate ratio based on original query words matched
        original_matched = set()
        for qw in query_words:
            qw_variants = expand_word_variants(qw)
            if qw_variants & expanded_model:
                original_matched.add(qw)

        ratio = len(original_matched) / len(query_words)

        # Exact name match bonus (also check expanded)
        name_lower = model_name.lower().replace("_", " ")
        query_lower = query_text.lower().strip()
        name_bonus = 0.0
        if query_lower in name_lower:
            name_bonus = 0.5
        else:
            # Check singular/plural in name
            for variant in expand_word_variants(query_lower):
                if variant in name_lower:
                    name_bonus = 0.4
                    break

        model_keyword_scores[model_name] = min(ratio + name_bonus, 1.0)
        model_matched_words[model_name] = list(original_matched)

    return model_keyword_scores, model_matched_words


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Find thumbnail for model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_thumbnail(model_name, best_view=None):
    """Find thumbnail across all render collection folders."""
    data = catalog.get(model_name, {})
    collection = data.get("collection", "")

    # Search in the collection's render folder
    for folder in os.listdir(RENDERS_ROOT):
        folder_path = os.path.join(RENDERS_ROOT, folder)
        model_dir = os.path.join(folder_path, model_name)
        if os.path.isdir(model_dir):
            safe_folder = folder.replace(" ", "_")
            if best_view:
                return f"/renders/{safe_folder}/{model_name}/{best_view}"
            # Look for front view
            for fname in os.listdir(model_dir):
                if "front" in fname and fname.endswith(".png"):
                    return f"/renders/{safe_folder}/{model_name}/{fname}"
            # Fallback to first PNG
            for fname in sorted(os.listdir(model_dir)):
                if fname.endswith(".png"):
                    return f"/renders/{safe_folder}/{model_name}/{fname}"
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HYBRID SEARCH — combines all signals + conflict detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def hybrid_search(
    query_text: str = None,
    image_vec=None,
    top_k: int = 50,
    category_filter: str = None
):
    """
    4-layer hybrid scoring with CONFLICT DETECTION.
    When image and text disagree, boosts image weight (user's image
    is more likely their intent than a mistyped word).
    """

    has_text = query_text and query_text.strip()
    has_image = image_vec is not None

    img_scores = {}
    img_best_views = {}
    txt_scores = {}
    kw_scores = {}
    kw_matched = {}

    if has_image:
        img_scores, img_best_views = get_image_scores(image_vec)

    if has_text:
        text_vec = encode_text(query_text)
        txt_scores = get_text_scores(text_vec)
        kw_scores, kw_matched = get_keyword_scores(query_text)

    # ── CONFLICT DETECTION (hybrid mode only) ──
    conflict = False
    conflict_msg = ""

    if has_text and has_image:
        # Top 5 by image shape
        top_image = sorted(img_scores.items(), key=lambda x: -x[1])[:5]
        top_image_names = set(n for n, _ in top_image)

        # Top 5 by keyword match
        top_keyword = sorted(kw_scores.items(), key=lambda x: -x[1])[:5]
        top_keyword_names = set(n for n, _ in top_keyword)

        # If zero overlap between image and text top results → conflict
        overlap = top_image_names & top_keyword_names
        if len(overlap) == 0 and len(top_image_names) > 0 and len(top_keyword_names) > 0:
            conflict = True
            img_top = top_image[0][0] if top_image else ""
            txt_top = top_keyword[0][0] if top_keyword else ""
            img_cat = catalog.get(img_top, {}).get("category", "unknown")
            conflict_msg = f"Your image looks like '{img_cat}' but text says '{query_text}'. Showing results weighted toward your image."

    # ── WEIGHTS ──
    if has_text and has_image:
        if conflict:
            # CONFLICT: image dominates — text is likely wrong
            w_shape = 0.85
            w_text = 0.05
            w_keyword = 0.05
            w_name = 0.05
        else:
            # AGREEMENT: standard hybrid
            w_shape = 0.30
            w_text = 0.20
            w_keyword = 0.30
            w_name = 0.20
    elif has_text:
        w_shape = 0.0
        w_text = 0.35
        w_keyword = 0.40
        w_name = 0.25
    else:
        w_shape = 1.0
        w_text = 0.0
        w_keyword = 0.0
        w_name = 0.0

    # ── SCORE EVERY MODEL ──
    results = []
    for model_name in catalog.keys():
        model_data = catalog[model_name]
        category = model_data.get("category", "Other")

        if category_filter and category.lower() != category_filter.lower():
            continue

        shape_score = img_scores.get(model_name, 0)
        text_score = txt_scores.get(model_name, 0)
        keyword_score = kw_scores.get(model_name, 0)

        final_score = (
            shape_score * w_shape +
            text_score * w_text +
            keyword_score * (w_keyword + w_name)
        )

        best_view = img_best_views.get(model_name)
        thumbnail = get_thumbnail(model_name, best_view)

        results.append({
            "model_name": model_name,
            "score": round(final_score * 100, 1),
            "shape_score": round(shape_score * 100, 1),
            "text_score": round(text_score * 100, 1),
            "keyword_score": round(keyword_score * 100, 1),
            "keyword_match": kw_matched.get(model_name, []),
            "thumbnail": thumbnail,
            "description": model_data.get("description", ""),
            "tags": model_data.get("tags", []),
            "category": category,
            "blend_file": model_data.get("blend_file", ""),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "results": results[:top_k],
        "conflict": conflict,
        "conflict_msg": conflict_msg
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASTAPI APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="3D Model Search API — Hybrid",
    description="Search 3D models by text, image, or both combined",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(RENDERS_ROOT):
    # Mount each render collection subfolder
    for folder in os.listdir(RENDERS_ROOT):
        folder_path = os.path.join(RENDERS_ROOT, folder)
        if os.path.isdir(folder_path):
            safe_name = folder.replace(" ", "_")
            app.mount(f"/renders/{safe_name}", StaticFiles(directory=folder_path), name=f"renders_{safe_name}")


# ── ENDPOINTS ───────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": len(catalog),
        "image_vectors": image_index.ntotal,
        "text_vectors": text_index.ntotal,
        "device": DEVICE,
        "search_version": "hybrid_v2"
    }


@app.get("/categories")
async def list_categories():
    cats = {}
    for data in catalog.values():
        cat = data.get("category", "Other")
        cats[cat] = cats.get(cat, 0) + 1
    return {"categories": [
        {"name": c, "count": n}
        for c, n in sorted(cats.items(), key=lambda x: x[0])
    ]}


@app.get("/suggest")
async def suggest_tags(query: str = Query(..., min_length=2)):
    """
    Returns related tags as you type.
    Type "bag" → suggests: bag, handbag, purse, backpack, etc.
    """
    q = query.lower().strip()
    q_variants = expand_word_variants(q)

    # Collect all tags from catalog with their frequency
    tag_freq = {}
    for data in catalog.values():
        for tag in data.get("tags", []):
            t = tag.lower()
            tag_freq[t] = tag_freq.get(t, 0) + 1

    # Find tags that:
    # 1. Start with the query (prefix match)
    # 2. Contain the query (substring match)
    # 3. Are found alongside matching tags (co-occurrence)
    suggestions = {}

    # Step 1: Prefix and substring matches
    for tag, freq in tag_freq.items():
        for variant in q_variants:
            if tag.startswith(variant) or variant.startswith(tag):
                suggestions[tag] = freq * 3  # Strong match
                break
            elif variant in tag or tag in variant:
                suggestions[tag] = freq * 2  # Medium match
                break

    # Step 2: Co-occurring tags (tags that appear with matching tags)
    matching_tags = set(suggestions.keys())
    for data in catalog.values():
        model_tags = set(t.lower() for t in data.get("tags", []))
        if model_tags & matching_tags:
            # This model has a matching tag — add all its other tags
            for tag in model_tags:
                if tag not in matching_tags and tag not in suggestions:
                    suggestions[tag] = tag_freq.get(tag, 0)

    # Sort by score descending, limit to 10
    sorted_suggestions = sorted(suggestions.items(), key=lambda x: -x[1])[:10]

    return {
        "query": query,
        "suggestions": [{"tag": tag, "count": count} for tag, count in sorted_suggestions]
    }


@app.get("/models")
async def list_models(category: str = None):
    results = []
    for model_name, data in catalog.items():
        cat = data.get("category", "Other")
        if category and cat.lower() != category.lower():
            continue
        thumbnail = get_thumbnail(model_name)
        results.append({
            "model_name": model_name,
            "thumbnail": thumbnail,
            "description": data.get("description", ""),
            "tags": data.get("tags", []),
            "category": cat,
            "blend_file": data.get("blend_file", ""),
        })
    return {"models": results, "count": len(results)}


@app.get("/search/text")
async def search_text(
    query: str = Query(...),
    top_k: int = Query(50, ge=1, le=100),
    category: str = Query(None)
):
    """Text-only search."""
    data = hybrid_search(query_text=query, top_k=top_k, category_filter=category)
    return {"query": query, "mode": "text", "results": data["results"], "count": len(data["results"]),
            "conflict": data.get("conflict", False), "conflict_msg": data.get("conflict_msg", "")}


@app.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Query(50, ge=1, le=100)
):
    """Image-only search (shape match)."""
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_vec = encode_image(image)
    data = hybrid_search(image_vec=image_vec, top_k=top_k)
    return {"query": file.filename, "mode": "image", "results": data["results"], "count": len(data["results"]),
            "conflict": data.get("conflict", False), "conflict_msg": data.get("conflict_msg", "")}


@app.post("/search/hybrid")
async def search_hybrid(
    query: str = Form(default=None),
    file: UploadFile = File(default=None),
    top_k: int = Query(50, ge=1, le=100),
    category: str = Query(None)
):
    """
    Hybrid search — text + image combined.
    Detects conflicts when image and text disagree.
    """
    image_vec = None
    if file:
        contents = await file.read()
        if contents:
            image = Image.open(BytesIO(contents)).convert("RGB")
            image_vec = encode_image(image)

    has_text = query and query.strip()
    has_image = image_vec is not None
    if has_text and has_image:
        mode = "hybrid"
    elif has_text:
        mode = "text"
    elif has_image:
        mode = "image"
    else:
        return {"error": "Provide either text query, image, or both", "results": []}

    data = hybrid_search(
        query_text=query if has_text else None,
        image_vec=image_vec,
        top_k=top_k,
        category_filter=category
    )

    return {
        "query": query or (file.filename if file else ""),
        "mode": mode,
        "results": data["results"],
        "count": len(data["results"]),
        "conflict": data.get("conflict", False),
        "conflict_msg": data.get("conflict_msg", "")
    }


@app.get("/model/{model_name}")
async def get_model(model_name: str):
    if model_name not in catalog:
        return JSONResponse(status_code=404, content={"error": f"Model '{model_name}' not found"})
    data = catalog[model_name]

    # Find renders across all collection render folders
    thumbnails = []
    for folder in os.listdir(RENDERS_ROOT):
        folder_path = os.path.join(RENDERS_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        model_dir = os.path.join(folder_path, model_name)
        if os.path.isdir(model_dir):
            safe_folder = folder.replace(" ", "_")
            for fname in sorted(os.listdir(model_dir)):
                if fname.lower().endswith('.png'):
                    thumbnails.append(f"/renders/{safe_folder}/{model_name}/{fname}")
            break

    return {
        "model_name": model_name,
        "description": data.get("description", ""),
        "tags": data.get("tags", []),
        "category": data.get("category", "Other"),
        "blend_file": data.get("blend_file", ""),
        "thumbnails": thumbnails,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════════╗
    ║  3D Model Search API — Hybrid v2             ║
    ║                                              ║
    ║  Models  : {len(catalog):<33}║
    ║  Images  : {image_index.ntotal:<33}║
    ║  Device  : {DEVICE:<33}║
    ║                                              ║
    ║  Server  : http://localhost:{PORT:<19}║
    ║  Docs    : http://localhost:{PORT}/docs{' '*12}║
    ╚══════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host=HOST, port=PORT)