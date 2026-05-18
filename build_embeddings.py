"""
CLIP Embedding + FAISS Index Builder
======================================
Builds two search indexes from your model_catalog_clean.json:
  1. IMAGE INDEX  — CLIP encodes all rendered PNGs (for image search)
  2. TEXT INDEX   — CLIP encodes descriptions + tags (for text search)

SETUP:
  pip install torch transformers pillow numpy faiss-cpu tqdm

USAGE:
  python build_embeddings.py --catalog ./model_catalog_clean.json --output ./embeddings

OUTPUT:
  embeddings/
    image_embeddings.npy     — All image vectors
    image_metadata.json      — Which image belongs to which model
    text_embeddings.npy      — All text vectors (one per model)
    text_metadata.json       — Model name + description for each vector
    image_index.faiss        — FAISS index for image search
    text_index.faiss         — FAISS index for text search
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    parser = argparse.ArgumentParser(description="Build CLIP embeddings + FAISS indexes")
    parser.add_argument("--catalog", default="./model_catalog_clean.json")
    parser.add_argument("--output", default="./embeddings")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD CLIP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_clip(model_name, device):
    print(f"  Loading CLIP: {model_name}")
    print(f"  Device: {device}\n")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    print(f"  CLIP loaded!\n")
    return model, processor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENCODE IMAGES IN BATCHES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def encode_images(image_paths, model, processor, device, batch_size=16):
    """Encode all images through CLIP, return normalized vectors."""
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Encoding images"):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"    WARNING: Failed to load {path}: {e}")
                images.append(Image.new("RGB", (512, 512), (128, 128, 128)))

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            # Handle both tensor and object return types
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output if hasattr(emb, 'pooler_output') else emb.last_hidden_state[:, 0]
            emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENCODE TEXT IN BATCHES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def encode_texts(texts, model, processor, device, batch_size=32):
    """Encode all text descriptions through CLIP, return normalized vectors."""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding text"):
        batch_texts = texts[i:i + batch_size]

        # CLIP has a 77-token limit — truncate long texts
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = model.get_text_features(**inputs)
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output if hasattr(emb, 'pooler_output') else emb.last_hidden_state[:, 0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD FAISS INDEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_faiss_index(embeddings):
    """Build a FAISS index for cosine similarity search."""
    import faiss

    dimension = embeddings.shape[1]

    if embeddings.shape[0] < 50000:
        # Flat index — exact search, fast enough for <50K vectors
        index = faiss.IndexFlatIP(dimension)
    else:
        # IVF index — approximate search for larger datasets
        nlist = int(np.sqrt(embeddings.shape[0]))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 10

    index.add(embeddings)
    return index


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD SEARCH TEXT FOR EACH MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_search_text(model_data):
    """
    Combine description + tags + category into one search string.
    This is what gets encoded by CLIP for text search.

    Example output:
      "A wall-mounted bathroom heater with ventilation grills.
       heater bathroom wall-mount ventilation appliance.
       HVAC & Climate"
    """
    parts = []

    # Description (most important)
    desc = model_data.get("description", "")
    if desc and "unanswerable" not in desc.lower():
        parts.append(desc)

    # Tags as space-separated words
    tags = model_data.get("tags", [])
    if tags:
        parts.append(" ".join(tags))

    # Category
    cat = model_data.get("category", "")
    if cat and cat != "Other":
        parts.append(cat)

    return ". ".join(parts) if parts else "3D object"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load catalog
    with open(args.catalog, "r") as f:
        catalog = json.load(f)

    print(f"""
    ╔══════════════════════════════════════════════╗
    ║  CLIP Embedding + FAISS Index Builder         ║
    ║                                              ║
    ║  Models in catalog: {len(catalog):<24}║
    ║  CLIP model       : {args.clip_model:<24}║
    ║  Device           : {device:<24}║
    ║  Output dir       : {args.output:<24}║
    ╚══════════════════════════════════════════════╝
    """)

    os.makedirs(args.output, exist_ok=True)

    # Load CLIP
    clip_model, processor = load_clip(args.clip_model, device)

    # ─────────────────────────────────────────
    # PART 1: IMAGE EMBEDDINGS
    # ─────────────────────────────────────────
    print("  ── Part 1: Image embeddings ──\n")

    image_paths = []
    image_metadata = []  # Track which image belongs to which model

    for model_name, data in catalog.items():
        renders = data.get("renders", [])
        for render_path in renders:
            if os.path.exists(render_path):
                image_paths.append(render_path)
                image_metadata.append({
                    "model_name": model_name,
                    "image_file": os.path.basename(render_path),
                    "image_path": render_path
                })

    print(f"  Found {len(image_paths)} images from {len(catalog)} models\n")

    if not image_paths:
        print("  ERROR: No image files found! Check render paths in catalog.")
        sys.exit(1)

    # Encode all images
    image_embeddings = encode_images(image_paths, clip_model, processor, device, args.batch_size)
    print(f"  Image embeddings shape: {image_embeddings.shape}")

    # Save image embeddings + metadata
    np.save(os.path.join(args.output, "image_embeddings.npy"), image_embeddings)
    with open(os.path.join(args.output, "image_metadata.json"), "w") as f:
        json.dump(image_metadata, f, indent=2)

    # Build FAISS image index
    import faiss
    image_index = build_faiss_index(image_embeddings)
    faiss.write_index(image_index, os.path.join(args.output, "image_index.faiss"))
    print(f"  Image FAISS index: {image_index.ntotal} vectors\n")

    # ─────────────────────────────────────────
    # PART 2: TEXT EMBEDDINGS
    # ─────────────────────────────────────────
    print("  ── Part 2: Text embeddings ──\n")

    text_entries = []
    text_metadata = []

    for model_name, data in catalog.items():
        search_text = build_search_text(data)
        text_entries.append(search_text)
        text_metadata.append({
            "model_name": model_name,
            "search_text": search_text,
            "category": data.get("category", "Other"),
            "tags": data.get("tags", []),
            "blend_file": data.get("blend_file", "")
        })

    print(f"  Prepared {len(text_entries)} text entries\n")

    # Encode all text
    text_embeddings = encode_texts(text_entries, clip_model, processor, device)
    print(f"  Text embeddings shape: {text_embeddings.shape}")

    # Save text embeddings + metadata
    np.save(os.path.join(args.output, "text_embeddings.npy"), text_embeddings)
    with open(os.path.join(args.output, "text_metadata.json"), "w") as f:
        json.dump(text_metadata, f, indent=2)

    # Build FAISS text index
    text_index = build_faiss_index(text_embeddings)
    faiss.write_index(text_index, os.path.join(args.output, "text_index.faiss"))
    print(f"  Text FAISS index: {text_index.ntotal} vectors\n")

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    total_size = 0
    for fname in os.listdir(args.output):
        fpath = os.path.join(args.output, fname)
        if os.path.isfile(fpath):
            total_size += os.path.getsize(fpath)

    print(f"""
    ╔══════════════════════════════════════════════╗
    ║  COMPLETE                                    ║
    ║                                              ║
    ║  Image vectors : {image_embeddings.shape[0]:<27}║
    ║  Text vectors  : {text_embeddings.shape[0]:<27}║
    ║  Vector dim    : {image_embeddings.shape[1]:<27}║
    ║  Total size    : {total_size / 1024 / 1024:.1f} MB{' '*22}║
    ║  Output dir    : {args.output:<27}║
    ╚══════════════════════════════════════════════╝

    Files created:
      {args.output}/image_embeddings.npy    — {image_embeddings.shape}
      {args.output}/image_metadata.json     — {len(image_metadata)} entries
      {args.output}/text_embeddings.npy     — {text_embeddings.shape}
      {args.output}/text_metadata.json      — {len(text_metadata)} entries
      {args.output}/image_index.faiss       — Image search index
      {args.output}/text_index.faiss        — Text search index
    """)


if __name__ == "__main__":
    main()