"""
PaliGemma Auto-Tagger for 3D Model Library
============================================
Generates 3 outputs per model:
  1. Filename tags       (from .blend name: Plush_Dinosaur_Toy → plush, dinosaur, toy)
  2. AI description      (PaliGemma detailed description for text search)
  3. AI keyword tags     (PaliGemma extracted keywords for filtering)

SETUP:
  pip install torch transformers pillow tqdm accelerate

USAGE:
  python tag_models.py --renders ./renders --output ./model_catalog.json

OPTIONS:
  --renders     Folder with rendered views from Step 1     (default: ./renders)
  --output      Output JSON file path                      (default: ./model_catalog.json)
  --models-dir  Original .blend files folder               (default: ./models)
  --device      cuda / cpu / auto                          (default: auto)
  --views       Which views to use (comma-separated)       (default: front,iso_front_right,back)
  --skip        Skip models already in existing catalog
"""

import os
import sys
import json
import re
import glob
import argparse
import time
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    parser = argparse.ArgumentParser(description="PaliGemma Auto-Tagger")
    parser.add_argument("--renders", default="./renders")
    parser.add_argument("--output", default="./model_catalog.json")
    parser.add_argument("--models-dir", default="./models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--views", default="front,iso_front_right,back")
    parser.add_argument("--skip", action="store_true")
    return parser.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD PALIGEMMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_model(device):
    """
    Load PaliGemma-3b-mix-448.
    First run downloads ~5.5GB. Cached after that.
    """
    model_name = "google/paligemma-3b-mix-448"
    print(f"  Loading PaliGemma: {model_name}")
    print(f"  Device: {device}")
    print(f"  (First run downloads ~5.5GB, please wait...)\n")

    processor = AutoProcessor.from_pretrained(model_name)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    print(f"  Model loaded!\n")
    return model, processor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASK PALIGEMMA A QUESTION ABOUT AN IMAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_paligemma(image, prompt, model, processor, device, max_tokens=256):
    """
    Send an image + text prompt to PaliGemma, get text response.
    """
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    # Decode — skip the input tokens to get only the generated part
    input_len = inputs["input_ids"].shape[-1]
    result = processor.decode(output[0][input_len:], skip_special_tokens=True)
    return result.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXTRACT TAGS FROM FILENAME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_filename_tags(model_name):
    """
    Extract keywords from .blend filename.
    'Plush_Dinosaur_Toy_002' → ['plush', 'dinosaur', 'toy']
    'Salt_and_pepper'        → ['salt', 'pepper']
    'ServingTray'            → ['serving', 'tray']
    'Photo_frame5'           → ['photo', 'frame']
    """
    # Remove trailing numbers like _001, _002, 5, 8
    clean = re.sub(r'[_]?\d+$', '', model_name)

    # Split on underscore and camelCase
    parts = re.sub(r'([a-z])([A-Z])', r'\1_\2', clean).split('_')

    tags = []
    for part in parts:
        word = part.lower().strip()
        if len(word) < 2:
            continue
        if word in {"and", "or", "the", "with", "for", "real", "new", "old"}:
            continue
        if word.isdigit():
            continue
        tags.append(word)

    return tags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXTRACT KEYWORD TAGS FROM AI TEXT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "that", "this", "these", "those", "it", "its",
    "image", "shows", "appears", "seems", "looks", "picture", "photo",
    "render", "rendering", "model", "object", "scene", "background",
    "gray", "grey", "white", "placed", "positioned", "sitting", "shown",
    "visible", "also", "which", "what", "there", "their", "they",
    "about", "up", "down", "left", "right", "front", "back", "side",
    "top", "bottom", "one", "two", "three", "four", "five", "like",
    "made", "used", "using", "typically", "usually", "often", "appear",
    "features", "designed", "likely", "small", "large", "simple"
}


def extract_tags_from_text(text):
    """Extract meaningful keywords from AI-generated text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    words = text.split()

    word_counts = {}
    for word in words:
        word = word.strip("-").strip()
        if len(word) < 3 or word in STOP_WORDS or word.isdigit():
            continue
        word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

    tags = []
    seen = set()
    for word, _ in sorted_words:
        if word not in seen:
            tags.append(word)
            seen.add(word)
        if len(tags) >= 15:
            break

    return tags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GUESS CATEGORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY_KEYWORDS = {
    "bathroom": ["bathroom", "bath", "shower", "toilet", "sink", "faucet", "towel", "soap", "loofah", "tub", "scrub"],
    "kitchen": ["kitchen", "stove", "oven", "microwave", "fridge", "dish", "pot", "pan", "kettle", "toaster", "plate", "serving", "rice", "salt", "pepper", "spoon", "fork", "knife", "cup", "mug", "bowl"],
    "furniture": ["chair", "table", "desk", "sofa", "couch", "bed", "shelf", "cabinet", "drawer", "bench", "stool", "wardrobe", "rack"],
    "lighting": ["lamp", "light", "bulb", "chandelier", "lantern", "sconce"],
    "electronics": ["tv", "monitor", "screen", "computer", "laptop", "phone", "speaker", "radio", "camera"],
    "appliance": ["heater", "fan", "conditioner", "washer", "dryer", "vacuum", "machine", "rowing"],
    "toys": ["toy", "plush", "stuffed", "dinosaur", "elephant", "giraffe", "rino", "doll", "plastic"],
    "decoration": ["vase", "frame", "picture", "mirror", "clock", "plant", "flower", "candle", "statue", "photo", "scarf"],
    "structural": ["railing", "niche", "door", "window", "curtain", "gate", "fence"],
    "container": ["jar", "bottle", "box", "holder", "tray", "packet", "roll"],
}


def guess_category(model_name, all_tags):
    """Guess category from model name + all collected tags."""
    all_text = model_name.lower() + " " + " ".join(all_tags)

    best_category = "uncategorized"
    best_score = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in all_text)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIND VIEW IMAGES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def find_view_images(model_dir, preferred_views):
    """Find preferred view images, fallback to first 3 PNGs."""
    found = []
    for view in preferred_views:
        pattern = os.path.join(model_dir, f"*_{view}.png")
        matches = glob.glob(pattern)
        if matches:
            found.append(matches[0])

    if not found:
        all_pngs = sorted(glob.glob(os.path.join(model_dir, "*.png")))
        found = all_pngs[:3]

    return found


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROCESS ONE MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_model(model_name, model_dir, models_base_dir, preferred_views, model, processor, device):
    """
    Generate complete metadata for one model:
      - Filename tags (from .blend name)
      - AI description (PaliGemma detailed caption)
      - AI keyword tags (PaliGemma extracted keywords)
      - Category (auto-guessed)
    """
    # Find view images
    view_images = find_view_images(model_dir, preferred_views)
    if not view_images:
        print(f"    WARNING: No images found for {model_name}")
        return None

    # ── SOURCE 1: Filename tags ──
    filename_tags = extract_filename_tags(model_name)

    # ── SOURCE 2: AI description (ask PaliGemma to describe the object) ──
    # Use the best view (front) for detailed description
    best_image = Image.open(view_images[0]).convert("RGB")

    description = ask_paligemma(
        best_image,
        "Describe this 3D object in detail. What is it? What shape, material, and features does it have? What is its purpose?",
        model, processor, device,
        max_tokens=200
    )

    # ── SOURCE 3: AI keyword tags (ask PaliGemma for keywords) ──
    # Caption multiple views and extract keywords
    all_ai_text = description
    for img_path in view_images[1:]:  # Skip first, already used above
        try:
            img = Image.open(img_path).convert("RGB")
            caption = ask_paligemma(
                img,
                "What is this object? Describe it in one sentence.",
                model, processor, device,
                max_tokens=80
            )
            all_ai_text += " " + caption
        except Exception as e:
            print(f"    WARNING: Failed on {img_path}: {e}")

    ai_tags = extract_tags_from_text(all_ai_text)

    # ── MERGE ALL TAGS ──
    # Priority: filename tags first, then AI tags (no duplicates)
    seen = set()
    merged_tags = []
    for tag in filename_tags + ai_tags:
        if tag not in seen:
            merged_tags.append(tag)
            seen.add(tag)

    # ── CATEGORY ──
    category = guess_category(model_name, merged_tags)

    # ── FIND .blend FILE PATH ──
    blend_path = ""
    matches = glob.glob(os.path.join(models_base_dir, "**", f"{model_name}.blend"), recursive=True)
    if matches:
        blend_path = matches[0]

    # ── ALL RENDER PATHS ──
    all_renders = sorted(glob.glob(os.path.join(model_dir, "*.png")))

    return {
        "blend_file": blend_path,
        "description": description,
        "filename_tags": filename_tags,
        "ai_tags": ai_tags,
        "tags": merged_tags,
        "category": category,
        "renders": all_renders,
        "views_used": [os.path.basename(v) for v in view_images]
    }


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

    preferred_views = [v.strip() for v in args.views.split(",")]

    # Find model directories
    model_dirs = sorted([
        d for d in os.listdir(args.renders)
        if os.path.isdir(os.path.join(args.renders, d))
    ])

    if not model_dirs:
        print(f"ERROR: No model folders found in {args.renders}")
        sys.exit(1)

    print(f"""
    ╔══════════════════════════════════════════════╗
    ║  PaliGemma Auto-Tagger                       ║
    ║                                              ║
    ║  Models found : {len(model_dirs):<28}║
    ║  Views/model  : {', '.join(preferred_views):<28}║
    ║  Device       : {device:<28}║
    ║  Output       : {args.output:<28}║
    ╚══════════════════════════════════════════════╝
    """)

    # Load existing catalog if --skip
    existing_catalog = {}
    if args.skip and os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing_catalog = json.load(f)
        print(f"  Loaded existing catalog: {len(existing_catalog)} models\n")

    # Load PaliGemma
    pg_model, processor = load_model(device)

    # Process each model
    catalog = dict(existing_catalog)
    failed = []
    start_time = time.time()

    def save_catalog():
        """Save current progress to disk."""
        with open(args.output, "w") as f:
            json.dump(catalog, f, indent=2)

    try:
        for i, model_name in enumerate(tqdm(model_dirs, desc="Tagging models")):
            if args.skip and model_name in catalog:
                continue

            model_dir = os.path.join(args.renders, model_name)

            try:
                result = process_model(
                    model_name, model_dir, args.models_dir,
                    preferred_views, pg_model, processor, device
                )

                if result:
                    catalog[model_name] = result
                else:
                    failed.append(model_name)

            except Exception as e:
                print(f"\n    FAIL: {model_name} — {e}")
                failed.append(model_name)

            # Checkpoint every 10 models
            if (i + 1) % 10 == 0:
                save_catalog()
                print(f"\n    [Saved: {len(catalog)} models]")

    except KeyboardInterrupt:
        # User pressed Ctrl+C — save everything before exiting
        print(f"\n\n    Interrupted! Saving {len(catalog)} models...")
        save_catalog()
        print(f"    Saved to {args.output}")
        print(f"    Resume with: python tag_models.py --skip")
        sys.exit(0)

    # Final save
    save_catalog()

    elapsed = time.time() - start_time

    print(f"""
    ╔══════════════════════════════════════════════╗
    ║  COMPLETE                                    ║
    ║                                              ║
    ║  Tagged        : {len(catalog):<27}║
    ║  Failed        : {len(failed):<27}║
    ║  Time          : {elapsed/60:.1f} min{' '*21}║
    ║  Output        : {args.output:<27}║
    ╚══════════════════════════════════════════════╝
    """)

    if failed:
        print("  Failed models:")
        for f_name in failed:
            print(f"    - {f_name}")

    # Print sample
    if catalog:
        sample_name = list(catalog.keys())[0]
        s = catalog[sample_name]
        print(f"\n  Sample — {sample_name}:")
        print(f"    Description  : {s['description'][:80]}...")
        print(f"    Filename tags: {', '.join(s['filename_tags'])}")
        print(f"    AI tags      : {', '.join(s['ai_tags'][:8])}")
        print(f"    All tags     : {', '.join(s['tags'][:10])}")
        print(f"    Category     : {s['category']}")


if __name__ == "__main__":
    main()