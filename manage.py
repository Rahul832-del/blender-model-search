"""
Product Collection Manager
============================
Reads collections.json for exact folder paths per collection.
Each collection has its own models, renders, tags, and embeddings folder.

SETUP:
  1. Edit collections.json — add your collection folder names
  2. Run commands below

COMMANDS:
  python manage.py list                                    — Show all collections + status
  python manage.py render  -c "Decorative 190"             — Render one collection
  python manage.py tag     -c "Decorative 190"             — Tag one collection
  python manage.py fix     -c "Decorative 190"             — Fix/clean tags
  python manage.py embed   -c "Decorative 190"             — Build embeddings
  python manage.py all     -c "Decorative 190"             — Full pipeline (render→tag→fix→embed)
  python manage.py merge                                   — Merge all into master index
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import shutil


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "collections.json")

MODELS_ROOT = os.path.join(BASE_DIR, "models")
RENDERS_ROOT = os.path.join(BASE_DIR, "renders")
TAGS_ROOT = os.path.join(BASE_DIR, "Tagged Json")
EMBEDDINGS_ROOT = os.path.join(BASE_DIR, "embeddings")
MASTER_DIR = os.path.join(EMBEDDINGS_ROOT, "master")

# Scripts
RENDER_SCRIPT = os.path.join(BASE_DIR, "renderview.py")
TAG_SCRIPT = os.path.join(BASE_DIR, "tag_models.py")
FIX_SCRIPT = os.path.join(BASE_DIR, "fix_catalog.py")
EMBED_SCRIPT = os.path.join(BASE_DIR, "build_embeddings.py")

# Blender
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"  ERROR: {CONFIG_FILE} not found!")
        print(f"  Create it with your collection folder names.")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        return json.load(f)


def get_collection(config, name):
    """Find a collection by name (case-insensitive partial match)."""
    for c in config["collections"]:
        if c["name"].lower() == name.lower():
            return c
    # Partial match
    matches = [c for c in config["collections"] if name.lower() in c["name"].lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"  Multiple matches: {[m['name'] for m in matches]}")
        return None
    print(f"  Collection '{name}' not found.")
    print(f"  Available: {[c['name'] for c in config['collections']]}")
    return None


def get_paths(collection):
    """Get full paths for a collection."""
    return {
        "name": collection["name"],
        "models": os.path.join(MODELS_ROOT, collection["models_folder"]),
        "renders": os.path.join(RENDERS_ROOT, collection["renders_folder"]),
        "tags": os.path.join(TAGS_ROOT, collection["tags_folder"]),
        "embeddings": os.path.join(EMBEDDINGS_ROOT, collection["embeddings_folder"]),
        "catalog_raw": os.path.join(TAGS_ROOT, collection["tags_folder"], "model_catalog.json"),
        "catalog_clean": os.path.join(TAGS_ROOT, collection["tags_folder"], "model_catalog_clean.json"),
        "renders_folder_name": collection["renders_folder"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_list(config):
    collections = config["collections"]
    print(f"\n  {'Collection':<25} {'Models':<10} {'Renders':<10} {'Tagged':<10} {'Embedded':<10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    total_models = 0
    for c in collections:
        p = get_paths(c)

        # Count .blend files
        blend_count = 0
        if os.path.isdir(p["models"]):
            blend_count = len(glob.glob(os.path.join(p["models"], "*.blend")))
        total_models += blend_count

        # Count rendered model folders
        render_count = 0
        if os.path.isdir(p["renders"]):
            render_count = len([d for d in os.listdir(p["renders"]) if os.path.isdir(os.path.join(p["renders"], d))])

        # Tag status
        if os.path.exists(p["catalog_clean"]):
            tagged = "Clean"
        elif os.path.exists(p["catalog_raw"]):
            tagged = "Raw"
        else:
            tagged = "No"

        # Embedding status
        embedded = "Yes" if os.path.exists(os.path.join(p["embeddings"], "image_index.faiss")) else "No"

        print(f"  {c['name']:<25} {blend_count:<10} {render_count:<10} {tagged:<10} {embedded:<10}")

    # Master status
    master_ok = os.path.exists(os.path.join(MASTER_DIR, "image_index.faiss"))
    print(f"\n  Total models: {total_models}")
    print(f"  Master index: {'Ready' if master_ok else 'Not built (run: python manage.py merge)'}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_render(p):
    if not os.path.isdir(p["models"]):
        print(f"  ERROR: Models folder not found: {p['models']}")
        return False

    print(f"\n  Rendering: {p['name']}")
    print(f"  Input:  {p['models']}")
    print(f"  Output: {p['renders']}\n")

    cmd = [
        BLENDER_EXE, "--background", "--python", RENDER_SCRIPT,
        "--", "--input", p["models"], "--output", p["renders"], "--skip"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_tag(p):
    if not os.path.isdir(p["renders"]):
        print(f"  ERROR: Renders not found: {p['renders']}")
        print(f"  Run render first.")
        return False

    os.makedirs(p["tags"], exist_ok=True)

    print(f"\n  Tagging: {p['name']}")
    print(f"  Renders: {p['renders']}")
    print(f"  Output:  {p['catalog_raw']}\n")

    cmd = [
        sys.executable, TAG_SCRIPT,
        "--renders", p["renders"],
        "--output", p["catalog_raw"],
        "--models-dir", p["models"],
        "--skip"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX TAGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_fix(p):
    if not os.path.exists(p["catalog_raw"]):
        print(f"  ERROR: Raw catalog not found: {p['catalog_raw']}")
        print(f"  Run tag first.")
        return False

    print(f"\n  Fixing tags: {p['name']}")
    print(f"  Input:  {p['catalog_raw']}")
    print(f"  Output: {p['catalog_clean']}\n")

    cmd = [
        sys.executable, FIX_SCRIPT,
        "--input", p["catalog_raw"],
        "--output", p["catalog_clean"]
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMBED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_embed(p):
    # Prefer clean catalog, fallback to raw
    catalog = p["catalog_clean"] if os.path.exists(p["catalog_clean"]) else p["catalog_raw"]
    if not os.path.exists(catalog):
        print(f"  ERROR: No catalog found. Run tag first.")
        return False

    os.makedirs(p["embeddings"], exist_ok=True)

    print(f"\n  Building embeddings: {p['name']}")
    print(f"  Catalog: {catalog}")
    print(f"  Output:  {p['embeddings']}\n")

    cmd = [
        sys.executable, EMBED_SCRIPT,
        "--catalog", catalog,
        "--output", p["embeddings"]
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALL (full pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_all(p):
    print(f"\n  ════════════════════════════════════════════")
    print(f"  Full pipeline: {p['name']}")
    print(f"  ════════════════════════════════════════════\n")

    print(f"  [1/4] Rendering...")
    if not cmd_render(p):
        print(f"  STOPPED: Render failed.")
        return

    print(f"\n  [2/4] Tagging...")
    if not cmd_tag(p):
        print(f"  STOPPED: Tagging failed.")
        return

    print(f"\n  [3/4] Fixing tags...")
    if not cmd_fix(p):
        print(f"  STOPPED: Fix failed.")
        return

    print(f"\n  [4/4] Building embeddings...")
    if not cmd_embed(p):
        print(f"  STOPPED: Embedding failed.")
        return

    print(f"\n  ════════════════════════════════════════════")
    print(f"  Pipeline complete: {p['name']}")
    print(f"  Now run: python manage.py merge")
    print(f"  ════════════════════════════════════════════\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MERGE ALL INTO MASTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_merge(config):
    import numpy as np
    import faiss

    print(f"\n  Merging all collections into master index...\n")

    all_img_emb = []
    all_txt_emb = []
    all_img_meta = []
    all_txt_meta = []
    master_catalog = {}
    loaded_count = 0

    for c in config["collections"]:
        p = get_paths(c)

        img_path = os.path.join(p["embeddings"], "image_embeddings.npy")
        txt_path = os.path.join(p["embeddings"], "text_embeddings.npy")
        img_meta_path = os.path.join(p["embeddings"], "image_metadata.json")
        txt_meta_path = os.path.join(p["embeddings"], "text_metadata.json")

        if not os.path.exists(img_path):
            print(f"  SKIP: {c['name']} (no embeddings)")
            continue

        print(f"  Loading: {c['name']}")
        loaded_count += 1

        # Load embeddings
        all_img_emb.append(np.load(img_path))
        all_txt_emb.append(np.load(txt_path))

        # Load metadata — add collection info
        with open(img_meta_path) as f:
            img_meta = json.load(f)
            for entry in img_meta:
                entry["collection"] = c["name"]
                entry["renders_folder"] = c["renders_folder"]
            all_img_meta.extend(img_meta)

        with open(txt_meta_path) as f:
            txt_meta = json.load(f)
            for entry in txt_meta:
                entry["collection"] = c["name"]
                entry["renders_folder"] = c["renders_folder"]
            all_txt_meta.extend(txt_meta)

        # Load catalog
        cat_path = p["catalog_clean"] if os.path.exists(p["catalog_clean"]) else p["catalog_raw"]
        if os.path.exists(cat_path):
            with open(cat_path) as f:
                cat = json.load(f)
                for model_name, data in cat.items():
                    data["collection"] = c["name"]
                    data["renders_folder"] = c["renders_folder"]
                    master_catalog[model_name] = data

    if not all_img_emb:
        print("  ERROR: No embeddings found!")
        return

    # Merge
    merged_img = np.vstack(all_img_emb).astype(np.float32)
    merged_txt = np.vstack(all_txt_emb).astype(np.float32)

    # Build FAISS
    os.makedirs(MASTER_DIR, exist_ok=True)

    img_index = faiss.IndexFlatIP(merged_img.shape[1])
    img_index.add(merged_img)
    faiss.write_index(img_index, os.path.join(MASTER_DIR, "image_index.faiss"))

    txt_index = faiss.IndexFlatIP(merged_txt.shape[1])
    txt_index.add(merged_txt)
    faiss.write_index(txt_index, os.path.join(MASTER_DIR, "text_index.faiss"))

    # Save
    np.save(os.path.join(MASTER_DIR, "image_embeddings.npy"), merged_img)
    np.save(os.path.join(MASTER_DIR, "text_embeddings.npy"), merged_txt)

    with open(os.path.join(MASTER_DIR, "image_metadata.json"), "w") as f:
        json.dump(all_img_meta, f, indent=2)
    with open(os.path.join(MASTER_DIR, "text_metadata.json"), "w") as f:
        json.dump(all_txt_meta, f, indent=2)
    with open(os.path.join(MASTER_DIR, "master_catalog.json"), "w") as f:
        json.dump(master_catalog, f, indent=2)

    print(f"""
    ╔══════════════════════════════════════════════╗
    ║  Master Index Built                          ║
    ║                                              ║
    ║  Collections : {loaded_count:<29}║
    ║  Models      : {len(master_catalog):<29}║
    ║  Image vecs  : {merged_img.shape[0]:<29}║
    ║  Text vecs   : {merged_txt.shape[0]:<29}║
    ║  Location    : {MASTER_DIR:<29}║
    ╚══════════════════════════════════════════════╝

    Now run: python search_api.py
    """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="Product Collection Manager")
    parser.add_argument("command", choices=["list", "render", "tag", "fix", "embed", "all", "merge"])
    parser.add_argument("--collection", "-c", help="Collection name")
    args = parser.parse_args()

    config = load_config()

    if args.command == "list":
        cmd_list(config)
        return

    if args.command == "merge":
        cmd_merge(config)
        return

    # All other commands need --collection
    if not args.collection:
        print("  ERROR: --collection required.")
        print(f"  Available: {[c['name'] for c in config['collections']]}")
        sys.exit(1)

    c = get_collection(config, args.collection)
    if not c:
        sys.exit(1)

    p = get_paths(c)

    if args.command == "render":
        cmd_render(p)
    elif args.command == "tag":
        cmd_tag(p)
    elif args.command == "fix":
        cmd_fix(p)
    elif args.command == "embed":
        cmd_embed(p)
    elif args.command == "all":
        cmd_all(p)


if __name__ == "__main__":
    main()