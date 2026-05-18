# Blender 3D Model Search Engine

A search engine for 3D Blender models using image embeddings and auto-tagging.

## How It Works
1. Blender `.blend` files are rendered into 16 images per model (`renderview.py`)
2. Images are auto-tagged using AI (`tag_models.py`)
3. Embeddings are generated from images + tags (`build_embeddings.py`)
4. Users search via text through a web UI (`search_ui.html`)

## Files
- `search_api.py` - Backend search API
- `search_ui.html` - Frontend search interface
- `build_embeddings.py` - Generates embeddings from rendered images
- `tag_models.py` - Auto-tags rendered images
- `renderview.py` - Renders 16 views from each .blend file
- `manage.py` - Project management script
- `collections.json` - Model collection metadata

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place .blend files in the `models/` folder
3. Run: `python manage.py`
4. Open `search_ui.html` to search

## Tech Stack
- Python
- CLIP / Sentence Transformers (for embeddings)
- Blender (for rendering)
