# Image & model assets

Binary images (~8.4 GB) and model checkpoints are **not** stored in this git repo.

## Restore images

Manifest: `scripts/image_data/image_manifest.jsonl`

```bash
# Golden / labelling images that have Azure Blob URLs
python scripts/download_images.py --urls-only

# All images from the original local zip (research outputs, stratified set, etc.)
python scripts/download_images.py --zip "/path/to/HomeDepotCV 2.zip"

# Prefer URLs, fall back to zip for everything else
python scripts/download_images.py --zip "/path/to/HomeDepotCV 2.zip"
```

## Restore model weights

```bash
python scripts/extract_models_from_zip.py --zip "/path/to/HomeDepotCV 2.zip"
```

## Refresh the manifest

```bash
python scripts/build_image_manifest.py --zip "/path/to/HomeDepotCV 2.zip"
```
