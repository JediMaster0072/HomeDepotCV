import json
from pathlib import Path

from PIL import Image
from google.cloud import vision


def run_google_ocr_words(image, current_image_name: str, debug: bool, debug_dir: Path, call_ocr_fn, parse_words_fn) -> list[dict]:
    """Call the local Google OCR wrapper and parse text annotations into SKU candidates."""
    annotations = call_ocr_fn(image)
    annotations = annotations["responses"][0].get("textAnnotations", [])

    if not annotations:
        return []

    save_google_ocr_annotations(
        annotations=annotations,
        debug=debug,
        debug_dir=debug_dir,
        file_name=Path(current_image_name).stem,
        output_name="google_ocr_text_annotations.json",
    )

    return parse_words_fn(annotations)


def run_google_ocr_raw(image, call_ocr_fn):
    """Call the local Google OCR wrapper and return the raw REST-style response."""
    annotations = call_ocr_fn(image)
    if not annotations:
        return []
    return annotations


def call_google_ocr_np(gcv_client, image, convert_image_bytes_fn):
    """Call Google Vision OCR using an in-memory NumPy/PIL image."""
    if image is None or image.size == 0:
        return None

    pil_image = Image.fromarray(image)
    jpg_image_bytes = convert_image_bytes_fn(pil_image)
    gcv_image = vision.Image(content=jpg_image_bytes)

    response = gcv_client.document_text_detection(
        image=gcv_image,
        image_context=vision.ImageContext(language_hints=["en-t-i0"]),
    )

    if response.error.message:
        raise RuntimeError(f"Google Vision OCR error: {response.error.message}")

    return list(response.text_annotations)


def save_google_ocr_annotations(
    annotations,
    debug: bool,
    debug_dir: Path,
    file_name: str = "",
    output_name: str = "google_ocr_annotations.json",
):
    """Save Google OCR annotations JSON for local debugging."""
    if not debug:
        return None

    image_stem = Path(file_name).stem if file_name else "in_memory_image"
    out_dir = debug_dir / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Saved Google OCR annotations: {out_path}")
    return out_path
