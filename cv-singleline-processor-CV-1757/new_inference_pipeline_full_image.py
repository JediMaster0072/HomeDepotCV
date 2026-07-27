import logging
import os
from typing import Union
from pathlib import Path

import cv2
import numpy as np

from google.cloud import vision
from services.image_inference import VertexModelInferenceService
from common_config import (
    BBOX_BUFFER_PCT,
    MAX_STRIPS,
    BoundingBox,
    OCRWordResult,
    PipelineResult,
    ClipTrack,
    CROPS_NOT_DETECTED,
    FINISHED,
    NO_SKUS_FOUND,
    CROPS_ENDPOINT_FAILED,
    SEG_ENDPOINT_FAILED,
)
from pipeline.strips import (
    center_pad as center_pad_strip,
    group_clip_items as group_clip_items_for_strips,
    create_strip as create_segmentation_strip,
)
from pipeline.assignment import (
    assign_word_to_track_in_original,
    bbox_center,
    point_in_bbox,
)
from pipeline.ocr_stage import (
    run_google_ocr_words,
    call_google_ocr_np,
)
from pipeline.detection_stage import (
    crop as crop_detection,
    preprocess as preprocess_crop,
    deblur as deblur_crop,
    denoise as denoise_crop,
)
from pipeline.detection_tracks import (
    predict_label_detections,
    build_detection_tracks,
    build_clip_items,
)
from pipeline.segmentation_stage import (
    mask_single_strip,
    predict_binary_strip_mask,
    process_binary_mask_with_rotation as expand_binary_mask_with_rotation,
    apply_mask_to_strip_preserve_unsegmented_clips,
    clip_has_segmentation,
    apply_mask_to_strip,
)
from pipeline.masked_original_builder import build_masked_original_image
from utils.validation_visualizer import save_validation_contact_sheet


# Pipeline order: 10
# Description: Owns the active production single-line image inference flow from label detection through OCR result assembly.
class HomeDepotInferencePipeline:
    # Pipeline order: 10
    # Description: Initializes pipeline thresholds, masking options, Vertex model access, and the Google OCR client.
    def __init__(
        self,
        bbox_buffer_pct: float = BBOX_BUFFER_PCT,
        max_strips: int = MAX_STRIPS,
        use_segmentation: bool = True,
        use_HD_OCR_parsing: bool = False,
        debug_output_dir=Path("./debug_outputs"),
        # iou_word_threshold: float = 0.10,
        # ---------------------------------------------------------------------------
        # Background / OCR-quality knobs for _build_masked_original_image
        # ---------------------------------------------------------------------------
        # "white"  – fill canvas with 255 (closest to document-scan appearance)
        # "black"  – fill canvas with 0   (original behaviour)
        # "blur"   – fill canvas with a heavily blurred copy of the raw image
        #            (natural image statistics; best OCR context preservation)
        background_mode: str = "white",
        # Expand every revealed ROI region by this many pixels on all sides AFTER
        # the seg mask is applied.  Gives OCR breathing room around character edges.
        # Set to 0 to disable.
        mask_roi_pad_px: int = 8,
        # If the ROI bounding-box area (in original-image pixels) is smaller than
        # this threshold, skip tight masking and reveal the full buffered crop
        # instead.  Prevents fine segmentation masks from hurting tiny labels.
        # Set to 0 to disable.
        small_label_area_threshold: int = 3000,  # 3000 # Change
        # Sigma for the Gaussian blur used when background_mode="blur".
        blur_sigma: int = 51,
    ) -> None:

        self.bbox_buffer_pct = bbox_buffer_pct
        self.max_strips = max_strips

        self._use_segmentation = use_segmentation
        self._use_HD_OCR_parsing = use_HD_OCR_parsing
        # self.iou_word_threshold = iou_word_threshold

        assert background_mode in ("white", "black", "blur"), f"background_mode must be 'white', 'black', or 'blur', got {background_mode!r}"
        self._background_mode = background_mode
        self._mask_roi_pad_px = mask_roi_pad_px
        self._small_label_area_threshold = small_label_area_threshold
        self._blur_sigma = blur_sigma

        self.debug = False
        self.debug_output_dir = Path(debug_output_dir)
        self.debug_validation = os.environ.get("CV_SINGLELINE_DEBUG_VALIDATION", "").lower() in ("1", "true", "yes")
        # self.iou_word_threshold = iou_word_threshold

        self.vertex_endpoint: VertexModelInferenceService = VertexModelInferenceService()

        self._gcv_client: vision.ImageAnnotatorClient = vision.ImageAnnotatorClient()

    # Pipeline order: Optional local/manual run
    # Description: Loads an image from disk and runs the active pipeline outside the Pub/Sub service.
    def run(self, image_path: str | Path) -> PipelineResult:
        image_path = Path(image_path)
        raw_image = self._load_image(image_path)
        return self.run_image(raw_image, file_name=str(image_path))

    @staticmethod
    # Pipeline order: Optional local/manual run
    # Description: Reads a local image file into an RGB NumPy image array.
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1]  # Convert BGR to RGB
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    # Pipeline order: 11
    # Description: Runs detection, crop extraction, strip creation, segmentation, masked-original OCR, and OCR-to-detection assignment for one image.
    def run_image(self, raw_image: np.ndarray, file_name: str = "", store_number: str = "") -> Union[str | Union[None, PipelineResult]]:
        print(f"Pipeline start → {file_name} | {store_number}")

        result = PipelineResult(metadata={"source": file_name or "<in-memory-image>"})

        # Step 1: Detect label regions in the full image.
        status, detections = self._predict_label_detections(raw_image, file_name, store_number)
        if status is not None:
            return status, None

        # Step 2: Extract buffered crops and initialize ClipTrack records.
        crops, tracked_clips = self._build_detection_tracks(raw_image, detections)

        print(f"Tracking initialized for {len(tracked_clips)} detection(s)")
        result.metadata["tracked_clips"] = tracked_clips

        # Step 3: Create strips of crops. Each strip contains a maximum of `self.max_strips` crops.
        # IMPORTANT:
        # Do not pad with a global max_h/max_w across all crops.
        # During training, padding was computed per strip/group, so we keep raw crops here
        # and let create_strip() compute group-local max_h/max_w.
        clip_items = build_clip_items(crops, tracked_clips)

        # Split raw crops into groups of N. Padding happens inside create_strip().
        groups = self.group_clip_items(clip_items, max_per_strip=self.max_strips)

        result.metadata["tracked_clips"] = tracked_clips
        result.metadata["num_groups"] = len(groups)

        # Step 4. For each group, run segmentation on the strip to locate SKU ROIs.
        # Instead of stacking strips for OCR, we paint the ROI regions directly onto
        # a black canvas of the original image — preserving full original resolution
        # and eliminating all strip coordinate remapping.

        # strip_batches = []
        all_groups = []
        for gid, group in enumerate(groups):
            strip = self.create_strip(group, strip_index=gid)
            _ = self._mask_single_strip(strip=strip, group=group, file_name=file_name, store_number=store_number)
            all_groups.append(group)

        # Step 5. Build a masked version of the original image.
        # Each segmented ROI region (mapped back to original image coordinates) is
        # kept at full resolution; everything else is blacked out.
        masked_original = self._build_masked_original_image(raw_image, all_groups)
        result.metadata["num_ocr_calls"] = 1

        # Step 6. One OCR call on the masked original image (full resolution, no stacking).
        ocr_words = self._run_google_ocr_words(masked_original)
        enhanced_retry_used = any(str(word.get("source", "")).endswith("_enhanced_retry") for word in ocr_words)
        rot180_retry_used = any(str(word.get("source", "")).endswith("_rot180_retry") for word in ocr_words)
        result.metadata["ocr_enhanced_retry_used"] = enhanced_retry_used
        result.metadata["ocr_rot180_retry_used"] = rot180_retry_used
        if rot180_retry_used:
            result.metadata["num_ocr_calls"] = 3
        elif enhanced_retry_used:
            result.metadata["num_ocr_calls"] = 2

        # Step 7. OCR bounding boxes are already in original image coordinates —
        # no strip remapping needed. Assign each word to the nearest track by IoU/containment.
        all_tracks = [item["track"] for group in all_groups for item in group]

        for word in ocr_words:
            orig_bbox = word["bbox"]
            track = self._assign_word_to_track_in_original(orig_bbox, all_tracks)

            if track is None:
                continue

            track.ocr_found = True

            result.raw_ocr_results.append(
                OCRWordResult(
                    text=word["text"],
                    raw_text=word["raw_text"],
                    det_id=track.det_id,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    source="seg+google_ocr",
                    strip_bbox=None,  # not applicable in this mode
                    original_bbox=orig_bbox,
                    confidence=word["confidence"],
                )
            )

        result.metadata["tracked_clips"] = tracked_clips
        self._save_validation_contact_sheet(
            raw_image=raw_image,
            tracked_clips=tracked_clips,
            all_groups=all_groups,
            masked_original=masked_original,
            result=result,
            file_name=file_name,
        )

        status = NO_SKUS_FOUND if not result.raw_ocr_results else FINISHED
        return status, result

    # Pipeline order: 17
    # Description: Extracts a rectangular crop from an image using a buffered detection box.
    def crop(self, img, box):
        return crop_detection(img, box)

    # Pipeline order: Optional debug validation
    # Description: Writes a contact sheet of detections, crops, segmentation clips, masked OCR input, and final OCR boxes.
    def _save_validation_contact_sheet(
        self,
        raw_image: np.ndarray,
        tracked_clips: list[ClipTrack],
        all_groups: list,
        masked_original: np.ndarray,
        result: PipelineResult,
        file_name: str,
    ) -> None:
        if not self.debug_validation:
            return

        source_name = Path(str(file_name)).stem or "in_memory_image"
        output_path = self.debug_output_dir / f"{source_name}_validation_contact_sheet.jpg"

        try:
            save_validation_contact_sheet(
                raw_image=raw_image,
                tracks=tracked_clips,
                output_path=output_path,
                groups=all_groups,
                masked_original=masked_original,
                ocr_results=result.raw_ocr_results,
            )
            print(f"Saved validation contact sheet: {output_path}")
        except Exception as exc:
            print(f"Failed to save validation contact sheet for {file_name}: {exc}")

    # Pipeline order: 14
    # Description: Calls the detection endpoint and returns a fallback status when label crops cannot be detected.
    def _predict_label_detections(self, raw_image: np.ndarray, file_name: str, store_number: str):
        return predict_label_detections(self.vertex_endpoint, raw_image, file_name, store_number)

    # Pipeline order: 17
    # Description: Converts raw detections into preprocessed crops and ClipTrack records.
    def _build_detection_tracks(self, raw_image: np.ndarray, detections) -> tuple[list[np.ndarray], list[ClipTrack]]:
        return build_detection_tracks(
            raw_image=raw_image,
            detections=detections,
            bbox_buffer_pct=self.bbox_buffer_pct,
            crop_fn=self.crop,
            preprocess_fn=self._preprocess,
        )

    # Pipeline order: 20.1
    # Description: Places one crop in the center of a fixed-size padded canvas.
    def center_pad(self, img, target_h, target_w, pad_value=0):
        return center_pad_strip(img, target_h, target_w, pad_value)

    # Pipeline order: 19
    # Description: Splits detected crop items into small groups that become segmentation strips.
    def group_clip_items(self, items, max_per_strip=5):
        return group_clip_items_for_strips(items, max_per_strip)

    # Pipeline order: 20
    # Description: Builds one horizontal segmentation strip and records where each crop sits inside it.
    def create_strip(self, group, strip_index: int) -> np.ndarray:
        return create_segmentation_strip(group, strip_index)

    # Pipeline order: 21
    # Description: Runs segmentation for one strip and stores per-crop masked-strip clips for later original-image reconstruction.
    def _mask_single_strip(
        self,
        strip: np.ndarray,
        group,
        file_name: str,
        store_number: str,
    ) -> np.ndarray:
        return mask_single_strip(
            vertex_endpoint=self.vertex_endpoint,
            strip=strip,
            group=group,
            file_name=file_name,
            store_number=store_number,
        )

    # Pipeline order: 35
    # Description: Assigns an original-coordinate OCR word bbox back to the detected label track that most likely produced it.
    def _assign_word_to_track_in_original(
        self,
        word_bbox: BoundingBox,
        all_tracks: list,
    ) -> "ClipTrack | None":
        return assign_word_to_track_in_original(word_bbox, all_tracks)

    # Pipeline order: 28
    # Description: Reconstructs a full-resolution image where only segmented SKU regions remain visible for OCR.
    def _build_masked_original_image(
        self,
        raw_image: np.ndarray,
        all_groups: list,
    ) -> np.ndarray:
        return build_masked_original_image(
            raw_image=raw_image,
            all_groups=all_groups,
            background_mode=self._background_mode,
            mask_roi_pad_px=self._mask_roi_pad_px,
            small_label_area_threshold=self._small_label_area_threshold,
            blur_sigma=self._blur_sigma,
        )

    # Pipeline order: Optional older strip-OCR path
    # Description: Stacks masked strips vertically for the older OCR strategy that does not use the masked original image.
    def _stack_masked_strips_vertically(
        self,
        strip_batches,
        y_gap: int = 30,
        pad_value: int = 0,
    ) -> tuple[np.ndarray, list[dict]]:
        if not strip_batches:
            return np.zeros((0, 0, 3), dtype=np.uint8), []

        max_w = max(batch["masked_strip"].shape[1] for batch in strip_batches)
        total_h = sum(batch["masked_strip"].shape[0] for batch in strip_batches)
        total_h += y_gap * (len(strip_batches) - 1)
        dtype = strip_batches[0]["masked_strip"].dtype

        stacked = np.full((total_h, max_w, 3), pad_value, dtype=dtype)
        layouts = []

        cursor_y = 0
        for batch in strip_batches:
            strip = batch["masked_strip"]
            h, w = strip.shape[:2]

            stacked[cursor_y : cursor_y + h, 0:w] = strip
            layouts.append({"group": batch["group"], "y_offset": cursor_y, "height": h, "width": w})

            cursor_y += h + y_gap

        return stacked, layouts

    # Pipeline order: Optional older strip-OCR path
    # Description: Finds which stacked strip contains a Google OCR word bbox.
    def _find_strip_layout_for_word(self, word_bbox: BoundingBox, strip_layouts: list[dict]) -> dict | None:
        cx, cy = self._bbox_center(word_bbox)

        for layout in strip_layouts:
            y1 = layout["y_offset"]
            y2 = y1 + layout["height"]

            if y1 <= cy < y2 and 0 <= cx < layout["width"]:
                return layout

        return None

    # Pipeline order: Optional older strip-OCR path
    # Description: Converts a word bbox from stacked-strip coordinates into local strip coordinates.
    def _stacked_bbox_to_strip_local(self, word_bbox: BoundingBox, y_offset: int) -> BoundingBox:
        return BoundingBox(
            x1=int(word_bbox.x1),
            y1=int(word_bbox.y1 - y_offset),
            x2=int(word_bbox.x2),
            y2=int(word_bbox.y2 - y_offset),
            confidence=word_bbox.confidence,
            class_id=word_bbox.class_id,
            class_name=word_bbox.class_name,
        )

    # Pipeline order: 22
    # Description: Calls the segmentation endpoint and normalizes its response into a binary strip mask.
    def _predict_binary_strip_mask(self, strip: np.ndarray, file_name: str = "", store_number: str = "") -> np.ndarray | None:
        return predict_binary_strip_mask(self.vertex_endpoint, strip, file_name, store_number)

    @staticmethod
    # Pipeline order: 26
    # Description: Expands rotated segmentation mask regions so OCR keeps enough character context.
    def process_binary_mask_with_rotation(binary_mask: np.ndarray, masks_np=None, scale_factor_w=1.5, scale_factor_h=1.35) -> np.ndarray:
        return expand_binary_mask_with_rotation(binary_mask, masks_np, scale_factor_w, scale_factor_h)

    # Pipeline order: 27
    # Description: Applies segmentation masks to crops with valid masks and stores masked clips for full-image reconstruction.
    def _apply_mask_to_strip_preserve_unsegmented_clips(
        self,
        strip: np.ndarray,
        mask: np.ndarray | None,
        group,
    ) -> np.ndarray:
        return apply_mask_to_strip_preserve_unsegmented_clips(strip, mask, group)

    # Pipeline order: 27.1
    # Description: Determines whether one crop has enough mask pixels to trust segmentation for that crop.
    def _clip_has_segmentation(
        self,
        mask: np.ndarray,
        track: ClipTrack,
        min_white_pixels: int = 20,
        min_white_ratio: float = 0.001,
    ) -> bool:
        return clip_has_segmentation(mask, track, min_white_pixels, min_white_ratio)

    # Pipeline order: Optional helper
    # Description: Applies a binary mask directly to a strip without per-crop fallback logic.
    def _apply_mask_to_strip(self, strip: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return apply_mask_to_strip(strip, mask)

    # def _run_google_ocr_words(self, image: np.ndarray) -> list[dict]:
    #     """
    #     Run Google Vision OCR on an in-memory NumPy image and return word-level boxes.

    #     Returns:
    #     [
    #         {
    #             "text": "...",
    #             "bbox": BoundingBox(...),
    #             "confidence": 1.0,
    #         },
    #         ...
    #     ]
    #     """
    #     annotations = self._call_google_ocr_np(image)

    #     if not annotations:
    #         return []

    #     words = []

    #     # annotations[0] = full text block
    #     # annotations[1:] = individual detected text elements
    #     for item in annotations[1:]:
    #         text = item.description.strip() if item.description else ""

    #         if not text:
    #             continue

    #         vertices = item.bounding_poly.vertices

    #         if len(vertices) < 4:
    #             continue

    #         xs = [v.x for v in vertices]
    #         ys = [v.y for v in vertices]

    #         x1, x2 = min(xs), max(xs)
    #         y1, y2 = min(ys), max(ys)

    #         words.append(
    #             {
    #                 "text": text,
    #                 "bbox": BoundingBox(
    #                     x1=int(round(x1)),
    #                     y1=int(round(y1)),
    #                     x2=int(round(x2)),
    #                     y2=int(round(y2)),
    #                 ),
    #                 "confidence": 1.0,
    #             }
    #         )

    #     return words

    # Pipeline order: 29
    # Description: Runs Google OCR on the masked original image and returns parsed SKU candidates.
    def _run_google_ocr_words(self, image: np.ndarray) -> list[dict]:
        return run_google_ocr_words(self._gcv_client, image)

    # Pipeline order: 30
    # Description: Encodes a NumPy image and calls Google Vision document text detection.
    def _call_google_ocr_np(self, image: np.ndarray) -> list[dict]:
        return call_google_ocr_np(self._gcv_client, image)

    # Pipeline order: Optional older strip-OCR path
    # Description: Assigns a strip-local OCR word bbox to the crop track containing its center point.
    def _assign_word_to_track(self, word_bbox: BoundingBox, group) -> ClipTrack | None:
        cx, cy = self._bbox_center(word_bbox)

        for item in group:
            track = item["track"]
            if track.crop_strip_bbox and self._point_in_bbox(cx, cy, track.crop_strip_bbox):
                return track

        return None

    # Pipeline order: Shared helper
    # Description: Computes the center point of a bounding box.
    def _bbox_center(self, bbox: BoundingBox) -> tuple[float, float]:
        return bbox_center(bbox)

    # Pipeline order: Shared helper
    # Description: Checks whether a point lies inside a bounding box.
    def _point_in_bbox(self, x: float, y: float, bbox: BoundingBox) -> bool:
        return point_in_bbox(x, y, bbox)

    # Pipeline order: Optional older strip-OCR path
    # Description: Maps OCR word coordinates from strip space back to original image space.
    def _map_strip_word_to_original(self, word_bbox: BoundingBox, track: ClipTrack) -> BoundingBox:
        """
        strip coords -> original image coords
        """
        if track.strip_bbox is None:
            raise ValueError("track.strip_bbox missing")
        if track.buffered_bbox is None:
            raise ValueError("track.buffered_bbox missing")

        strip_x1 = track.strip_bbox.x1
        strip_y1 = track.strip_bbox.y1

        pad_x = track.pad_offset_x or 0
        pad_y = track.pad_offset_y or 0

        crop_orig_x1 = track.buffered_bbox.x1
        crop_orig_y1 = track.buffered_bbox.y1

        return BoundingBox(
            x1=int(word_bbox.x1 - strip_x1 - pad_x + crop_orig_x1),
            y1=int(word_bbox.y1 - strip_y1 - pad_y + crop_orig_y1),
            x2=int(word_bbox.x2 - strip_x1 - pad_x + crop_orig_x1),
            y2=int(word_bbox.y2 - strip_y1 - pad_y + crop_orig_y1),
            # confidence=track.confidence,
            class_id=track.class_id,
            class_name=track.class_name,
        )

    # Pipeline order: 18
    # Description: Applies crop-level cleanup before segmentation and OCR.
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess_crop(image)

    @staticmethod
    # Pipeline order: 18.1
    # Description: Sharpens a crop by subtracting a blurred version from the original.
    def _deblur(image: np.ndarray) -> np.ndarray:
        return deblur_crop(image)

    @staticmethod
    # Pipeline order: 18.2
    # Description: Removes image noise from grayscale or color crops using OpenCV denoising.
    def _denoise(image: np.ndarray) -> np.ndarray:
        return denoise_crop(image)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python inference_pipeline.py <image_path>")
        sys.exit(1)

    pipeline = HomeDepotInferencePipeline(
        gcp_project_id="YOUR_GCP_PROJECT_ID",
        gcp_location="us-central1",
        od_endpoint_id="1234567890123456789",
        seg_endpoint_id="9876543210987654321",
        multiline_seg_endpoint_id="1122334455667788990",
        od_class_names={0: "Pallet", 1: "RDC", 2: "Printed_on_Box", 3: "Handwritten", 4: "Multiline_Label", 5: "Other"},
        seg_class_names={
            0: "RDC_SKU",
            1: "Pallet_SKU",
            2: "Printed_on_Box_SKU",
            3: "Handwritten_SKU",
        },
        multiline_class_name="Multiline_Label",
        od_conf_threshold=0.50,
        seg_conf_threshold=0.45,
        bbox_buffer_pct=0.15,
        max_strips=5,
        # iou_word_threshold=0.10,
    )

    result = pipeline.run(sys.argv[1])

    print(f"\n{'─' * 60}")
    print(f"  Fallback used   : {result.fallback_used}")
    print(f"  Strip fallbacks : {result.strip_fallbacks}")
    print(f"  OCR results     : {len(result.raw_ocr_results)}")
    print(f"{'─' * 60}")
    for r in result.raw_ocr_results:
        print(f"  [{r.class_name:<14}]  {r.text!r:<40}  ({r.source})")
