import logging
from typing import Union
from pathlib import Path

import cv2
import numpy as np

from google.cloud import vision
from services.image_processor import numpy_image_to_base64_png
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
from pipeline.ocr_stage import run_google_ocr_words


# Pipeline order: 10
# Description: Owns the single-line image inference flow from detected label crops through OCR result assembly.
class HomeDepotInferencePipeline:
    # Pipeline order: 10
    # Description: Initializes pipeline thresholds, Vertex model access, and the Google OCR client.
    def __init__(
        self,
        bbox_buffer_pct: float = BBOX_BUFFER_PCT,
        max_strips: int = MAX_STRIPS,
        # iou_word_threshold: float = 0.10,
    ) -> None:

        self.bbox_buffer_pct = bbox_buffer_pct
        self.max_strips = max_strips

        self.debug = False
        # self.iou_word_threshold = iou_word_threshold

        self.vertex_endpoint: VertexModelInferenceService = VertexModelInferenceService()

        self._gcv_client: vision.ImageAnnotatorClient = vision.ImageAnnotatorClient()

    # Pipeline order: Optional local/manual run
    # Description: Loads an image from disk and runs the pipeline outside the Pub/Sub service.
    def run(self, image_path: str | Path) -> PipelineResult:
        image_path = Path(image_path)
        raw_image = self._load_image(image_path)
        return self.run_image(raw_image, file_name=str(image_path))

    @staticmethod
    # Pipeline order: Optional local/manual run
    # Description: Reads a local image file into a NumPy image array.
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    # Pipeline order: 11
    # Description: Runs detection, crop extraction, strip creation, segmentation, OCR, coordinate mapping, and result assembly for one image.
    def run_image(self, raw_image: np.ndarray, file_name: str = "", store_number: str = "") -> Union[str | Union[None, PipelineResult]]:
        print(f"Pipeline start → {file_name} | {store_number}")

        result = PipelineResult(metadata={"source": file_name or "<in-memory-image>"})

        # Step 1: Get OD predictions on the image
        try:
            detections = self.vertex_endpoint.predict_detection(raw_image, file_name, store_number)
        except Exception as exc:
            print(f"OD endpoint unreachable: {exc} – falling back to original codebase.")
            detections = []
            status = CROPS_ENDPOINT_FAILED
            return status, None

        # Step 1.1: If no detections fallback on the original pipeline
        if not len(detections):
            status = CROPS_NOT_DETECTED
            print("No detections – full-image Google OCR fallback.")
            return status, None

        # Get image height and width.
        img_h, img_w = raw_image.shape[:2]

        # For each image, we record the
        # 1. ROI from OD model
        # 2. Additional info such as location, zero-pad offset, etc.
        crops = []
        tracked_clips: list[ClipTrack] = []

        # -----------------------------
        # Step 2: EXTRACT BUFFERED CROPS + INIT TRACKING
        # -----------------------------
        for det_id, det in enumerate(detections):
            # Step 2.1: add buffere region to the predicted bbox region crop before extraction.
            buffered_det = det.apply_buffer(img_w, img_h, pct=self.bbox_buffer_pct)

            # Step 2.2: Extract region
            crop = self.crop(raw_image, buffered_det)
            crop = self._preprocess(crop)

            # # Ensure crop has a valid size.
            # if crop.size == 0:
            #     continue

            # Get buffered crop's height and width
            h, w = crop.shape[:2]

            # Step 2.3: Record:
            #   detection id
            #   class
            #   original bbox coordinates
            #   buffered bbox coordinates
            #   Buffered Crop height and width

            track = ClipTrack(
                det_id=det_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                orig_bbox=det,
                buffered_bbox=buffered_det,
                clip_h=h,
                clip_w=w,
            )

            tracked_clips.append(track)

            # Save crop for later process.
            crops.append(crop)

        print(f"Tracking initialized for {len(tracked_clips)} detection(s)")
        result.metadata["tracked_clips"] = tracked_clips

        # # Step 2.4: If no valid crops available, fallback to original solution.
        # if not crops:
        #     logger.warning("All buffered crops were empty – full-image Google OCR fallback.")
        #     result.fallback_used = True
        #     result.raw_ocr_results = self._google_ocr_full_image(raw_image, class_name="unknown")
        #     return result

        # Step 3: Create strips of crops. Each strip contains a maximum of `self.max_strips` crops.
        # IMPORTANT:
        # Do not pad with a global max_h/max_w across all crops.
        # During training, padding was computed per strip/group, so we keep raw crops here
        # and let create_strip() compute group-local max_h/max_w.
        clip_items = []
        for track, crop in zip(tracked_clips, crops):
            clip_items.append(
                {
                    "track": track,
                    "crop": crop,
                }
            )

        # Split raw crops into groups of N. Padding happens inside create_strip().
        groups = self.group_clip_items(clip_items, max_per_strip=self.max_strips)

        result.metadata["tracked_clips"] = tracked_clips
        result.metadata["num_groups"] = len(groups)

        # Step 4. For each group, run segmentation and build masked strips.
        # OCR will run only once on all strips stacked vertically.
        strip_batches = []
        for gid, group in enumerate(groups):
            strip = self.create_strip(group, strip_index=gid)
            masked_strip = self._mask_single_strip(strip=strip, group=group, file_name=file_name, store_number=store_number)
            strip_batches.append({"group": group, "masked_strip": masked_strip})

        stacked_strip, strip_layouts = self._stack_masked_strips_vertically(strip_batches)
        result.metadata["num_ocr_calls"] = 1 if strip_layouts else 0

        # Step 5. One OCR call on the stacked strip image.
        ocr_words = self._run_google_ocr_words(stacked_strip)

        # Step 6. Map OCR words from stacked-strip coordinates -> strip-local -> original image.
        for word in ocr_words:
            stacked_bbox = word["bbox"]
            layout = self._find_strip_layout_for_word(stacked_bbox, strip_layouts)

            if layout is None:
                continue

            local_bbox = self._stacked_bbox_to_strip_local(stacked_bbox, y_offset=layout["y_offset"])
            track = self._assign_word_to_track(local_bbox, layout["group"])

            if track is None:
                continue

            orig_bbox = self._map_strip_word_to_original(local_bbox, track)
            track.ocr_found = True

            result.raw_ocr_results.append(
                OCRWordResult(
                    text=word["text"],
                    raw_text=word["raw_text"],
                    det_id=track.det_id,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    source="seg+google_ocr",
                    strip_bbox=local_bbox,
                    original_bbox=orig_bbox,
                    confidence=word["confidence"],
                )
            )

        result.metadata["tracked_clips"] = tracked_clips

        status = NO_SKUS_FOUND if not result.raw_ocr_results else FINISHED
        return status, result

    # Pipeline order: 17
    # Description: Extracts a rectangular crop from an image using a buffered detection box.
    def crop(self, img, box):
        return img[box.y1 : box.y2, box.x1 : box.x2]

    # Pipeline order: 20.1
    # Description: Places one crop in the center of a fixed-size padded canvas.
    def center_pad(self, img, target_h, target_w, pad_value=0):
        h, w = img.shape[:2]

        padded = np.full((target_h, target_w, 3), pad_value, dtype=img.dtype)

        y_offset = (target_h - h) // 2
        x_offset = (target_w - w) // 2

        padded[y_offset : y_offset + h, x_offset : x_offset + w] = img
        return padded, x_offset, y_offset

    # Pipeline order: 19
    # Description: Splits detected crop items into small groups that become segmentation strips.
    def group_clip_items(self, items, max_per_strip=5):
        return [items[i : i + max_per_strip] for i in range(0, len(items), max_per_strip)]

    # Pipeline order: 20
    # Description: Builds one horizontal segmentation strip and records where each crop sits inside it.
    def create_strip(self, group, strip_index: int) -> np.ndarray:
        """
        Build one segmentation strip the same way as training:
        - compute max_h/max_w only from clips inside this group
        - center-pad each clip to that group-local size
        - concatenate padded clips horizontally

        group: list[{"track": ClipTrack, "crop": np.ndarray}]
        """
        if not group:
            raise ValueError("Empty group passed to create_strip")

        group_max_h = max(item["crop"].shape[0] for item in group)
        group_max_w = max(item["crop"].shape[1] for item in group)

        strip_h = group_max_h
        strip_w = group_max_w * len(group)
        dtype = group[0]["crop"].dtype

        strip = np.zeros((strip_h, strip_w, 3), dtype=dtype)

        cursor_x = 0
        for slot_idx, item in enumerate(group):
            track = item["track"]
            crop = item["crop"]

            padded, x_offset, y_offset = self.center_pad(
                crop,
                target_h=group_max_h,
                target_w=group_max_w,
                pad_value=0,
            )

            # Keep this for debug/inspection compatibility if needed later.
            item["padded_crop"] = padded

            # These offsets are now group-local, not image-global.
            track.pad_h = group_max_h
            track.pad_w = group_max_w
            track.pad_offset_x = x_offset
            track.pad_offset_y = y_offset

            strip[:, cursor_x : cursor_x + group_max_w] = padded

            track.strip_index = strip_index
            track.strip_slot = slot_idx
            track.strip_bbox = BoundingBox(
                x1=cursor_x,
                y1=0,
                x2=cursor_x + group_max_w,
                y2=group_max_h,
                confidence=track.confidence,
                class_id=track.class_id,
                class_name=track.class_name,
            )

            track.crop_strip_bbox = BoundingBox(
                x1=cursor_x + x_offset,
                y1=y_offset,
                x2=cursor_x + x_offset + (track.clip_w or crop.shape[1]),
                y2=y_offset + (track.clip_h or crop.shape[0]),
                confidence=track.confidence,
                class_id=track.class_id,
                class_name=track.class_name,
            )

            cursor_x += group_max_w

        return strip

    # Pipeline order: 21
    # Description: Runs segmentation for one strip and returns a masked strip while preserving unsegmented crops.
    def _mask_single_strip(
        self,
        strip: np.ndarray,
        group,
        file_name: str,
        store_number: str,
    ) -> np.ndarray:

        # 1. get one binary mask for the whole strip
        mask = self._predict_binary_strip_mask(strip, file_name=file_name, store_number=store_number)

        # # fallback: no segmentation mask
        # if mask is None or mask.sum() == 0:
        #     if self.debug:
        #         vis = strip.copy()
        #         cv2.putText(
        #             vis,
        #             "No mask detected",
        #             (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7,
        #             (0, 0, 255),
        #             2,
        #         )
        #         self.save_debug_image(vis, "strips_viz", f"{strip_name}_no_mask.jpg")

        #     return results

        # # mark segmentation found for all clips in this strip
        # for item in group:
        #     item["track"].seg_found = True

        # if self.debug:
        #     self.save_debug_image(mask, "strips_viz", f"{strip_name}_mask.jpg")

        # # 2. apply mask to strip
        # masked_strip = self._apply_mask_to_strip(strip, mask)

        # =================================
        # If no mask at all, preserve the full strip and still run OCR.
        # This means no clip is blacked out.
        if mask is None or mask.sum() == 0:
            for item in group:
                item["track"].seg_found = False

            masked_strip = strip.copy()

        else:
            mask = self.process_binary_mask_with_rotation(mask)

            # if self.debug:
            #     self.save_debug_image(mask, "strips_viz", f"{file_name}_{store_number}_mask.jpg")

            # 2. apply mask to strip
            # Apply mask only to clips where segmentation exists.
            # Preserve original image region for clips without segmentation.
            masked_strip = self._apply_mask_to_strip_preserve_unsegmented_clips(
                strip=strip,
                mask=mask,
                group=group,
            )

        # if self.debug:
        #     self.save_debug_image(masked_strip, "rois", f"{file_name}_{store_number}_masked.jpg")

        return masked_strip

    # Pipeline order: 31
    # Description: Stacks all masked strips vertically so Google OCR can process them in one call.
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

    # Pipeline order: 34
    # Description: Finds which stacked strip contains a Google OCR word bbox.
    def _find_strip_layout_for_word(self, word_bbox: BoundingBox, strip_layouts: list[dict]) -> dict | None:
        cx, cy = self._bbox_center(word_bbox)

        for layout in strip_layouts:
            y1 = layout["y_offset"]
            y2 = y1 + layout["height"]

            if y1 <= cy < y2 and 0 <= cx < layout["width"]:
                return layout

        return None

    # Pipeline order: 35
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
        """
        Expected segmentation output:
        - a single binary mask of shape [H, W]
        - or something convertible into that
        """

        try:
            seg_output = self.vertex_endpoint.predict_segmentation(strip, file_name, store_number)
        except Exception as exc:
            status = SEG_ENDPOINT_FAILED
            print(f"{status} - SEG endpoint unreachable for {file_name} | {store_number}: {exc} – Continuing with no segmentation results.")
            return None

        if seg_output is None:
            print(f"parse_segmentation_outputs returned None for {file_name} | {store_number}")
            return None

        if isinstance(seg_output, np.ndarray):
            print(f"Successfully obtained segmentation mask with shape {seg_output.shape} for {file_name}")
            mask = seg_output
        else:
            print(f"Unexpected return type from parse_segmentation_outputs: {type(seg_output)}. Expected np.ndarray or None.")
            return None

        try:
            if mask.ndim == 3:
                mask = np.squeeze(mask)

            if mask.shape[:2] != strip.shape[:2]:
                raise ValueError(f"Mask shape {mask.shape[:2]} does not match strip shape {strip.shape[:2]} for {file_name} | {store_number}.")

            # masks_np = (masks_np > 0.5).astype(np.uint8) * 255
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        except ValueError:
            raise
        except Exception as e:
            print(f"Error processing segmentation mask: {type(e).__name__}: {e}")
            return None

    @staticmethod
    # Pipeline order: 26
    # Description: Expands rotated segmentation mask regions so OCR keeps enough character context.
    def process_binary_mask_with_rotation(binary_mask: np.ndarray, masks_np=None, scale_factor_w=1.5, scale_factor_h=1.35) -> np.ndarray:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue

            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)

            (cx, cy), (w, h), angle = rect

            # Increase box size : W 50%, H 35%
            w *= scale_factor_w
            h *= scale_factor_h

            enlarged_rect = ((cx, cy), (w, h), angle)

            # Get enlarged box points and clip to image boundaries.
            # Clipping after boxPoints is correct for any rotation angle —
            # the axis-aligned cx±w/2 / cy±h/2 approach only works at angle=0.
            img_h, img_w = binary_mask.shape[:2]
            box = cv2.boxPoints(enlarged_rect)
            box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
            box = box.astype(np.int32)

            # Fill enlarged rotated box
            binary_mask = cv2.drawContours(binary_mask, [box], 0, 255, cv2.FILLED)

        return binary_mask

    # Pipeline order: 27.1
    # Description: Determines whether one crop has enough mask pixels to trust segmentation for that crop.
    def _clip_has_segmentation(
        self,
        mask: np.ndarray,
        track: ClipTrack,
        min_white_pixels: int = 20,
        min_white_ratio: float = 0.001,
    ) -> bool:
        """
        Decide whether this clip has a valid segmentation region.

        Checks white pixels inside the actual crop region within the strip,
        not the full padded slot.
        """
        if track.crop_strip_bbox is None:
            return False

        box = track.crop_strip_bbox

        h, w = mask.shape[:2]

        x1 = max(0, box.x1)
        y1 = max(0, box.y1)
        x2 = min(w, box.x2)
        y2 = min(h, box.y2)

        if x2 <= x1 or y2 <= y1:
            return False

        clip_mask = mask[y1:y2, x1:x2]

        white_pixels = int(np.count_nonzero(clip_mask > 0))
        total_pixels = clip_mask.shape[0] * clip_mask.shape[1]

        if total_pixels == 0:
            return False

        white_ratio = white_pixels / total_pixels

        return white_pixels >= min_white_pixels and white_ratio >= min_white_ratio

    # Pipeline order: 27
    # Description: Applies the segmentation mask to crops with valid masks and leaves other crop regions visible.
    def _apply_mask_to_strip_preserve_unsegmented_clips(
        self,
        strip: np.ndarray,
        mask: np.ndarray,
        group,
    ) -> np.ndarray:
        """
        Apply segmentation mask only for clips where segmentation exists.

        For clips without segmentation, preserve the original clip region.
        """
        output = strip.copy()

        for item in group:
            track = item["track"]

            if track.crop_strip_bbox is None:
                continue

            box = track.crop_strip_bbox

            h, w = strip.shape[:2]

            x1 = max(0, box.x1)
            y1 = max(0, box.y1)
            x2 = min(w, box.x2)
            y2 = min(h, box.y2)

            if x2 <= x1 or y2 <= y1:
                continue

            has_seg = self._clip_has_segmentation(mask, track)

            if has_seg:
                track.seg_found = True

                clip = strip[y1:y2, x1:x2]
                clip_mask = mask[y1:y2, x1:x2]

                masked_clip = cv2.bitwise_and(clip, clip, mask=clip_mask)
                output[y1:y2, x1:x2] = masked_clip

            else:
                track.seg_found = False
                # Important: leave this clip region unchanged.
                output[y1:y2, x1:x2] = strip[y1:y2, x1:x2]

        return output

    # Pipeline order: Optional helper
    # Description: Applies a binary mask directly to a strip without per-crop fallback logic.
    def _apply_mask_to_strip(self, strip: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(strip, strip, mask=mask)

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

    # Pipeline order: 32
    # Description: Runs Google OCR on the stacked strip image and returns parsed SKU candidates.
    def _run_google_ocr_words(self, image: np.ndarray) -> list[dict]:
        """
        Run Google Vision OCR on an in-memory NumPy image and return validated SKU results.
        Uses upright → enhanced → ±5°/±10°/180° rotation retries with short-circuit.
        """
        return run_google_ocr_words(self._gcv_client, image)

    # Pipeline order: 32.1
    # Description: Encodes a NumPy image and calls Google Vision document text detection.
    def _call_google_ocr_np(self, image: np.ndarray) -> list[dict]:
        """
        Calls Google Vision OCR using an in-memory OpenCV / NumPy image.

        image: np.ndarray in OpenCV BGR format
        """
        if image is None or image.size == 0:
            return None

        _, png_image_bytes = numpy_image_to_base64_png(image, return_bytes=True)

        gcv_image = vision.Image(content=png_image_bytes)

        response = self._gcv_client.document_text_detection(
            image=gcv_image,
            image_context=vision.ImageContext(language_hints=["en-t-i0"]),
        )

        if response.error.message:
            raise RuntimeError(f"Google Vision OCR error: {response.error.message}")

        return list(response.text_annotations)

    # Pipeline order: 36
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
        return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)

    # Pipeline order: Shared helper
    # Description: Checks whether a point lies inside a bounding box.
    def _point_in_bbox(self, x: float, y: float, bbox: BoundingBox) -> bool:
        return bbox.x1 <= x < bbox.x2 and bbox.y1 <= y < bbox.y2

    # Pipeline order: 37
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
        image = self._deblur(image)
        image = self._denoise(image)
        # image = self._correct_tilt(image)
        return image

    @staticmethod
    # Pipeline order: 18.1
    # Description: Sharpens a crop by subtracting a blurred version from the original.
    def _deblur(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    @staticmethod
    # Pipeline order: 18.2
    # Description: Removes image noise from grayscale or color crops using OpenCV denoising.
    def _denoise(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(image, h=10)
        return cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)


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
