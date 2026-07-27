import os
import logging
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision

from call_google_ocr import call_google_ocr_api
from yolo_adaptor import LocalYOLOv7Endpoint, switch_repo

# from services.image_inference import VertexModelInferenceService
from services.image_processor import convert_image_bytes
from common_config import (
    BBOX_BUFFER_PCT,
    MAX_STRIPS,
    BoundingBox,
    OCRWordResult,
    PipelineResult,
    ClipTrack,
    NO_CROPS_DETECTED,
)

# from utils.google_ocr_utils import parse_google_ocr_words
from utils.google_ocr_utils_2 import parse_google_ocr_words
from app_2 import find_sku_entities, prepare_sku_result_json
from B64_image_encoding_decoding import numpy_image_to_base64_png, base64_png_to_numpy_image
from local_pipeline.assignment import (
    assign_word_to_track,
    assign_word_to_track_in_original,
    bbox_center,
    map_strip_word_to_original,
    point_in_bbox,
)
from local_pipeline.debug_outputs import (
    save_debug_image as save_local_debug_image,
    save_ocr_results_on_original_image as save_local_ocr_results_on_original_image,
)
from local_pipeline.detection_stage import (
    crop as crop_detection,
    deblur as deblur_crop,
    denoise as denoise_crop,
    enhance_to_gray,
    preprocess as preprocess_crop,
)
from local_pipeline.detection_tracks import (
    build_clip_items,
    build_detection_tracks,
    predict_local_detections,
)
from local_pipeline.masked_original_builder import build_masked_original_image
from local_pipeline.ocr_stage import (
    call_google_ocr_np,
    run_google_ocr_raw,
    run_google_ocr_words,
    save_google_ocr_annotations as save_local_google_ocr_annotations,
)
from local_pipeline.segmentation_stage import (
    apply_mask_to_strip,
    apply_mask_to_strip_preserve_unsegmented_clips,
    clip_has_segmentation,
    mask_single_strip,
    predict_binary_strip_mask,
)
from local_pipeline.strips import (
    center_pad as center_pad_strip,
    create_strip as create_segmentation_strip,
    group_clip_items as group_clip_items_for_strips,
)


logger = logging.getLogger(__name__)


class HomeDepotInferencePipeline:
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
        small_label_area_threshold: int = 3000,
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

        # switch_repo("/data/saivijaay/yolov7")
        switch_repo("/home/saivijaay.vk/CV/yolov7")
        self.od = LocalYOLOv7Endpoint(
            # weights="/data/saivijaay/Inference_Home_Depot/yolov7_singleline_best_220426.pt",
            weights="/data/saranjit/Harish_HD_dataset/Best_yolox_18may_trimmed_A4500.pt",
            class_names={
                0: "Pallet",
                1: "RDC",
                2: "Printed_on_Box",
                3: "Handwritten",
                4: "Multiline_Label",
                5: "Other",
            },
            device="cuda:0",
            conf=0.25,
            is_seg=False,
            yolov7_repo_path="/home/saivijaay.vk/CV/yolov7",
            half=True,
            imgsz=1280,
        )

        # Load SEG
        if self._use_segmentation:
            # switch_repo("/data/saivijaay/yolov7-segmentation")
            switch_repo("/home/saivijaay.vk/CV/yolov7-seg-mlflow")
            self.seg = LocalYOLOv7Endpoint(
                # weights="/home/saivijaay.vk/CV/Inference_Home_Depot/singleline-pipeline-seg-models_best.pt",
                weights="/data/saranjit/Harish_HD_dataset/20260518_segmentation_single_cls_best.pt",  # previous
                class_names={0: "SKU"},
                # class_names={
                #     0: "RDC_SKU",
                #     1: "Pallet_SKU",
                #     2: "Printed_on_Box_SKU",
                #     3: "Handwritten_SKU",
                # },
                device="cuda:1",
                # conf=0.35,
                conf=0.25,
                is_seg=True,
                imgsz=608,
                yolov7_repo_path="/home/saivijaay.vk/CV/yolov7-seg-mlflow",
            )

        self.debug = True
        self.debug_dir = debug_output_dir

        # self.vertex_endpoint: VertexModelInferenceService = VertexModelInferenceService()

        # self._gcv_client: vision.ImageAnnotatorClient = vision.ImageAnnotatorClient()

    def save_debug_image(self, img, subdir, name, image_name=None):
        return save_local_debug_image(
            img=img,
            debug=self.debug,
            debug_dir=self.debug_dir,
            current_image_name=self.current_image_name,
            subdir=subdir,
            name=name,
            image_name=image_name,
        )

    def run(self, image_path: str | Path) -> PipelineResult:
        image_path = Path(image_path)
        self.current_image_name = image_path.name
        raw_image = self._load_image(image_path)
        result = self.run_image(raw_image, source_name=str(image_path))
        self.save_ocr_results_on_original_image(
            raw_image=raw_image,
            result=result,
            file_path=image_path,
            output_name="ocr_original_bboxes.jpg",
        )
        return result

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1]
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        logger.debug("Loaded %s  shape=%s", path.name, img.shape)
        return img

    def run_image(self, raw_image: np.ndarray, source_name: str = "") -> PipelineResult:
        logger.info("Pipeline start → %s", source_name or "<in-memory-image>")

        result = PipelineResult(metadata={"source": source_name or "<in-memory-image>"})

        # Step 1: Get OD predictions on the image.
        status, detections = self._predict_local_detections(raw_image)
        if status is not None:
            return status

        # Step 2: Extract buffered crops and initialize ClipTrack records.
        crops, tracked_clips = self._build_detection_tracks(raw_image, detections)

        logger.info("Tracking initialized for %d detection(s)", len(tracked_clips))
        result.metadata["tracked_clips"] = tracked_clips

        # Step 2.4: If no valid crops available, fallback to original solution.
        if not crops:
            logger.warning("All buffered crops were empty – full-image Google OCR fallback.")
            result.fallback_used = True
            # result.raw_ocr_results = self._google_ocr_full_image(raw_image, class_name="unknown")
            return result

        # Step 3: Create strips of crops. Each strip contains a maximum of `self.max_strips` crops.
        #
        # IMPORTANT:
        # Do NOT pad using a global max_h/max_w across all crops.
        # During training, each strip/group used its own max_h/max_w computed only from
        # the clips inside that group. So we first group raw crops, then `create_strip`
        # computes group-local padding and records pad/strip metadata.
        clip_items = build_clip_items(crops, tracked_clips)

        # Split raw clips into groups of N. Padding happens inside create_strip().
        groups = self.group_clip_items(clip_items, max_per_strip=self.max_strips)

        result.metadata["tracked_clips"] = tracked_clips
        result.metadata["num_groups"] = len(groups)

        # Step 4. For each group, run segmentation on the strip to locate SKU ROIs.
        # Instead of stacking strips for OCR, we paint the ROI regions directly onto
        # a black canvas of the original image — preserving full original resolution
        # and eliminating all strip coordinate remapping.
        all_groups = []
        for gid, group in enumerate(groups):
            strip = self.create_strip(group, strip_index=gid)
            if self.debug:
                self.save_debug_image(
                    strip,
                    "strips",
                    f"strip_{gid}.jpg",
                    self.current_image_name,
                )

            masked_strip = self._mask_single_strip(strip=strip, group=group, strip_name=gid)

            if self.debug:
                self.save_debug_image(
                    masked_strip,
                    "rois",
                    f"strip_{gid}_masked.jpg",
                    self.current_image_name,
                )

            all_groups.append(group)

        # Step 5. Build a masked version of the original image.
        # Each segmented ROI region (mapped back to original image coordinates) is
        # kept at full resolution; everything else is blacked out.
        masked_original = self._build_masked_original_image(raw_image, all_groups)
        result.metadata["num_ocr_calls"] = 1

        if self.debug:
            self.save_debug_image(
                masked_original,
                "rois",
                "masked_original.jpg",
                self.current_image_name,
            )

        # Step 6. One OCR call on the masked original image (full resolution, no stacking).
        if self._use_HD_OCR_parsing:
            sku_text = self._run_google_ocr_words_orig(masked_original)
            skus_output = []

            sku_annotation_text = sku_text.get("responses")[0].get("fullTextAnnotation", False)

            if sku_annotation_text:
                skus, recovered_skus = find_sku_entities(sku_text, 10, "test")
                if len(skus) > 0:
                    sku_data, _ = prepare_sku_result_json(
                        skus,
                        "x",
                        sku_text,
                        recovered_skus,
                    )
                    sku_list = sku_data.skus_list
                    skus_output = [i["sku"] for i in sku_list if i.get("sku", None)]

            return skus_output
        else:
            ocr_words = self._run_google_ocr_words(masked_original)

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
            return result

    def crop(self, img, box):
        return crop_detection(img, box)

    def _predict_local_detections(self, raw_image: np.ndarray):
        return predict_local_detections(
            od_endpoint=self.od,
            raw_image=raw_image,
            encode_image_fn=numpy_image_to_base64_png,
        )

    def _build_detection_tracks(self, raw_image: np.ndarray, detections) -> tuple[list[np.ndarray], list[ClipTrack]]:
        return build_detection_tracks(
            raw_image=raw_image,
            detections=detections,
            bbox_buffer_pct=self.bbox_buffer_pct,
            crop_fn=self.crop,
            preprocess_fn=self._preprocess,
        )

    def center_pad(self, img, target_h, target_w, pad_value=0):
        return center_pad_strip(img, target_h, target_w, pad_value)

    def group_clip_items(self, items, max_per_strip=5):
        return group_clip_items_for_strips(items, max_per_strip)

    def create_strip(self, group, strip_index: int) -> np.ndarray:
        return create_segmentation_strip(
            group=group,
            strip_index=strip_index,
            debug_save_fn=self.save_debug_image if self.debug else None,
            current_image_name=self.current_image_name,
        )

    def _mask_single_strip(self, strip: np.ndarray, group, strip_name: str = "strip") -> np.ndarray:
        return mask_single_strip(
            seg_endpoint=self.seg,
            strip=strip,
            group=group,
            encode_image_fn=numpy_image_to_base64_png,
            decode_image_fn=base64_png_to_numpy_image,
            use_segmentation=self._use_segmentation,
            debug_save_fn=self.save_debug_image if self.debug else None,
            current_image_name=self.current_image_name,
            strip_name=strip_name,
        )

    # def _stack_masked_strips_vertically(self, strip_batches) -> tuple[np.ndarray, list[dict]]:
    #     if not strip_batches:
    #         return np.zeros((0, 0, 3), dtype=np.uint8), []

    #     max_w = max(batch["masked_strip"].shape[1] for batch in strip_batches)
    #     total_h = sum(batch["masked_strip"].shape[0] for batch in strip_batches)
    #     dtype = strip_batches[0]["masked_strip"].dtype

    #     stacked = np.zeros((total_h, max_w, 3), dtype=dtype)
    #     layouts = []

    #     cursor_y = 0
    #     for batch in strip_batches:
    #         strip = batch["masked_strip"]
    #         h, w = strip.shape[:2]

    #         stacked[cursor_y : cursor_y + h, 0:w] = strip
    #         layouts.append({"group": batch["group"], "y_offset": cursor_y, "height": h, "width": w})
    #         cursor_y += h

    #     return stacked, layouts

    def _stack_masked_strips_vertically(
        self,
        strip_batches,
        y_gap: int = 30,
        pad_value: int = 0,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Stack masked strips vertically for a single Google OCR call.

        Coordinate mapping remains valid because each layout stores the strip's
        y_offset, height, and real width.

        y_gap adds separation between strips so OCR does not accidentally merge
        text across strip boundaries.
        """
        if not strip_batches:
            return np.zeros((0, 0, 3), dtype=np.uint8), []

        max_w = max(batch["masked_strip"].shape[1] for batch in strip_batches)

        total_h = sum(batch["masked_strip"].shape[0] for batch in strip_batches)
        total_h += y_gap * (len(strip_batches) - 1)

        dtype = strip_batches[0]["masked_strip"].dtype

        stacked = np.full((total_h, max_w), pad_value, dtype=dtype)
        layouts = []

        cursor_y = 0
        for batch in strip_batches:
            strip = batch["masked_strip"]
            h, w = strip.shape[:2]

            stacked[cursor_y : cursor_y + h, 0:w] = strip

            layouts.append(
                {
                    "group": batch["group"],
                    "y_offset": cursor_y,
                    "height": h,
                    "width": w,
                }
            )

            cursor_y += h + y_gap

        return stacked, layouts

    def _find_strip_layout_for_word(self, word_bbox: BoundingBox, strip_layouts: list[dict]) -> dict | None:
        cx, cy = self._bbox_center(word_bbox)

        for layout in strip_layouts:
            y1 = layout["y_offset"]
            y2 = y1 + layout["height"]

            if y1 <= cy < y2 and 0 <= cx < layout["width"]:
                return layout

        return None

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

    def _predict_binary_strip_mask(self, strip: str) -> tuple[np.ndarray | None, np.ndarray]:
        return predict_binary_strip_mask(
            seg_endpoint=self.seg,
            strip=strip,
            decode_image_fn=base64_png_to_numpy_image,
            use_segmentation=self._use_segmentation,
        )

    def _apply_mask_to_strip_preserve_unsegmented_clips(
        self,
        strip: np.ndarray,
        mask: np.ndarray | None,
        group,
    ) -> np.ndarray:
        return apply_mask_to_strip_preserve_unsegmented_clips(strip, mask, group)

    def _clip_has_segmentation(
        self,
        mask: np.ndarray,
        track: ClipTrack,
        min_white_pixels: int = 40,
        min_white_ratio: float = 0.001,
    ) -> bool:
        return clip_has_segmentation(mask, track, min_white_pixels, min_white_ratio)

    def _apply_mask_to_strip(self, strip: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return apply_mask_to_strip(strip, mask)

    def _run_google_ocr_words(self, image: np.ndarray) -> list[dict]:
        return run_google_ocr_words(
            image=image,
            current_image_name=self.current_image_name,
            debug=self.debug,
            debug_dir=self.debug_dir,
            call_ocr_fn=call_google_ocr_api,
            parse_words_fn=parse_google_ocr_words,
        )

    def save_google_ocr_annotations(
        self,
        annotations,
        file_name: str = "",
        output_name: str = "google_ocr_annotations.json",
    ):
        return save_local_google_ocr_annotations(
            annotations=annotations,
            debug=getattr(self, "debug", False),
            debug_dir=Path(getattr(self, "debug_dir", Path("./debug_outputs"))),
            file_name=file_name,
            output_name=output_name,
        )

    def _run_google_ocr_words_orig(self, image: np.ndarray) -> list[dict]:
        return run_google_ocr_raw(image, call_google_ocr_api)

    def _call_google_ocr_np(self, image: np.ndarray) -> list[dict]:
        return call_google_ocr_np(self._gcv_client, image, convert_image_bytes)

    def _assign_word_to_track(self, word_bbox: BoundingBox, group) -> ClipTrack | None:
        return assign_word_to_track(word_bbox, group)

    def _bbox_center(self, bbox: BoundingBox) -> tuple[float, float]:
        return bbox_center(bbox)

    def _point_in_bbox(self, x: float, y: float, bbox: BoundingBox) -> bool:
        return point_in_bbox(x, y, bbox)

    def _map_strip_word_to_original(self, word_bbox: BoundingBox, track: ClipTrack) -> BoundingBox:
        return map_strip_word_to_original(word_bbox, track)

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

    def _assign_word_to_track_in_original(
        self,
        word_bbox: BoundingBox,
        all_tracks: list,
    ) -> "ClipTrack | None":
        return assign_word_to_track_in_original(word_bbox, all_tracks)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess_crop(image)

    @staticmethod
    def _deblur(image: np.ndarray) -> np.ndarray:
        return deblur_crop(image)

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        return denoise_crop(image)

    @staticmethod
    def _img_enhance(image: np.ndarray) -> np.ndarray:
        return enhance_to_gray(image)

    def save_ocr_results_on_original_image(
        self,
        raw_image: np.ndarray,
        result: PipelineResult,
        file_path: str = "",
        output_name: str = "ocr_original_bboxes.jpg",
    ) -> Path | None:
        return save_local_ocr_results_on_original_image(
            raw_image=raw_image,
            result=result,
            debug_dir=self.debug_dir,
            file_path=file_path,
            output_name=output_name,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    base_img_dir = "/home/saivijaay.vk/CV/Single-Line-Data/Test/Golden_Dataset_overhead_eval_orig"
    # base_img_dir = "/home/saivijaay.vk/CV/Single-Line-Data/Test/Single_Image"

    # output_root = Path("Inference_Home_Depot/debug_outputs_With_buffer/orig_ocr_parse/without_segmentation")

    imgs = os.listdir(base_img_dir)
    # imgs = ["1770341131504_0244_1116_11-013.jpg"] # 1770341131504_0244_1116_11-013, 1770339128729_0244_1169_08-003

    for use_segmentation, use_HD_OCR_parsing, output_root in (
        (True, False, Path("/home/saivijaay.vk/CV/Inference_Home_Depot/20260404_with_seg_mask_scaling_GOCR_orig_image_2/new_ocr_parse")),
    ):
        print("----------------------------------------------------------------------------------------")
        print(f"{use_HD_OCR_parsing = }, {use_segmentation = }, {output_root = }")
        print("----------------------------------------------------------------------------------------")

        pipeline = HomeDepotInferencePipeline(
            use_segmentation=use_segmentation,
            use_HD_OCR_parsing=use_HD_OCR_parsing,
            debug_output_dir=output_root,
            # ── Background / OCR-quality knobs ──────────────────────────────
            # Choose ONE of: "white" | "blur" | "black"
            background_mode="white",
            # Extra pixels to dilate the seg mask before painting (~8–12 works well)
            mask_roi_pad_px=8,
            # Reveal full crop for ROIs smaller than this area (px²); 0 = disable
            small_label_area_threshold=3000,
            # Blur sigma used when background_mode="blur" (must be odd, or set even
            # and the code will auto-correct to the next odd value)
            blur_sigma=51,
        )
        output_root.mkdir(exist_ok=True, parents=True, mode=511)

        for i in range(len(imgs)):  # len(imgs)
            img_name = imgs[i]
            img_path = os.path.join(base_img_dir, img_name)

            if not os.path.isfile(img_path):
                continue

            print(f"\nRunning inference on image {i + 1}/{len(imgs)}")
            # already_done = os.listdir(output_root)
            # if img_name.split(".")[0] in already_done:
            #     print(f"\nRunning inference on image {i + 1}/{len(imgs)} already done")
            #     continue

            result = pipeline.run(img_path)
            # print(f"{result = }")

            # 👉 Create folder with image name (without extension)
            img_stem = Path(img_name).stem
            img_folder = output_root / img_stem
            img_folder.mkdir(parents=True, exist_ok=True)

            # 👉 Create txt file inside that folder
            txt_path = img_folder / "results.txt"

            with open(txt_path, "w") as f:
                f.write(f"{img_path}\n")
                f.write(f"{'─' * 60}\n")
                f.write(f"Fallback used   : {result.fallback_used}\n")
                f.write(f"Strip fallbacks : {result.strip_fallbacks}\n")
                f.write(f"OCR results     : {len(result.raw_ocr_results)}\n")
                f.write(f"{'─' * 60}\n")

                for r in result.raw_ocr_results:
                    if use_HD_OCR_parsing:
                        f.write(f"[{'dummy':<14}]  {r!r:<40}  ({'dummy'})\n")
                    else:
                        f.write(f"[{r.class_name:<14}]  {r.text!r:<40}  ({r.source})\n")

            # break

        print("----------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------")
