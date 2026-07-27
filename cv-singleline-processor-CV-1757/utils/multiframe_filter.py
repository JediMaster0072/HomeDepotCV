"""
Multiframe SKU filtering module for handling beam-based position filtering.

This module processes beam position information from multi-frame images to filter out
SKUs that are outside the current bay/beam boundaries. It handles different frame types
(center, before, after) and beam configurations.

Enhanced features:
- Tilted beam support: Handles beams that appear tilted due to camera angles
- Sky-shelf top handling: Properly filters items at the top of shelves
- Robust validation: Validates bbox coordinates and handles edge cases
"""

import json
import logging
import math
from typing import Dict, List, Optional, Tuple

from google.cloud import storage

logger = logging.getLogger("sku_pipeline.multiframe_filter")


# Pipeline order: Optional multiframe stage
# Description: Validates that a bbox has four finite coordinates in the expected order.
def _validate_bbox(bbox: List[float]) -> bool:
    """
    Validate that a bounding box has valid coordinates.

    Args:
        bbox: [xmin, ymin, xmax, ymax]

    Returns:
        True if bbox is valid, False otherwise
    """
    try:
        if len(bbox) != 4:
            return False
        xmin, ymin, xmax, ymax = bbox
        # Check that coordinates are valid numbers and ordered correctly
        valid_coords = all(isinstance(v, (int, float)) for v in bbox)
        if not valid_coords:
            return False

        # Check for inf and nan without numpy
        for v in bbox:
            if math.isinf(v) or math.isnan(v):
                return False

        # Check ordering
        return xmin <= xmax and ymin <= ymax
    except (TypeError, ValueError):
        return False


# Pipeline order: Optional multiframe stage
# Description: Infers whether a beam should use left-leaning or right-leaning geometry based on image position.
def _infer_beam_side(beam_bbox: List[float], image_width: int) -> str:
    """
    Infer the correct diagonal direction for a beam based on its position in the image.

    Due to camera perspective, beams lean toward the bay center as they go up:
    - Beam in the right half of the image → leans bottom-right to top-left → "right"
    - Beam in the left half of the image  → leans bottom-left to top-right → "left"

    Args:
        beam_bbox: [xmin, ymin, xmax, ymax]
        image_width: Full width of the image in pixels

    Returns:
        "right" if beam center is in the right half of the image, "left" otherwise
    """
    if image_width <= 0:
        logger.warning(f"image_width={image_width} is invalid; defaulting beam_side to 'left'")
        return "left"
    beam_cx = (beam_bbox[0] + beam_bbox[2]) / 2.0
    return "right" if beam_cx > image_width / 2.0 else "left"


# Pipeline order: Optional multiframe stage
# Description: Reads the overhead-hints metadata from the GCS blob for the image.
def get_blob_metadata(bucket_name: str, blob_name: str) -> Optional[str]:
    """
    Retrieve the 'overheadhints' custom metadata value from a GCS blob.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name/path of the blob

    Returns:
        The raw JSON string stored under the 'overheadhints' metadata key,
        or None if the blob / key is not found.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name)

        if blob is None:
            logger.warning(f"Blob not found: {blob_name} in bucket {bucket_name}")
            return None

        if blob.metadata:
            value = blob.metadata.get("overheadhints")
            if value is None:
                logger.warning(f"No 'overheadhints' key in metadata for blob: {blob_name}")
            return value
        else:
            logger.warning(f"No custom metadata found for blob: {blob_name}")
            return None

    except Exception as e:
        logger.error(f"Error retrieving blob metadata for {blob_name}: {e}")
        return None


# Pipeline order: Optional multiframe stage
# Description: Parses frame position and beam detections from overhead-hints metadata.
def parse_beam_metadata(overheadhints: str) -> Tuple[Optional[str], Optional[List[List[float]]]]:
    """
    Parse position and beam_detections from the 'overheadhints' JSON string.

    Args:
        overheadhints: JSON string returned by get_blob_metadata, e.g.
            '{"position": "right-center", "beam_detections": [[x1,y1,x2,y2], ...]}'

    Returns:
        Tuple of (position, beam_detections) where:
        - position: str like "left-center", "right-center", "left-before", etc.
        - beam_detections: List of beam bboxes [[xmin, ymin, xmax, ymax], ...]
    """
    try:
        data = json.loads(overheadhints)
        position = data.get("position")
        beam_detections = data.get("beam_detections")
        return position, beam_detections

    except Exception as e:
        logger.error(f"Error parsing overheadhints metadata: {e}")
        return None, None


# Pipeline order: Optional multiframe stage
# Description: Computes a simple center line for a beam bbox.
def get_beam_center_line(beam_bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Calculate the center line of a beam bbox.

    For a tilted beam (due to camera angle), we compute a center line using the midpoints
    of the bbox edges rather than just using a single coordinate.

    Args:
        beam_bbox: [xmin, ymin, xmax, ymax]

    Returns:
        Tuple of (x_left, y_left, x_right, y_right) representing the center line
    """
    xmin, ymin, xmax, ymax = beam_bbox

    # Center line: from the middle of the left edge to the middle of the right edge
    x_left = xmin
    y_left = (ymin + ymax) / 2.0
    x_right = xmax
    y_right = (ymin + ymax) / 2.0

    return x_left, y_left, x_right, y_right


# Pipeline order: Optional multiframe stage
# Description: Builds the perspective-aware diagonal beam boundary used for SKU filtering.
def calculate_beam_slope_and_line(
    beam_bbox: List[float], beam_side: str = "left"
) -> Tuple[Optional[float], Tuple[float, float, float, float]]:
    """
    Calculate slope and dividing line of a potentially tilted beam.

    Due to perspective/camera angle, beams lean toward the bay center as they go up.
    The dividing line uses diagonal corners of the bbox:
    - Left beam  (left bay boundary): bottom-left → top-right  i.e. (xmin,ymax)→(xmax,ymin)
    - Right beam (right bay boundary): bottom-right → top-left i.e. (xmax,ymax)→(xmin,ymin)

    Args:
        beam_bbox: [xmin, ymin, xmax, ymax]
        beam_side: "left" for the left boundary of a bay, "right" for the right boundary

    Returns:
        Tuple of (slope, (x1, y1, x2, y2)) where:
        - slope: slope of the dividing line (None if dx ≈ 0)
        - (x1, y1, x2, y2): endpoints of the dividing line
    """
    xmin, ymin, xmax, ymax = beam_bbox

    if beam_side == "right":
        # Right boundary: lean left as you go up → bottom-right to top-left
        x1, y1 = xmax, ymax  # bottom-right
        x2, y2 = xmin, ymin  # top-left
    else:
        # Left boundary (default): lean right as you go up → bottom-left to top-right
        x1, y1 = xmin, ymax  # bottom-left
        x2, y2 = xmax, ymin  # top-right

    dx = x2 - x1
    slope = None if abs(dx) < 1e-6 else (y2 - y1) / dx

    return slope, (x1, y1, x2, y2)


# Pipeline order: Optional multiframe stage
# Description: Determines whether a point lies left of a tilted beam boundary.
def is_point_left_of_beam_line(
    point_x: float,
    point_y: float,
    beam_bbox: List[float],
    slope: Optional[float],
    line_endpoints: Tuple[float, float, float, float],
) -> bool:
    """
    Determine if a point is to the left of a beam line, handling tilted beams.

    Strategy:
    1. For points within the beam bbox horizontal range: use the tilted line
    2. For points outside: use vertical line extension at the bbox edge

    This handles the "sky-shelf top" case where items on top of shelves
    need to be correctly assigned to bays based on horizontal position.

    Args:
        point_x, point_y: Coordinates of the point to check
        beam_bbox: [xmin, ymin, xmax, ymax] of the beam
        slope: Slope of beam center line (None if vertical)
        line_endpoints: (x1, y1, x2, y2) of the beam center line

    Returns:
        True if point is to the left of the beam line, False otherwise
    """
    xmin, ymin, xmax, ymax = beam_bbox
    x1, y1, x2, y2 = line_endpoints

    if point_x < xmin:
        # Outside bbox on the left: always to the left of the beam
        return True
    elif point_x > xmax:
        # Outside bbox on the right: always to the right of the beam
        return False
    else:
        # Within beam bbox horizontal range: interpolate the dividing x at this point_y.
        # x_line = x1 + (x2-x1) * t  where  t = (point_y - y1) / (y2 - y1)
        # A point is "left" when its x is less than the dividing line's x at that height.
        dy = y2 - y1
        if abs(dy) < 1e-6:
            # Degenerate (near-horizontal) dividing line: use x-midpoint
            x_line = (x1 + x2) / 2.0
        else:
            t = (point_y - y1) / dy
            x_line = x1 + (x2 - x1) * t
        return point_x < x_line


# Pipeline order: Optional multiframe stage
# Description: Determines whether a point lies right of a tilted beam boundary.
def is_point_right_of_beam_line(
    point_x: float,
    point_y: float,
    beam_bbox: List[float],
    slope: Optional[float],
    line_endpoints: Tuple[float, float, float, float],
) -> bool:
    """
    Determine if a point is to the right of a beam line, handling tilted beams.

    This is the inverse of is_point_left_of_beam_line.

    Args:
        point_x, point_y: Coordinates of the point to check
        beam_bbox: [xmin, ymin, xmax, ymax] of the beam
        slope: Slope of beam center line (None if vertical)
        line_endpoints: (x1, y1, x2, y2) of the beam center line

    Returns:
        True if point is to the right of the beam line, False otherwise
    """
    return not is_point_left_of_beam_line(point_x, point_y, beam_bbox, slope, line_endpoints)


# Pipeline order: Optional multiframe stage
# Description: Determines whether a SKU bbox center should be considered left of a beam.
def is_sku_left_of_beam(sku_bbox: List[float], beam_bbox: List[float], beam_side: str = "left") -> bool:
    """
    Check if a SKU bbox center is to the left of the beam, considering beam tilt.

    Uses the center point of the SKU bbox so that a label straddling the beam
    boundary is judged by where the majority of it lies rather than requiring
    all four corners to be on the same side.

    Strategy:
    1. If SKU's right edge is left of beam's left edge (simple case): return True
    2. Otherwise: check the SKU center point against the tilted beam line

    Args:
        sku_bbox: [xmin, ymin, xmax, ymax]
        beam_bbox: [xmin, ymin, xmax, ymax]
        beam_side: "left" or "right" — controls which diagonal to use for tilt

    Returns:
        True if SKU center is to the left of beam, False otherwise
    """
    try:
        beam_xmin = beam_bbox[0]
        sku_xmin, sku_ymin, sku_xmax, sku_ymax = sku_bbox

        # Simple case: SKU is completely left of beam's left edge
        if sku_xmax < beam_xmin:
            return True

        # Use center point of SKU bbox against the tilted beam line
        slope, line_endpoints = calculate_beam_slope_and_line(beam_bbox, beam_side)
        cx = (sku_xmin + sku_xmax) / 2.0
        cy = (sku_ymin + sku_ymax) / 2.0
        return is_point_left_of_beam_line(cx, cy, beam_bbox, slope, line_endpoints)

    except Exception as e:
        logger.warning(f"Error in is_sku_left_of_beam: {e}. Falling back to x-coordinate only.")
        sku_xmax = sku_bbox[2]
        beam_xmin = beam_bbox[0]
        return sku_xmax < beam_xmin


# Pipeline order: Optional multiframe stage
# Description: Determines whether a SKU bbox center should be considered right of a beam.
def is_sku_right_of_beam(sku_bbox: List[float], beam_bbox: List[float], beam_side: str = "left") -> bool:
    """
    Check if a SKU bbox center is to the right of the beam, considering beam tilt.

    Uses the center point of the SKU bbox so that a label straddling the beam
    boundary is judged by where the majority of it lies rather than requiring
    all four corners to be on the same side.

    Strategy:
    1. If SKU's left edge is right of beam's right edge (simple case): return True
    2. Otherwise: check the SKU center point against the tilted beam line

    Args:
        sku_bbox: [xmin, ymin, xmax, ymax]
        beam_bbox: [xmin, ymin, xmax, ymax]
        beam_side: "left" or "right" — controls which diagonal to use for tilt

    Returns:
        True if SKU center is to the right of beam, False otherwise
    """
    try:
        beam_xmax = beam_bbox[2]
        sku_xmin, sku_ymin, sku_xmax, sku_ymax = sku_bbox

        # Simple case: SKU is completely right of beam's right edge
        if sku_xmin > beam_xmax:
            return True

        # Use center point of SKU bbox against the tilted beam line
        slope, line_endpoints = calculate_beam_slope_and_line(beam_bbox, beam_side)
        cx = (sku_xmin + sku_xmax) / 2.0
        cy = (sku_ymin + sku_ymax) / 2.0
        return is_point_right_of_beam_line(cx, cy, beam_bbox, slope, line_endpoints)

    except Exception as e:
        logger.warning(f"Error in is_sku_right_of_beam: {e}. Falling back to x-coordinate only.")
        sku_xmin = sku_bbox[0]
        beam_xmax = beam_bbox[2]
        return sku_xmin > beam_xmax


# Pipeline order: Optional multiframe stage
# Description: Keeps center-frame SKUs that fall between the two detected bay beams.
def filter_skus_center_frame(
    skus: List[str], bboxes: List[List[float]], beam_detections: List[List[float]]
) -> Tuple[List[str], List[List[float]]]:
    """
    Filter SKUs for center frame (left-center or right-center).

    Rules:
    - 2 beams: Keep only SKUs between the two beams (not left of left beam, not right of right beam)
    - 0 beams: No filtering needed
    - 3+ beams or other cases: Log warning and return all SKUs

    For tilted beams (camera angle effects), uses the beam's center line for comparison,
    with special handling for SKUs outside the beam bbox horizontal range.

    Args:
        skus: List of SKU strings
        bboxes: List of SKU bboxes
        beam_detections: List of beam bboxes

    Returns:
        Tuple of (filtered_skus, filtered_bboxes)
    """
    if not beam_detections:
        # No beams detected, no filtering
        logger.debug("Center frame with 0 beams: no filtering applied")
        return skus, bboxes

    if len(beam_detections) == 2:
        # Sort beams by x-center so [0] is always the left beam and [1] the right beam
        beam_detections_sorted = sorted(beam_detections, key=lambda b: (b[0] + b[2]) / 2.0)
        left_beam = beam_detections_sorted[0]
        right_beam = beam_detections_sorted[1]

        # Validate beam bboxes
        if not _validate_bbox(left_beam) or not _validate_bbox(right_beam):
            logger.warning("Invalid beam bbox detected in center frame with 2 beams, returning all SKUs")
            return skus, bboxes

        filtered_skus = []
        filtered_bboxes = []

        for sku, bbox in zip(skus, bboxes):
            # Keep if: NOT left of left beam AND NOT right of right beam
            # Left beam leans bottom-left→top-right; right beam leans bottom-right→top-left
            is_not_left_of_left = not is_sku_left_of_beam(bbox, left_beam, beam_side="left")
            is_not_right_of_right = not is_sku_right_of_beam(bbox, right_beam, beam_side="right")

            if is_not_left_of_left and is_not_right_of_right:
                filtered_skus.append(sku)
                filtered_bboxes.append(bbox)

        logger.debug(
            f"Center frame with 2 beams: filtered {len(skus)} SKUs -> {len(filtered_skus)} "
            f"(left_beam=[{left_beam[0]:.1f},{left_beam[1]:.1f},{left_beam[2]:.1f},{left_beam[3]:.1f}], "
            f"right_beam=[{right_beam[0]:.1f},{right_beam[1]:.1f},{right_beam[2]:.1f},{right_beam[3]:.1f}])"
        )
        return filtered_skus, filtered_bboxes

    elif len(beam_detections) == 1:
        # Single beam in center frame is unusual, log warning
        logger.warning("Center frame with 1 beam: unexpected configuration, no filtering applied")
        return skus, bboxes

    else:
        # 3 or more beams: log warning and return all
        logger.warning(f"Center frame with {len(beam_detections)} beams: not yet supported, returning all SKUs")
        return skus, bboxes


# Pipeline order: Optional multiframe stage
# Description: Filters before-frame SKUs according to the visible beam boundary and frame position.
def filter_skus_before_frame(
    skus: List[str], bboxes: List[List[float]], beam_detections: List[List[float]], position: str,
    image_width: int = 0
) -> Tuple[List[str], List[List[float]]]:
    """
    Filter SKUs for before frame (left-before or right-before).

    Only handles single beam case:
    - left-before: ignore SKUs to the left of the beam (keep right side and beam region)
    - right-before: ignore SKUs to the right of the beam (keep left side and beam region)

    The beam diagonal (beam_side) is inferred from the beam's x-position in the image
    rather than from the position string, because camera perspective causes beams to lean
    toward the bay center: beams on the right half lean ↙, beams on the left half lean ↗.

    Args:
        skus: List of SKU strings
        bboxes: List of SKU bboxes
        beam_detections: List of beam bboxes
        position: "left-before" or "right-before"
        image_width: Full image width in pixels (used to infer beam diagonal direction)

    Returns:
        Tuple of (filtered_skus, filtered_bboxes)
    """
    if not beam_detections:
        logger.warning(f"Before frame ({position}) with 0 beams: no filtering applied")
        return skus, bboxes

    if len(beam_detections) == 1:
        beam = beam_detections[0]

        # Validate beam bbox
        if not _validate_bbox(beam):
            logger.warning(f"Invalid beam bbox in before frame ({position}), returning all SKUs")
            return skus, bboxes

        filtered_skus = []
        filtered_bboxes = []
        beam_side = _infer_beam_side(beam, image_width)

        for sku, bbox in zip(skus, bboxes):
            if position == "left-before":
                # Keep right side and beam region: ignore SKUs to the left of beam
                if not is_sku_left_of_beam(bbox, beam, beam_side=beam_side):
                    filtered_skus.append(sku)
                    filtered_bboxes.append(bbox)
            elif position == "right-before":
                # Keep left side and beam region: ignore SKUs to the right of beam
                if not is_sku_right_of_beam(bbox, beam, beam_side=beam_side):
                    filtered_skus.append(sku)
                    filtered_bboxes.append(bbox)

        logger.debug(
            f"Before frame ({position}) with 1 beam: filtered {len(skus)} -> {len(filtered_skus)} SKUs "
            f"(beam=[{beam[0]:.1f},{beam[1]:.1f},{beam[2]:.1f},{beam[3]:.1f}], "
            f"beam_side='{beam_side}' inferred from image_width={image_width})"
        )
        return filtered_skus, filtered_bboxes

    else:
        logger.warning(
            f"Before frame ({position}) with {len(beam_detections)} beams: only 1 beam supported, returning all SKUs"
        )
        return skus, bboxes


# Pipeline order: Optional multiframe stage
# Description: Filters after-frame SKUs according to the visible beam boundary and frame position.
def filter_skus_after_frame(
    skus: List[str], bboxes: List[List[float]], beam_detections: List[List[float]], position: str,
    image_width: int = 0
) -> Tuple[List[str], List[List[float]]]:
    """
    Filter SKUs for after frame (left-after or right-after).

    Only handles single beam case:
    - left-after: ignore SKUs to the right of the beam (keep left side and beam region)
    - right-after: ignore SKUs to the left of the beam (keep right side and beam region)

    The beam diagonal (beam_side) is inferred from the beam's x-position in the image
    rather than from the position string, because camera perspective causes beams to lean
    toward the bay center: beams on the right half lean ↙, beams on the left half lean ↗.

    Args:
        skus: List of SKU strings
        bboxes: List of SKU bboxes
        beam_detections: List of beam bboxes
        position: "left-after" or "right-after"
        image_width: Full image width in pixels (used to infer beam diagonal direction)

    Returns:
        Tuple of (filtered_skus, filtered_bboxes)
    """
    if not beam_detections:
        logger.warning(f"After frame ({position}) with 0 beams: no filtering applied")
        return skus, bboxes

    if len(beam_detections) == 1:
        beam = beam_detections[0]

        # Validate beam bbox
        if not _validate_bbox(beam):
            logger.warning(f"Invalid beam bbox in after frame ({position}), returning all SKUs")
            return skus, bboxes

        filtered_skus = []
        filtered_bboxes = []
        beam_side = _infer_beam_side(beam, image_width)

        for sku, bbox in zip(skus, bboxes):
            if position == "left-after":
                # Keep left side and beam region: ignore SKUs to the right of beam
                if not is_sku_right_of_beam(bbox, beam, beam_side=beam_side):
                    filtered_skus.append(sku)
                    filtered_bboxes.append(bbox)
            elif position == "right-after":
                # Keep right side and beam region: ignore SKUs to the left of beam
                if not is_sku_left_of_beam(bbox, beam, beam_side=beam_side):
                    filtered_skus.append(sku)
                    filtered_bboxes.append(bbox)

        logger.debug(
            f"After frame ({position}) with 1 beam: filtered {len(skus)} -> {len(filtered_skus)} SKUs "
            f"(beam=[{beam[0]:.1f},{beam[1]:.1f},{beam[2]:.1f},{beam[3]:.1f}], "
            f"beam_side='{beam_side}' inferred from image_width={image_width})"
        )
        return filtered_skus, filtered_bboxes

    else:
        logger.warning(
            f"After frame ({position}) with {len(beam_detections)} beams: only 1 beam supported, returning all SKUs"
        )
        return skus, bboxes


# Pipeline order: Optional multiframe stage
# Description: Main multiframe entry point that filters SKU strings and bboxes using beam metadata.
def apply_multiframe_filter(
    skus: List[str], bboxes: List[List[float]], bucket_name: str, file_name: str,
    image_width: int = 0
) -> Tuple[List[str], List[List[float]]]:
    """
    Apply multiframe-based SKU filtering based on beam position information.

    This is the main entry point for multiframe filtering. It:
    1. Retrieves beam metadata from the blob
    2. Determines the frame type (center/before/after)
    3. Applies appropriate filtering rules considering beam tilt

    The filtering uses an enhanced algorithm that handles:
    - Tilted beams due to camera angles (via center line and slope calculation)
    - Sky-shelf top items (vertical line extension for points outside bbox)
    - Different position types (left/right, center/before/after)

    Args:
        skus: List of detected SKU strings
        bboxes: List of SKU bounding boxes [[x1,y1,x2,y2], ...]
        bucket_name: GCS bucket name
        file_name: GCS blob name
        image_width: Full image width in pixels from Vision API (used to infer beam diagonal)

    Returns:
        Tuple of (filtered_skus, filtered_bboxes)
    """
    try:
        # Retrieve metadata from blob
        metadata = get_blob_metadata(bucket_name, file_name)
        if not metadata:
            logger.debug(f"No metadata for {file_name}, returning original results")
            return skus, bboxes

        # Parse position and beam info
        position, beam_detections = parse_beam_metadata(metadata)
        if not position or not beam_detections:
            logger.debug(
                f"Missing position or beam_detections for {file_name} (position={position}), "
                "returning original results"
            )
            return skus, bboxes

        logger.debug(
            f"Applying multiframe filter: file={file_name}, position={position}, "
            f"num_beams={len(beam_detections)}, num_skus={len(skus)}"
        )

        # Apply filtering based on frame type
        if position in ["left-center", "right-center"]:
            return filter_skus_center_frame(skus, bboxes, beam_detections)

        elif position in ["left-before", "right-before"]:
            return filter_skus_before_frame(skus, bboxes, beam_detections, position, image_width)

        elif position in ["left-after", "right-after"]:
            return filter_skus_after_frame(skus, bboxes, beam_detections, position, image_width)

        else:
            logger.warning(f"Unknown position type: {position}, returning original results")
            return skus, bboxes

    except Exception as e:
        logger.error(f"Error applying multiframe filter: {e}", exc_info=True)
        # On error, return original results to avoid breaking the pipeline
        return skus, bboxes


# Pipeline order: Optional multiframe stage
# Description: Applies beam-based multiframe filtering to the final SKU-to-bboxes dictionary.
def apply_multiframe_filter_to_sku_data(
    sku_data: Dict[str, List[List[float]]], bucket_name: str, file_name: str,
    image_width: int = 0
) -> Dict[str, List[List[float]]]:
    """
    Apply multiframe filtering to sku_data using beam position information.

    This function filters sku_data (dictionary mapping normalized SKU to list of bboxes)
    based on beam detections from GCS blob metadata. It handles tilted beams and
    different position types (center/before/after).

    The function is designed to be called after prepare_sku_result_json for flexible
    filtering at different pipeline stages or in other services.

    Args:
        sku_data: Dictionary mapping normalized SKU (str) to list of bboxes
                 Example: {"0000123456": [[x1,y1,x2,y2], [...]], ...}
        bucket_name: GCS bucket name
        file_name: GCS blob name

    Returns:
        Filtered sku_data with same structure. Returns original data on error
        to avoid breaking the pipeline.

    Example:
        >>> sku_data = {"0000123456": [[100, 200, 150, 250]]}
        >>> filtered = apply_multiframe_filter_to_sku_data(sku_data, "bucket", "file")
    """
    try:
        # Retrieve metadata from blob
        metadata = get_blob_metadata(bucket_name, file_name)
        if not metadata:
            logger.debug(f"No metadata for {file_name}, returning original sku_data")
            return sku_data

        # Parse position and beam info
        position, beam_detections = parse_beam_metadata(metadata)
        if not position or not beam_detections:
            logger.debug(f"Missing position or beam_detections for {file_name}, returning original sku_data")
            return sku_data

        # Flatten sku_data for filtering
        all_skus = []
        all_bboxes = []
        sku_to_original_key = []  # Track original normalized SKU keys

        for sku_key, bboxes_list in sku_data.items():
            for bbox in bboxes_list:
                all_skus.append(sku_key)
                all_bboxes.append(bbox)
                sku_to_original_key.append(sku_key)

        if not all_skus:
            # No SKUs to filter
            logger.debug(f"No SKUs in sku_data for {file_name}")
            return sku_data

        logger.debug(
            f"Filtering sku_data for {file_name}: {len(sku_data)} unique SKUs, "
            f"{len(all_skus)} total detections, position={position}, beams={len(beam_detections)}"
        )

        # Apply filtering directly using already-fetched metadata (avoids a second GCS call)
        if position in ["left-center", "right-center"]:
            filtered_skus, filtered_bboxes = filter_skus_center_frame(all_skus, all_bboxes, beam_detections)
        elif position in ["left-before", "right-before"]:
            filtered_skus, filtered_bboxes = filter_skus_before_frame(all_skus, all_bboxes, beam_detections, position, image_width)
        elif position in ["left-after", "right-after"]:
            filtered_skus, filtered_bboxes = filter_skus_after_frame(all_skus, all_bboxes, beam_detections, position, image_width)
        else:
            logger.warning(f"Unknown position type: {position}, returning original sku_data")
            return sku_data

        # Rebuild sku_data from filtered results
        filtered_sku_data = {}
        for sku, bbox in zip(filtered_skus, filtered_bboxes):
            if sku not in filtered_sku_data:
                filtered_sku_data[sku] = []
            filtered_sku_data[sku].append(bbox)

        num_removed = len(all_skus) - len(filtered_bboxes)
        logger.debug(
            f"Filtered sku_data result: {len(sku_data)} -> {len(filtered_sku_data)} unique SKUs, "
            f"{len(all_skus)} -> {len(filtered_bboxes)} total detections "
            f"(removed {num_removed} detections)"
        )
        return filtered_sku_data

    except Exception as e:
        logger.error(f"Error filtering sku_data: {e}", exc_info=True)
        # On error, return original data to avoid breaking the pipeline
        return sku_data
