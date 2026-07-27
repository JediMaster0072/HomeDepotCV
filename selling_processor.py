import os
import cv2
import copy
import numpy as np
from shapely.geometry import Polygon, box
import logging
import re
import math
import metrics
import sys
from time import time
from db import BigTableClient, BigQueryClient
from services.model_interfaces.model_interface_base import SellingModelBase
from services.model_interfaces.common_model_functions import retry_model_predict
from configuration import Settings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import io
import faulthandler
faulthandler.enable()


@dataclass
class SellingProcessResult:
    price_grid: Any
    sku_grid: Any
    sku_grid_info: Any
    empty_grid_2d: Any
    empty_grid_info: Any
    empty_bboxes: Any
    status_code_array: Any
    beam_bboxes: Any
    image_score: Any
    image_angle: Any
    price_label_coverage: Any
    price_label_bboxes: Any
    price_grid_conf: Any
    empty_bboxes_grid: Any
    edge_empty_grid: Any
    beam_count: Any


# Move all cloud/db/model imports inside functions
def print_log(*args, **kwargs):
    from common import print_log as _print_log
    return _print_log(*args, **kwargs)

def get_time():
    """Get the current time in milliseconds"""
    return int(time() * 1000)


def size_ls(ls):
    """Returns the storage size of a list in megabytes."""
    total_size = sys.getsizeof(ls)  # Size of the list object itself
    for item in ls:
        total_size += sys.getsizeof(item)  # Size of each element in the list
    return total_size / 1024 / 1024  # Convert bytes to megabytes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pad_image(image_np: np.ndarray, padding_width: int = 300) -> np.ndarray:
    """Apply image padding."""
    if image_np is None:
        raise ValueError("Image could not be loaded.")

    return cv2.copyMakeBorder(
        image_np,
        top=padding_width,
        bottom=padding_width,
        left=padding_width,
        right=padding_width,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

def unpad_bboxes(bboxes: list, padding_width: int = 300) -> list:
    """
    Remove padding from bounding box coordinates.

    Args:
        bboxes: List of bboxes in [x_min, y_min, x_max, y_max] format.
        padding_width: Padding width used during preprocessing.

    Returns:
        List of adjusted bboxes with padding removed.
    """
    unpadded_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        unpadded_bbox = [
            max(x_min - padding_width, 0),
            max(y_min - padding_width, 0),
            max(x_max - padding_width, 0),
            max(y_max - padding_width, 0),
        ]
        unpadded_bboxes.append(unpadded_bbox)
    return unpadded_bboxes


def extract_and_save_masks(processing_key, padded_image_bytes: bytes, segment_model_interface):
    """Extract segmentation masks from image using model.
    
    Args:
        processing_key: Unique key for logging
        padded_image_bytes: Padded image as raw bytes in JPEG format
        segment_model_interface: Model interface for segmentation
        
    Returns:
        Tuple of (status_code, detections dict) where detections contains polygon masks by class name
    """
    status_code, results = retry_model_predict(
        segment_model_interface.predict,
        processing_key,
        model_name="segment_model_interface",
        image_bytes=padded_image_bytes,
        iou=0.5
    )

    # Dictionary to hold results, keys are class names and values are lists of masks
    detections = {}

    # Class ID to name mapping
    class_names = {0: 'SHELF', 1: 'BEAM', 2: 'LOWER_SHELF', 3: 'PRICE_LABEL'}

    if status_code == 200:
        # Extract classes and mask coordinates
        if results.masks is not None:
            for mask, cls in zip(results.masks, results.class_list):
                class_name = class_names[cls]  # Convert class ID to class name
                mask_points = np.array(mask, dtype=np.int32).tolist()  # Convert mask to list of integer points

                # Check if class name is already in the dictionary
                if class_name not in detections:
                    detections[class_name] = []
                detections[class_name].append(mask_points)
        else:
            logging.warning(f"[{processing_key}] : No masks detected in the image.")

        if not detections:
            logging.warning(f"[{processing_key}] : No objects detected in the image.")

        print_log('debug', processing_key, "Detected objects",
                  f"{', '.join([f'{k}: {len(v)}' for k, v in detections.items()])}")
    return status_code, detections

def extract_and_annotate_boxes(processing_key, padded_image_bytes: bytes, combined_model_interface) -> Dict[str, List[List[float]]]:
    """Extract bounding boxes using YOLOv8 model.
    
    Args:
        processing_key: Unique key for logging
        padded_image_bytes: Padded image as raw bytes in JPEG format
        combined_model_interface: Model interface for object detection
        
    Returns:
        Dict mapping class names to lists of bounding boxes [x1, y1, x2, y2]
    """
    status_code, results = retry_model_predict(
        combined_model_interface.predict,
        processing_key,
        model_name="combined_model_interface",
        image_bytes=padded_image_bytes,
        iou=0.5,
        classes=[0, 1, 3, 5, 6, 7, 8]
    )
    
    confidences = {}
    detections = {}
    class_names = {
        0: 'PRICE_LABEL',
        1: 'EMPTY',
        3: 'BEAM',
        5: 'HANGING_EMPTY',
        6: 'LOWER_EMPTY',  # Adding new class
        7: 'EMPTY_CASE_TRAY',
        8: 'EMPTY_BOX'
    }
    empty_group =  ['LOWER_EMPTY', 'EMPTY_CASE_TRAY', 'EMPTY_BOX']
    
    if status_code == 200:
        # Validate results structure before accessing attributes
        has_boxes = hasattr(results, 'boxes') and results.boxes is not None
        has_class_list = hasattr(results, 'class_list') and results.class_list is not None
        has_confidences = hasattr(results, 'confidences') and results.confidences is not None
        
        # Only iterate if all attributes exist and have content
        if has_boxes and has_class_list and has_confidences:
            for bounding_box, cls, conf in zip(results.boxes, results.class_list, results.confidences):
                class_name = class_names[cls]
                box_coords = bounding_box
                
                # Filter out invalid BEAM detections that span >50% of image width or height
                
                # Combining LOWER_EMPTY detections with EMPTY
                if class_name in empty_group:
                    class_name = 'EMPTY'

                if class_name not in detections:
                    detections[class_name] = []
                    confidences[class_name] = []
                detections[class_name].append(box_coords)
                confidences[class_name].append(conf)
        
        # Log detection counts for debugging
        print_log('debug', processing_key, "extract_and_annotate_boxes detections",
                  f"classes={list(detections.keys())}, counts={[(k, len(v)) for k, v in detections.items()]}")
        return status_code, detections, confidences

def cluster_labels_by_y_coordinate(processing_key, bboxes, y_threshold=100):
    """
    Fallback function to cluster price labels by Y-coordinate when shelf masks fail.
    Groups labels into rows based on vertical proximity.
    """
    price_labels = bboxes.get('PRICE_LABEL', [])
    if not price_labels:
        logging.warning(f"[{processing_key}] : No price labels to cluster.")
        return {}
    
    # Calculate Y-center for each label
    labels_with_y = [(bbox, (bbox[1] + bbox[3]) / 2) for bbox in price_labels]
    labels_with_y.sort(key=lambda x: x[1])  # Sort by Y-coordinate
    
    # Cluster labels by Y-coordinate proximity
    clusters = []
    current_cluster = [labels_with_y[0][0]]
    last_y = labels_with_y[0][1]
    
    for bbox, y_center in labels_with_y[1:]:
        if y_center - last_y < y_threshold:
            current_cluster.append(bbox)
        else:
            clusters.append(current_cluster)
            current_cluster = [bbox]
        last_y = y_center
    clusters.append(current_cluster)
    
    # Create sorted_shelves structure
    sorted_shelves = {}
    for i, cluster in enumerate(clusters):
        sorted_shelves[i + 1] = []
        # Sort labels within cluster by X-coordinate (left to right)
        cluster_sorted = sorted(cluster, key=lambda bbox: (bbox[0] + bbox[2]) / 2)
        for bbox in cluster_sorted:
            sorted_shelves[i + 1].append(list(bbox))
    
    print_log('debug', processing_key, 
             f"Y-coordinate clustering created {len(clusters)} rows from {len(price_labels)} labels",
             "no_value")
    
    return sorted_shelves


def assign_labels_with_optional_beam_extension(processing_key, bboxes, masks, str_nbr=None, confidences=None):
    """
    Assigns price label bounding boxes to shelves using polygon intersection and optional beam extension.
    Also handles hanging product detection and virtual shelf creation for labels above shelves.

    Args:
        processing_key: Unique identifier for the current image/process.
        bboxes: Dict mapping class names to lists of bounding boxes.
        masks: Dict mapping class names to lists of polygon masks.
        str_nbr: Optional store number for context.
        confidences: Optional dict mapping class names to confidence scores.

    Returns:
        Tuple[Dict[int, List[List[float]]], Dict[int, List[float]]]: 
            - Dict mapping shelf numbers to sorted lists of label bounding boxes
            - Dict mapping shelf numbers to corresponding confidence values
    """
    total_price_labels = len(bboxes.get('PRICE_LABEL', []))
    print_log('debug', processing_key, "Number of shelves detected",
              len(masks.get('SHELF', [])))
    print_log('debug', processing_key, "Number of lower shelves detected",
              len(masks.get('LOWER_SHELF', [])))
    print_log('debug', processing_key, "Number of beams detected",
              len(masks.get('BEAM', [])))
    print_log('debug', processing_key, "Total PRICE_LABEL bboxes to assign",
              total_price_labels)
    
    # Build confidence map from input confidences (if provided)
    # Maps bbox tuple -> confidence float
    confidence_map = {}
    
    price_label_bboxes = bboxes.get('PRICE_LABEL', [])
    price_label_confs = confidences.get('PRICE_LABEL', [])
    for idx, (bbox, conf) in enumerate(zip(price_label_bboxes, price_label_confs)):
        # Ensure bbox is a list or compatible sequence before converting to tuple
        if isinstance(bbox, (list, tuple, np.ndarray)):
            bbox_tuple = tuple(bbox) if not isinstance(bbox, tuple) else bbox
        else:
            bbox_tuple = tuple([bbox])
        
        confidence_map[bbox_tuple] = float(conf)

    beam_x_positions = [point[0] for mask in masks.get('BEAM', []) for point in mask]
    # Only use beam extension if we have valid beams and they span a reasonable width
    use_beam_extension = len(beam_x_positions) >= 2
    leftmost_beam = min(beam_x_positions) if use_beam_extension else None
    rightmost_beam = max(beam_x_positions) if use_beam_extension else None

    def process_shelf_mask(mask):
        top = min(mask, key=lambda x: x[1])[1]
        bottom = max(mask, key=lambda x: x[1])[1]
        min_x = min(mask, key=lambda x: x[0])[0]
        max_x = max(mask, key=lambda x: x[0])[0]

        if use_beam_extension and (min_x >= leftmost_beam and max_x <= rightmost_beam):
            return Polygon([
                (leftmost_beam, top), (rightmost_beam, top),
                (rightmost_beam, bottom), (leftmost_beam, bottom)
            ]), top, bottom
        else:
            return Polygon(mask), top, bottom

    shelves = []
    lower_shelf = None
    
    for i, mask in enumerate(masks.get('SHELF', [])):
        try:
            shelf_polygon, top, bottom = process_shelf_mask(mask)
            shelves.append((shelf_polygon, top, bottom))
        except ValueError as e:
            logging.error(f"[{processing_key}] : Error processing shelf {i + 1} : {str(e)}")

    if 'LOWER_SHELF' in masks and masks['LOWER_SHELF']:
        try:
            lower_shelf_polygon, top, bottom = process_shelf_mask(masks['LOWER_SHELF'][0])
            lower_shelf = (lower_shelf_polygon, top, bottom)
            print_log('debug', processing_key, "Lower shelf processed successfully", "no_value")
        except ValueError as e:
            logging.error(f"[{processing_key}] : Error processing lower shelf : {str(e)}")

    if not shelves and not lower_shelf:
        logging.warning(f"[{processing_key}] : No valid shelves detected. Using Y-coordinate clustering fallback.")
        # Fallback: cluster price labels by Y-coordinate when masks fail
        fallback_shelves = cluster_labels_by_y_coordinate(processing_key, bboxes)
        # Build confidence map for fallback
        fallback_confidence_map = {}
        for shelf, labels in fallback_shelves.items():
            fallback_confidence_map[shelf] = []
            for label in labels:
                bbox_tuple = tuple(label)
                conf = confidence_map.get(bbox_tuple, 0.0)
                fallback_confidence_map[shelf].append(conf)
        return fallback_shelves, fallback_confidence_map

    shelves.sort(key=lambda x: x[1])  # Sorting by top coordinate

    sorted_shelves = {i + 1: [] for i in range(len(shelves))}
    if lower_shelf:
        sorted_shelves[len(shelves) + 1] = []  # Adding lower shelf as the last shelf
        print_log('debug', processing_key, "LOWER_SHELF detected and added", 
                  f"lower_shelf_position={len(shelves)+1}, top={lower_shelf[1]}, bottom={lower_shelf[2]}")

    # Assigning labels to all shelves including lower shelf
    unassigned_labels = []
        
    for bbox in bboxes.get('PRICE_LABEL', []):
        box_tuple = tuple(bbox)
        x1, y1, x2, y2 = box_tuple
        label_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        
        assigned = False
        for i, (shelf_polygon, _, _) in enumerate(shelves + ([lower_shelf] if lower_shelf else [])):
            if shelf_polygon.intersects(label_polygon):
                overlap_area = shelf_polygon.intersection(label_polygon).area
                sorted_shelves[i + 1].append((box_tuple, overlap_area, (x1 + x2) / 2))
                assigned = True
                break
        
        if not assigned:
            unassigned_labels.append(box_tuple)
    
    # Log unassigned labels for debugging
    hanging_rows = []  # Initialize outside conditional blocks
    if unassigned_labels:
        print_log('debug', processing_key, 
                 f"Unassigned labels: {len(unassigned_labels)} labels not matched to any shelf mask",
                 "no_value")
        
        hanging_labels = []
        other_unassigned = []
        
        if shelves:
            topmost_shelf_y = min(shelf_top for _, shelf_top, _ in shelves)
            
            for box_tuple in unassigned_labels:
                x1, y1, x2, y2 = box_tuple
                label_y_center = (y1 + y2) / 2
                
                # Labels significantly above the topmost shelf are likely hanging products
                if label_y_center < topmost_shelf_y - 100:  # 100px buffer
                    hanging_labels.append(box_tuple)
                else:
                    other_unassigned.append(box_tuple)
        else:
            other_unassigned = unassigned_labels
        
        print_log('debug', processing_key,
                    f"Categorized unassigned: {len(hanging_labels)} hanging, {len(other_unassigned)} other",
                    "no_value")
        
        # Create virtual "hanging shelf" for hanging products
        if hanging_labels:
            # Cluster hanging labels by Y-coordinate (they might be on multiple peg rows)
            hanging_labels.sort(key=lambda b: (b[1] + b[3]) / 2)
            
            # Group into rows using Y-proximity
            current_row = [hanging_labels[0]]
            last_y = (hanging_labels[0][1] + hanging_labels[0][3]) / 2
            
            for box_tuple in hanging_labels[1:]:
                y_center = (box_tuple[1] + box_tuple[3]) / 2
                if abs(y_center - last_y) < 150:  # Same row threshold
                    current_row.append(box_tuple)
                else:
                    hanging_rows.append(current_row)
                    current_row = [box_tuple]
                    last_y = y_center
            hanging_rows.append(current_row)
            
            print_log('debug', processing_key,
                        f"Created {len(hanging_rows)} hanging product rows with {[len(r) for r in hanging_rows]} labels each",
                        "no_value")
            
            # Insert hanging rows at the beginning of shelf structure
            new_sorted_shelves = {}
            for row_idx, row in enumerate(hanging_rows):
                new_shelf_num = row_idx + 1
                # Sort labels left-to-right within row
                sorted_row = sorted(row, key=lambda b: (b[0] + b[2]) / 2)
                # Use 0.0 for area since hanging labels don't have shelf intersection area
                new_sorted_shelves[new_shelf_num] = [(box, 0.0, (box[0] + box[2]) / 2) for box in sorted_row]
            
            # Shift existing shelf numbers
            shift = len(hanging_rows)
            for old_shelf, labels in sorted_shelves.items():
                new_sorted_shelves[old_shelf + shift] = labels
            
            sorted_shelves = new_sorted_shelves
            print_log('debug', processing_key,
                        f"Integrated {len(hanging_labels)} hanging labels into grid structure",
                        "no_value")
        
        # Try to assign other unassigned labels to nearest shelf by Y-coordinate
        for box_tuple in other_unassigned:
            x1, y1, x2, y2 = box_tuple
            label_y_center = (y1 + y2) / 2
            
            # Find closest shelf by Y-coordinate
            closest_shelf = None
            min_distance = float('inf')
            for i, (_, shelf_top, shelf_bottom) in enumerate(shelves + ([lower_shelf] if lower_shelf else [])):
                shelf_y_center = (shelf_top + shelf_bottom) / 2
                distance = abs(label_y_center - shelf_y_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_shelf = i + 1 + len(hanging_rows)  # Account for hanging rows shift
            
            if closest_shelf and min_distance < 300:  # within 300px
                sorted_shelves[closest_shelf].append((box_tuple, 0.0, (x1 + x2) / 2))
                print_log('debug', processing_key,
                         f"Assigned unmatched label to shelf {closest_shelf} by Y-proximity (distance={min_distance:.1f}px)",
                         "no_value")

    # Resolving conflicts for labels that overlap multiple shelves
    conflicted_labels = {}
    for shelf, labels in sorted_shelves.items():
        for label in labels:
            box, area, x_center = label
            if box in conflicted_labels:
                if area > conflicted_labels[box][1]:
                    conflicted_labels[box] = (shelf, area, x_center)
            else:
                conflicted_labels[box] = (shelf, area, x_center)

    # Reassigning labels based on resolved conflicts
    final_shelves = {i: [] for i in sorted_shelves.keys()}
    for box, (shelf, _, x_center) in conflicted_labels.items():
        final_shelves[shelf].append((box, x_center))

    # Sorting labels within each shelf from left to right
    for shelf in final_shelves:
        final_shelves[shelf] = [list(box) for box, _ in sorted(final_shelves[shelf], key=lambda x: x[1])]
    
    # Build final confidence map from final_shelves (shelf -> list of confidence values)
    final_confidence_map = {}
    for shelf, labels in final_shelves.items():
        final_confidence_map[shelf] = []
        for label in labels:
            bbox_tuple = tuple(label)
            conf = confidence_map.get(bbox_tuple, 0.0)  # Default to 0.0 if not found
            final_confidence_map[shelf].append(conf)
    
    total_assigned = sum(len(labels) for labels in final_shelves.values())
    print_log('debug', processing_key, "Number of shelves in final result",
              len(final_shelves))
    print_log('debug', processing_key, "Shelves with labels",
              [key for key, value in final_shelves.items() if value])
    print_log('debug', processing_key, f"Label assignment summary: {total_assigned}/{total_price_labels} assigned",
              f"({100.0 * total_assigned / total_price_labels if total_price_labels else 0:.1f}% coverage)")
    
    if total_assigned < total_price_labels:
        print_log('debug', processing_key, 
                 f"WARNING: {total_price_labels - total_assigned} price labels were dropped during shelf assignment",
                 "no_value")
    
    return final_shelves, final_confidence_map


def find_matching_empty_box(label_centroid, empty_boxes, all_empty_bboxes):
    """Return the coordinates of the empty box containing the label centroid, or empty string if not found."""
    try:
        for empty_box_obj, empty_box_coords in zip(empty_boxes, all_empty_bboxes):
            if empty_box_obj.contains(label_centroid):
                return empty_box_coords
    except Exception as e:
        logging.error(f"Error in find_matching_empty_box: {str(e)}")
    return ""


def create_price_label_grid(processing_key, bboxes: Dict[str, List[List[float]]], sorted_shelves: Dict[int, List[List[float]]],
                            padding_width: int = 300) -> tuple[Dict[int, List[str]], List[List[float]], List[List[Tuple[float, float, float, float]]]]:
    """Creating a grid of price labels, marking empty spots."""
    # empty_boxes = [box(*bbox) for bbox in bboxes.get('EMPTY', [])]
    # all_empty_bboxes = bboxes.get('EMPTY', []) + bboxes.get('HANGING_EMPTY', [])
    
    all_empty_bboxes = bboxes.get('EMPTY', []) + bboxes.get('HANGING_EMPTY', [])
    empty_boxes = [box(*bbox) for bbox in all_empty_bboxes]
    price_label_bboxes = []
    
    print_log('debug', processing_key, "create_price_label_grid input", 
              f"EMPTY_count={len(bboxes.get('EMPTY', []))}, HANGING_EMPTY_count={len(bboxes.get('HANGING_EMPTY', []))}, total_empty={len(all_empty_bboxes)}, shelves_count={len(sorted_shelves)}")

    # Adjust empty_bboxes coordinates by subtracting padding
    adjusted_empty_bboxes = [
        [x1 - padding_width, y1 - padding_width, x2 - padding_width, y2 - padding_width]
        for x1, y1, x2, y2 in all_empty_bboxes
    ]
    
    #print_log('debug', processing_key, "TEST empty_boxes", empty_boxes)
    #print_log('debug', processing_key, "TEST all_empty_bboxes", all_empty_bboxes)
    #print_log('debug', processing_key, "TEST adjusted_empty_bboxes", adjusted_empty_bboxes)

    print_log('debug', processing_key, "Adjusted empty_bboxes", 
              f"count={len(adjusted_empty_bboxes)}, unique_count={len(set(tuple(bbox) for bbox in adjusted_empty_bboxes))}")

    grid = {shelf: ['' for _ in labels] for shelf, labels in sorted_shelves.items()}
    empty_bboxes_grid = {shelf: ['' for _ in labels] for shelf, labels in sorted_shelves.items()}

    for shelf, labels in sorted_shelves.items():
        for idx, label in enumerate(labels):
            label_box = box(*label)
            label_centroid = label_box.centroid
            is_empty, empty_box_coords = get_empty_details(processing_key, label_centroid, empty_boxes, adjusted_empty_bboxes)
            if is_empty:
                grid[shelf][idx] = "Empty"
                empty_bboxes_grid[shelf][idx] = empty_box_coords
        
        adjusted_price_label_bboxes = [
            (x1 - padding_width, y1 - padding_width, x2 - padding_width, y2 - padding_width)
            for x1, y1, x2, y2 in labels
        ]
        price_label_bboxes.append(adjusted_price_label_bboxes)

    print_log('debug', processing_key, "Created empty grid", grid)
    print_log('debug', processing_key, "Created empty bboxes grid", empty_bboxes_grid)
    print_log('debug', processing_key, "create_price_label_grid output", 
              f"adjusted_empty_bboxes_count={len(adjusted_empty_bboxes)}, "
              f"adjusted_empty_bboxes_unique={len(set(tuple(bbox) for bbox in adjusted_empty_bboxes))}, "
              f"price_label_bboxes_shelves={len(price_label_bboxes)}")
    return grid, adjusted_empty_bboxes, price_label_bboxes, empty_bboxes_grid

def get_empty_details(processing_key,label_centroid, empty_boxes, adjusted_empty_bboxes):
    """Check if label centroid is inside any adjusted empty bbox. Returns (is_empty, empty_box_coords)."""
    try:
        for idx, empty_bbox in enumerate(empty_boxes):
            if empty_bbox.contains(label_centroid):
                try:
                    selected_empty_box = adjusted_empty_bboxes[idx]
                    print_log('debug', processing_key, "TEST label_centroid", label_centroid)
                    print_log('debug', processing_key, "TEST selected_empty_box", selected_empty_box)
                    return True, selected_empty_box
                except Exception as empty_err:
                    print_log('debug', processing_key, "TEST label_centroid", label_centroid)
                    print_log('debug', processing_key, "TEST empty_bbox", empty_bbox)
                    logging.error(f"Error in selecting empty box Index {idx} , Error :{empty_err}")
                    return True, ""
        return False, ""
    except Exception as e:
        logging.error(f"Error in get_empty_details: {str(e)}")
        return False, ""

def create_empty_grids(processing_key, empty_grid: Dict[int, List[str]],
                       sku_grid_info: List[List[Tuple[str, float, float, int]]]) -> tuple[List[List[str]], List[List[Tuple[str, float, float, int]]]]:
    # Creating empty_grid_2d from empty_grid
    empty_grid_2d = [empty_grid[shelf] for shelf in sorted(empty_grid.keys())]

    # Creating empty_grid_info based on sku_grid_info and empty_grid_2d
    empty_grid_info = []
    for shelf_idx, shelf in enumerate(empty_grid_2d):
        shelf_info = []
        for spot_idx, spot in enumerate(shelf):
            if spot == 'Empty':
                # Checking if there's a corresponding entry in sku_grid_info
                if shelf_idx < len(sku_grid_info) and spot_idx < len(sku_grid_info[shelf_idx]):
                    sku, image_price, retail_price, location_id, confidence = sku_grid_info[shelf_idx][spot_idx]
                    shelf_info.append(('Empty', sku, image_price, retail_price, location_id, confidence))
                else:
                    # If no corresponding entry, use placeholder values
                    shelf_info.append(('Empty', '', 0.0, 0.0, 0, 0.0))
            else:
                # For non-empty spots, use placeholder values
                shelf_info.append(('', '', 0.0, 0.0, 0, 0.0))
        empty_grid_info.append(shelf_info)
    print_log('debug', processing_key, "Create empty grid complete", "no_value")
    return empty_grid_2d, empty_grid_info

def process_shelf_collages(processing_key, shelf_collages: List[bytes], digit_model_interface: SellingModelBase, label_confidence_map: Dict[int, List[float]] = None) -> Tuple[List[List[str]], List[List[float]]]:
    """Process each shelf collage with YOLOv8 for digit detection.
    
    Args:
        processing_key: Unique key for logging
        shelf_collages: List of collage images as JPEG bytes
        digit_model_interface: Model interface for OCR/digit detection
        
    Returns:
        Tuple[List[List[str]], List[List[float]]]:
            - 2D price grid where each row represents a shelf and contains detected prices
            - 2D confidence grid with corresponding confidence values for each price
    """
    price_grid = []
    confidence_grid = []

    for i, collage in enumerate(shelf_collages):
        start_time = get_time()
        shelf_prices, shelf_confidences = process_image_with_yolov8(processing_key, collage, digit_model_interface)
        elapsed_time = get_time() - start_time
        print_log('debug', processing_key, f"Shelf collage {i + 1} processing time (ms)", elapsed_time)
        shelf_prices_flattened = [price for row in shelf_prices for price in row]
        shelf_confidences_flattened = [conf for row in shelf_confidences for conf in row]
        
        # Get price tag confidences for this shelf from label_confidence_map
        tag_confs = label_confidence_map.get(i+1, []) if label_confidence_map else []
        
        # Fuse digit and tag confidences
        fused_confidences = []
        for j, digit_conf in enumerate(shelf_confidences_flattened):
            tag_conf = tag_confs[j] if j < len(tag_confs) else 0.0
            # Defensive cast: ensure both confidences are floats, default to 0.0 if None or invalid
            digit_conf = float(digit_conf) if digit_conf is not None else 0.0
            tag_conf = float(tag_conf) if tag_conf is not None else 0.0
            # Weighted fusion: 60% tag confidence + 40% digit confidence
            fused_conf = 0.6 * tag_conf + 0.4 * digit_conf
            fused_confidences.append(fused_conf)
        
        price_grid.append(shelf_prices_flattened)
        confidence_grid.append(fused_confidences)
        print_log('debug', processing_key, f"Processed shelf collage {i+1} successfully", f"prices={shelf_prices}")

    print_log('debug', processing_key, "process_shelf_collages completed",
              f"total_shelves={len(price_grid)}, price_grid={price_grid}, "
              f"total_confidence_values={sum(len(row) for row in confidence_grid)}, "
              f"avg_confidence={sum(sum(row) for row in confidence_grid) / sum(len(row) for row in confidence_grid) if sum(len(row) for row in confidence_grid) > 0 else 0:.3f}")
    return price_grid, confidence_grid


def process_image_with_yolov8(processing_key, image_bytes: bytes, digit_model_interface: SellingModelBase) -> Tuple[List[List[str]], List[List[float]]]:
    """Process image with YOLOv8 for digit detection.
    
    Args:
        processing_key: Unique key for logging
        image_bytes: Image as raw bytes in JPEG format
        digit_model_interface: Model interface for OCR/digit detection
        
    Returns:
        2D grid of detected price strings
    """
    status_code, results = retry_model_predict(
        digit_model_interface.predict,
        processing_key,
        model_name="digit_model_interface",
        image_bytes=image_bytes,
        iou=0.5
    )
    
    if status_code == 200 and results is not None:
        bboxes = results.boxes
        classes = [str(int(cls)) for cls in results.class_list]
        confidences = list(results.confidences) if hasattr(results, 'confidences') else [0.0] * len(bboxes)
        print_log('debug', processing_key, "Detected No of objects in the image", len(bboxes))
        return process_image_output(processing_key, bboxes, classes, confidences)
    else:
        return process_image_output(processing_key, [], [], [])


def process_image_output(processing_key, bboxes: List[List[float]], classes: List[str], confidences: List[float] = None) -> Tuple[List[List[str]], List[List[float]]]:
    """Processing the output of YOLOv8 digit detection. Returns (price_grid, confidence_grid)."""
    try:
        if confidences is None:
            confidences = [0.0] * len(bboxes)
        classwise_bboxes = get_classwise_bboxes(processing_key, bboxes, classes)
        classwise_confidences = get_classwise_confidences(processing_key, confidences, classes)
        price_grid, confidence_grid = extract_prices(processing_key, classwise_bboxes, classwise_confidences)
        return price_grid, confidence_grid
    except Exception as e:
        logging.error(f"[{processing_key}] : An error occurred in process_image_output: {str(e)}", exc_info=True)
        return [], []


def get_classwise_bboxes(processing_key, bboxes: List[List[float]], classes: List[str]) -> Dict[str, List[List[float]]]:
    """Organizing bounding boxes by class."""
    if not bboxes or not classes or len(bboxes) != len(classes):
        raise ValueError('Bounding boxes and classes are either empty or of different lengths.')

    classwise_bboxes = {str(i): [] for i in range(15)}  # Updated to 15 classes with recent changes in model
    for bbox, cls in zip(bboxes, classes):
        if cls not in classwise_bboxes:
            logging.error(f'[{processing_key}] : Class {cls} is not a valid class identifier.')
            continue
        classwise_bboxes[cls].append(bbox)
    return classwise_bboxes


def get_classwise_confidences(processing_key, confidences: List[float], classes: List[str]) -> Dict[str, List[float]]:
    """Organizing confidences by class, parallel to get_classwise_bboxes."""
    if not confidences or not classes or len(confidences) != len(classes):
        raise ValueError('Confidences and classes are either empty or of different lengths.')

    classwise_confidences = {str(i): [] for i in range(15)}
    for conf, cls in zip(confidences, classes):
        if cls not in classwise_confidences:
            logging.error(f'[{processing_key}] : Class {cls} is not a valid class identifier.')
            continue
        classwise_confidences[cls].append(conf)
    return classwise_confidences


def extract_prices(processing_key, classwise_bboxes: Dict[str, List[List[float]]], classwise_confidences: Dict[str, List[float]] = None) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Extract prices from classified bounding boxes.

    Args:
        processing_key: Unique identifier for the current image/process.
        classwise_bboxes: Dictionary mapping class labels to lists of bounding boxes.
        classwise_confidences: Dictionary mapping class labels to lists of confidence scores (optional).

    Returns:
        Tuple[List[List[str]], List[List[float]]]:
            - price_grid: 2D list of extracted prices organized by price label rows
            - confidence_grid: 2D list of average confidence values for each price
    """
    if classwise_confidences is None:
        classwise_confidences = {str(i): [] for i in range(15)}
    sorted_price_labels_rows = sort_bboxes(classwise_bboxes, '10')
    sorted_dollar_bboxes = sort_bboxes(classwise_bboxes, '11')
    sorted_cent_bboxes = sort_bboxes(classwise_bboxes, '12')
    sorted_cent_symbols = sort_bboxes(classwise_bboxes, '13')
    sorted_cent_boxes = sort_bboxes(classwise_bboxes, '14')

    sorted_digits = []
    sorted_digit_confidences = []
    for digit_class in map(str, range(10)):
        if digit_class in classwise_bboxes:
            class_bboxes = classwise_bboxes[digit_class]
            class_confs = classwise_confidences.get(digit_class, [0.0] * len(class_bboxes))
            sorted_digits.extend([(bbox, digit_class) for bbox in class_bboxes])
            sorted_digit_confidences.extend(class_confs)
    
    # Sort digits by y-coordinate, maintaining parallel confidence list
    sorted_indices = sorted(range(len(sorted_digits)), key=lambda i: sorted_digits[i][0][1])
    sorted_digits = [sorted_digits[i] for i in sorted_indices]
    sorted_digit_confidences = [sorted_digit_confidences[i] for i in sorted_indices]

    price_grid = []
    confidence_grid = []
    for sorted_price_labels in sorted_price_labels_rows:
        current_row_prices = []
        current_row_confidences = []
        for label in sorted_price_labels:
            price_label_digits = [(bbox, digit) for bbox, digit in sorted_digits if centroid_inside(bbox, label)]
            # Get corresponding confidences for these digits
            digit_indices = []
            for bbox, digit in price_label_digits:
                for idx, (sorted_bbox, _) in enumerate(sorted_digits):
                    if sorted_bbox == bbox:
                        digit_indices.append(idx)
                        break
            label_confidences = [sorted_digit_confidences[idx] for idx in digit_indices] if digit_indices else [0.0]

            # Checking for cents-only scenario
            has_cent_symbol = any(centroid_inside(bbox, label) for bbox in sorted_cent_symbols)
            has_cent_box = any(centroid_inside(bbox, label) for bbox in sorted_cent_boxes)
            has_dollar_box = any(centroid_inside(bbox, label) for bbox in sorted_dollar_bboxes)

            if (has_cent_symbol or has_cent_box) and not has_dollar_box and price_label_digits:
                # Processing as cents-only price
                cent_digits = sorted(price_label_digits, key=lambda x: x[0][0])
                cent_part = ''.join([digit[1] for digit in cent_digits]).zfill(2)
                price = f"$0.{cent_part}"
                print_log('debug', processing_key, "Detected cents-only price", price)
            else:
                # Processing as normal price
                dollar_digits = [digit for digit in price_label_digits if
                                 any(centroid_inside(digit[0], dollar_bbox) for dollar_bbox in sorted_dollar_bboxes)]
                cent_digits = [digit for digit in price_label_digits if
                               any(centroid_inside(digit[0], cent_bbox) for cent_bbox in sorted_cent_bboxes)]

                dollar_digits.sort(key=lambda x: x[0][0])
                cent_digits.sort(key=lambda x: x[0][0])

                dollar_part = ''.join([digit[1] for digit in dollar_digits])
                cent_part = ''.join([digit[1] for digit in cent_digits]).zfill(2)

                price = f"${dollar_part if dollar_part else '0'}.{cent_part}"
                print_log('debug', processing_key, "Detected normal price", price)

            current_row_prices.append(price)
            # Calculate confidence as average of digit confidences, default to 0.0 if no digits
            avg_confidence = sum(label_confidences) / len(label_confidences) if label_confidences else 0.0
            current_row_confidences.append(avg_confidence)

        price_grid.append(current_row_prices)
        confidence_grid.append(current_row_confidences)

    return price_grid, confidence_grid


def sort_bboxes(classwise_bboxes: Dict[str, List[List[float]]], cls: str) -> List[List[float]]:
    """Sorting bounding boxes for a specific class."""
    if cls not in classwise_bboxes:
        raise ValueError(f'No {cls} labels in the input data.')

    bboxes = classwise_bboxes[cls]
    if cls == '10':  # if sorting price labels
        return sort_labels(bboxes)

    bboxes.sort(key=lambda x: x[0])
    return bboxes


def sort_labels(bboxes: List[List[float]]) -> List[List[List[float]]]:
    """Sorting price label bounding boxes into rows."""
    bboxes.sort(key=lambda x: x[1])
    rows = []
    current_row = [bboxes[0]]
    row_y = bboxes[0][1]

    for label in bboxes[1:]:
        if abs(label[1] - row_y) < 100:  # if the label is in the same row
            current_row.append(label)
        else:  # if the label is in the next row
            rows.append(
                sorted(current_row, key=lambda x: x[0]))  # sorting the current row by x-coordinate and add to rows
            current_row = [label]
            row_y = label[1]

    rows.append(sorted(current_row, key=lambda x: x[0]))  # adding the last row
    return rows


def centroid_inside(bbox: List[float], outer_bbox: List[float]) -> bool:
    """Checking if the centroid of a bbox is inside another bbox."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    oxmin, oymin, oxmax, oymax = outer_bbox
    return oxmin <= cx <= oxmax and oymin <= cy <= oymax


def extract_details(filename: str) -> Tuple[str, str]:
    """Extracting store number and inventory location from filename."""
    match = re.match(r'^\d+#(\d{4})#.*?#(\d{2}-\d{3})(\.jpg|\.jpeg|\.webp)?$', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def sequence_data_needs_refresh(csv_data: Optional[List[Dict]]) -> bool:
    """
    Detect cache responses that are empty or demonstrably incomplete.

    A missing final row cannot be inferred from row contents alone. When the
    data source supplies an expected-count field, use it; otherwise detect
    interior LOCATIONID gaps because the matcher requires contiguous sequence.
    """
    if not csv_data:
        return True

    expected_count_keys = (
        "EXPECTED_RECORD_COUNT",
        "TOTAL_RECORD_COUNT",
        "SEQUENCE_RECORD_COUNT",
    )
    for key in expected_count_keys:
        raw_expected = csv_data[0].get(key)
        if raw_expected is None:
            continue
        try:
            if len(csv_data) < int(raw_expected):
                return True
        except (TypeError, ValueError):
            pass

    locations = []
    for item in csv_data:
        location = item.get("LOCATIONID")
        if location is None:
            continue
        try:
            locations.append(int(location))
        except (TypeError, ValueError):
            return True

    unique_locations = sorted(set(locations))
    return any(
        right - left > 1
        for left, right in zip(unique_locations, unique_locations[1:])
    )


def prepare_data_for_matching(processing_key, csv_data: List[Dict]) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, str]]:
    dollar_amount_to_locationIDs = {}
    locationID_to_sku = {}
    print_log('debug', processing_key, "Number of items in csv_data", len(csv_data))

    # Sort data by location ID to maintain proper sequence
    sorted_data = sorted(csv_data,
                         key=lambda x: x['LOCATIONID'] if x['LOCATIONID'] is not None
                         else float('inf'))

    for item in sorted_data:
        try:
            curr_retl_amt = item['CURR_RETL_AMT']
            if isinstance(curr_retl_amt, str):
                curr_retl_amt = float(curr_retl_amt)

            if math.isnan(curr_retl_amt):
                logging.warning(f"[{processing_key}] : Skipping item due to NaN CURR_RETL_AMT: {item}")
                continue

            # Storing cent values for prices less than $1, otherwise store dollar amount
            amount_key = int(curr_retl_amt * 100) if curr_retl_amt < 1 else int(curr_retl_amt)
            sku = item['SKU_NBR']
            location = item['LOCATIONID']

            # Check if this is a null location item
            is_null_location = location is None

            if amount_key not in dollar_amount_to_locationIDs:
                dollar_amount_to_locationIDs[amount_key] = []

            if is_null_location:
                # For null locations, use None
                dollar_amount_to_locationIDs[amount_key].append((None, sku))
            else:
                # For known locations, use location ID
                dollar_amount_to_locationIDs[amount_key].append((location, sku))
                locationID_to_sku[location] = sku

        except (ValueError, KeyError) as e:
            logging.error(f"[{processing_key}] : Error processing item: {item}. Error: {str(e)}")
            continue
    print_log('debug', processing_key, "Number of unique dollar amounts",
              len(dollar_amount_to_locationIDs))
    print_log('debug', processing_key, "Number of known locations",
              len(locationID_to_sku))

    return dollar_amount_to_locationIDs, locationID_to_sku

def match_non_empty_flat(flattened_grid: List[str], dollar_amount_to_locationIDs: Dict[int, List[Tuple[int, str]]], csv_data: List[Dict]) -> Tuple[List[str], List[Tuple[str, float, float, int]]]:
    # Pure matching logic, no cloud/db/model dependencies
    res_grid, res_info = _match_non_empty_core("test", flattened_grid, dollar_amount_to_locationIDs, csv_data)
    # Backwards-compatibility: tests expect 4-tuple info (sku, grid_price, data_price, location_id)
    compact_info = [(sku, grid_p, data_p, loc) for (sku, grid_p, data_p, loc, *_rest) in res_info]
    return res_grid, compact_info
    # Pure matching logic, no cloud/db/model dependencies
    # This is a wrapper for offline tests
    


def match_non_empty(*args, **kwargs):
    """
    Flexible wrapper around the core non-empty matching logic.

    Supported call signatures:
      - match_non_empty(processing_key, flattened_grid, dollar_amount_to_locationIDs, csv_data)
      - match_non_empty(price_grid, sequence_data)  # convenience for older tests

    Returns same tuple as _match_non_empty_core.
    """
    # Four-argument form (processing_key provided)
    if len(args) >= 4:
        processing_key, flattened_grid, dollar_amount_to_locationIDs, csv_data = args[0], args[1], args[2], args[3]
        return _match_non_empty_core(processing_key, flattened_grid, dollar_amount_to_locationIDs, csv_data)

    # Two-argument legacy form: (price_grid, sequence_data)
    if len(args) == 2:
        price_grid, sequence_data = args
        # Flatten price grid (row-major) and prepare dollar->locations map
        flattened_grid = [cell for row in price_grid for cell in row]
        # prepare_data_for_matching expects (processing_key, csv_data)
        dollar_amount_to_locationIDs, _ = prepare_data_for_matching("test", sequence_data)
        return _match_non_empty_core("test", flattened_grid, dollar_amount_to_locationIDs, sequence_data)

    raise TypeError("match_non_empty expects either 4 args (processing_key, flattened_grid, dollar_map, csv_data) or 2 args (price_grid, sequence_data)")

def _match_non_empty_core(processing_key,
    flattened_grid: List[str],
    dollar_amount_to_locationIDs: Dict[int, List[Tuple[int, str]]],  
    csv_data: List[Dict],
) -> Tuple[List[str], List[Tuple[str, float, float, int]]]:
    """

      *Exact sequence search
      *Greedy runs only if exact sequence search produced zero anchors
      *Gap bridge
      *Anchor expansion (skips unreadable)
      *Long-gap auto-fill
      *Facing sweep (unreadable tags stop scan)
      *Post-facing expansion & long-gap

    Returns
    -------
    result_grid : list[str]
    result_grid_info : list[(sku, grid_price, data_price, location, confidence)]
    """

    #Parse grid
    result_grid: List[str] = ["No Match"] * len(flattened_grid)
    result_grid_info: List[Tuple[str, float, float, int, float]] = [
        ("No Match", 0.0, 0.0, 0, 0.0) for _ in flattened_grid
    ]
    original_prices: Dict[int, float] = {}

    for i, txt in enumerate(flattened_grid):
        txt = (txt or "").strip()
        if txt in ("", "Empty"):
            result_grid[i] = "No SKU"
            result_grid_info[i] = ("No SKU", 0.0, 0.0, 0, 0.0)
            continue
        try:
            val = float(txt.replace("$", ""))
            original_prices[i] = val
            result_grid_info[i] = ("No Match", val, 0.0, 0, 0.0)
        except ValueError:
            result_grid[i] = "Invalid Price"
            result_grid_info[i] = ("Invalid Price", 0.0, 0.0, 0, 0.0)

    if not original_prices:
        print_log('debug', processing_key, "No valid prices found – nothing to match.", "no_value")
        return result_grid, result_grid_info
    print_log('debug', processing_key, "Total price tags to match", len(original_prices))

    #Helpers from csv_data
    loc2sku = {
        r["LOCATIONID"]: r["SKU_NBR"] for r in csv_data if r.get("LOCATIONID") is not None
    }
    sku2ret = {r["SKU_NBR"]: r.get("CURR_RETL_AMT", 0.0) for r in csv_data}
    loc2ret = {
        r["LOCATIONID"]: r["CURR_RETL_AMT"] for r in csv_data if r.get("LOCATIONID") is not None
    }
    loc2dol = {
        loc: (int(p) if p >= 1 else int(round(p * 100))) for loc, p in loc2ret.items()
    }

    price2locs: Dict[int, List[int]] = {}
    for loc, d in loc2dol.items():
        price2locs.setdefault(d, []).append(loc)

    #Utility helpers
    def dollar_int(p: float) -> int:
        return int(p) if p >= 1 else int(round(p * 100))

    def dollar_sub(x: float, y: float) -> bool:
        sx, sy = str(abs(dollar_int(x))), str(abs(dollar_int(y)))
        return sx != sy and (sx in sy or sy in sx)
    
    def price_match_score(grid_price: float, data_price: float) -> float:
        """
        Enhanced price matching with fuzzy tolerance for OCR errors.
        Returns score: 1.0 = exact, 0.7 = substring, 0.5 = close, -1.0 = mismatch
        """
        if grid_price == 0:
            return 0.3
        
        g_int = dollar_int(grid_price)
        d_int = dollar_int(data_price)
        
        # Exact match
        if g_int == d_int:
            return 1.0
        
        # Substring match (e.g., 14 vs 140, 7 vs 79)
        if dollar_sub(grid_price, data_price):
            return 0.7
        
        # Close match within ±2 (OCR often misreads single digits)
        # e.g., 14.97 misread as 14.90 due to OCR error on cents
        diff = abs(grid_price - data_price)
        if diff <= 0.10:  # Within 10 cents
            return 0.9
        elif diff <= 0.50:  # Within 50 cents
            return 0.6
        elif diff <= 1.00:  # Within $1
            return 0.5
        
        # Dollar-only match (cents misread but dollars correct)
        # e.g., $14.97 vs $14.00 - OCR might miss cents entirely
        if int(grid_price) == int(data_price) and abs(diff) < 2.0:
            return 0.6
        
        return -1.0

    assigned, used_locs = set(), set()

    def anchors():
        return sorted(
            [(idx, info[3]) for idx, info in enumerate(result_grid_info) if idx in assigned],
            key=lambda t: t[0],
        )

    def assign_real(pos: int, loc: int):
        sku = loc2sku[loc]
        result_grid[pos] = sku
        result_grid_info[pos] = (
            sku,
            original_prices.get(pos, 0.0),
            sku2ret.get(sku, 0.0),
            loc,
            0
        )
        assigned.add(pos)
        used_locs.add(loc)

    def assign_facing(pos: int, loc: int):
        sku = loc2sku[loc]
        result_grid[pos] = sku
        result_grid_info[pos] = (
            sku,
            original_prices.get(pos, 0.0),
            sku2ret.get(sku, 0.0),
            loc,
            0
        )
        assigned.add(pos)  

    #Exact sequence
    MIN_LEN, MAX_LEN = 3, 8
    pos_sorted = sorted(original_prices)
    seq_specs: List[Tuple[List[int], List[int]]] = []
    for i in range(len(pos_sorted)):
        for L in range(MIN_LEN, min(MAX_LEN, len(pos_sorted) - i) + 1):
            seg = pos_sorted[i : i + L]
            seg_d = [dollar_int(original_prices[p]) for p in seg]
            if len(set(seg_d)) < 2:
                continue
            seq_specs.append((seg, seg_d))
    seq_specs.sort(key=lambda s: -len(s[0]))

    seq_matches: List[Tuple[float, List[int], List[int]]] = []
    for g_pos, g_d in seq_specs:
        first = g_d[0]
        if first not in price2locs:
            continue
        for start_loc in price2locs[first]:
            loc_seq = [start_loc + k for k in range(len(g_d))]
            if all(loc2dol.get(lc) == g_d[k] for k, lc in enumerate(loc_seq)):
                uniq = len(set(g_d))
                conf = 0.7 * (len(g_d) / MAX_LEN) + 0.3 * (uniq / len(g_d))
                seq_matches.append((conf, g_pos, loc_seq))
    seq_matches.sort(reverse=True)
    for _, g_pos, loc_seq in seq_matches:
        if any(p in assigned or lc in used_locs for p, lc in zip(g_pos, loc_seq)):
            continue
        for p, loc in zip(g_pos, loc_seq):
            assign_real(p, loc)

    print_log('debug', processing_key, "After (exact sequence) match", len(assigned))

    #Greedy, when zero anchors
    if not anchors():
        grid_positions = sorted(original_prices)
        ordered_locs = sorted(loc2ret)

        best_start = None
        best_len = 0
        best_score = 0.0

        for loc_start in ordered_locs:
            score_sum = 0.0
            length = 0
            for offset, pos in enumerate(grid_positions):
                loc = loc_start + offset
                if loc not in loc2ret:
                    break
                g_price = original_prices[pos]
                d_price = loc2ret[loc]
                
                # Use enhanced price matching
                s = price_match_score(g_price, d_price)
                
                if s < 0:
                    break
                score_sum += s
                length += 1
            if length >= 3:
                avg = score_sum / length
                # Lower threshold from 0.6 to 0.5 to allow more fuzzy matches
                if avg >= 0.5 and length > best_len:
                    best_start, best_len, best_score = loc_start, length, avg

        if best_start is not None:
            for k in range(best_len):
                pos = grid_positions[k]
                loc = best_start + k
                if loc in loc2sku and loc not in used_locs:
                    assign_real(pos, loc)
        print_log('debug', processing_key, f"Greedy applied: start={best_start} "
                    f"len={best_len} score={best_score} – matches now {len(assigned)}"
                    , "no_value")

    #Gap-bridge
    for (lp, ll), (rp, rl) in zip(anchors(), anchors()[1:]):
        if rp - lp <= 1 or rl - ll <= 1:
            continue
        gap_pos = [p for p in range(lp + 1, rp) if p not in assigned]
        exp_loc = list(range(ll + 1, rl))
        if len(gap_pos) != len(exp_loc):
            continue
        ok = True
        for gp, el in zip(gap_pos, exp_loc):
            gp_price = original_prices.get(gp, 0.0)
            dp_price = loc2ret.get(el, 0.0)
            if gp_price == 0:
                continue
            # Use enhanced price matching with threshold of 0.5
            score = price_match_score(gp_price, dp_price)
            if score >= 0.5:
                continue
            ok = False
            break
        if ok:
            for gp, el in zip(gap_pos, exp_loc):
                if el not in used_locs:
                    assign_real(gp, el)

    print_log('debug', processing_key, "After (gap-bridge) matches", len(assigned))

    #Anchor expansion
    def expand(a_pos: int, a_loc: int, step: int) -> int:
        added = 0
        pos = a_pos + step
        while (
            0 <= pos < len(flattened_grid)
            and pos in assigned
            and result_grid_info[pos][3] == a_loc
        ):
            pos += step  # skip facings
        loc = a_loc + step
        while 0 <= pos < len(flattened_grid) and loc not in used_locs:
            if pos in assigned:
                break
            g_price = original_prices.get(pos, 0.0)
            if g_price == 0.0:
                pos += step
                loc += step
                continue
            d_price = loc2ret.get(loc)
            if d_price is None:
                break
            # Use enhanced price matching with threshold of 0.6 for expansion
            score = price_match_score(g_price, d_price)
            if score >= 0.6:
                assign_real(pos, loc)
                added += 1
                pos += step
                loc += step
            else:
                break
        return added

    for _ in range(10):
        if not any(expand(p, loc_var, -1) + expand(p, loc_var, 1) for p, loc_var in anchors()):
            break

    print_log('debug', processing_key, "After (expansion) matches", len(assigned))

    #Long-gap auto-fill
    for (lp, ll), (rp, rl) in zip(anchors(), anchors()[1:]):
        pg, lg = rp - lp - 1, rl - ll - 1
        if pg <= 0 or pg != lg:
            continue
        for off, gp in enumerate([p for p in range(lp + 1, rp) if p not in assigned], 1):
            loc = ll + off
            if loc not in used_locs and loc in loc2sku:
                assign_real(gp, loc)

    print_log('debug', processing_key, "After (long-gap fill) matches", len(assigned))

    #Early-exit on full coverage
    if len(assigned) == len(original_prices):
        print_log('debug', processing_key, "Full coverage achieved after pass 4 – early exit.", "no_value")
        print_log('debug', processing_key, "Final coverage", (100.0 * len(assigned) / len(original_prices)))
        return result_grid, result_grid_info

    #Facing sweep
    def facing_sweep() -> int:
        made = 0
        Q = anchors()
        i = 0
        while i < len(Q):
            a_pos, a_loc = Q[i]
            a_price, a_dol = loc2ret[a_loc], loc2dol[a_loc]
            for step in (-1, 1):
                pos, loc = a_pos + step, a_loc
                while 0 <= pos < len(flattened_grid):
                    if pos in assigned and result_grid_info[pos][3] != loc:
                        break
                    if pos in assigned:
                        pos += step
                        continue
                    g_price = original_prices.get(pos, 0.0)
                    if g_price == 0.0:
                        break  # stop, leave as No Match
                    g_dol = dollar_int(g_price)

                    nxt_loc = loc + step
                    if nxt_loc in loc2ret and nxt_loc not in used_locs:
                        nxt_price, nxt_dol = loc2ret[nxt_loc], loc2dol[nxt_loc]
                        if g_dol == nxt_dol or dollar_sub(g_price, nxt_price):
                            assign_real(pos, nxt_loc)
                            made += 1
                            Q.append((pos, nxt_loc))
                            break

                    if g_dol == a_dol or dollar_sub(g_price, a_price):
                        assign_facing(pos, loc)
                        made += 1
                        pos += step
                        continue
                    break
            i += 1
        return made

    facing_sweep()
    print_log('debug', processing_key, "After (facing sweep) Matches", len(assigned))

    #Post-facing expansion & long-gap
    for _ in range(10):
        if not any(expand(p, loc_value, -1) + expand(p, loc_value, 1) for p, loc_value in anchors()):
            break

    for (lp, ll), (rp, rl) in zip(anchors(), anchors()[1:]):
        pg, lg = rp - lp - 1, rl - ll - 1
        if pg <= 0 or pg != lg:
            continue
        for off, gp in enumerate([p for p in range(lp + 1, rp) if p not in assigned], 1):
            loc = ll + off
            if loc not in used_locs and loc in loc2sku:
                assign_real(gp, loc)

    coverage = 100.0 * len(assigned) / len(original_prices)

    print_log('debug', processing_key, f"Final assignment count: {len(assigned)} of {len(original_prices)}"
                                       f" (coverage {coverage})", "no_value")

    return result_grid, result_grid_info

def grid_position_to_flat_index(price_grid: List[List[str]], row: int, col: int) -> int:
    """Convert a row/column position to a row-major index for ragged grids."""
    if not 0 <= row < len(price_grid) or not 0 <= col < len(price_grid[row]):
        raise IndexError(f"Grid position out of bounds: ({row}, {col})")
    return sum(len(grid_row) for grid_row in price_grid[:row]) + col


def match_empty_spots(price_grid: List[List[str]], sequence_data: List[Dict]) -> List[Dict]:
    """
    Processes empty spots using enhanced pattern matching.
    Returns a list of dictionaries, each containing:
      - 'position': (row, col)
      - 'sku': matched SKU,
      - 'confidence': score,
      - 'details': breakdown of scores.
    """
    empty_spots = find_empty_spots(price_grid)
    matches = []
    for empty_spot in empty_spots:
        match = find_best_match(empty_spot, price_grid, sequence_data)
        if match:
            # Add the row-major sequence index for later mapping.
            row, col = empty_spot['position']
            empty_spot['grid_sequence'] = grid_position_to_flat_index(
                price_grid, row, col
            )
            matches.append({
                'position': empty_spot['position'],
                'grid_sequence': empty_spot['grid_sequence'],
                'sku': match['sku'],
                'location_id': match.get('location_id', 0),
                'confidence': match['confidence'],
                'details': match['details']
            })
    return matches


def find_empty_spots(price_grid: List[List[str]]) -> List[Dict]:
    """
    Identifies all empty spots in the price grid.
    """
    spots = []
    for row_idx, row in enumerate(price_grid):
        for col_idx, price_str in enumerate(row):
            if price_str is not None and 'Empty' in price_str:
                spots.append({
                    'position': (row_idx, col_idx),
                    'price': parse_price(price_str)
                })
    return spots


def find_best_match(empty_spot: Dict,
                    price_grid: List[List[str]],
                    sequence_data: List[Dict]) -> Optional[Dict]:
    """
    For a given empty spot, extract surrounding patterns and score candidates from sequence_data.
    """
    row, col = empty_spot['position']
    empty_price = empty_spot['price']
    pattern_info = get_pattern_info_empty(row, col, price_grid)
    patterns = extract_patterns_empty(empty_spot, price_grid, pattern_info)
    candidates = find_candidates_empty(empty_price, sequence_data)
    if not candidates:
        return None
    scored_candidates = []
    for candidate in candidates:
        score = score_candidate_empty(empty_spot, patterns, candidate, pattern_info)
        if score['total_score'] >= get_minimum_confidence_empty(pattern_info):
            scored_candidates.append({
                'sku': candidate.get('SKU_NBR', candidate.get('sku_nbr', "No SKU")),
                'location_id': candidate.get('LOCATIONID', candidate.get('locationid', 0)),
                'confidence': score['total_score'],
                'details': score
            })
    if scored_candidates:
        return max(scored_candidates, key=lambda x: x['confidence'])
    return None


def get_pattern_info_empty(row: int, col: int, price_grid: List[List[str]]) -> Dict:
    """
    Gets pattern info for an empty spot.
    """
    max_row = len(price_grid) - 1
    max_col = len(price_grid[row]) - 1
    return {
        'is_edge': row == 0 or row == max_row or col == 0 or col == max_col,
        'core_size': 3,
        'extended_size': min(5, max(3, min(col + 3, (max_col - col) + 2)))
    }


def extract_patterns_empty(empty_spot: Dict, price_grid: List[List[str]], pattern_info: Dict) -> Dict:
    """
    Extracts a core and extended pattern around an empty spot.
    """
    row, col = empty_spot['position']
    patterns = {
        'core': extract_pattern_window_empty(price_grid, row, col, pattern_info['core_size'])
    }
    if pattern_info['extended_size'] > 3:
        patterns['extended'] = extract_pattern_window_empty(price_grid, row, col, pattern_info['extended_size'])
    return patterns


def extract_pattern_window_empty(price_grid: List[List[str]], row: int, col: int, window_size: int) -> List[Dict]:
    """
    Extracts a horizontal window for empty matching.
    """
    pattern = []
    half_window = window_size // 2
    for i in range(-half_window, half_window + 1):
        current_col = col + i
        if 0 <= current_col < len(price_grid[row]):
            price_str = price_grid[row][current_col]
            pattern.append(parse_price(price_str))
        else:
            pattern.append({'valid': False})
    return pattern


def _sequence_retail_amount(candidate: Dict) -> float:
    """Read retail amount from production or legacy sequence row schemas."""
    value = candidate.get('CURR_RETL_AMT', candidate.get('curr_retl_amt', 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def find_candidates_empty(empty_price: Dict, sequence_data: List[Dict]) -> List[Dict]:
    """
    Filters candidates from sequence_data based on an exact dollar match.
    """
    target_dollars = empty_price.get('dollars')
    if target_dollars is None:
        return []
    return [
        candidate
        for candidate in sequence_data
        if int(_sequence_retail_amount(candidate)) == target_dollars
    ]


def score_pattern_match_empty(pattern: List[Dict], candidate: Dict, is_core: bool) -> float:
    """
    Scores how well a pattern matches a candidate (dollar amounts).
    """
    score = 0.0
    weight = 1.0 if is_core else 0.5
    candidate_price = _sequence_retail_amount(candidate)
    candidate_dollars = int(candidate_price)
    valid_elements = 0
    for element in pattern:
        if element.get('valid'):
            valid_elements += 1
            if element.get('dollars') == candidate_dollars:
                score += weight
    return score / valid_elements if valid_elements else 0.0


def score_cents_match_empty(price1: float, price2: float) -> float:
    """
    Scores cents matching for empty cells in a position-aware manner.
    """

    def get_cents_digits(price: float) -> Tuple[int, int]:
        cents = int(round((price % 1) * 100))
        return divmod(cents, 10)

    tens1, ones1 = get_cents_digits(price1)
    tens2, ones2 = get_cents_digits(price2)
    score = 0.0
    if tens1 == tens2:
        score += 0.6
    if ones1 == ones2:
        score += 0.4
    return score


def calculate_total_score_empty(core_score: float, extended_score: float, cents_score: float,
                                pattern_info: Dict) -> float:
    """
    Combines core, extended, and cents scores for empty matching.
    """
    CORE_WEIGHT = 0.7
    EXTENDED_WEIGHT = 0.2
    CENTS_WEIGHT = 0.1
    if pattern_info['is_edge']:
        CORE_WEIGHT = 0.8
        EXTENDED_WEIGHT = 0.15
        CENTS_WEIGHT = 0.05
    return core_score * CORE_WEIGHT + extended_score * EXTENDED_WEIGHT + cents_score * CENTS_WEIGHT


def score_candidate_empty(empty_spot: Dict, patterns: Dict, candidate: Dict, pattern_info: Dict) -> Dict:
    """
    Computes the composite score for a candidate for an empty spot.
    """
    core_score = score_pattern_match_empty(patterns['core'], candidate, is_core=True)
    extended_score = 0.0
    if 'extended' in patterns:
        extended_score = score_pattern_match_empty(patterns['extended'], candidate, is_core=False)
    cents_score = score_cents_match_empty(
        empty_spot['price']['full_price'],
        _sequence_retail_amount(candidate),
    )
    total_score = calculate_total_score_empty(core_score, extended_score, cents_score, pattern_info)
    return {
        'total_score': total_score,
        'core_score': core_score,
        'extended_score': extended_score,
        'cents_score': cents_score
    }


def get_minimum_confidence_empty(pattern_info: Dict) -> float:
    """
    Minimum confidence threshold for empty matching.
    """
    return 0.90 if pattern_info['is_edge'] else 0.9


def parse_price(price_str: str) -> Dict:
    """
    Parses a price string (e.g. "$7.99") into a structured dictionary.
    """
    try:
        if not price_str or price_str.strip() == "":
            return {'valid': False}
        price = float(price_str.replace('$', ''))
        return {
            'valid': True,
            'full_price': price,
            'dollars': int(price),
            'cents': int(round((price % 1) * 100))
        }
    except ValueError:
        return {'valid': False}

def get_image_angles(processing_key, polygons, padded_image: np.ndarray, padding_width = 300):
    """
    Calculate image angles from polygon masks using PCA analysis.
    
    Args:
        processing_key: Unique key for logging
        polygons: List of polygon coordinates
        padded_image: NP array (already padded)
        padding_width: Width of padding applied to image
        
    Returns:
        Average angle in degrees, or None if no valid angles found
    """
    def get_pca_angle(contour_points):
        if len(contour_points) < 5:
            return None

        data = np.array(contour_points, dtype=np.float32)
        mean = np.mean(data, axis=0)
        centered = data - mean

        cov = np.cov(centered.T)
        _, eig_vecs = np.linalg.eigh(cov)  # eigenvectors in ascending order
        principal_vec = eig_vecs[:, -1]

        angle_rad = np.arctan2(principal_vec[1], principal_vec[0])
        angle_deg = np.degrees(angle_rad)
        angle_deg = abs(angle_deg)

        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        return angle_deg
    angles = []
    for i, polygon in enumerate(polygons):
        try:
            # Validate polygon data
            if not polygon or len(polygon) < 3:  # Need at least 3 points to form a polygon
                print_log('debug', processing_key, f"Invalid polygon {i}: insufficient points. Skipping.", "no_value")
                continue
            
            # Convert to numpy array and adjust for padding
            poly_np = np.array(polygon, dtype=np.int32) - padding_width
            poly_np = poly_np.reshape((-1, 1, 2))
            
            # Validate coordinates are within image bounds after padding adjustment
            height, width = padded_image.shape[:2]
            if np.any(poly_np < 0) or np.any(poly_np[:, :, 0] >= width) or np.any(poly_np[:, :, 1] >= height):
                print_log('debug', processing_key, f"Polygon {i} coordinates out of bounds after padding adjustment. Skipping.", "no_value")
                continue

            temp_mask = np.zeros(padded_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(temp_mask, [poly_np], 1)

            contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                contour = contour.squeeze()
                if contour.ndim == 2:
                    angle = get_pca_angle(contour)
                    if angle is not None:
                        angles.append(angle)
            else:
                logging.warning(f"No valid contours found for polygon {i}. Skipping.")

        except Exception as e:
            logging.error(f"Error processing polygon {i}: {e}")
    # print(f"image angles: {angles}")
    return round(abs(np.mean(angles)), 3) if angles else None

def get_beam_score(beam_count):
    # Beam score
    if beam_count and beam_count >= 2:
        # If the number of beams is >=2, get 50% score
        beam_score = 0.5
    elif beam_count and beam_count == 1:
        # If the number of beams = 1, get 35% score
        beam_score = 0.35
    else:
        # If the number of beams = 0, get 0% score
        beam_score = 0
    return beam_score

def get_label_score(sku_grid_info, total_sku_count):
    # SKU score
    score = 0
    match_count = sum(1 for group in sku_grid_info for _ in group)
    total_count = total_sku_count
    if total_count == match_count:
        # if prices labels detected / items in the sequence data = 100% would add 50% to score
        score = 0.5
    else:
        # If the matching percentage is less or greater than 100, 
        # would subtract the difference from the score, divided by 2. 
        # So if you are 10 items over a shelf of 100 items, you would subtract 5% from the final score.
        if total_count:
            dif = abs(match_count - total_count)
            score = 0.5 - (float(dif) / 2 / max(total_count, match_count))
            score = round(score, 3)
            return score
        else:
            logging.error(f"invalid total_count: {total_count}")
    return score

    
def match_skus_to_grid(processing_key, price_grid: List[List[str]],
                       dollar_amount_to_locationIDs: Dict[int, List[Tuple[int, str]]],
                       locationID_to_sku: Dict[int, str],
                       csv_data: List[Dict],
                       ) -> Tuple[List[List[str]], List[List[Tuple[str, float, float, int]]]]:
    """
    Matching function:
      1. Runs the  two-pass matching logic (for non-empty cells) on the entire grid.
      2. Then, for each cell marked "Empty", runs the enhanced empty-spot matching logic (using a fresh candidate data copy)
         and overwrite the SKU assignment at that position.
    Returns:
      - sku_grid: 2D list of SKUs.
      - sku_grid_info: 2D list of tuples: (SKU, cell price, candidate retail price, location)
    """
    #Copy the data
    sequence_data = copy.deepcopy(csv_data)

    # Step 1: Flattens the grid and run the original matching.
    flattened_grid = []
    grid_structure = []
    for row in price_grid:
        start = len(flattened_grid)
        flattened_grid.extend(row)
        end = len(flattened_grid)
        grid_structure.append((start, end))

    # Runs matching for all cells.
    original_result, original_info = match_non_empty(processing_key, flattened_grid, dollar_amount_to_locationIDs, csv_data)

    # Make a fresh copy of candidate data for empty matching.
    # ('sequence_data' is a fresh copy of csv_data and is not affected by candidate removal.)
    empty_matches = match_empty_spots(price_grid, sequence_data)

    print_log('debug', processing_key, "Called empty spots", "no_value")
    # Overwrites empty cell assignments with the empty matching output.
    for match in empty_matches:
        row, col = match['position']
        pos = grid_position_to_flat_index(price_grid, row, col)
        original_result[pos] = match['sku']
        # Preserve the sequence LOCATIONID in the standard five-field metadata tuple.
        cell_price = parse_price(price_grid[row][col])['full_price'] if parse_price(price_grid[row][col]).get(
            'valid') else 0.0
        original_info[pos] = (
            match['sku'],
            cell_price,
            match['details'].get('total_score', 0.0),
            match.get('location_id', 0),
            match.get('confidence', 0.0),
        )

    # Reconstructs the grid.
    sku_grid = []
    sku_grid_info = []
    for start, end in grid_structure:
        sku_grid.append(original_result[start:end])
        sku_grid_info.append(original_info[start:end])

    return sku_grid, sku_grid_info


def save_results(filename: str, empty_grid: Dict[int, List[str]], price_grid: List[List[str]],
                 sku_grid: List[List[str]], sku_grid_info: List[List[Tuple[str, float, float, int]]],
                 empty_grid_2d: List[List[str]], empty_grid_info: List[List[Tuple[str, str, float, float, int]]], price_label_bboxes: List[List[Tuple[float, float, float, float]]]):
    output_filename = f"{os.path.splitext(filename)[0]}_results.txt"

    with open(output_filename, 'w') as f:
        f.write("Empty Grid:\n")
        for shelf, row in empty_grid.items():
            f.write(f"Shelf {shelf}: {row}\n")

        f.write("\nEmpty Grid 2D:\n")
        for row in empty_grid_2d:
            f.write(f"{row}\n")

        f.write("\nEmpty Grid Info:\n")
        for row in empty_grid_info:
            f.write(f"{row}\n")

        f.write("\nPrice Grid:\n")
        for row in price_grid:
            f.write(f"{row}\n")

        f.write("\nPrice Label BBoxes:\n")
        for row in price_label_bboxes:
            f.write(f"{row}\n")

        f.write("\nSKU Grid:\n")
        for row in sku_grid:
            f.write(f"{row}\n")

        f.write("\nSKU Grid Info:\n")
        for row in sku_grid_info:
            f.write(f"{row}\n")

        # Calculating match statistics  #This is just for my tests can be totally removed 
        total_cells = sum(len(row) for row in price_grid)
        matched_cells = sum(1 for row in sku_grid for cell in row if cell not in ["No SKU", "No Match"])

        f.write("\nMatch Statistics:\n")
        f.write(f"Total cells: {total_cells}\n")
        f.write(f"Matched cells: {matched_cells}\n")

        if total_cells > 0:
            match_percentage = (matched_cells / total_cells) * 100
            f.write(f"Match percentage: {match_percentage:.2f}%\n")
        else:
            f.write("Match percentage: N/A (no cells)\n")

        # Additional statistics
        price_mismatches = sum(1 for row in sku_grid_info for cell in row if
                               cell[0] not in ["No SKU", "No Match"] and abs(cell[1] - cell[2]) > 0.01)
        f.write(f"Price mismatches: {price_mismatches}\n")

        if matched_cells > 0:
            f.write(f"Price mismatch percentage: {(price_mismatches / matched_cells) * 100:.2f}%\n")
        else:
            f.write("Price mismatch percentage: N/A (no matched cells)\n")

        # Location mismatches
        location_mismatches = sum(1 for i, row in enumerate(sku_grid_info) for j, cell in enumerate(row) if
                                  cell[0] not in ["No SKU", "No Match"] and cell[3] != (i * len(row) + j + 1))
        f.write(f"Location mismatches: {location_mismatches}\n")

        if matched_cells > 0:
            f.write(f"Location mismatch percentage: {(location_mismatches / matched_cells) * 100:.2f}%\n")
        else:
            f.write("Location mismatch percentage: N/A (no matched cells)\n")

    logging.info(f"Results saved to {output_filename}")


def _shrink_bbox_centered(bbox_coords, shrink_factor=0.2):
    x1, y1, x2, y2 = bbox_coords
    width = x2 - x1
    height = y2 - y1
    width_reduction = width * shrink_factor / 2
    height_reduction = height * shrink_factor / 2
    
    return [
        x1 + width_reduction,   # new x1 (move right)
        y1 + height_reduction,  # new y1 (move down)
        x2 - width_reduction,   # new x2 (move left)
        y2 - height_reduction   # new y2 (move up)
    ]


def _calculate_empty_grid_statistics(processing_key, edge_empty_grid, empty_grid_2d, hints_data):

    try:
        #Empty counts across empty grids and edge empty grid and hints data and empty state with empty
        
        edge_empty_grid_count = sum(1 for row in edge_empty_grid for cell in row if cell == 'Empty') if edge_empty_grid else 0
        empty_grid_count = sum(1 for row in empty_grid_2d for cell in row if cell == 'Empty') if empty_grid_2d else 0
        hints_data_count = len(hints_data) if hints_data and isinstance(hints_data, list) else 0
        empty_state_count = sum(1 for hint in hints_data if hint.get('empty_state') == 'Empty') if hints_data and isinstance(hints_data, list) else 0

        print_log('debug', processing_key, "Empty grid analysis and final counts", 
                f"Empty counts : empty_grid={empty_grid_count}, edge_empty_grid_count={edge_empty_grid_count}, \
                hints_data_count={hints_data_count}, empty_state_count={empty_state_count}")
    except Exception as e:
        logging.error(f"Error calculating empty grid statistics: {e}")

    return


def build_edge_empty_grid(processing_key, empty_grid_2d, empty_bboxes_grid, hints_data):
    
    edge_empty_grid = [[None for _ in row] for row in empty_grid_2d]  if empty_grid_2d else None
    
    try:
        # hints_data is expected to be a list of hint dictionaries
        if hints_data and isinstance(hints_data, list) and empty_bboxes_grid:
            for hint in hints_data:
                print_log('debug', processing_key, "hint", hint)       
                hint_coords = hint.get('coordinates')
                empty_state = hint.get('empty_state')
                if not hint_coords:
                    continue
                shrink_hints_coords = _shrink_bbox_centered(hint_coords, shrink_factor=0.2)
                # Search for a matching bbox in empty_bboxes_grid (dict format: shelf_num -> list of bboxes)
                for shelf_num, bbox_list in empty_bboxes_grid.items():
                    for j, bbox in enumerate(bbox_list):
                        # bbox is either '' (empty string) or [x1, y1, x2, y2] (list of coordinates)
                        if bbox and isinstance(bbox, list) and len(bbox) == 4:
                            print_log('debug', processing_key, f"bbox : {bbox}", f"hint_coords : {hint_coords}, shrink_hints_coords : {shrink_hints_coords}") 
                            if _bbox_coords_match(bbox, shrink_hints_coords):
                                print_log('debug', processing_key, "Inside bbox match", empty_state)
                                # shelf_num is 1-indexed, convert to 0-indexed for edge_empty_grid
                                shelf_idx = shelf_num - 1
                                edge_empty_grid[shelf_idx][j] = empty_state
        else:
            print_log('debug', processing_key, "No hints_data or empty_bboxes_grid provided", f"hints_data: {hints_data}, empty_bboxes_grid: {empty_bboxes_grid}")  
    except Exception as e:
        logging.error(f"Error building edge_empty_grid: {e}")

    print_log('debug', processing_key, "edge_empty_grid structure", edge_empty_grid)

    _calculate_empty_grid_statistics(processing_key, edge_empty_grid, empty_grid_2d, hints_data)

    return edge_empty_grid

def _bbox_coords_match(bbox, coords, tol=2.0):
    """
    Returns True if coords box is contained within bbox (with tolerance).
    bbox: [x1, y1, x2, y2] (list or tuple) - the container box
    coords: [x1, y1, x2, y2] (list or tuple) - the box to check if inside
    tol: float, allowable tolerance for containment check
    """
    if not bbox or not coords or len(bbox) != 4 or len(coords) != 4:
        return False
    
    # Check if coords box is within bbox (with tolerance)
    return (float(coords[0]) >= float(bbox[0]) - tol and
            float(coords[1]) >= float(bbox[1]) - tol and
            float(coords[2]) <= float(bbox[2]) + tol and
            float(coords[3]) <= float(bbox[3]) + tol)

class SellingProcessor:
    
    def __init__(self, segment_model_interface: SellingModelBase, combined_model_interface: SellingModelBase, digit_model_interface: SellingModelBase, big_table_client: BigTableClient, big_query_client: BigQueryClient, settings: Settings):
        self.big_table_client = big_table_client
        self.big_query_client = big_query_client
        self.segment_model_interface = segment_model_interface
        self.combined_model_interface = combined_model_interface
        self.digit_model_interface = digit_model_interface
        self.settings = settings
    
    def load_image_from_bytes_and_pad(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Load image from raw bytes and apply 300px padding.
        Args:
            image_bytes: Raw image bytes in JPEG format
        Returns:
            Padded image as numpy array with 300px black border, or None if image is broken/corrupt
        """
        try:
            with io.BytesIO(image_bytes) as image_io:
                with Image.open(image_io) as img:
                    image_np = np.array(img)
            padded_image = pad_image(image_np)
            return padded_image
        except Exception as e:
            logging.error(f"Error loading image from bytes: {e}")
            metrics.total_image_failure_count.labels(
                self.settings.experience, self.settings.sub_experience, self.settings.application, self.settings.environment).inc()
            return None

    def convert_image_to_bytes(self, image: np.ndarray, image_format='JPEG') -> Optional[bytes]:
        """Convert numpy array to raw image bytes.
        Args:
            image: Numpy array representing the image
            image_format: Output format (default: 'JPEG')
        Returns:
            Raw image bytes in specified format, or None if conversion fails
        """
        try:
            with io.BytesIO() as output_io:
                pil_image = Image.fromarray(image)
                pil_image.save(output_io, format=image_format)
                image_bytes = output_io.getvalue()
                pil_image.close()
            return image_bytes
        except Exception as e:
            logging.error(f"Error converting image to bytes: {e}")
            return None

    def create_shelf_collages(self, processing_key, sorted_shelves: Dict[int, List[List[float]]], padded_image: np.ndarray,
                          horizontal_padding: int = 200, label_size: Tuple[int, int] = (1280, 1280)) -> List[
    bytes]:
        """Create collages of price labels for each shelf using numpy/cv2.
        
        Args:
            processing_key: Unique key for logging
            sorted_shelves: Dict mapping shelf numbers to lists of label bounding boxes
            padded_image: Padded image as numpy array
            horizontal_padding: Horizontal spacing between labels in pixels (default: 200)
            label_size: Target size for each resized label (default: 1280x1280)
            
        Returns:
            List of shelf collages as JPEG bytes, one per shelf
        """
        shelf_collages = []

        for shelf_number in sorted(sorted_shelves.keys()):
            labels = sorted_shelves[shelf_number]
            print_log('debug', processing_key, f"Processing shelf {shelf_number}",
                    f" labels found {len(labels)} labels")

            if not labels:
                logging.warning(f"[{processing_key}] : Shelf {shelf_number} has no labels, skipping")
                continue

            shelf_image = None

            for bbox in labels:
                x1, y1, x2, y2 = map(int, bbox)
                if x1 < x2 and y1 < y2:
                    cropped_image = padded_image[y1:y2, x1:x2]
                    if cropped_image.size > 0:
                        resized_image = cv2.resize(cropped_image, label_size)
                        if shelf_image is None:
                            shelf_image = resized_image
                        else:
                            shelf_image = np.hstack(
                                [shelf_image, np.zeros((label_size[1], horizontal_padding, 3), dtype=np.uint8),
                                resized_image])

            if shelf_image is not None:
                shelf_collages.append(self.convert_image_to_bytes(shelf_image))

        print_log('debug', processing_key, "No of shelf collages created",
                f"{len(shelf_collages)} collages")
        return shelf_collages

    def default_process_result(self, processing_key):

        print_log('debug', processing_key, "default_process_result","Intiated")

        price_grid = []
        sku_grid = []
        sku_grid_info = []
        empty_grid_2d = []
        empty_grid_info = []
        empty_bboxes = []
        empty_bboxes_grid = []
        beam_bboxes = []
        image_score = 0.0
        image_angle = 0.0
        price_label_coverage = 0.0
        price_label_bboxes = []
        price_grid_conf = []
        edge_empty_grid = []
        beam_count = 0
        masks_status_code = 500
        bbox_status_code = 500
        status_code_array = (masks_status_code, bbox_status_code)
        
        return SellingProcessResult(price_grid, sku_grid, sku_grid_info, empty_grid_2d, empty_grid_info, empty_bboxes, status_code_array, beam_bboxes, image_score, image_angle, price_label_coverage,
            price_label_bboxes, price_grid_conf, empty_bboxes_grid, edge_empty_grid, beam_count)



    def process_single_image(self, processing_key, image_bytes: bytes, filename: str, str_nbr: str, inv_loc_nm: str, hints_data: Dict = None,
                            debug=False):
        """Process a single image through the complete V2 selling pipeline.
        
        Args:
            processing_key: Unique processing identifier for logging
            image_bytes: Raw image bytes in JPEG format (unpadded)
            filename: Image filename for reference
            str_nbr: Store number
            inv_loc_nm: Inventory location name (aisle-bay format)
            debug: Enable debug output and save intermediate results (default: False)
            
        Returns:
            Tuple containing processing results:
            - empty_grid: Dict mapping shelf numbers to empty spot indicators
            - price_grid: 2D list of detected prices
            - sku_grid: 2D list of matched SKUs
            - sku_grid_info: 2D list of SKU metadata tuples
            - empty_grid_2d: 2D empty grid representation
            - empty_grid_info: 2D empty spot metadata
            - empty_bboxes: List of empty spot bounding boxes
            - status_code_array: Tuple of (mask_status, bbox_status)
            - beam_bboxes: List of detected beam bounding boxes
            - image_score: Quality score for the image
            - image_angle: Detected shelf angle in degrees
            - price_label_coverage: Percentage of price labels detected
            - price_label_bboxes: 2D list of price label bounding boxes
        """

        try:
            csv_data = []
            image_score = 0
            price_label_coverage = 0
            image_angle = None
            price_grid_conf = []

            experience = self.settings.experience
            sub_experience = self.settings.sub_experience
            application = self.settings.application
            environment = self.settings.environment

            bt_cache_table_base = self.settings.bt_cache_table_base
            bt_cache_table = self.settings.bt_cache_table
            cache_store_list = self.settings.cache_store_list

            # Use the config flag
            if self.settings.bt_cache:
                original_time = get_time()
                # the configuration module supplies bt_cache_table_base/bt_cache_table and cache_store_list at module-level
                # from configuration import bt_cache_table_base, bt_cache_table, cache_store_list
                # Stores in cache_store_list use the cache table. If cache_store_list is empty,
                # treat that as "all stores" and use the cache_table for every store.
                use_cache_table = not cache_store_list or str_nbr in cache_store_list
                bt_table = bt_cache_table if use_cache_table else bt_cache_table_base
                print_log('debug', processing_key, f"Using Bigtable {bt_table} as cache for store number {str_nbr}",
                        f"inventory location {inv_loc_nm}")
                csv_data = self.big_table_client.read_sequence_data_retry(processing_key, str_nbr, inv_loc_nm, bt_table)
                print_log('debug', processing_key, f"Time taken to read from BT cache: {get_time() - original_time} ms",
                        f"size : {size_ls(csv_data)} MB")
                cache_needs_refresh = sequence_data_needs_refresh(csv_data)
                if cache_needs_refresh:
                    print_log(
                        'debug',
                        processing_key,
                        f"[{str_nbr}#{inv_loc_nm}]",
                        "CACHE MISS OR INCOMPLETE; refreshing from BigQuery base data",
                    )
                    try:
                        fresh_csv_data = self.big_query_client.read_sequence_data_retry(
                            processing_key,
                            str_nbr,
                            inv_loc_nm,
                            bt_cache_table,
                        )
                    except Exception as refresh_error:
                        fresh_csv_data = None
                        logging.warning(
                            f"[{processing_key}] : BigQuery sequence refresh failed; "
                            f"retaining cache response: {refresh_error}"
                        )
                    if fresh_csv_data and (
                        not csv_data or len(fresh_csv_data) >= len(csv_data)
                    ):
                        csv_data = fresh_csv_data

                if csv_data is None or csv_data == []:
                    metrics.total_processing_error_count.labels(experience, sub_experience, application, environment, "BT_NO_CSV").inc()
                    print_log('debug', processing_key, f"[{str_nbr}#{inv_loc_nm}]", " CACHE MISS")
                else:
                    source = "BASE TABLE REFRESH" if cache_needs_refresh else "CACHE HIT"
                    print_log('debug', processing_key, f"[{str_nbr}#{inv_loc_nm}]", source)
            else:
                # fetch_bigquery_data is a local wrapper defined above which forwards to the db interface
                csv_data = self.big_query_client.read_sequence_data_retry(processing_key, str_nbr, inv_loc_nm, bt_cache_table)
                if csv_data is None or csv_data == []:
                    metrics.total_processing_error_count.labels(experience, sub_experience, application, environment, "BQ_NO_CSV").inc()
                    print_log('debug', processing_key, f"[{str_nbr}#{inv_loc_nm}]", " BigQuery RESULT NOT FOUND.")
                else:
                    print_log('debug', processing_key, f"[{str_nbr}#{inv_loc_nm}]", " BigQuery RESULT FETCHED.")

            if self.settings.print_csv:
                print_log('debug', processing_key, f"[{str_nbr}#{inv_loc_nm}]", f"csv_data - {csv_data}")

            # Extracting filename details
            filename = os.path.basename(filename)

            # Loading and padding of the image
            padded_image_bytes = self.convert_image_to_bytes(self.load_image_from_bytes_and_pad(image_bytes))
            print_log('debug', processing_key, "Image has been padded successfully", "no_value")

            if padded_image_bytes is None:
                logging.error(f"[{processing_key}] : Failed to load and pad image, resulting in None bytes")
                metrics.total_processing_error_count.labels(experience, sub_experience, application, environment,
                                                            "IMAGE_LOAD_PAD_FAILURE").inc()
                return self.default_process_result(processing_key)

            # Extracting masks and bounding boxes
            start_time = get_time()
            masks_status_code, masks = extract_and_save_masks(processing_key, padded_image_bytes, self.segment_model_interface)
            print_log('debug', processing_key, f"Time taken to extract masks: {get_time() - start_time} ms", "no_value")
            start_time = get_time()
            bbox_status_code, bbox, detection_confidence = extract_and_annotate_boxes(processing_key, padded_image_bytes, self.combined_model_interface)
            print_log('debug', processing_key, "BBoxes Base", bbox)
            print_log('debug', processing_key, f"Time taken to extract bbox: {get_time() - start_time} ms", "no_value")
            print_log('debug', processing_key, "Masks and bbox extracted successfully", "no_value")
            status_code_array = (masks_status_code, bbox_status_code)
            # Free padded_image_bytes after model inference - no longer needed
            del padded_image_bytes

            beam_bboxes = unpad_bboxes(bbox.get('BEAM', []))
            beam_count = len(beam_bboxes)
            print_log('debug', processing_key, "Beam bboxes after unpadding",
                    f"count={len(beam_bboxes)}, unique={len(set(tuple(b) for b in beam_bboxes))}")
            # Get beam score 
            image_score += get_beam_score(len(beam_bboxes))
            # Assigning labels to shelves
            assigned_and_sorted_labels, label_confidence_map = assign_labels_with_optional_beam_extension(
            processing_key, bbox, masks, str_nbr, confidences=detection_confidence)

            if not assigned_and_sorted_labels:
                logging.error(f"[{processing_key}] : No valid shelves detected in the image")
                metrics.total_processing_error_count.labels(experience, sub_experience, application, environment,
                                                            "NO_SHELVES").inc()
                return SellingProcessResult(price_grid=None, sku_grid=None, sku_grid_info=None, empty_grid_2d=None, empty_grid_info=None, empty_bboxes=None, 
                                            status_code_array=status_code_array, beam_bboxes=beam_bboxes, image_score=image_score, image_angle=image_angle, 
                                            price_label_coverage=price_label_coverage, price_label_bboxes=None, price_grid_conf=price_grid_conf, empty_bboxes_grid=None, 
                                            edge_empty_grid=None, beam_count=beam_count)
            print_log('debug', processing_key, "Shelves detected and labels are sorted now", "no_value")

            # Creating empty grid
            print_log('debug', processing_key, "Before create_price_label_grid - bbox check",
                    f"EMPTY_count={len(bbox.get('EMPTY', []))}, HANGING_EMPTY_count={len(bbox.get('HANGING_EMPTY', []))}, "
                    f"EMPTY_unique={len(set(tuple(b) for b in bbox.get('EMPTY', [])))}, "
                    f"HANGING_EMPTY_unique={len(set(tuple(b) for b in bbox.get('HANGING_EMPTY', [])))}")
            empty_grid, empty_bboxes, price_label_bboxes, empty_bboxes_grid = create_price_label_grid(processing_key, bbox, assigned_and_sorted_labels)
            print_log('debug', processing_key, "After create_price_label_grid - empty_bboxes check",
                    f"empty_bboxes_count={len(empty_bboxes)}, empty_bboxes_unique={len(set(tuple(b) for b in empty_bboxes))}")
            print_log('debug', processing_key, "Empty_grid created", "no_value")

            padded_image = self.load_image_from_bytes_and_pad(image_bytes)
            # Get image angle
            original_time = get_time()
            image_angle = get_image_angles(processing_key, masks.get('SHELF', []), padded_image)
            print_log('debug', processing_key, f"Time taken to get image_angle: {get_time() - original_time} ms, image_angle structure: {image_angle}", "no_value")

            # Creating price label collages for each shelf
            shelf_collages = self.create_shelf_collages(processing_key, assigned_and_sorted_labels, padded_image)

            
            # Free padded_image after collage creation - no longer needed
            del padded_image
            
            print_log('debug', processing_key, "Price label collages for each shelf created", "no_value")

            # Saving individual shelf collages
            for i, collage in enumerate(shelf_collages):
                collage_filename = f"{os.path.splitext(filename)[0]}_shelf_{i + 1}_collage.jpg"

                if debug:
                    collage_path = os.path.join(os.path.dirname(filename), collage_filename)
                    with open(collage_path, 'wb') as f:
                        f.write(collage)
                    print_log('debug', processing_key, f"Shelf {i + 1} collage saved as", collage_filename)

            # Processing the shelf collages with YOLOv8 for digit detection
            if shelf_collages:
                try:
                    price_grid, price_grid_conf = process_shelf_collages(processing_key, shelf_collages, self.digit_model_interface, label_confidence_map)
                    # Validate price_grid_conf structure for price_grid_conf feature
                    structure_valid = len(price_grid) == len(price_grid_conf) and all(
                        len(price_grid[i]) == len(price_grid_conf[i]) for i in range(len(price_grid))
                    )
                    total_conf_values = sum(len(row) for row in price_grid_conf)
                    avg_conf = sum(sum(row) for row in price_grid_conf) / total_conf_values if total_conf_values > 0 else 0
                    print_log('debug', processing_key, "process_shelf_collages completed successfully",
                                f"shelves_in_price_grid={len(price_grid)}, shelves_in_conf_grid={len(price_grid_conf)}, total_conf_values={total_conf_values}, avg_conf={avg_conf:.3f}, structure_valid={structure_valid}")
                except Exception as e:
                    logging.error(f"Exception occurred while processing shelf collages {str(e)}")
                    metrics.total_processing_error_count.labels(
                        experience, sub_experience, application, environment,
                        "PROCESS_SHELF_COLLAGES_EXCEPTION"
                    ).inc()
                    price_grid = []
                    return SellingProcessResult(price_grid=None, sku_grid=None, sku_grid_info=None, empty_grid_2d=None, empty_grid_info=None, empty_bboxes=None, 
                                            status_code_array=status_code_array, beam_bboxes=beam_bboxes, image_score=image_score, image_angle=image_angle, 
                                            price_label_coverage=price_label_coverage, price_label_bboxes=price_label_bboxes, price_grid_conf=price_grid_conf, 
                                            empty_bboxes_grid=None, edge_empty_grid=None, beam_count=beam_count)
            else:
                price_grid = []
                logging.error("Unable to process price grid due to missing collages")
                metrics.total_processing_error_count.labels(experience, sub_experience, application, environment,
                                                            "MISSING_COLLAGES").inc()
                return SellingProcessResult(price_grid=None, sku_grid=None, sku_grid_info=None, empty_grid_2d=None, empty_grid_info=None, empty_bboxes=None, 
                                        status_code_array=status_code_array, beam_bboxes=beam_bboxes, image_score=image_score, image_angle=image_angle, 
                                        price_label_coverage=price_label_coverage, price_label_bboxes=price_label_bboxes, price_grid_conf=price_grid_conf, 
                                        empty_bboxes_grid=None, edge_empty_grid=None, beam_count=beam_count)
 

            # Free shelf_collages after processing - no longer needed
            del shelf_collages

            print_log('debug', processing_key, "Processed the shelf collages with YOLOv8 for digit detection", "no_value")

            #print("Ran the query successfully and have the data")
            # Preparing data for matching
            dollar_amount_to_locationIDs, locationID_to_sku = prepare_data_for_matching(processing_key, csv_data)
            
            #print("Preaparing data for matching")
            # Matching SKUs to grid
            sku_grid, sku_grid_info = match_skus_to_grid(processing_key, price_grid, dollar_amount_to_locationIDs, locationID_to_sku, csv_data)
            # Get price_label_coverage:
            price_label_coverage = sum(len(row) for row in price_grid) / len(csv_data) if len(csv_data) != 0 else 0
            print_log('debug', processing_key, "price_label_coverage", (sum(len(row) for row in price_grid), len(csv_data)))
            # Get image_score:
            image_score += get_label_score(sku_grid_info = sku_grid_info, total_sku_count = len(csv_data))
            #print("Matching of SKUs to grid completed")
            print_log('debug', processing_key, "Empty_grid_bboxes structure", empty_bboxes_grid)
            print_log('debug', processing_key, "Empty_grid structure", empty_grid)
            print_log('debug', processing_key, "SKU_grid_info structure", sku_grid_info)
            print_log('debug', processing_key, "image_score structure", image_score)
            
            empty_grid_2d, empty_grid_info = create_empty_grids(processing_key, empty_grid, sku_grid_info)
            
            edge_empty_grid = build_edge_empty_grid(processing_key,empty_grid_2d,empty_bboxes_grid, hints_data)

            print_log('debug', processing_key, "empty_grid_2d structure", empty_grid_2d)
            print_log('debug', processing_key, "empty_grid_info structure", empty_grid_info)
            
            if len(empty_grid) == 0:
                metrics.total_processing_error_count.labels(experience, sub_experience, application, environment,
                                                            "EMPTY_GRID").inc()
            else:
                metrics.total_images_empty_count.labels(experience, sub_experience, application, environment).inc()
                for each_grid in empty_grid:
                    for each_value in empty_grid[each_grid]:
                        if each_value == "empty":
                            metrics.total_empty_count.labels(experience, sub_experience, application, environment).inc()

        
            structure_matches = len(price_grid) == len(price_grid_conf)
            all_rows_match = all(len(price_grid[i]) == len(price_grid_conf[i]) for i in range(len(price_grid))) if structure_matches else False
            total_conf_count = sum(len(row) for row in price_grid_conf)
            print_log('debug', processing_key, "Final price_grid_conf validation",
                    f"structure_matches={structure_matches}, all_rows_match={all_rows_match}, total_conf_values={total_conf_count}, price_grid_rows={len(price_grid)}, conf_grid_rows={len(price_grid_conf)}")
            if len(sku_grid_info) == 0:
                metrics.total_processing_error_count.labels(experience, sub_experience, application, environment,
                                                            "SKU_GRID").inc()
            # Calculating and saving statistics for myself
            if debug:
                save_results(filename, empty_grid, price_grid, sku_grid, sku_grid_info, empty_grid_2d, empty_grid_info, price_label_bboxes)
                print_log('debug', processing_key, "Stats have been calculated and saved", "no_value")
            
            print_log('debug', processing_key, "Final empty_bboxes before return",
                    f"count={len(empty_bboxes)}, unique={len(set(tuple(b) for b in empty_bboxes))}, type={type(empty_bboxes).__name__}")
            return SellingProcessResult(price_grid, sku_grid, sku_grid_info, empty_grid_2d, empty_grid_info, empty_bboxes, status_code_array, beam_bboxes, image_score, image_angle, price_label_coverage,
                price_label_bboxes, price_grid_conf, empty_bboxes_grid, edge_empty_grid, beam_count)
        except Exception as e:
                logging.error(f"Exception in process_single_image: {e}")
                print_log('debug', processing_key, "Error - Exception in process_single_image:", e)                
                return self.default_process_result(processing_key)