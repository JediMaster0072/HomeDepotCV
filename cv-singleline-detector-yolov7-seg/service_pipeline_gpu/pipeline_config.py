"""
SKU Reading Pipeline — Configuration
=====================================
Single source of truth for all pipeline parameters.
No hardcoded values in stage functions — everything flows from here.
"""

CONFIG = {
    # ─── Detection Model — Stage 1 ───────────────────────────────────────
    "detection_conf_thresh": 0.25,
    "detection_nms_thresh": 0.5,
    "detection_input_size": 1280,

    # ─── Segmentation Model — Stage 4 ────────────────────────────────────
    "seg_conf_thresh": 0.45,
    "seg_nms_thresh": 0.55,
    "seg_input_size": 768,
    "seg_class_mapping": {
        0: "label",
        1: "first_line",
        2: "second_line",
        3: "quantity",
    },



    # ─── Pipeline ─────────────────────────────────────────────────────────
    "device": "gpu",
    "debug": False,
    "save_debug_artifacts": False,
    "save_result_json": False,
    "save_annotated_image": False,
    "log_label_reads": True,        # final per-label output lines for validation/integration
    "log_label_reads_level": "INFO",
    "log_assignment_diagnostics": True,
    "log_assignment_diagnostics_level": "INFO",
}


# ─── Detection Model Class Definitions ────────────────────────────────────
# These are fixed by training and should not change.

DETECTION_CLASSES = {
    0: "Label",
    1: "First_Line",
    2: "Second_Line",
    3: "Quantity",
    4: "Beam",
}

DETECTION_CLASS_LABEL       = 0
DETECTION_CLASS_FIRST_LINE  = 1
DETECTION_CLASS_SECOND_LINE = 2
DETECTION_CLASS_QUANTITY    = 3
DETECTION_CLASS_BEAM        = 4

DETECTION_SUB_ELEMENT_IDS = {
    DETECTION_CLASS_FIRST_LINE,
    DETECTION_CLASS_SECOND_LINE,
    DETECTION_CLASS_QUANTITY,
}
