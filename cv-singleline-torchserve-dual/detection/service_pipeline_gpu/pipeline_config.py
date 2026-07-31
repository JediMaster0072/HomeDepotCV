"""Detection-only pipeline defaults."""

CONFIG = {
    "detection_conf_thresh": 0.25,
    "detection_nms_thresh": 0.5,
    "detection_input_size": 1280,
    "device": "gpu",
    "debug": False,
    "save_debug_artifacts": False,
    "save_result_json": False,
    "save_annotated_image": False,
}

DETECTION_CLASSES = {
    0: "Label",
    1: "First_Line",
    2: "Second_Line",
    3: "Quantity",
    4: "Beam",
}

DETECTION_CLASS_LABEL = 0
DETECTION_CLASS_FIRST_LINE = 1
DETECTION_CLASS_SECOND_LINE = 2
DETECTION_CLASS_QUANTITY = 3
DETECTION_CLASS_BEAM = 4

DETECTION_SUB_ELEMENT_IDS = {
    DETECTION_CLASS_FIRST_LINE,
    DETECTION_CLASS_SECOND_LINE,
    DETECTION_CLASS_QUANTITY,
}
