"""Segmentation-only pipeline defaults."""

CONFIG = {
    "seg_conf_thresh": 0.45,
    "seg_nms_thresh": 0.55,
    "seg_input_size": 768,
    "seg_class_mapping": {
        0: "label",
        1: "first_line",
        2: "second_line",
        3: "quantity",
    },
    "device": "gpu",
    "debug": False,
    "save_debug_artifacts": False,
    "save_result_json": False,
    "save_annotated_image": False,
}
