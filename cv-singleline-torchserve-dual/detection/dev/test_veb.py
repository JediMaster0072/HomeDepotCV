import requests
import base64
import json
import io
import cv2
from PIL import Image
import numpy as np


IMAGE_PATH = "test_img_new.jpg"
URL = "http://localhost:9000/predictions/yolov7"


def numpy_image_to_base64_png(image: np.ndarray) -> str:
    """
    Encode a NumPy RGB/grayscale image as PNG base64 string.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")

    if image.dtype == np.bool_:
        image = image.astype(np.uint8)

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or bool image, got {image.dtype}")

    if image.ndim not in (2, 3):
        raise ValueError(f"Expected HxW or HxWxC image, got shape {image.shape}")

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {image.shape}")

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    image = np.ascontiguousarray(image)

    pil_image = Image.fromarray(image)

    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

    return base64.b64encode(png_bytes).decode("ascii")


# cv2 loads as BGR
image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# Convert BGR → RGB because your TorchServe code does PIL Image.open(...).convert("RGB")
image_rgb = image_bgr[:, :, ::-1]

print(
    "client_image_rgb:",
    "shape=",
    image_rgb.shape,
    "dtype=",
    image_rgb.dtype,
    "sum=",
    image_rgb.sum(),
    "min=",
    image_rgb.min(),
    "max=",
    image_rgb.max(),
)

b64_image = numpy_image_to_base64_png(image_rgb)

payload = {"instances": [{"file": b64_image}]}

response = requests.post(URL, json=payload)

print("Status:", response.status_code)

try:
    print("Response:", json.dumps(response.json(), indent=2))
except Exception:
    print("Raw response:", response.text)
