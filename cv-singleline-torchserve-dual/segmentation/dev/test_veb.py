import requests
import base64
import json
import io
import cv2
from PIL import Image
import numpy as np


IMAGE_PATH = "/data/saranjit/Project_files/yolo7-deployment/yolo7-seg/strip_0.jpg"
URL = "http://localhost:10000/predictions/yolov7"


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


def base64_png_to_numpy_image(image_base64: str) -> np.ndarray:
    """
    Decode a base64 PNG string back into a NumPy array.

    Returns:
    - Grayscale PNG -> H x W uint8
    - RGB PNG -> H x W x 3 uint8
    - RGBA PNG -> H x W x 4 uint8
    """

    if not isinstance(image_base64, str):
        raise TypeError(f"Expected base64 string, got {type(image_base64)}")

    image_bytes = base64.b64decode(image_base64.encode("ascii"))

    with io.BytesIO(image_bytes) as buffer:
        pil_image = Image.open(buffer)
        pil_image.load()  # Force loading before buffer closes

    return np.array(pil_image)


def process_binary_mask_with_rotation(binary_mask: np.ndarray, masks_np=None):
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        # print(box)
        box = box.astype(np.int32)
        binary_mask = cv2.drawContours(binary_mask, [box], 0, 255, cv2.FILLED)
        # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        # # Fill actual contour region
        # binary_mask = cv2.drawContours(binary_mask, [contour], -1, (0, 0, 255), thickness=-1)

    # return {"img": base64.b64encode(binary_mask.tobytes()).decode("utf-8"), "height": masks_np.shape[1], "width": masks_np.shape[2]}
    return binary_mask


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

# payload = {"instances": [{"file": b64_image}]}

# response = requests.post(URL, json=payload)

# print("Status:", response.status_code)

# try:
#     print("Response:", json.dumps(response.json(), indent=2))
# except Exception:
#     print("Raw response:", response.text)

encoded_image = "iVBORw0KGgoAAAANSUhEUgAACVsAAAFDCAAAAAAL4hOCAAAFZ0lEQVR4nO3dW3LbMBAEQCrl+19Z+YnzkClFkgfcBdh9AhaWYU3NIta2AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMB5XaofAAAoc61+gAWjyI/qBwAAWIhsBQCQI1sBAOQst+QEAJ7S4K7VHXOnE70VAECObAUAkDN36wYAvKvvTvDGZGFFbwUAkCNbAQDkTFazAQAh0+wEv2odX/RWAAA5shUAQI5sBQCQ03phCQAMM/F9q8eKw43eCgAgR7YCAMixEwSA81l2H/ipMODorQAAcmQrAIAcO0EAOJ/ld4LbVhZy9FYAADmyFQBAjmwFAJDzUf0AAMCxTnHZqo7eCgAgR7YCAMiRrQAAcmQrAIAc2QoAIEe2AgDIka0AAHJkKwCAHNkKACBHtgIAyJGtAAByZCsAgJxL9QMAACWW/8nmopCjtwIAyJGtAAByZCsAgBz3rQCAfRPfyCoMOHorAIAc2QoAIMdOEAAYo3CnaCcIALAE2QoAIMdOEAA42tBtYXG40VsBAOTIVgAAObIVAECO+1YAQL3UDaz6ZKO3AgDIka0AAHLqm7M/ev0iZKeT4RWB98jwm8l+G4wXFvLr89Dr37XeCgAgR7YCAMjp0qL12gfe6HJIPBR+h0y9ymEfAyMGxtBbAQDkyFYAADmyFQBATpcbB63vW33qclj8a8zLY9pHq/gImDIwgN4KACBHtgIAyPmofoCZ7OwsrBSqTLFF5j4DBJaltwIAyJGtAAByuuy0ll4QdDnkiTR5H0wupMk8vzLhlB4jNk+a0FsBAOTIVgAAObIVAEBOl/V0j2X9QF0OegL93gXD+45+8/yL0Sb0G7G5UkxvBQCQI1sBAOT4u+wAvKnfPhAa0FsBAOTIVgAAObIVAECO+1YALOW6+TsMlNJbAQDkyFYAADl2gvCIxQIAr9FbAQDkyFYAADl2gsA52fcCY+itAAByZCsAgBzZCgAgp8t9q8vmB9Vpx4WcJRlrkE837NBbAQDkyFYAADlddoLQiKXResz0TEybYnorAIAc2QoAIKfTTvCQGtf/aFmaVcDyjLifm5kUf2S9ITSgtwIAyJGtAAByZCsAgByr6d/GXhJw0E/7/yAc5mzMlKD7r5PXiCb0VgAAObIVAECOCnXPgPWgg37J/gQc4qzMEzgTvRUAQI5sBQCQo5V/wndXhA75LVcnB8CE9FYAADmyFQBAjmwFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwJn9BPZ4Md177RXTAAAAAElFTkSuQmCC"

np_image = base64_png_to_numpy_image(encoded_image)


processed_image = process_binary_mask_with_rotation(np_image)
print(np_image.dtype, np_image.shape)
print(processed_image.dtype, processed_image.shape)

cv2.imwrite("output_image.jpg", np_image)
cv2.imwrite("output_image_additional.jpg", processed_image)
