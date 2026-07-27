import io
import base64

import cv2
import numpy as np
from PIL import Image
from google.cloud import storage
# import services.metrics as metrics

from common_config import project_id, gce_name


# Pipeline order: 07.0
# Description: Creates or reuses the Google Cloud Storage client used to download image bytes.
def storage_client():
    client = getattr(storage_client, "storage_client", storage.Client(project=project_id))
    storage_client.storage_client = client
    return client


# def convert_image_bytes(image, format="JPEG"):
#     """
#     Converts a PIL.Image or np.ndarray to image bytes (default: JPEG).
#     """

#     # If input is a NumPy array, convert to PIL Image
#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image)

#     with io.BytesIO() as output:
#         image.save(output, format=format)
#         image_bytes = output.getvalue()

#     return image_bytes


# def base64_to_pil(image_base64: str) -> Image.Image:
#     """
#     Decodes a base64 string back into a PIL Image.
#     """
#     image_bytes = base64.b64decode(image_base64)
#     buffer = io.BytesIO(image_bytes)
#     image = Image.open(buffer)
#     return image


# def base64_to_numpy(image_base64: str) -> np.ndarray:
#     """
#     Decodes a base64 string back into a NumPy array.
#     """
#     pil_image = base64_to_pil(image_base64)
#     return np.array(pil_image)


# Pipeline order: 14.1 and 24.1
# Description: Encodes a NumPy image or mask as lossless base64 PNG for Vertex requests or Google OCR image bytes.
def numpy_image_to_base64_png(image: np.ndarray, return_bytes: bool = False) -> str | tuple[str, bytes]:
    """
    Encode a NumPy image/mask as lossless PNG.

    Returns:
    - return_bytes=False: base64-encoded PNG string
    - return_bytes=True: tuple of (base64-encoded PNG string, raw PNG bytes)

    Supports:
    - Grayscale: H x W
    - RGB/RGBA: H x W x 3 / H x W x 4
    - uint8 images/masks
    - bool masks, converted to uint8 0/1

    Notes:
    - This preserves pixel values exactly for uint8 images/masks.
    - It avoids OpenCV BGR/RGB confusion.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")

    image = np.asarray(image)

    if image.dtype == np.bool_:
        image = image.astype(np.uint8)

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 or bool image for PNG encoding, got {image.dtype}. "
            "If you have float data, normalize/convert it explicitly before encoding."
        )

    if image.ndim not in (2, 3):
        raise ValueError(f"Expected image shape HxW or HxWxC, got shape {image.shape}")

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {image.shape}")

    # PIL does not need a channel dimension for grayscale
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    image = np.ascontiguousarray(image)

    pil_image = Image.fromarray(image)

    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

    encoded = base64.b64encode(png_bytes).decode("ascii")

    if return_bytes:
        return encoded, png_bytes

    return encoded


# Pipeline order: 25.1
# Description: Decodes a base64 PNG endpoint response into a NumPy image or mask.
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


# Pipeline order: 07.2
# Description: Converts a downloaded PIL image into the NumPy array format used by the pipeline.
def convert_pil_to_numpy(image):
    img_np = np.asarray(image)
    return img_np


# @metrics.download_image_time.labels(experience, sub_experience, application, environment).time()
# Pipeline order: 07
# Description: Downloads the input image from GCS and returns it as an RGB NumPy array.
def download_image(bucket_name: str, file_name: str):
    try:
        bucket = storage_client().bucket(bucket_name)
        blob = bucket.get_blob(file_name)

        if blob is None:
            print(f"[{file_name}] Download Image Error: blob is None!")
            return None

        # blob_content = blob.download_as_bytes()
        blob_content = bucket.blob(file_name).download_as_string()

        image = Image.open(io.BytesIO(blob_content)).convert("RGB")
        print(f"[{file_name}, {gce_name}] Download Image Completed")

        return convert_pil_to_numpy(image)

    except Exception as e:
        print(f"[{file_name}] download image failed. Error: {e}")
        # metrics.total_error_count.labels(experience, sub_experience, application, environment, "ImageDownload").inc()
    return None


# Pipeline order: Optional utility
# Description: Uploads generated image arrays to a GCS bucket for debugging or downstream use.
async def write_image_to_bucket(images, filename, bucket_id="np-store-sim-multiline-stitched-images"):
    storage_client_instance = storage.Client(project=project_id)
    bucket = storage_client_instance.bucket(bucket_id)

    for index, image in enumerate(images):
        image_en = cv2.imencode(".jpg", image)[1].tobytes()
        blob = bucket.blob(filename + str(index) + ".jpg")

        print("Uploading file: {} to bucket: {}".format(filename, bucket_id))

        blob.upload_from_string(image_en, content_type="image/jpg")
