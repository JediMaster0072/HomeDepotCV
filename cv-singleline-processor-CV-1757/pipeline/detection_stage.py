import cv2
import numpy as np


# Pipeline order: 17
# Description: Extracts a rectangular crop from an image using a buffered detection box.
def crop(img: np.ndarray, box):
    return img[box.y1 : box.y2, box.x1 : box.x2]


# Pipeline order: 18
# Description: Applies crop-level cleanup before segmentation and OCR.
def preprocess(image: np.ndarray) -> np.ndarray:
    image = deblur(image)
    image = denoise(image)
    # image = correct_tilt(image)
    return image


# Pipeline order: 18.1
# Description: Sharpens a crop by subtracting a blurred version from the original.
def deblur(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


# Pipeline order: 18.2
# Description: Removes image noise from grayscale or color crops using OpenCV denoising.
def denoise(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, h=10)
    return cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)
