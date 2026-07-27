import cv2
import numpy as np


def crop(img: np.ndarray, box):
    """Extract a rectangular crop using a buffered detection box."""
    return img[box.y1 : box.y2, box.x1 : box.x2]


def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply the same local crop cleanup used before segmentation/OCR."""
    image = deblur(image)
    image = denoise(image)
    return image


def deblur(image: np.ndarray) -> np.ndarray:
    """Sharpen a crop by subtracting a blurred version of itself."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def denoise(image: np.ndarray) -> np.ndarray:
    """Denoise grayscale or color crops with OpenCV's non-local means filters."""
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, h=10)
    return cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)


def enhance_to_gray(image: np.ndarray) -> np.ndarray:
    """Return the local grayscale enhancement used for debugging/experimentation."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
