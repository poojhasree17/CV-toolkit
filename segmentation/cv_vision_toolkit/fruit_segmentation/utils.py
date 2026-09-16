"""
CV Vision Toolkit - Fruit Segmentation Utilities
================================================
Utility functions for fruit segmentation, including HSV color masking,
Otsu thresholding, watershed algorithm, morphological cleanup, contour
labeling, and comparison-image composition.

Author: CV Vision Toolkit
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

# ----------------------------------------------------------------------
# Default HSV lower/upper bounds for common fruit colours.
# These are used by the HSV colour-based segmentation method.
# NOTE: Hue range in OpenCV is 0-179.
# ----------------------------------------------------------------------
HSV_RANGES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "apple_red":  (np.array([0, 50, 50]),    np.array([10, 255, 255])),
    "apple_red2": (np.array([160, 50, 50]),  np.array([180, 255, 255])),
    "orange":     (np.array([10, 80, 100]),  np.array([25, 255, 255])),
    "banana":     (np.array([20, 80, 100]),  np.array([35, 255, 255])),
    "green":      (np.array([35, 50, 50]),   np.array([85, 255, 255])),
}


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to HSV colour space.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    np.ndarray
        HSV image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def segment_by_hsv(
    hsv: np.ndarray,
    ranges: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> np.ndarray:
    """
    Create a binary mask using HSV colour ranges.

    All ranges are OR-ed together so multiple fruit colours can be
    captured in a single mask.

    Parameters
    ----------
    hsv : np.ndarray
        HSV image.
    ranges : dict, optional
        Mapping of name -> (lower_bound, upper_bound). Defaults to the
        module-level HSV_RANGES dict.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255) where fruit pixels are white.
    """
    if ranges is None:
        ranges = HSV_RANGES

    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for name, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_or(combined, mask)
    return combined


def segment_by_otsu(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask using Otsu's thresholding on blurred grayscale.

    Otsu automatically finds the threshold that best separates
    foreground from background based on pixel-intensity variance.

    Both polarities are produced (objects darker than the background and
    objects brighter than the background). The polarity that marks fewer
    pixels as white is returned, based on the common assumption that the
    object(s) of interest occupy the minority of the scene.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255) where the segmented object pixels
        are white.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    # Objects are assumed to cover the smaller part of the scene, so pick
    # the polarity that produces the smaller white region.
    if np.count_nonzero(binary) <= np.count_nonzero(inverted):
        return binary
    return inverted


def segment_by_watershed(image: np.ndarray) -> np.ndarray:
    """
    Segment an image using the marker-based Watershed algorithm.

    Step 1: threshold the grayscale image and use a distance transform
    to build "sure foreground" markers.
    Step 2: run cv2.watershed on the colour image to separate touching
    objects.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.

    Returns
    -------
    np.ndarray
        Binary mask (uint8). Each object is white; background is black.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up threshold with morphology
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Distance transform to find "sure foreground"
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown = thresh minus sure_fg
    unknown = cv2.subtract(thresh, sure_fg)

    # Label connected components -> markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # 1 reserved for background
    markers[unknown == 255] = 0    # unknown regions become 0

    # Run watershed
    markers = cv2.watershed(image, markers)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[markers > 1] = 255
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean a binary mask.

    A single opening (erosion then dilation) removes isolated noise, and a
    single closing (dilation then erosion) fills small holes. Both use a
    5x5 kernel. The iterations are kept low so that nearby-but-distinct
    regions (e.g. objects touching only at watershed boundary lines) are
    not accidentally fused together.

    Parameters
    ----------
    mask : np.ndarray
        Binary input mask (0 or 255).

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing


def find_fruit_contours(
    mask: np.ndarray,
    min_area: int = 500,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Find and filter contours from a binary segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255).
    min_area : int
        Minimum contour area (pixels) to keep.

    Returns
    -------
    Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]
        (contours, bounding_boxes) where bounding boxes are (x, y, w, h).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered.append(contour)
            boxes.append(cv2.boundingRect(contour))
    return filtered, boxes


def draw_fruit_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
) -> np.ndarray:
    """
    Draw contours around each segmented fruit and label each with an ID.

    Parameters
    ----------
    image : np.ndarray
        BGR image to annotate (a copy is made internally).
    contours : List[np.ndarray]
        Contours returned by find_fruit_contours.

    Returns
    -------
    np.ndarray
        Annotated image with green contours and numbered labels.
    """
    output = image.copy()
    for idx, contour in enumerate(contours, start=1):
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

        label = str(idx)
        font_scale, thickness = 0.9, 2
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(output, (cx - tw // 2 - 3, cy - th - 3),
                      (cx + tw // 2 + 3, cy + 3), (0, 0, 255), -1)
        cv2.putText(output, label, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return output


def make_comparison_image(
    original: np.ndarray,
    mask: np.ndarray,
    segmented: np.ndarray,
    title: str,
) -> np.ndarray:
    """
    Composite original, binary mask, and segmented result side-by-side
    using Matplotlib and return it as a numpy array.

    Parameters
    ----------
    original : np.ndarray
        BGR original image.
    mask : np.ndarray
        Binary segmentation mask.
    segmented : np.ndarray
        BGR image with contours drawn.
    title : str
        Figure title.

    Returns
    -------
    np.ndarray
        The composed RGB image containing the three panels.
    """
    # Imported here so cv2-only usage stays fast and headless-safe.
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless runs
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Binary Mask")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Segmented Result")
    axes[2].axis("off")

    plt.tight_layout()

    # Render the matplotlib figure to a numpy array
    fig.canvas.draw()
    # Use buffer_rgba() which is available in matplotlib >= 3.5
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf, dtype=np.uint8).copy()
    # Convert RGBA to RGB (drop alpha channel)
    data = data[:, :, :3]
    plt.close(fig)
    return data