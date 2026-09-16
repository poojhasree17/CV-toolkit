"""
CV Vision Toolkit - Coin Detection Utilities
=============================================
Utility functions for coin detection, including image preprocessing,
Hough Circle Transform, contour-based detection, and visualization.

Author: CV Vision Toolkit
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


def preprocess_image(
    image: np.ndarray,
    blur_kernel: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an image for coin detection.

    Converts the image to grayscale and applies Gaussian blur to reduce
    noise and improve circle detection accuracy.

    Parameters
    ----------
    image : np.ndarray
        The input BGR image loaded via OpenCV.
    blur_kernel : int, optional
        The size of the Gaussian blur kernel. Must be odd. Default is 9.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of (grayscale_image, blurred_image).
    """
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 2)
    return gray, blurred


def detect_circles_hough(
    blurred: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 30,
    param1: int = 100,
    param2: int = 40,
    min_radius: int = 15,
    max_radius: int = 200,
) -> np.ndarray:
    """
    Detect circles using the Hough Circle Transform.

    Parameters
    ----------
    blurred : np.ndarray
        The preprocessed (blurred grayscale) image.
    dp : float
        Inverse ratio of accumulator resolution to image resolution.
    min_dist : int
        Minimum distance between detected circle centers.
    param1 : int
        Upper threshold for the internal Canny edge detector.
    param2 : int
        Accumulator threshold (lower = more circles detected).
    min_radius : int
        Minimum circle radius to detect.
    max_radius : int
        Maximum circle radius to detect.

    Returns
    -------
    np.ndarray
        Array of detected circles [x, y, radius]. Empty if none found.
    """
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    else:
        circles = np.array([])
    return circles


def detect_circles_contour(
    blurred: np.ndarray,
    min_area: int = 500,
    max_area: int = 150000,
    circularity_threshold: float = 0.6,
) -> List[Dict]:
    """
    Detect coin-like shapes using contour detection with a circularity check.

    Fallback method when Hough Circles doesn't perform well. Finds contours
    and filters by area and circularity (4*pi*area / perimeter^2).

    Parameters
    ----------
    blurred : np.ndarray
        The preprocessed (blurred grayscale) image.
    min_area : int
        Minimum contour area to consider (in pixels).
    max_area : int
        Maximum contour area to consider (in pixels).
    circularity_threshold : float
        Minimum circularity score (0-1). A perfect circle has 1.0.

    Returns
    -------
    List[Dict]
        List of dicts with keys: 'center' (x,y), 'radius', 'contour'.
    """
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity >= circularity_threshold:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            radius = int(np.sqrt(area / np.pi))
            detected.append({
                "center": (cx, cy),
                "radius": radius,
                "contour": contour,
            })
    return detected


def classify_coins_by_size(
    circles: np.ndarray,
    small_threshold: int = 25,
    large_threshold: int = 50,
) -> Dict[str, List]:
    """
    Classify detected coins into size categories based on radius.

    Parameters
    ----------
    circles : np.ndarray
        Array of detected circles [x, y, radius].
    small_threshold : int
        Maximum radius to be classified as 'small'.
    large_threshold : int
        Minimum radius to be classified as 'large'.

    Returns
    -------
    Dict[str, List]
        Dictionary with keys 'small', 'medium', 'large', each mapping
        to a list of [x, y, radius] entries.
    """
    classification: Dict[str, List] = {"small": [], "medium": [], "large": []}
    for circle in circles:
        x, y, r = circle
        if r < small_threshold:
            classification["small"].append(circle)
        elif r >= large_threshold:
            classification["large"].append(circle)
        else:
            classification["medium"].append(circle)
    return classification


def draw_detections(
    image: np.ndarray,
    circles: np.ndarray,
    contour_detections: Optional[List[Dict]] = None,
    method: str = "hough",
) -> np.ndarray:
    """
    Draw detected coins on the image with bounding circles and numbered labels.

    Parameters
    ----------
    image : np.ndarray
        The original BGR image to annotate (a copy is made internally).
    circles : np.ndarray
        Detected circles [x, y, radius] for the Hough method.
    contour_detections : List[Dict], optional
        Contour-based detections for the contour fallback method.
    method : str
        'hough' or 'contour'.

    Returns
    -------
    np.ndarray
        Annotated image with numbered circles drawn.
    """
    output = image.copy()

    if method == "hough" and circles is not None and len(circles) > 0:
        for idx, (x, y, r) in enumerate(circles, start=1):
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)
            label = f"#{idx}"
            font_scale, thickness = 0.6, 2
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            lx = x - tw // 2
            ly = y - r - 10
            if ly < th + 5:
                ly = y + r + 15
            cv2.putText(output, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 0, 0), thickness)

    elif method == "contour" and contour_detections:
        for idx, det in enumerate(contour_detections, start=1):
            cx, cy = det["center"]
            r = det["radius"]
            cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
            cv2.drawContours(output, [det["contour"]], -1, (255, 255, 0), 1)
            label = f"#{idx}"
            font_scale, thickness = 0.6, 2
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            lx = cx - tw // 2
            ly = cy - r - 10
            if ly < th + 5:
                ly = cy + r + 15
            cv2.putText(output, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 0, 0), thickness)

    return output


def overlay_count(image: np.ndarray, count: int) -> np.ndarray:
    """
    Overlay the total coin count on the image.

    Parameters
    ----------
    image : np.ndarray
        The annotated image.
    count : int
        Number of coins detected.

    Returns
    -------
    np.ndarray
        Image with the count text overlay in the top-left corner.
    """
    output = image.copy()
    text = f"Total Coins: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 1.0, 3
    color = (0, 0, 255)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(output, (5, 5), (tw + 15, th + 15 + baseline), (255, 255, 255), -1)
    cv2.putText(output, text, (10, th + 10), font, font_scale, color, thickness)
    return output