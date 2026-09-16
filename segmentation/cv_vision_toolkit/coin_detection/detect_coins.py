"""
CV Vision Toolkit - Coin Detection & Counting
=============================================
Detects and counts coins in an image using either the Hough Circle
Transform (default) or a contour-based circularity fallback.

Usage:
    python coin_detection/detect_coins.py [--image path/to/image.jpg]
        [--method hough|contour] [--blur 9] [--min-radius 15]
        [--max-radius 120] [--min-dist 30] [--param1 100] [--param2 30]
        [--min-area 500] [--circularity 0.6] [--save path]

Author: CV Vision Toolkit
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np

# Allow running from project root or from inside the module folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Local utility imports
from coin_detection.utils import (
    preprocess_image,
    detect_circles_hough,
    detect_circles_contour,
    classify_coins_by_size,
    draw_detections,
    overlay_count,
)

# ---------------- Tunable Parameters (also settable via CLI args) --------
BLUR_KERNEL = 9          # Gaussian blur kernel size (odd)
HOUGH_DP = 1.2           # Accumulator resolution divisor
HOUGH_MIN_DIST = 30      # Min distance between circle centers
HOUGH_PARAM1 = 100       # Canny high threshold used internally
HOUGH_PARAM2 = 35        # Accumulator threshold (lower = more circles)
HOUGH_MIN_RADIUS = 15    # Smallest radius to look for
HOUGH_MAX_RADIUS = 120   # Largest radius to look for

# Contour fallback parameters
CONTOUR_MIN_AREA = 800
CONTOUR_MAX_AREA = 120000
MIN_CIRCULARITY = 0.75

# Size classification thresholds (in pixels of radius)
SMALL_RADIUS = 25        # radius < this  -> small
LARGE_RADIUS = 50        # radius >= this -> large

# File paths
DEFAULT_IMAGE = os.path.join(BASE_DIR, "sample_images", "coins", "coins_sample.jpg")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output", "coins_result.jpg")


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk with error handling.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        The loaded image in BGR format.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file exists but cannot be read by OpenCV.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}. "
            "Run 'python generate_test_images.py' to create sample images."
        )
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Could not read image (not a valid image file): {image_path}"
        )
    return image


def run_coin_detection(args: argparse.Namespace) -> int:
    """
    Main pipeline for coin detection and counting.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        The number of coins detected.
    """
    # 1. Load image
    image = load_image(args.image)
    logging.info("Loaded image: %s (shape=%s)", args.image, image.shape)
    height, width = image.shape[:2]

    # 2. Preprocess
    _, blurred = preprocess_image(image, blur_kernel=args.blur)
    logging.info("Preprocessed image (grayscale + Gaussian blur k=%d)", args.blur)

    # 3. Detect circles
    circles = np.array([])
    contour_dets = None
    method = args.method

    if method == "hough":
        circles = detect_circles_hough(
            blurred,
            dp=args.dp,
            min_dist=args.min_dist,
            param1=args.param1,
            param2=args.param2,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
        )
        if circles.size == 0:
            logging.warning("HoughCircle found no circles; falling back to contour method.")
            method = "contour"
        else:
            logging.info("HoughCircle found %d coin(s).", len(circles))

    if method == "contour":
        contour_dets = detect_circles_contour(
            blurred,
            min_area=args.min_area,
            max_area=CONTOUR_MAX_AREA,
            circularity_threshold=args.circularity,
        )
        circles = np.array([[d["center"][0], d["center"][1], d["radius"]]
                            for d in contour_dets])
        logging.info("Contour method found %d coin(s).", len(circles))

    # 4. Draw detections + overlay count
    annotated = image.copy()
    if circles.size > 0:
        annotated = draw_detections(image, circles, contour_dets, method=method)
        annotated = overlay_count(annotated, len(circles))
        logging.info("Annotated image with %d coin(s).", len(circles))

        # 5. Size classification breakdown
        classification = classify_coins_by_size(
            circles, small_threshold=SMALL_RADIUS, large_threshold=LARGE_RADIUS)
        print("Size breakdown -> "
              f"small: {len(classification['small'])}, "
              f"medium: {len(classification['medium'])}, "
              f"large: {len(classification['large'])}")
        for size_key, entries in classification.items():
            logging.info("  %-6s : %d coin(s)", size_key, len(entries))
    else:
        logging.error("No coins detected in %s", args.image)

    # 6. Save output
    output_path = args.save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, annotated)
    logging.info("Saved annotated image to %s", output_path)
    print(f"Total Coins: {len(circles)} -> saved to {output_path}")

    # 7. Optional interactive display
    if getattr(args, "display", False):
        resized = cv2.resize(annotated,
                             (min(width, 900),
                              int(height * min(width, 900) / width)))
        cv2.imshow("Coin Detection Result", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(circles)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for coin detection."""
    parser = argparse.ArgumentParser(
        description="Detect and count coins in an image.")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE,
                        help="Path to input image (default: sample image).")
    parser.add_argument("--method", type=str, default="hough",
                        choices=["hough", "contour"],
                        help="Detection method (default: hough).")
    parser.add_argument("--blur", type=int, default=BLUR_KERNEL,
                        help="Gaussian blur kernel size (must be odd).")
    parser.add_argument("--dp", type=float, default=HOUGH_DP,
                        help="HoughCircles dp parameter.")
    parser.add_argument("--min-dist", type=int, default=HOUGH_MIN_DIST,
                        help="HoughCircles minDist parameter.")
    parser.add_argument("--param1", type=int, default=HOUGH_PARAM1,
                        help="HoughCircles param1 (Canny threshold).")
    parser.add_argument("--param2", type=int, default=HOUGH_PARAM2,
                        help="HoughCircles param2 (accumulator threshold).")
    parser.add_argument("--min-radius", type=int, default=HOUGH_MIN_RADIUS,
                        help="HoughCircles minRadius.")
    parser.add_argument("--max-radius", type=int, default=HOUGH_MAX_RADIUS,
                        help="HoughCircles maxRadius.")
    parser.add_argument("--min-area", type=int, default=CONTOUR_MIN_AREA,
                        help="Contour method minimum area.")
    parser.add_argument("--circularity", type=float, default=MIN_CIRCULARITY,
                        help="Contour method minimum circularity (0-1).")
    parser.add_argument("--save", type=str, default=DEFAULT_OUTPUT,
                        help="Output image path.")
    parser.add_argument("--display", action="store_true",
                        help="Show the annotated image in a window.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Entry point for the coin detection module."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    try:
        run_coin_detection(args)
    except (FileNotFoundError, ValueError, cv2.error) as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()