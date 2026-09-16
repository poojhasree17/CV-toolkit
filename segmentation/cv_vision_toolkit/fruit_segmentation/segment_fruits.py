"""
CV Vision Toolkit - Fruit Image Segmentation
============================================
Segments fruits in an image using one of three selectable methods:

  * hsv       - colour-based segmentation in HSV space
  * otsu      - Otsu's adaptive thresholding
  * watershed - marker-based watershed for touching/overlapping objects

Usage:
    python fruit_segmentation/segment_fruits.py [--image path/to/image.jpg]
        [--method hsv|otsu|watershed] [--min-area 500] [--save path]

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

# Local utility imports (available because BASE_DIR is on sys.path)
from fruit_segmentation.utils import (
    convert_to_hsv,
    segment_by_hsv,
    segment_by_otsu,
    segment_by_watershed,
    clean_mask,
    find_fruit_contours,
    draw_fruit_contours,
    make_comparison_image,
)

# ---------------- Tunable Parameters ----------------
DEFAULT_METHOD = "hsv"           # hsv | otsu | watershed
MIN_CONTOUR_AREA = 500           # ignore small speckles from the mask

# File paths
DEFAULT_IMAGE = os.path.join(
    BASE_DIR, "sample_images", "fruits", "fruits_sample.jpg")
DEFAULT_OUTPUT = os.path.join(
    BASE_DIR, "output", "fruit_segmentation_result.jpg")


def segment_by_hsv_on_image(image: np.ndarray) -> np.ndarray:
    """
    HSV segmentation entry point accepting a BGR image.

    Converts the image to HSV internally and delegates to the shared
    segment_by_hsv utility.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255).
    """
    hsv = convert_to_hsv(image)
    return segment_by_hsv(hsv)


# Map CLI method names to segmentation functions.
# Every callable takes a BGR image and returns a binary mask.
METHOD_FUNCTIONS = {
    "hsv": segment_by_hsv_on_image,
    "otsu": segment_by_otsu,
    "watershed": segment_by_watershed,
}


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
        Loaded image in BGR format.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the file exists but cannot be read as an image.
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


def run_fruit_segmentation(args: argparse.Namespace) -> int:
    """
    Main pipeline for fruit segmentation.

    Steps: load image -> segment (hsv/otsu/watershed) -> morphological
    cleanup -> find contours -> annotate -> save comparison image.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Number of fruits detected.
    """
    # 1. Load image
    image = load_image(args.image)
    logging.info("Loaded image: %s (shape=%s)", args.image, image.shape)
    height, width = image.shape[:2]

    # 2. Choose and run the segmentation method
    seg_func = METHOD_FUNCTIONS[args.method]
    raw_mask = seg_func(image)
    logging.info("Segmentation method '%s' produced raw mask.", args.method)

    # 3. Clean the mask with morphology (opening + closing)
    mask = clean_mask(raw_mask)
    logging.info("Applied morphological cleanup to mask.")

    # 4. Find contours
    contours, boxes = find_fruit_contours(mask, min_area=args.min_area)
    fruit_count = len(contours)
    logging.info("Found %d fruit object(s).", fruit_count)
    print(f"Total Fruits Detected: {fruit_count}")

    # 5. Draw contours + labels on the original image
    if fruit_count > 0:
        segmented = draw_fruit_contours(image, contours)
    else:
        logging.warning("No fruits segmented; keeping an unannotated copy.")
        segmented = image.copy()

    # 6. Compose side-by-side comparison (original | mask | result)
    comparison = make_comparison_image(
        image, mask, segmented,
        title=f"Fruit Segmentation ({args.method.upper()} method) - "
              f"{fruit_count} fruit(s)",
    )
    logging.info("Composed side-by-side comparison image.")

    # 7. Save the comparison image and the segmented overlay
    save_path = args.save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, comparison)
    logging.info("Saved comparison image to %s", save_path)

    seg_path = os.path.join(
        os.path.dirname(save_path) or ".",
        os.path.splitext(os.path.basename(save_path))[0] + "_segmented.jpg",
    )
    cv2.imwrite(seg_path, segmented)
    logging.info("Saved segmented overlay to %s", seg_path)
    print(f"Results saved to {save_path} and {seg_path}")

    # 8. Optional interactive display
    if getattr(args, "display", False):
        resized = cv2.resize(
            comparison, (min(1200, width * 3),
                         int(height * min(1200, width * 3) / (width * 3))))
        cv2.imshow("Fruit Segmentation Result", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return fruit_count


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for fruit segmentation."""
    parser = argparse.ArgumentParser(
        description="Segment fruits in an image (HSV / Otsu / Watershed).")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE,
                        help="Path to input image (default: sample image).")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD,
                        choices=list(METHOD_FUNCTIONS.keys()),
                        help="Segmentation method (default: hsv).")
    parser.add_argument("--min-area", type=int, default=MIN_CONTOUR_AREA,
                        help="Minimum contour area to keep as a fruit.")
    parser.add_argument("--save", type=str, default=DEFAULT_OUTPUT,
                        help="Output comparison image path.")
    parser.add_argument("--display", action="store_true",
                        help="Show the result in a window.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Entry point for the fruit segmentation module."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    try:
        run_fruit_segmentation(args)
    except (FileNotFoundError, ValueError, cv2.error) as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()