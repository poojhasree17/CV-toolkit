"""
CV Vision Toolkit - Synthetic Test Image Generator
==================================================
Generates simple synthetic sample images (no external downloads needed):

  * sample_images/coins/coins_sample.jpg   - drawn circles as fake coins
  * sample_images/fruits/fruits_sample.jpg - drawn coloured shapes as fruits

The images are designed so that the coin detection and fruit segmentation
modules can be tested immediately.

Usage:
    python generate_test_images.py [--overwrite]

Author: CV Vision Toolkit
"""

import argparse
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COINS_DIR = os.path.join(BASE_DIR, "sample_images", "coins")
FRUITS_DIR = os.path.join(BASE_DIR, "sample_images", "fruits")
COINS_PATH = os.path.join(COINS_DIR, "coins_sample.jpg")
FRUITS_PATH = os.path.join(FRUITS_DIR, "fruits_sample.jpg")


def add_noise_gradient(image: np.ndarray, intensity: float = 12.0) -> np.ndarray:
    """
    Add mild Gaussian noise and a soft gradient so the synthetic image
    looks less artificial (helps exercise the blur step).

    Parameters
    ----------
    image : np.ndarray
        BGR base image.
    intensity : float
        Standard deviation of Gaussian noise.

    Returns
    -------
    np.ndarray
        Noisy gradient-adjusted image.
    """
    rows, cols = image.shape[:2]
    grad = np.linspace(0.85, 1.0, cols, dtype=np.float32)
    gradient = np.tile(grad, (rows, 1))[:, :, np.newaxis]
    noisy = image.astype(np.float32) * gradient
    noise = np.random.normal(0, intensity, image.shape)
    return np.clip(noisy + noise, 0, 255).astype(np.uint8)


def generate_coins_image() -> np.ndarray:
    """
    Create a synthetic coins image by drawing filled circles.

    Uses 8 coins of different sizes and metallic shades on a dark
    background to mimic a simple coin-counting scene.

    Returns
    -------
    np.ndarray
        Synthetic coins image (BGR).
    """
    height, width = 600, 800
    image = np.full((height, width, 3), 25, dtype=np.uint8)

    # Format: (center, radius, colour in BGR)
    coins = [
        ((120, 140), 40, (196, 196, 196)),    # small silver
        ((260, 100), 35, (160, 170, 190)),    # small grey-silver
        ((420, 190), 45, (190, 200, 210)),    # medium silver
        ((560, 150), 38, (150, 160, 175)),    # small nickel
        ((140, 380), 65, (205, 175, 150)),    # large copper
        ((320, 330), 52, (220, 210, 190)),    # medium gold-ish
        ((500, 360), 70, (180, 150, 110)),    # large bronze
        ((660, 300), 48, (210, 220, 230)),    # medium bright silver
    ]

    for (cx, cy), radius, colour in coins:
        cv2.circle(image, (cx, cy), radius, colour, -1)
        inner = tuple(max(0, int(c * 0.82)) for c in colour)
        cv2.circle(image, (cx, cy), int(radius * 0.55), inner, 2)
        hl = tuple(min(255, int(c + 40)) for c in colour)
        cv2.circle(image, (cx - radius // 4, cy - radius // 4),
                   max(3, radius // 5), hl, -1)

    return add_noise_gradient(image, intensity=6.0)


def generate_fruits_image() -> np.ndarray:
    """
    Create a synthetic fruits image with drawn coloured shapes.

    Includes three red apples, one orange, two green fruits, and a
    banana-colour block so all three segmentation methods have something
    to find. Two red apples slightly overlap to exercise the Watershed.

    Returns
    -------
    np.ndarray
        Synthetic fruits image (BGR).
    """
    height, width = 600, 800
    image = np.full((height, width, 3), 230, dtype=np.uint8)  # light bg

    def draw_fruit(cx, cy, radius, bgr, shade_factor=0.85, ratio=1.0):
        """Draw a filled elliptical 'fruit' with subtle shading."""
        axes = (radius, int(radius * ratio))
        cv2.ellipse(image, (cx, cy), axes, 0, 0, 360, bgr, -1)
        shade = tuple(max(0, int(c * shade_factor)) for c in bgr)
        cv2.ellipse(image, (cx, cy), (int(radius * 0.85),
                     int(radius * 0.85 * ratio)), 0, 0, 360, shade, -1)
        hl = tuple(min(255, int(c + 40)) for c in bgr)
        cv2.ellipse(image, (cx - radius // 3, cy - radius // 3),
                    (radius // 3, int(radius // 3 * ratio)),
                    0, 0, 360, hl, -1)

    # Apple red (dark red BGR = (60, 40, 200))
    draw_fruit(160, 180, 75, (60, 40, 200))
    draw_fruit(300, 210, 80, (50, 30, 190))   # touching second apple
    draw_fruit(120, 420, 65, (70, 50, 210))   # third red apple

    # Orange (BGR = (20, 140, 255))
    draw_fruit(430, 150, 70, (20, 140, 255))

    # Green fruits (approx BGR green (60, 180, 60))
    draw_fruit(560, 260, 68, (60, 180, 60))
    draw_fruit(430, 380, 62, (70, 170, 80))

    # Banana-yellow block (BGR = (20, 200, 240))
    draw_fruit(640, 430, 55, (20, 200, 240), ratio=0.7)

    return add_noise_gradient(image, intensity=4.0)


def main(argv=None) -> None:
    """Generate the sample images and print their paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing sample images.")
    args = parser.parse_args(argv)

    os.makedirs(COINS_DIR, exist_ok=True)
    os.makedirs(FRUITS_DIR, exist_ok=True)

    if os.path.exists(COINS_PATH) and not args.overwrite:
        print(f"Skipping (already exists): {COINS_PATH}")
    else:
        cv2.imwrite(COINS_PATH, generate_coins_image())
        print(f"Generated coins image: {COINS_PATH}")

    if os.path.exists(FRUITS_PATH) and not args.overwrite:
        print(f"Skipping (already exists): {FRUITS_PATH}")
    else:
        cv2.imwrite(FRUITS_PATH, generate_fruits_image())
        print(f"Generated fruits image: {FRUITS_PATH}")


if __name__ == "__main__":
    main()