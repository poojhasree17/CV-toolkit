# CV-Vision-Toolkit

A small, dependency-light **OpenCV** project with two independent computer
vision modules:

1. **Coin Detection & Counting** — detects circular coins in an image and
   counts them, classified by size.
2. **Fruit Image Segmentation** — segments fruit regions from a photo using
   three different algorithms (HSV color masking, Otsu thresholding, and the
   marker-based Watershed).

Synthetic test images are generated locally by the included script, so the
project runs out of the box with **no downloads and no internet access**.

---

## Features

| Module              | Capabilities                                                                  |
| ------------------- | ----------------------------------------------------------------------------- |
| Coin Detection      | Hough Circle Transform **and** contour-based detection, size classification   |
| Fruit Segmentation  | HSV color-range masking, Otsu global thresholding, marker-based Watershed     |
| Test images         | Synthetic, reproducible scenes with noise/gradient for realistic algorithm runs |

---

## Folder Structure

```
cv_vision_toolkit/
├── coin_detection/
│   ├── __init__.py
│   ├── detect_coins.py        # CLI: pipeline for coin detection & counting
│   └── utils.py               # preprocess, Hough, contour, classify, draw, overlay
├── fruit_segmentation/
│   ├── __init__.py
│   ├── segment_fruits.py      # CLI: pipeline for fruit segmentation
│   └── utils.py               # HSV / Otsu / Watershed + morphology + contour helpers
├── sample_images/
│   ├── coins/coins_sample.jpg        # 8 synthetic coins
│   └── fruits/fruits_sample.jpg      # 7 synthetic fruits
├── output/
│   ├── coins_result.jpg              # annotated coin image (generated)
│   └── fruit_segmentation_result*.jpg  # comparison + overlay (generated)
├── generate_test_images.py    # CLI: creates the synthetic sample images
├── requirements.txt           # pinned dependencies
└── README.md
```

---

## Installation

Requires **Python 3.9+** (developed against Python 3.14, Windows).

```bash
cd cv_vision_toolkit
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

`requirements.txt` pins:

- `opencv-python==4.13.0.92`
- `numpy==2.4.3`
- `matplotlib==3.10.8`

> Matplotlib is only required for composing the *side-by-side comparison
> image* in fruit segmentation. Coin detection uses OpenCV only.

---

## Quick Start

Generate the synthetic test images, then run each module:

```bash
# 1) Create sample images
python generate_test_images.py --overwrite

# 2) Coin detection (Hough, default)
python coin_detection/detect_coins.py

# 3) Fruit segmentation (HSV, default)
python fruit_segmentation/segment_fruits.py
```

---

## Usage

### Generate test images

```bash
python generate_test_images.py            # only creates missing images
python generate_test_images.py --overwrite  # always regenerate
```

Creates:

- `sample_images/coins/coins_sample.jpg` — 8 coins of three size classes on a
  dark background.
- `sample_images/fruits/fruits_sample.jpg` — 7 fruits (3 red apples with two
  overlapping, 1 orange, 2 green fruits, 1 banana) on a light background.

Both images include subtle Gaussian noise and a horizontal illumination
gradient so the algorithms have to work a little.

### Coin detection & counting

```bash
python coin_detection/detect_coins.py
python coin_detection/detect_coins.py --method contour          # contour fallback
python coin_detection/detect_coins.py --method hough --verbose  # debug logging
```

Optional arguments (see `python coin_detection/detect_coins.py --help`):

| Flag            | Default | Meaning                              |
| --------------- | ------- | ------------------------------------ |
| `--image`       | sample  | Input image path                     |
| `--method`      | hough   | `hough` or `contour`                 |
| `--blur`        | 9       | Gaussian blur kernel (odd)           |
| `--dp`          | 1.2     | HoughCircles `dp`                    |
| `--min-dist`    | 30      | HoughCircles `minDist`               |
| `--param1`      | 100     | HoughCircles `param1` (Canny high)   |
| `--param2`      | 35      | HoughCircles `param2` (accumulator)  |
| `--min-radius`  | 15      | Smallest search radius               |
| `--max-radius`  | 120     | Largest search radius                |
| `--min-area`    | 800     | Contour method minimum blob area     |
| `--circularity` | 0.75    | Contour method minimum circularity   |
| `--save`        | sample  | Output annotated image path          |
| `--display`     | off     | Show result in a window              |
| `--verbose`     | off     | DEBUG logging                        |

Coins are classified by radius:

- `radius < 25` → **small**
- `25 <= radius < 50` → **medium**
- `radius >= 50` → **large**

### Fruit segmentation

```bash
python fruit_segmentation/segment_fruits.py --method hsv        # default
python fruit_segmentation/segment_fruits.py --method otsu
python fruit_segmentation/segment_fruits.py --method watershed
```

Optional arguments (see `python fruit_segmentation/segment_fruits.py --help`):

| Flag         | Default | Meaning                            |
| ------------ | ------- | ---------------------------------- |
| `--image`    | sample  | Input image path                   |
| `--method`   | hsv     | `hsv`, `otsu`, or `watershed`      |
| `--min-area` | 500     | Minimum contour area to keep       |
| `--save`     | sample  | Output comparison image path       |
| `--display`  | off     | Show result in a window            |
| `--verbose`  | off     | DEBUG logging                      |

Each run writes two files:

- `output/fruit_segmentation_result.jpg` — side-by-side comparison
  (original | binary mask | annotated result).
- `output/fruit_segmentation_result_segmented.jpg` — the annotated overlay
  only.

---

## Algorithms

### Coin Detection

**1. Preprocessing** — BGR → grayscale, then a 9×9 Gaussian blur to suppress
noise so the contour/Hough operators see clean edges.

**2a. Hough Circle Transform** (`method=hough`) — `cv2.HoughCircles` on the
blurred grayscale image votes for candidate circle centers/radii in a
parameter space. Voters below the accumulator threshold are discarded; nearby
centers are merged using `minDist`. Tuned parameters are exposed as CLI flags.

**2b. Contour-based detection** (`method=contour`) — thresholded binary
contours are extracted with `cv2.findContours`. Each contour is kept only if
it is large enough (`min_area`) and circular enough (`circularity` =
`4π·area / perimeter²`), then the best-fitting circle is drawn from its
centroid and radius.

**3. Size classification** — radii are mapped to *small / medium / large*
using the radius thresholds in the table above.

**4. Annotate & count** — each coin is drawn with a colored circle and the
final count is reported on the image (`Total Coins: N`).

### Fruit Segmentation

All three methods produce a **binary mask** (white = fruit), which is then
cleaned with morphological *opening* (remove speckles) and *closing* (fill
holes) using a 5×5 kernel, and turned into numbered contours.

**1. HSV color masking** (`method=hsv`) — the image is converted from BGR to
HSV. Hand-tuned hue/saturation/value ranges for red (two ranges, because red
wraps around 0°/180°), orange, yellow, and green are each applied with
`cv2.inRange` and OR-ed together into one mask. A great color-based approach,
but it cannot separate two objects that physically touch and share/wrap
colors.

**2. Otsu thresholding** (`method=otsu`) — a grayscale histogram is
automatically split at the value that maximizes *between-class variance*,
separating bright background from dark objects. Because objects can be
darker *or* brighter than the background, both polarities are computed and
the one covering the smaller area is selected as the foreground — a simple,
robust heuristic for typical photos.

**3. Marker-based Watershed** (`method=watershed`) — after Otsu binarization,
a distance transform identifies *sure foreground* markers; unknown boundary
pixels are marked and `cv2.watershed` floods the gradient image from those
markers. Touching objects get separate regions, which makes Watershed the
only included method that can split the overlapping apples-orange pair.

---

## Results on the Synthetic Samples

| Module / method       | Objects found | Notes                                          |
| --------------------- | ------------- | ---------------------------------------------- |
| Coins — Hough         | 8 / 8         | 5 medium + 3 large (no small coins in scene)   |
| Coins — Contour       | 8 / 8         | Same tally as Hough                            |
| Fruits — HSV          | 5             | Overlapping apples + orange read as one region |
| Fruits — Otsu         | 4             | Banana blends into the bright background       |
| Fruits — Watershed    | 5             | Separates the touching apples from the orange  |

The fruit scene places 7 fruits such that a red apple touches a second apple
*(and that apple touches the orange)*, so the expected maximum object count
is **6 distinct regions**; methods that cannot split touching objects count
that cluster as one.

---

## Known Limitations

- **HSV** cannot separate objects whose masks touch — the overlapping
  apple+orange cluster is counted once.
- **Otsu / Watershed** operate on grayscale; bright-yellow objects on light
  backgrounds (like the synthetic banana) have nearly equal luminance and are
  easily missed.
- Hough/contour coin parameters are tuned for the supplied synthetic image;
  real-world photos may need `--param2`, `--min-area`, or `--circularity`
  adjusted to avoid misses or duplicates.
- The Otsu foreground-polarity heuristic assumes the object(s) of interest
  occupy the minority of the scene (true for most photos).

---

## Screenshots
## Results

### Coin Detection
![Coin detection result](output/coins_result.jpg)

### Fruit Segmentation
![Fruit segmentation result](output/fruit_segmentation_result.jpg)

```
![Coin detection](output/coins_result.jpg)
![Fruit segmentation comparison](output/fruit_segmentation_result.jpg)
```

The pipeline automatically writes annotated images to the `output/` folder
every time you run either module, so you can drop them in directly.

---

## License

Free to use, modify, and distribute for educational purposes.
