# Computer Vision

A collection of Computer Vision laboratory programs implemented using **Python, OpenCV, NumPy, and Matplotlib**.

## Programs

### 1. Normalisation and Contrast Enhancement

**Folder:** `normalisation/`

**File:** `contrast_enhancement.py`

This program enhances the contrast of a low-contrast image using image normalisation and contrast enhancement techniques.

**Input Image:** `lowcontrast.jpeg`

**Concepts covered:**

* Image normalisation
* Contrast enhancement
* Pixel intensity adjustment
* Low-contrast image processing

### 2. Histogram

**Folder:** `histogram/`

**File:** `histogram.py`

This program calculates and displays the histogram of an image.

The histogram represents the distribution of pixel intensity values and helps in analyzing the brightness and contrast of the image.

**Input Image:** `Pic2.png`

**Concepts covered:**

* Image histograms
* Pixel intensity distribution
* Histogram calculation
* Histogram visualization

### 3. Nearest-Neighbor Interpolation

**Folder:** `interpolation_nearest/`

**File:** `nearest_neighbor.py`

This program resizes an image using **nearest-neighbor interpolation**.

Nearest-neighbor interpolation assigns the value of the nearest pixel to each new pixel in the resized image.

**Input Image:** `Pic3.png`

**Concepts covered:**

* Image resizing
* Nearest-neighbor interpolation
* Image scaling
* Fixed output dimensions
* Aspect ratio

### 4. Bilinear Interpolation

**Folder:** `interpolation_linear/`

**File:** `bilinear_interpolation.py`

This program resizes an image using **bilinear interpolation**.

Bilinear interpolation calculates the value of a new pixel using the weighted average of neighboring pixels, producing smoother results than nearest-neighbor interpolation.

**Input Image:** `leaf.png`

**Concepts covered:**

* Image resizing
* Bilinear interpolation
* Image scaling
* Pixel interpolation
* Aspect ratio

### 5. Filtering

**Folder:** `filtering/`

**File:** `filtering.py`

This program demonstrates image filtering techniques used to process and enhance images.

Filtering can be used to reduce noise, smooth images, and improve image quality.

**Input Image:** `Pic1.png`

**Concepts covered:**

* Image filtering
* Image smoothing
* Noise reduction
* Kernel-based operations

### 6. Translation

**Folder:** `translation/`

**File:** `translation.py`

This program demonstrates **image translation using an affine transformation**.

The image is shifted horizontally and vertically by applying a translation matrix.

**Input Image:** `bird.jpeg`

**Concepts covered:**

* Geometric transformation
* Image translation
* Affine transformation
* Translation matrix
* `cv.warpAffine()`

### 7. Rotation

**Folder:** `rotation/`

**File:** `rotation.py`

This program demonstrates image rotation using OpenCV's rotation transformation.

The image is rotated around its center using a rotation matrix.

**Input Image:** `bird.jpeg`

**Concepts covered:**

* Image rotation
* Rotation matrix
* Rotation around image center
* Affine transformation
* Image transformation

## Project Structure

```text
Computer-Vision/
│
├── filtering/
│   ├── Pic1.png
│   └── filtering.py
│
├── histogram/
│   ├── Pic2.png
│   └── histogram.py
│
├── interpolation_linear/
│   ├── bilinear_interpolation.py
│   └── leaf.png
│
├── interpolation_nearest/
│   ├── Pic3.png
│   └── nearest_neighbor.py
│
├── normalisation/
│   ├── contrast_enhancement.py
│   └── lowcontrast.jpeg
│
├── rotation/
│   ├── bird.jpeg
│   └── rotation.py
│
├── translation/
│   ├── bird.jpeg
│   └── translation.py
│
└── README.md
```

## Requirements

Make sure Python is installed. Install the required libraries using:

```bash
pip install opencv-python numpy matplotlib
```

## How to Run

Open the project folder in VS Code and run any Python program from its respective folder.

For example:

```bash
python normalisation/contrast_enhancement.py
```

You can also run the other programs:

```bash
python filtering/filtering.py
python histogram/histogram.py
python interpolation_nearest/nearest_neighbor.py
python interpolation_linear/bilinear_interpolation.py
python rotation/rotation.py
python translation/translation.py
```

Make sure the input image used by each program is available in the correct folder or update the image filename/path in the code.

## Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**

## Topics Covered

* Image normalisation
* Contrast enhancement
* Histogram analysis
* Image interpolation
* Nearest-neighbor interpolation
* Bilinear interpolation
* Image filtering
* Image resizing
* Image translation
* Image rotation
* Affine transformations
