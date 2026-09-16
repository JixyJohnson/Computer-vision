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


### Single Layer Perceptron for Image Classification

## 📌 Overview

This project implements a **Single Layer Perceptron (SLP)** from scratch using Python for a simple computer vision classification problem.

The model is trained to classify handwritten digit images into **two classes: 0 and 1**.

The project demonstrates the basic working of a perceptron, including:

* Loading an image dataset
* Preprocessing image pixels
* Splitting data into training and testing sets
* Training a Single Layer Perceptron
* Making predictions
* Calculating training and testing accuracy

---

## 🎯 Objective

The main objective of this project is to understand how a **Single Layer Perceptron** can be used for binary image classification.

The input is an **8 × 8 grayscale image** of a handwritten digit.

The image is converted into **64 pixel features**, which are given as input to the perceptron.

The model predicts whether the image belongs to:

* **Class 0 → Digit 0**
* **Class 1 → Digit 1**

---

## 📊 Dataset

The project uses the **Digits dataset** available through `scikit-learn`.

### Dataset characteristics

| Property           | Description                  |
| ------------------ | ---------------------------- |
| Image type         | Grayscale handwritten digits |
| Image size         | 8 × 8 pixels                 |
| Number of features | 64                           |
| Classes used       | 0 and 1                      |
| Samples used       | 40                           |
| Training samples   | 30                           |
| Testing samples    | 10                           |

Only digits **0 and 1** are selected because a Single Layer Perceptron is designed for binary classification.

---

## 🧠 Single Layer Perceptron

A perceptron calculates a weighted sum of the input features:

$$
z = W^T X + b
$$

where:

* \(X\) = input pixel values
* \(W\) = weights
* \(b\) = bias
* \(z\) = weighted sum

A **step activation function** is then applied:

$$
\hat{y} =
\begin{cases}
1, & z \geq 0\\
0, & z < 0
\end{cases}
$$

The prediction is therefore either **0 or 1**.

---

## 🔄 Working Process

```text
        Handwritten Image
               ↓
          8 × 8 Image
               ↓
      Normalize Pixel Values
               ↓
       Flatten Image
               ↓
       64 Input Features
               ↓
     Single Layer Perceptron
               ↓
       Weighted Sum + Bias
               ↓
       Step Activation
               ↓
       Prediction: 0 or 1
               ↓
          Accuracy
```

---

## ⚙️ Perceptron Training

The perceptron updates its weights when the prediction is incorrect.

### Error

$$
Error = Target - Prediction
$$

### Weight update

$$
W_{new} = W_{old} + \eta(Error)X
$$

### Bias update

$$
b_{new} = b_{old} + \eta(Error)
$$

where \(\eta\) represents the learning rate.

In this project:

```text
Learning Rate = 0.1
Epochs = 20
```

---

## 🛠️ Technologies Used

* Python
* NumPy
* Scikit-learn

---

## 📁 Project Structure

```text
computer vision/
│
├── slp.py
└── README.md
```

---

## 📦 Installation

Make sure Python is installed.

Install the required libraries using:

```bash
python -m pip install numpy scikit-learn
```

---

## ▶️ How to Run

Open the terminal in the project folder:

```bash
cd "D:\computer vision"
```

Run the program:

```bash
python slp.py
```

---

## 📈 Output

The program displays:

* Total number of samples
* Number of training samples
* Number of testing samples
* Image size
* Number of features
* Training accuracy
* Testing accuracy

Example:

```text
====================================
 SINGLE LAYER PERCEPTRON
====================================
Total samples: 40
Training samples: 30
Testing samples: 10
Image size: 8 x 8
Number of features: 64
------------------------------------
Training Accuracy: 100.0 %
Testing Accuracy: 90.0 %
```

The exact testing accuracy may vary depending on the training/test split and training process.

---

## 📌 Advantages

* Simple and easy to understand
* Fast training for small datasets
* Easy to implement
* Suitable for binary classification
* Demonstrates the basic concept of neural networks

---

## ⚠️ Limitations

* A Single Layer Perceptron can only learn **linearly separable** problems.
* It cannot solve problems such as XOR.
* It is not suitable for complex image classification tasks.
* It does not automatically learn spatial features like a CNN.

---

## 🔍 SLP in Computer Vision

For an image classification problem, the image pixels are treated as input features.

For example:

```text
8 × 8 image
    ↓
64 pixel values
    ↓
Weights × Pixels
    ↓
Weighted Sum + Bias
    ↓
Step Function
    ↓
0 or 1
```

Unlike a **Convolutional Neural Network (CNN)**, the perceptron does not learn local spatial features such as edges, shapes, or textures.

---

## 🆚 SLP vs CNN

| Feature            | SLP                             | CNN                                  |
| ------------------ | ------------------------------- | ------------------------------------ |
| Architecture       | Single layer                    | Multiple layers                      |
| Input processing   | Flattened pixels                | Image structure preserved            |
| Feature extraction | No automatic feature extraction | Automatic feature extraction         |
| Classification     | Simple                          | Complex                              |
| Suitable for       | Simple binary problems          | Complex image problems               |
| XOR                | Cannot solve                    | Can solve with suitable architecture |
| Image recognition  | Limited                         | Highly suitable                      |

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

1. How image pixels are represented numerically.
2. How an image is converted into input features.
3. How a Single Layer Perceptron works.
4. How weights and bias are initialized.
5. How the step activation function works.
6. How perceptron weights are updated during training.
7. How training and testing datasets are used.
8. How classification accuracy is calculated.


