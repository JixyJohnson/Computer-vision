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


### 8. Single Layer Perceptron for Image Classification

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


# Program 1 – Basic CNN Image Classification Using MobileNetV3

## Project Overview

This project implements an image classification system using **MobileNetV3-Small** and the **CIFAR-10 dataset** with PyTorch. It uses transfer learning and fine-tuning to classify images into 10 different categories.

The model is trained using CPU and saves the best-performing model for predicting new images.

## Objectives

- Implement image classification using a pretrained CNN architecture.
- Apply transfer learning and fine-tuning.
- Train and evaluate the model using CIFAR-10.
- Generate training and evaluation visualizations.
- Predict the category of new input images.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Pillow)

## Project Structure

```text
Pogram1_Basic_CNN/
│
├── dataset/
│   └── cifar-10-batches-py/
│
├── models/
│   └── best_cnn.pth
│
├── results/
│   ├── training_graph.png
│   ├── confusion_matrix.png
│   └── predictions.png
│
├── train.py
├── predict.py
└── test_installation.py
```

## 1. Dataset

**Dataset:** CIFAR-10

CIFAR-10 is a benchmark image classification dataset containing 60,000 colour images divided into 10 classes.

| Dataset | Number of Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |
| **Total** | **60,000** |

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is automatically downloaded using Torchvision if it is not already available in the `dataset/` directory.

## 2. Model Architecture

**MobileNetV3-Small**

MobileNetV3-Small is a lightweight Convolutional Neural Network designed for efficient image classification.

This project uses pretrained ImageNet weights and replaces the original classification layer with a new layer containing 10 output classes for CIFAR-10.

### Training Approach

**Phase 1 – Transfer Learning**
- Load pretrained MobileNetV3-Small.
- Freeze the feature extraction layers.
- Train the newly added classification layer.
- Train for 5 epochs.

**Phase 2 – Fine-Tuning**
- Unfreeze the last two feature extraction blocks.
- Fine-tune the model using a lower learning rate.
- Train for another 5 epochs.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV3-Small |
| Framework | PyTorch |
| Pretrained Weights | ImageNet |
| Input Image Size | 96 × 96 |
| Batch Size | 64 |
| Head Training Epochs | 5 |
| Fine-Tuning Epochs | 5 |
| Optimizer | Adam |
| Loss Function | Cross Entropy Loss |
| Device | CPU |
| Number of Classes | 10 |

The best model is selected based on validation accuracy and saved as `best_cnn.pth` inside the `models/` directory.

## 3. Programs Description

### train.py – Model Training

- Downloads and loads the CIFAR-10 dataset.
- Splits the training dataset into training and validation sets.
- Applies image transformations and data augmentation.
- Loads pretrained MobileNetV3-Small.
- Performs transfer learning and fine-tuning.
- Evaluates the model using the test dataset.
- Saves the best model checkpoint.
- Generates training graphs, a confusion matrix, and sample predictions.

### predict.py – Image Prediction

- Loads the saved MobileNetV3-Small model.
- Accepts an image path from the user.
- Preprocesses the input image.
- Predicts its category.
- Displays the prediction confidence.
- Displays the top 3 predicted classes.

### test_installation.py – Environment Testing

Checks whether the required computer vision libraries are installed correctly.

It verifies:
- PyTorch installation.
- Torchvision installation.
- Matplotlib installation.
- Scikit-learn installation.
- GPU availability.
- MobileNetV3-Small architecture availability.
- Pretrained weights configuration availability.

## 4. Results

The `results/` directory stores the visual outputs generated during model training and evaluation.

### Training Graph

Displays the training and validation loss and accuracy across epochs.

File: `training_graph.png`

### Confusion Matrix

Visualizes the classification performance across the 10 CIFAR-10 classes.

File: `confusion_matrix.png`

### Sample Predictions

Displays sample test images along with their actual class, predicted class, and prediction confidence.

File: `predictions.png`



## 5. How to Run

**Step 1 – Test the installation**

```bash
python test_installation.py
```

**Step 2 – Train the model**

```bash
python train.py
```

The CIFAR-10 dataset will be downloaded automatically if necessary. After training, the best model will be saved in the `models/` directory, and the visual results will be stored in `results/`.

**Step 3 – Predict a new image**

```bash
python predict.py
```

Enter the image path when prompted.

Example:

```text
Enter image path (example: dog.jpg): dog.jpg
```

The program displays the predicted category, confidence score, and top 3 predictions.

## 6. Evaluation Metrics

The model performance is evaluated using:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Training Loss
- Validation Loss
- Test Loss
- Confusion Matrix
- Prediction Confidence

Actual performance values are obtained after running the training program.

# Program 2 – CNN Architecture Comparison Using Transfer Learning and Fine-Tuning

## Project Overview

This project compares three pretrained Convolutional Neural Network (CNN) architectures: **ResNet18, DenseNet121, and EfficientNetB0** for image classification using the CIFAR-10 dataset.

The implementation uses PyTorch and applies transfer learning and fine-tuning to evaluate the performance of each architecture.

The experiments are conducted using CPU with smaller subsets of CIFAR-10 to reduce computational requirements.

The models are compared based on:
- Classification accuracy
- Test loss
- Training time
- Model size

## Objectives

- Compare different pretrained CNN architectures.
- Implement transfer learning using pretrained ImageNet weights.
- Apply fine-tuning to selected feature extraction layers.
- Evaluate model performance using CIFAR-10.
- Compare accuracy, training time, and model size.
- Generate graphical representations of the comparison results.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Pandas
- Matplotlib
- NumPy

## Project Structure

```text
Program2_CNN_Comparison/
│
├── dataset/
│   └── cifar-10-batches-py/
│
├── models/
│   ├── ResNet18.pth
│   ├── DenseNet121.pth
│   └── EfficientNetB0.pth
│
├── results/
│   ├── model_comparison.csv
│   ├── accuracy_comparison.png
│   ├── training_time_comparison.png
│   └── model_size_comparison.png
│
└── train.py
```

## 1. Dataset

**Dataset: CIFAR-10**

CIFAR-10 is a widely used image classification dataset containing 60,000 colour images belonging to 10 different classes.

### Dataset Details

| Dataset | Number of Images |
|---|---:|
| Training Dataset | 50,000 |
| Testing Dataset | 10,000 |
| Total | 60,000 |

For this experiment, smaller subsets are selected to reduce CPU training time.

| Parameter | Value |
|---|---:|
| Training Subset | 2,000 images |
| Testing Subset | 500 images |
| Image Size | 96 × 96 |
| Batch Size | 32 |

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The program automatically searches for an existing CIFAR-10 dataset in the project and parent directories. If it is not found, the dataset is downloaded automatically.

## 2. CNN Architectures

This project compares three pretrained CNN architectures.

### 1. ResNet18

ResNet18 is a residual neural network that uses shortcut connections to improve information and gradient flow during training.

### 2. DenseNet121

DenseNet121 connects each layer to subsequent layers through dense connections, encouraging feature reuse and efficient information flow.

### 3. EfficientNetB0

EfficientNetB0 uses a compound scaling approach to balance network depth, width, and input resolution while maintaining computational efficiency.

## 3. Training Methodology

The project uses two training stages for each architecture.

### Stage A – Transfer Learning (Feature Extraction)

- Load pretrained ImageNet weights.
- Freeze the pretrained feature extraction layers.
- Replace the original classification layer with a new 10-class classification layer.
- Train only the new classification layer.
- Number of epochs: 2.
- Learning rate: 0.001.

### Stage B – Fine-Tuning

- Unfreeze the selected final feature extraction block.
- Train the selected layers along with the classification layer.
- Use a lower learning rate to adjust pretrained features.
- Number of epochs: 1.
- Learning rate: 0.0001.

### Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Device | CPU |
| Architectures | ResNet18, DenseNet121, EfficientNetB0 |
| Pretrained Weights | ImageNet |
| Batch Size | 32 |
| Image Size | 96 × 96 |
| Training Images | 2,000 |
| Testing Images | 500 |
| Feature Extraction Epochs | 2 |
| Fine-Tuning Epochs | 1 |
| Optimizer | AdamW |
| Loss Function | Cross Entropy Loss |

## 4. Program Description

### train.py – CNN Architecture Comparison

The main Python program performs the following operations:

1. Loads and preprocesses the CIFAR-10 dataset.
2. Selects smaller training and testing subsets.
3. Loads pretrained ResNet18, DenseNet121, and EfficientNetB0.
4. Freezes pretrained layers and trains the classification layers.
5. Unfreezes selected final feature extraction blocks for fine-tuning.
6. Evaluates each architecture using the test dataset.
7. Measures total experiment time and model size.
8. Saves the trained model weights.
9. Generates comparison tables and graphs.

Each architecture is trained and evaluated independently using the same dataset subsets and training configuration.

## 5. Evaluation Metrics

The architectures are compared using the following metrics:

| Metric | Description |
|---|---|
| Test Accuracy | Percentage of correctly classified test images |
| Test Loss | Measures classification prediction error |
| Training Time | Total time taken for the complete model experiment |
| Model Size | Storage size of the saved model weights in MB |

## 6. Results and Visualization

The `results/` directory contains the comparison outputs generated after executing the program.

### 1. Model Comparison Table

File: `model_comparison.csv`

Stores the final performance metrics of all three CNN architectures.

It includes:
- Model name
- Accuracy (%)
- Test loss
- Training time (seconds)
- Model size (MB)

### 2. Accuracy Comparison

File: `accuracy_comparison.png`

A bar chart comparing the test accuracy of ResNet18, DenseNet121, and EfficientNetB0.

### 3. Training Time Comparison

File: `training_time_comparison.png`

A bar chart comparing the total experiment time of the three architectures.

### 4. Model Size Comparison

File: `model_size_comparison.png`

A bar chart comparing the saved model sizes in megabytes.

**Note:** Actual accuracy, loss, training time, and model size values are generated when the program is executed. They are not predetermined in the source code.


## 7. How to Run

Execute the following command:

```bash
python train.py
```

The program will:

1. Check for the CIFAR-10 dataset.
2. Download it if necessary.
3. Train ResNet18.
4. Train DenseNet121.
5. Train EfficientNetB0.
6. Evaluate all three architectures.
7. Save trained model weights in the `models/` directory.
8. Save the comparison table and graphs in the `results/` directory.

## 8. Expected Output Files

After successful execution, the following files will be generated:

**Models**
- `ResNet18.pth`
- `DenseNet121.pth`
- `EfficientNetB0.pth`

**Results**
- `model_comparison.csv`
- `accuracy_comparison.png`
- `training_time_comparison.png`
- `model_size_comparison.png`

## 9. Conclusion

This project provides a comparative study of three popular pretrained CNN architectures using transfer learning and fine-tuning.

ResNet18, DenseNet121, and EfficientNetB0 are evaluated using the same CIFAR-10 subsets. Their classification accuracy, test loss, training time, and model size are compared to understand their performance and computational requirements.

# Program 3 – Object Detection Using Faster R-CNN with FPN

## Project Overview

This project implements an object detection system using **Faster R-CNN with ResNet50-FPN** (Feature Pyramid Network) using PyTorch and Torchvision.

The system uses a model pretrained on the **COCO dataset** to identify and locate multiple objects in an input image.

It detects objects, draws bounding boxes around them, calculates confidence scores, counts detected objects, and generates detection reports.

The implementation uses CPU for object detection.

## Objectives

- Implement object detection using Faster R-CNN.
- Understand the Faster R-CNN architecture with ResNet50-FPN.
- Use pretrained COCO weights for object detection.
- Identify multiple objects in an image.
- Draw bounding boxes with object labels and confidence scores.
- Count detected objects belonging to each category.
- Extract and display FPN feature map information.
- Generate detection reports and visualizations.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- Pandas
- Pillow (PIL)

## Project Structure

```text
Program3_object_detection/
│
├── input_images/
│   └── sample.jpg
│
├── models/
│
├── results/
│   ├── detected_objects.jpg
│   ├── detection_results.csv
│   └── detection_visualization.png
│
└── detect.py
```

## 1. Dataset and Pretrained Weights

**Dataset: COCO (Common Objects in Context)**

This project uses Faster R-CNN pretrained weights trained on the COCO dataset.

COCO is a large-scale object detection dataset containing images with multiple object categories and annotated bounding boxes.

The pretrained model supports **80 object categories**.

The program loads the category names directly from the pretrained model's weight metadata.

### Important

This project uses pretrained COCO weights for inference. It does not implement a separate training process or download and prepare a custom object detection dataset.

## 2. Model Architecture

**Faster R-CNN with ResNet50-FPN**

Faster R-CNN is a two-stage object detection architecture.

### Main Components

**1. ResNet50 Backbone**

- Extracts important visual features from input images.
- Produces feature representations for object detection.

**2. Feature Pyramid Network (FPN)**

- Combines feature information from different spatial resolutions.
- Helps detect objects of different sizes.
- Produces multiple feature pyramid levels.

**3. Region Proposal Network (RPN)**

- Generates candidate object regions.
- Identifies possible locations containing objects.

**4. RoI Heads**

- Classifies proposed regions into object categories.
- Refines the bounding box coordinates.
- Produces confidence scores for detected objects.

### Detection Configuration

| Parameter | Value |
|---|---|
| Architecture | Faster R-CNN |
| Backbone | ResNet50 |
| Feature Extractor | FPN |
| Pretrained Dataset | COCO |
| Framework | PyTorch |
| Device | CPU |
| Confidence Threshold | 0.50 |
| Number of Categories | 80 |

## 3. Program Description

### detect.py – Object Detection

The main Python program performs the following operations:

1. Loads the pretrained Faster R-CNN model.
2. Loads the COCO object category names.
3. Reads images from the `input_images/` directory.
4. Selects the first available image.
5. Converts the image into a PyTorch tensor.
6. Extracts feature pyramid information using a forward hook.
7. Performs object detection.
8. Filters detections using a confidence threshold of 50%.
9. Draws bounding boxes around detected objects.
10. Displays object labels and confidence scores.
11. Counts the detected objects in each category.
12. Displays FPN feature map shapes.
13. Saves the annotated image and detection report.
14. Displays the final detection visualization.

## 4. Object Detection Process

The system follows this workflow:

```text
        Input Image
             |
             v
     Image Preprocessing
             |
             v
       ResNet50 Backbone
             |
             v
    Feature Pyramid Network
             |
             v
 Region Proposal Network
             |
             v
       RoI Heads
             |
             v
      Object Detection
             |
             v
   Confidence Filtering
       (Threshold 0.50)
             |
             v
    Bounding Box Drawing
             |
             v
     Object Classification
             |
             v
       Object Counting
             |
             v
       Output Results
```

## 5. Detection Results

The program produces the following information:

- Detected object names.
- Confidence percentage for each detected object.
- Bounding box coordinates (X1, Y1, X2, Y2).
- Total number of detected objects.
- Category-wise object counts.
- Feature pyramid output shapes.

Detections with confidence scores below 50% are excluded from the final results.

## 6. Results and Visualization

All generated outputs are stored inside the `results/` directory.

### 1. Detected Objects Image

File: `detected_objects.jpg`

Contains the original input image with bounding boxes drawn around detected objects.

Each bounding box includes:
- Object category.
- Confidence percentage.

### 2. Detection Results CSV

File: `detection_results.csv`

Stores the object detection details in tabular format.

| Column | Description |
|---|---|
| Object | Detected object category |
| Confidence (%) | Detection confidence |
| X1 | Top-left X coordinate |
| Y1 | Top-left Y coordinate |
| X2 | Bottom-right X coordinate |
| Y2 | Bottom-right Y coordinate |

### 3. Detection Visualization

File: `detection_visualization.png`

Displays the final object detection output with bounding boxes and labels.

## 7. How to Run

**Step 1 – Add an input image**

Place one or more JPG, JPEG, PNG, or BMP images inside:

```text
input_images/
```

**Step 2 – Execute the detection program**

```bash
python detect.py
```

**Step 3 – View the output**

After successful execution, the detected image, CSV report, and visualization will be saved inside the `results/` directory.

The program automatically selects the first image in alphabetical order from the `input_images/` directory.

## 8. Conclusion

This project demonstrates object detection using Faster R-CNN with a ResNet50 backbone and Feature Pyramid Network.

By using pretrained COCO weights, the system can detect multiple objects, locate them using bounding boxes, calculate confidence scores, and count objects by category.

The project also provides information about the feature pyramid outputs and generates visual and tabular detection results.

# Program 4 – Fruit Image Classification Using CNN

## Project Overview

This project implements a fruit image classification system using a custom Convolutional Neural Network (CNN) with PyTorch.

The system uses a fruit image dataset containing six classes. It trains a CNN model to learn visual features from fruit images and classify them into their corresponding categories.

The project includes dataset preprocessing, training, validation, testing, model saving, and performance visualization.

## Objectives

- Implement image classification using a custom CNN architecture.
- Load and preprocess a fruit image dataset.
- Apply image augmentation techniques.
- Train and evaluate the CNN model.
- Save the best-performing model based on validation accuracy.
- Generate training loss and validation accuracy graphs.
- Evaluate the final model using a separate test dataset.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- CSV
- Pillow (PIL)

## Project Structure

```text
Program4_Image_Segmentation/
│
├── dataset/
│   ├── Fruit_Class_1/
│   ├── Fruit_Class_2/
│   ├── Fruit_Class_3/
│   ├── Fruit_Class_4/
│   ├── Fruit_Class_5/
│   └── Fruit_Class_6/
│
├── models/
│   └── best_cnn.pth
│
├── results/
│   ├── training_metrics.csv
│   ├── training_loss.png
│   └── validation_accuracy.png
│
└── segmentation.py
```

*Replace the example class folder names with the actual six class names in your dataset.*

## 1. Dataset

**Dataset: Fruit Image Dataset**

The project uses a custom fruit image dataset containing six different classes.

The dataset is loaded using the `ImageFolder` function from Torchvision.

### Dataset Organization

Each fruit category must be stored in a separate subfolder inside the `dataset/` directory.

Example:

```text
dataset/
│
├── Apple/
├── Banana/
├── Orange/
├── Mango/
├── Grapes/
└── Other_Fruit/
```

The above names are illustrative. The actual folder names determine the class labels used by the model.

### Dataset Selection

The program selects a maximum of 150 images from each class.

| Dataset Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The images are selected and split separately for each class.

### Image Preprocessing

The following transformations are applied:

- Resize images to 64 × 64 pixels.
- Random horizontal flipping.
- Random rotation up to 10 degrees.
- Convert images into tensors.
- Normalize image pixel values.

## 2. CNN Model Architecture

**Custom Convolutional Neural Network (FruitCNN)**

The project uses a custom CNN architecture to extract visual features and classify fruit images.

### Architecture Components

**Feature Extraction Layers**

1. First Convolution Block:
   - Input channels: 3
   - Output channels: 32
   - ReLU activation
   - Max pooling

2. Second Convolution Block:
   - Input channels: 32
   - Output channels: 64
   - ReLU activation
   - Max pooling

3. Third Convolution Block:
   - Input channels: 64
   - Output channels: 128
   - ReLU activation
   - Max pooling

**Classification Layers**

- Flatten layer.
- Fully connected layer with 128 neurons.
- ReLU activation.
- Dropout with probability 0.3.
- Final fully connected layer with six output classes.

### Model Configuration

| Parameter | Value |
|---|---|
| Model | Custom CNN |
| Input Image Size | 64 × 64 |
| Number of Classes | 6 |
| Batch Size | 16 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Cross Entropy Loss |
| Device | CUDA if available, otherwise CPU |

## 3. Training Methodology

The model follows these steps:

1. Load the fruit image dataset.
2. Identify the six fruit classes.
3. Select up to 150 images per class.
4. Divide the dataset into training, validation, and testing sets.
5. Apply image preprocessing and augmentation.
6. Create the custom CNN architecture.
7. Train the CNN using the training dataset.
8. Evaluate the model using the validation dataset after each epoch.
9. Save the best model based on validation accuracy.
10. Evaluate the saved model using the test dataset.

## 4. Program Description

### segmentation.py – Fruit Image Classification

The main Python program performs the following operations:

- Loads the fruit image dataset.
- Applies preprocessing and augmentation.
- Creates training, validation, and testing subsets.
- Builds a custom CNN model.
- Trains the model for five epochs.
- Calculates training loss and validation accuracy.
- Saves the best model checkpoint.
- Evaluates the model using the test dataset.
- Generates training and validation graphs.
- Saves training metrics in CSV format.

## 5. Evaluation Metrics

The model performance is evaluated using:

| Metric | Description |
|---|---|
| Training Loss | Measures prediction error during training |
| Validation Loss | Measures prediction error on validation images |
| Validation Accuracy | Percentage of correctly classified validation images |
| Test Loss | Measures prediction error on test images |
| Test Accuracy | Percentage of correctly classified test images |

## 6. Results and Visualization

All generated outputs are stored inside the `results/` directory.

### 1. Training Metrics

File: `training_metrics.csv`

Contains the following information for each epoch:

- Epoch number.
- Training loss.
- Validation accuracy.

### 2. Training Loss Graph

File: `training_loss.png`

Displays the training loss across five epochs and helps observe how the model learns during training.

### 3. Validation Accuracy Graph

File: `validation_accuracy.png`

Displays the validation accuracy across five epochs and helps evaluate the model's classification performance.

## 7. Model Saving

The best model is saved in:

```text
models/best_cnn.pth
```

The checkpoint is selected based on the highest validation accuracy achieved during training.

## 8. How to Run

**Step 1 – Prepare the dataset**

Place the extracted fruit dataset inside the `dataset/` directory.

Ensure that it contains exactly six class folders.

**Step 2 – Execute the program**

```bash
python segmentation.py
```

**Step 3 – View the results**

After successful execution:

- The best trained model is saved inside `models/`.
- Training metrics are saved inside `results/`.
- Training loss and validation accuracy graphs are generated inside `results/`.
- Final test accuracy and test loss are displayed in the terminal.

## 9. Conclusion

This project demonstrates fruit image classification using a custom Convolutional Neural Network.

The model learns visual features from six fruit categories and classifies input images into their corresponding classes. Training, validation, and testing are performed to evaluate its performance.

The project also generates performance graphs and saves the best-trained CNN model for future use.

# Program 5: Fruit Image Classification Using CNN and CBAM Attention

## Overview

This project compares two Convolutional Neural Network (CNN) models for fruit image classification:

1. **Normal CNN** – A basic CNN architecture without an attention mechanism.
2. **CNN with CBAM Attention** – A CNN enhanced with the Convolutional Block Attention Module (CBAM).

The objective is to evaluate whether incorporating attention improves classification accuracy compared with a normal CNN.

The dataset contains fruit quality images, including Apple, Banana, and Lime, with Good and Bad quality categories.

The program trains both models, evaluates their performance, compares their accuracy and training time, and generates attention visualizations.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)

## Project Structure

```text
Program5_Attention_Mechanism/
│
├── dataset/
│   ├── Apple_Good/
│   ├── Apple_Bad/
│   ├── Banana_Good/
│   ├── Banana_Bad/
│   ├── Lime_Good/
│   └── Lime_Bad/
│
├── models/
│   ├── normal_cnn.pth
│   └── attention_cnn.pth
│
├── results/
│   ├── sample_fruit_images.png
│   ├── model_comparison.csv
│   ├── accuracy_comparison.png
│   ├── training_loss.png
│   └── attention_visualization.png
│
└── fruit_attention.py
```

*The dataset folder names above are examples. Use the actual class folder names from your dataset. The program automatically detects the class folders and their labels.*

## Dataset Description

The program loads fruit images from the `dataset/` directory.

- Fruit categories: Apple, Banana, and Lime.
- Quality categories: Good and Bad.
- Expected total classes: Six.
- Maximum images selected per class: 150.

The program automatically detects class folders containing images. Class names are assigned according to alphabetical folder order.

### Dataset Splitting

The selected images are divided into three sets separately for each class.

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

This splitting approach helps maintain the distribution of each class across the datasets.

## Model Architecture

### 1. Normal CNN

The normal CNN extracts image features using three convolutional blocks.

- Convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Adaptive Average Pooling
- Fully Connected Classification Layers

The model predicts the fruit quality class without using an attention mechanism.

### 2. CNN with CBAM Attention

The second model uses the same CNN feature extraction architecture with an additional CBAM attention module.

CBAM contains two attention mechanisms:

**Channel Attention**
- Identifies important feature channels.
- Uses average pooling and maximum pooling.
- Applies a shared MLP and sigmoid activation.

**Spatial Attention**
- Identifies important spatial regions in the feature maps.
- Uses channel-wise average and maximum information.
- Applies a convolution layer and sigmoid activation.

The CBAM module applies channel attention first, followed by spatial attention.

The resulting feature maps are passed to the classification layers.

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 64 × 64 |
| Batch Size | 16 |
| Number of Epochs | 5 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Cross Entropy Loss |
| Maximum Images per Class | 150 |
| Random Seed | 42 |
| Device | GPU if available, otherwise CPU |

Both models are trained separately using the same training and validation datasets.

The best-performing checkpoint for each model is saved based on validation accuracy.

## Model Evaluation

After training, both models are evaluated using the test dataset.

The program calculates:

- Test Loss
- Test Accuracy
- Training Time

The results are saved in `model_comparison.csv`.

The accuracy comparison helps determine whether the CBAM attention mechanism improves the classification performance.

## Attention Visualization

The program generates an attention-related visualization using the trained CNN with CBAM.

It displays three images:

1. **Input Fruit Image** – The original fruit image.
2. **Attention Feature Map** – A heatmap generated from the average absolute feature activations after the CBAM module.
3. **Attention Overlay** – The feature-map heatmap overlaid on the original image.

This visualization helps inspect which spatial regions have stronger feature activations. It is a feature-map visualization, not a direct display of the raw CBAM attention weights.

## Generated Results

The program automatically creates the following output files.

| File | Description |
|---|---|
| `sample_fruit_images.png` | Displays sample images from the dataset. |
| `model_comparison.csv` | Stores test loss, test accuracy, and training time for both models. |
| `accuracy_comparison.png` | Compares the test accuracy of both models. |
| `training_loss.png` | Displays training and validation loss across epochs. |
| `attention_visualization.png` | Displays the input image, attention feature map, and overlay. |

## How to Run

### Step 1: Prepare the Dataset

Extract the fruit dataset inside the `dataset/` directory.

Ensure that the class folders directly contain the corresponding fruit images. The program can also detect one additional nested dataset directory.

### Step 2: Run the Program

Execute:

```bash
python fruit_attention.py
```

### Step 3: Observe the Output

The program will:

1. Detect the dataset and class folders.
2. Select a maximum of 150 images per class.
3. Split the images into training, validation, and testing sets.
4. Train the normal CNN.
5. Train the CNN with CBAM attention.
6. Evaluate both models using the test dataset.
7. Compare their accuracy and training time.
8. Generate attention visualizations.
9. Save the trained models and experimental results.

## Expected Output

The terminal displays:

- Detected dataset classes.
- Number of selected images.
- Training, validation, and testing image counts.
- Training and validation loss for each epoch.
- Validation accuracy for both models.
- Final test accuracy and training time.
- Comparison result indicating which model achieved higher test accuracy.

**Note:** Actual accuracy and training time depend on the dataset, hardware, and training process. They are determined when the program is executed.

## Conclusion

This project demonstrates the implementation and comparison of a normal CNN and a CNN integrated with CBAM attention for fruit image classification.

By comparing their test accuracy, loss, and training time, the experiment evaluates the effect of channel and spatial attention on CNN-based fruit quality classification.
