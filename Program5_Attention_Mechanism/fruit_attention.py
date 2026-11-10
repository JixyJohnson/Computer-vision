
"""
FRUIT IMAGE CLASSIFICATION
Normal CNN vs CNN with CBAM Attention

Dataset: Fruit Quality - Apple, Banana and Lime
Classes: Good and Bad for each fruit

Features:
1. Dataset loading
2. Limited dataset selection
3. Train/validation/test split
4. Normal CNN
5. CNN with CBAM Attention
6. Model training and evaluation
7. Accuracy comparison
8. Loss graph
9. Attention visualization
10. Save models and results
"""

import os
import csv
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =====================================================
# 1. SETTINGS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_ROOT = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

MAX_IMAGES_PER_CLASS = 150

RANDOM_SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("=" * 60)
print("FRUIT CLASSIFICATION USING CNN AND ATTENTION")
print("=" * 60)

print("Device:", DEVICE)
print("Dataset location:", DATASET_ROOT)


# =====================================================
# 2. DATASET CHECK
# =====================================================

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        "Dataset folder not found!\n"
        "Please extract the fruit dataset into:\n"
        f"{DATASET_ROOT}"
    )

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def contains_images(folder):

    return any(
        file.suffix.lower() in IMAGE_EXTENSIONS
        for file in folder.iterdir()
        if file.is_file()
    )


# Detect the folder that directly contains class folders.
def find_dataset_root(folder):

    candidates = [folder]

    candidates.extend(
        item for item in folder.iterdir()
        if item.is_dir()
    )

    for candidate in candidates:

        class_folders = [
            item for item in candidate.iterdir()
            if item.is_dir() and contains_images(item)
        ]

        if len(class_folders) >= 2:
            return candidate

    return None


detected_root = find_dataset_root(DATASET_ROOT)

if detected_root is None:

    raise RuntimeError(
        "Could not find fruit class folders. "
        "Check your extracted dataset structure."
    )

print("\nDataset found:", detected_root)


# =====================================================
# 3. LOAD DATASET IMAGES
# =====================================================

class_folders = sorted(
    [
        folder for folder in detected_root.iterdir()
        if folder.is_dir() and contains_images(folder)
    ],
    key=lambda folder: folder.name
)

CLASS_NAMES = [
    folder.name for folder in class_folders
]

NUM_CLASSES = len(CLASS_NAMES)

class_to_index = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

samples = []

for folder in class_folders:

    image_files = [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(image_files)

    selected_images = image_files[:MAX_IMAGES_PER_CLASS]

    for image_path in selected_images:

        label = class_to_index[folder.name]

        samples.append((image_path, label))


print("\nDetected Classes:", CLASS_NAMES)
print("Number of Classes:", NUM_CLASSES)
print("Total Selected Images:", len(samples))

if NUM_CLASSES < 2 or len(samples) < 20:
    raise RuntimeError(
        "Not enough classes or images were found."
    )


# =====================================================
# 4. DATASET SPLITTING
# =====================================================

# Split each class separately to maintain class balance.

train_samples = []
validation_samples = []
test_samples = []

for class_index in range(NUM_CLASSES):

    class_samples = [
        sample for sample in samples
        if sample[1] == class_index
    ]

    random.shuffle(class_samples)

    total = len(class_samples)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_samples.extend(class_samples[:train_end])

    validation_samples.extend(
        class_samples[train_end:validation_end]
    )

    test_samples.extend(
        class_samples[validation_end:]
    )


print("\nDataset Split:")
print("Training:", len(train_samples))
print("Validation:", len(validation_samples))
print("Testing:", len(test_samples))


# =====================================================
# 5. IMAGE TRANSFORMATIONS
# =====================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

])


test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

])


# =====================================================
# 6. CUSTOM DATASET CLASS
# =====================================================

class FruitDataset(Dataset):

    def __init__(self, samples, transform):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        with Image.open(image_path) as image:

            image = image.convert("RGB")

        image = self.transform(image)

        return image, label


train_dataset = FruitDataset(
    train_samples,
    train_transform
)

validation_dataset = FruitDataset(
    validation_samples,
    test_transform
)

test_dataset = FruitDataset(
    test_samples,
    test_transform
)


# =====================================================
# 7. DATA LOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =====================================================
# 8. SAVE SAMPLE IMAGES
# =====================================================

def save_sample_images():

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for ax in axes.flat:
        ax.axis("off")

    for index in range(min(8, len(samples))):

        image_path, label = samples[index]

        with Image.open(image_path) as image:

            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        axes.flat[index].imshow(image)

        axes.flat[index].set_title(
            CLASS_NAMES[label]
        )

        axes.flat[index].axis("off")

    plt.suptitle("Fruit Dataset Samples")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "sample_fruit_images.png"
    )

    plt.close()


save_sample_images()


# =====================================================
# 9. CHANNEL ATTENTION
# =====================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=8):

        super().__init__()

        hidden = max(channels // reduction, 4)

        self.shared_mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=False
            )

        )

    def forward(self, x):

        average = F.adaptive_avg_pool2d(x, 1)

        maximum = F.adaptive_max_pool2d(x, 1)

        attention = torch.sigmoid(
            self.shared_mlp(average)
            +
            self.shared_mlp(maximum)
        )

        return x * attention


# =====================================================
# 10. SPATIAL ATTENTION
# =====================================================

class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

    def forward(self, x):

        average = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        maximum = torch.max(
            x,
            dim=1,
            keepdim=True
        )[0]

        combined = torch.cat(
            [average, maximum],
            dim=1
        )

        attention = torch.sigmoid(
            self.conv(combined)
        )

        return x * attention


# =====================================================
# 11. CBAM ATTENTION MODULE
# =====================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = ChannelAttention(channels)

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


# =====================================================
# 12. CNN MODEL
# =====================================================

class CNNModel(nn.Module):

    def __init__(self, use_attention=False):

        super().__init__()

        self.use_attention = use_attention

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        if use_attention:

            self.attention = CBAM(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, NUM_CLASSES)

        )

    def forward(self, x):

        x = self.features(x)

        if self.use_attention:

            x = self.attention(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


# =====================================================
# 13. MODEL EVALUATION
# =====================================================

def evaluate_model(model, loader, criterion):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / max(total, 1)

    accuracy = correct / max(total, 1)

    return average_loss, accuracy


# =====================================================
# 14. MODEL TRAINING
# =====================================================

def train_model(model, model_name):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    best_accuracy = -1

    checkpoint = MODEL_DIR / f"{model_name}.pth"

    start_time = time.time()

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

        train_loss = (
            running_loss / len(train_dataset)
        )

        validation_loss, validation_accuracy = evaluate_model(
            model,
            validation_loader,
            criterion
        )

        history["train_loss"].append(train_loss)

        history["validation_loss"].append(validation_loss)

        history["validation_accuracy"].append(
            validation_accuracy
        )

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                checkpoint
            )

        print(
            f"{model_name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Validation Loss: {validation_loss:.4f} "
            f"| Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

    training_time = time.time() - start_time

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE
        )
    )

    return model, history, training_time


# =====================================================
# 15. TRAIN NORMAL CNN
# =====================================================

print("\n" + "=" * 60)

print("TRAINING NORMAL CNN")

print("=" * 60)

normal_cnn = CNNModel(
    use_attention=False
)

normal_cnn, normal_history, normal_time = train_model(
    normal_cnn,
    "normal_cnn"
)


# =====================================================
# 16. TRAIN CNN WITH ATTENTION
# =====================================================

print("\n" + "=" * 60)

print("TRAINING CNN WITH CBAM ATTENTION")

print("=" * 60)

attention_cnn = CNNModel(
    use_attention=True
)

attention_cnn, attention_history, attention_time = train_model(
    attention_cnn,
    "attention_cnn"
)


# =====================================================
# 17. FINAL TEST EVALUATION
# =====================================================

criterion = nn.CrossEntropyLoss()

normal_loss, normal_accuracy = evaluate_model(
    normal_cnn,
    test_loader,
    criterion
)

attention_loss, attention_accuracy = evaluate_model(
    attention_cnn,
    test_loader,
    criterion
)


# =====================================================
# 18. SAVE COMPARISON CSV
# =====================================================

with open(
    RESULTS_DIR / "model_comparison.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Test Loss",
        "Test Accuracy",
        "Training Time"
    ])

    writer.writerow([
        "Normal CNN",
        normal_loss,
        normal_accuracy * 100,
        normal_time
    ])

    writer.writerow([
        "CNN with Attention",
        attention_loss,
        attention_accuracy * 100,
        attention_time
    ])


# =====================================================
# 19. ACCURACY COMPARISON GRAPH
# =====================================================

model_names = [
    "Normal CNN",
    "CNN with Attention"
]

accuracies = [
    normal_accuracy * 100,
    attention_accuracy * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    model_names,
    accuracies
)

plt.title("CNN vs CNN with Attention")

plt.ylabel("Test Accuracy (%)")

plt.ylim(0, 100)

for bar, value in zip(bars, accuracies):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_comparison.png"
)

plt.close()


# =====================================================
# 20. TRAINING LOSS GRAPH
# =====================================================

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    normal_history["train_loss"],
    marker="o",
    label="Normal CNN Training"
)

plt.plot(
    epochs,
    attention_history["train_loss"],
    marker="o",
    label="Attention CNN Training"
)

plt.plot(
    epochs,
    normal_history["validation_loss"],
    marker="s",
    linestyle="--",
    label="Normal CNN Validation"
)

plt.plot(
    epochs,
    attention_history["validation_loss"],
    marker="s",
    linestyle="--",
    label="Attention CNN Validation"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "training_loss.png"
)

plt.close()


# =====================================================
# 21. ATTENTION VISUALIZATION
# =====================================================

def save_attention_visualization(model):

    model.eval()

    image_path, actual_label = test_samples[0]

    with Image.open(image_path) as image:

        original_image = image.convert("RGB")

    tensor_image = test_transform(
        original_image
    ).unsqueeze(0).to(DEVICE)

    captured = {}

    def attention_hook(module, inputs, output):

        captured["features"] = output.detach()

    hook = model.attention.register_forward_hook(
        attention_hook
    )

    with torch.no_grad():

        prediction = model(
            tensor_image
        ).argmax(dim=1).item()

    hook.remove()

    feature_map = captured["features"][0]

    attention_map = (
        feature_map.abs()
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    original_array = np.array(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(12, 4)
    )

    axes[0].imshow(original_array)

    axes[0].set_title("Input Fruit Image")

    axes[0].axis("off")

    axes[1].imshow(
        attention_map,
        cmap="hot"
    )

    axes[1].set_title("Attention Feature Map")

    axes[1].axis("off")

    axes[2].imshow(original_array)

    axes[2].imshow(
        attention_map,
        cmap="jet",
        alpha=0.45,
        extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0)
    )

    axes[2].set_title("Attention Overlay")

    axes[2].axis("off")

    plt.suptitle(
        f"Actual: {CLASS_NAMES[actual_label]} | "
        f"Predicted: {CLASS_NAMES[prediction]}"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "attention_visualization.png"
    )

    plt.close()


save_attention_visualization(attention_cnn)


# =====================================================
# 22. FINAL OUTPUT
# =====================================================

print("\n" + "=" * 60)

print("EXPERIMENT COMPLETED SUCCESSFULLY")

print("=" * 60)

print("\nNumber of Classes:", NUM_CLASSES)

print("Classes:", CLASS_NAMES)

print(
    f"\nNormal CNN Accuracy: "
    f"{normal_accuracy * 100:.2f}%"
)

print(
    f"CNN with Attention Accuracy: "
    f"{attention_accuracy * 100:.2f}%"
)

print(
    f"Normal CNN Training Time: "
    f"{normal_time:.2f} seconds"
)

print(
    f"Attention CNN Training Time: "
    f"{attention_time:.2f} seconds"
)

if attention_accuracy > normal_accuracy:

    print(
        "\nResult: CNN with Attention performed better."
    )

elif attention_accuracy < normal_accuracy:

    print(
        "\nResult: Normal CNN performed better."
    )

else:

    print(
        "\nResult: Both models achieved equal accuracy."
    )


# =====================================================
# 23. GENERATED OUTPUT FILES
# =====================================================

output_files = [

    RESULTS_DIR / "sample_fruit_images.png",

    RESULTS_DIR / "model_comparison.csv",

    RESULTS_DIR / "accuracy_comparison.png",

    RESULTS_DIR / "training_loss.png",

    RESULTS_DIR / "attention_visualization.png",

    MODEL_DIR / "normal_cnn.pth",

    MODEL_DIR / "attention_cnn.pth"

]

print("\nGenerated Files:")

for file in output_files:

    if file.exists():

        print("[SAVED]", file)

    else:

        print("[MISSING]", file)


print("\nModels Folder:", MODEL_DIR)

print("Results Folder:", RESULTS_DIR)

print("\nAll available outputs are saved.")

print("=" * 60)
