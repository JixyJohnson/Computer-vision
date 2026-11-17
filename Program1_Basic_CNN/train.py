
# ============================================================
# CIFAR-10 IMAGE CLASSIFICATION USING PRETRAINED MOBILENETV3
# Framework: PyTorch
# Training: CPU
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 64

HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 5

IMAGE_SIZE = 96

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

torch.manual_seed(42)
np.random.seed(42)

print("Device:", DEVICE)


# ============================================================
# 2. IMAGE TRANSFORMATIONS
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomCrop(32, padding=4),

    transforms.RandomHorizontalFlip(),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])

test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])


# ============================================================
# 3. LOAD CIFAR-10 DATASET
# ============================================================

print("\nLoading CIFAR-10 dataset...")

train_full = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=train_transform
)

validation_full = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=test_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=test_transform
)

# Split training data into training and validation
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(train_full),
    generator=generator
).tolist()

train_indices = indices[5000:]
validation_indices = indices[:5000]

train_dataset = Subset(
    train_full,
    train_indices
)

validation_dataset = Subset(
    validation_full,
    validation_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Testing images:", len(test_dataset))


# ============================================================
# 4. LOAD PRETRAINED MOBILENETV3
# ============================================================

print("\nLoading pretrained MobileNetV3-Small...")

weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(weights=weights)

# Freeze all feature extraction layers initially
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace original ImageNet classifier with 10 CIFAR-10 classes
input_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    input_features,
    10
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()


# ============================================================
# 5. TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_accuracy = 0.0


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(loader)

    accuracy = 100 * correct / total

    return average_loss, accuracy


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_model(epochs, optimizer, phase):

    global best_accuracy

    for epoch in range(epochs):

        model.train()

        # During classifier-only training, keep frozen
        # feature extraction layers in evaluation mode.
        if phase == "HEAD":
            model.features.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)

        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate_model(
            validation_loader
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"\n{phase} Epoch [{epoch + 1}/{epochs}]"
            f"\nTraining Loss: {train_loss:.4f}"
            f"\nTraining Accuracy: {train_accuracy:.2f}%"
            f"\nValidation Loss: {val_loss:.4f}"
            f"\nValidation Accuracy: {val_accuracy:.2f}%"
        )

        # Save the best validation checkpoint
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD
                },
                MODEL_DIR / "best_cnn.pth"
            )

            print("Best model saved!")

    print("\nPhase completed:", phase)


# ============================================================
# 8. PLOT TRAINING RESULTS
# ============================================================

def plot_results():

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(epochs, train_losses, label="Training Loss")

    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.title("Loss Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.title("Accuracy Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(RESULT_DIR / "training_graph.png")

    plt.show()


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

def show_confusion_matrix():

    model.eval()

    actual = []
    predicted_all = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.numpy())

            predicted_all.extend(predictions.numpy())

    cm = confusion_matrix(
        actual,
        predicted_all,
        labels=list(range(10))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    display.plot(
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False
    )

    plt.title("MobileNetV3 Confusion Matrix")

    plt.tight_layout()

    plt.savefig(RESULT_DIR / "confusion_matrix.png")

    plt.show()


# ============================================================
# 10. SAMPLE PREDICTIONS
# ============================================================

def show_predictions():

    model.eval()

    images, labels = next(iter(test_loader))

    images = images[:10]
    labels = labels[:10]

    with torch.no_grad():

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = probabilities.max(dim=1)

    mean = torch.tensor(MEAN).view(3, 1, 1)

    std = torch.tensor(STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, 5, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        image = images[i] * std + mean

        image = image.clamp(0, 1)

        ax.imshow(image.permute(1, 2, 0))

        actual = CLASSES[labels[i].item()]

        predicted = CLASSES[predictions[i].item()]

        score = confidence[i].item() * 100

        ax.set_title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}\n"
            f"Confidence: {score:.1f}%"
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(RESULT_DIR / "predictions.png")

    plt.show()


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\nStarting classifier training...")

    # Phase 1: Train the new classification layer
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    train_model(
        HEAD_EPOCHS,
        optimizer,
        "HEAD"
    )

    # Phase 2: Fine-tune the last feature extraction blocks
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=0.0001,
        weight_decay=0.0001
    )

    train_model(
        FINETUNE_EPOCHS,
        optimizer,
        "FINE-TUNING"
    )

    # Load best checkpoint
    checkpoint = torch.load(
        MODEL_DIR / "best_cnn.pth",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nBest model loaded!")

    # Final test evaluation
    test_loss, test_accuracy = evaluate_model(test_loader)

    print("\nFINAL TEST RESULTS")

    print("Test Loss:", round(test_loss, 4))

    print("Test Accuracy:", round(test_accuracy, 2), "%")

    print(
        "Best Validation Accuracy:",
        round(best_accuracy, 2),
        "%"
    )

    print("\nGenerating graphs...")

    plot_results()

    print("\nGenerating confusion matrix...")

    show_confusion_matrix()

    print("\nGenerating sample predictions...")

    show_predictions()

    print("\nTRAINING COMPLETED SUCCESSFULLY!")
