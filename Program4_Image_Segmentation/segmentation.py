
import os
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

MAX_IMAGES_PER_CLASS = 150

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Device:", DEVICE)


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}\n"
        "Place your extracted dataset inside data/fruit_dataset."
    )

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# Load dataset
full_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH)
)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)

print("\nFruit classes:", class_names)

if NUM_CLASSES != 6:
    raise ValueError(
        f"Expected 6 fruit classes, but found {NUM_CLASSES}. "
        "Check your dataset folders."
    )


# Select limited images from each class
class_indices = {i: [] for i in range(NUM_CLASSES)}

for index, (_, label) in enumerate(full_dataset.samples):
    class_indices[label].append(index)

selected_indices = []

for class_id, indices in class_indices.items():

    random.shuffle(indices)

    selected = indices[:MAX_IMAGES_PER_CLASS]

    selected_indices.extend(selected)

    print(
        class_names[class_id],
        "images selected:",
        len(selected)
    )


# Split images class-wise into train, validation and test
train_indices = []
validation_indices = []
test_indices = []

for class_id in range(NUM_CLASSES):

    indices = [
        i for i in selected_indices
        if full_dataset.targets[i] == class_id
    ]

    random.shuffle(indices)

    total = len(indices)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_indices.extend(indices[:train_end])

    validation_indices.extend(
        indices[train_end:validation_end]
    )

    test_indices.extend(
        indices[validation_end:]
    )


# Create datasets with separate transformations
train_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH),
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH),
    transform=validation_transform
)

test_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH),
    transform=validation_transform
)

train_subset = Subset(train_dataset, train_indices)
validation_subset = Subset(validation_dataset, validation_indices)
test_subset = Subset(test_dataset, test_indices)


train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTraining images:", len(train_subset))
print("Validation images:", len(validation_subset))
print("Testing images:", len(test_subset))


# --------------------------------------------------
# 3. CNN MODEL
# --------------------------------------------------

class FruitCNN(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = FruitCNN(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nFruit CNN model created successfully!")


# --------------------------------------------------
# 4. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0

    average_loss = (
        total_loss / len(loader) if len(loader) > 0 else 0
    )

    return accuracy, average_loss


# --------------------------------------------------
# 5. TRAINING
# --------------------------------------------------

train_losses = []
validation_accuracies = []

best_accuracy = -1.0

MODEL_PATH = MODEL_DIR / "best_cnn.pth"

print("\nStarting Fruit CNN training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    validation_accuracy, validation_loss = evaluate(
        model,
        validation_loader
    )

    train_losses.append(epoch_loss)
    validation_accuracies.append(validation_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Training Loss: {epoch_loss:.4f} "
        f"Validation Loss: {validation_loss:.4f} "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )

    # Save the best model
    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved successfully!")


print("\nTraining completed!")


# --------------------------------------------------
# 6. TESTING
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

test_accuracy, test_loss = evaluate(
    model,
    test_loader
)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))


# --------------------------------------------------
# 7. SAVE TRAINING RESULTS
# --------------------------------------------------

with open(
    RESULT_DIR / "training_metrics.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Training Loss",
        "Validation Accuracy"
    ])

    for i in range(EPOCHS):

        writer.writerow([
            i + 1,
            train_losses[i],
            validation_accuracies[i]
        ])


# --------------------------------------------------
# 8. TRAINING GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Fruit CNN Training Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    RESULT_DIR / "training_loss.png"
)

plt.close()


# --------------------------------------------------
# 9. ACCURACY GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    validation_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Fruit CNN Validation Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    RESULT_DIR / "validation_accuracy.png"
)

plt.close()


# --------------------------------------------------
# 10. FINAL OUTPUT
# --------------------------------------------------

print("\n--------------------------------")
print("FRUIT QUALITY CLASSIFICATION")
print("--------------------------------")

print("Fruit classes:", class_names)

print(
    "Best validation accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)

print(
    "Test accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("\nModel saved at:", MODEL_PATH)

print("Results saved inside:", RESULT_DIR)

print("\nAll operations completed successfully!")
