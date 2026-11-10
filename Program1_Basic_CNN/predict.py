
# ============================================================
# IMAGE PREDICTION USING TRAINED MOBILENETV3
# ============================================================

import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small

from torchvision import transforms

from PIL import Image

from pathlib import Path

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_cnn.pth"

DEVICE = torch.device("cpu")


# ============================================================
# 2. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

CLASSES = checkpoint["classes"]

IMAGE_SIZE = checkpoint["image_size"]

MEAN = checkpoint["mean"]

STD = checkpoint["std"]


# ============================================================
# 3. REBUILD MODEL
# ============================================================

model = mobilenet_v3_small(weights=None)

input_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    input_features,
    len(CLASSES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# 4. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )

])


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        print("Error: Image not found!")

        return

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = probabilities.max(dim=1)

    predicted_class = CLASSES[predicted.item()]

    confidence_score = confidence.item() * 100

    top_values, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=1
    )

    # Display results
    print("\n========== CNN PREDICTION ==========")

    print("Image:", image_path.name)

    print("Predicted category:", predicted_class)

    print("Confidence:", round(confidence_score, 2), "%")

    print("\nTop 3 predictions:")

    for score, index in zip(
        top_values[0],
        top_indices[0]
    ):

        class_name = CLASSES[index.item()]

        percentage = score.item() * 100

        print(
            class_name,
            "-",
            round(percentage, 2),
            "%"
        )

    print("====================================")


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    image_name = input(
        "Enter image path (example: dog.jpg): "
    ).strip().strip('"')

    predict_image(image_name)
