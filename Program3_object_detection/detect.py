
# ==========================================================
# PROGRAM 3: OBJECT DETECTION USING FASTER R-CNN WITH FPN
# Dataset: COCO pretrained weights
# Framework: PyTorch and Torchvision
# Device: CPU
# ==========================================================

import torch
import torchvision

import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from pathlib import Path
from collections import Counter


# ==========================================================
# 1. PROJECT SETTINGS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input_images"
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

CONFIDENCE_THRESHOLD = 0.50

print("=" * 60)
print("FASTER R-CNN OBJECT DETECTION WITH FPN")
print("=" * 60)

print("Device:", DEVICE)


# ==========================================================
# 2. LOAD PRETRAINED FASTER R-CNN MODEL
# ==========================================================

print("\nLoading pretrained Faster R-CNN model...")

weights = (
    torchvision.models.detection
    .FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)

model = (
    torchvision.models.detection
    .fasterrcnn_resnet50_fpn(
        weights=weights
    )
)

model = model.to(DEVICE)

model.eval()

print("Pretrained model loaded successfully!")

print("\nArchitecture: Faster R-CNN with ResNet50-FPN")


# ==========================================================
# 3. LOAD COCO CLASS NAMES
# ==========================================================

CLASS_NAMES = weights.meta["categories"]

print("\nTotal available categories:", len(CLASS_NAMES))


# ==========================================================
# 4. SELECT AN IMAGE
# ==========================================================

print("\nAvailable images:")

image_files = sorted([
    f for f in INPUT_DIR.iterdir()
    if f.is_file()
    and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
])

for file in image_files:
    print("-", file.name)

if not image_files:
    raise FileNotFoundError(
        "No image found! Add a JPG or PNG image "
        "inside the input_images folder."
    )

# Select the first available image

image_path = image_files[0]

print("\nSelected image:", image_path.name)


# ==========================================================
# 5. IMAGE PREPROCESSING
# ==========================================================

try:
    image = Image.open(image_path).convert("RGB")

except Exception as error:
    raise RuntimeError(
        f"Unable to open image: {error}"
    )

original_width, original_height = image.size

print("\nImage dimensions:")
print("Width:", original_width)
print("Height:", original_height)

# Convert image into PyTorch tensor

image_tensor = to_tensor(image).to(DEVICE)


# ==========================================================
# 6. CAPTURE FEATURE PYRAMID OUTPUTS
# ==========================================================

feature_shapes = {}


def capture_fpn_features(module, inputs, output):

    if isinstance(output, dict):

        for name, feature in output.items():

            feature_shapes[name] = tuple(feature.shape)


# Register a hook to observe FPN feature maps

hook = model.backbone.register_forward_hook(
    capture_fpn_features
)


# ==========================================================
# 7. PERFORM OBJECT DETECTION
# ==========================================================

print("\nDetecting objects...")

try:

    with torch.no_grad():

        predictions = model([image_tensor])

finally:

    hook.remove()


prediction = predictions[0]

boxes = prediction["boxes"].cpu()

labels = prediction["labels"].cpu()

scores = prediction["scores"].cpu()


# ==========================================================
# 8. FILTER DETECTIONS USING CONFIDENCE
# ==========================================================

detections = []

for box, label, score in zip(boxes, labels, scores):

    confidence = score.item()

    if confidence >= CONFIDENCE_THRESHOLD:

        class_id = label.item()

        class_name = CLASS_NAMES[class_id]

        x1, y1, x2, y2 = box.tolist()

        detections.append({

            "Object": class_name,

            "Confidence (%)": round(
                confidence * 100, 2
            ),

            "X1": round(x1, 2),

            "Y1": round(y1, 2),

            "X2": round(x2, 2),

            "Y2": round(y2, 2)

        })


# ==========================================================
# 9. DRAW BOUNDING BOXES
# ==========================================================

annotated_image = image.copy()

draw = ImageDraw.Draw(annotated_image)

# Load font

try:

    font = ImageFont.truetype("arial.ttf", 18)

except OSError:

    font = ImageFont.load_default()


for item in detections:

    x1 = item["X1"]
    y1 = item["Y1"]
    x2 = item["X2"]
    y2 = item["Y2"]

    class_name = item["Object"]

    confidence = item["Confidence (%)"]

    # Draw object boundary

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=4
    )

    # Add object name and confidence

    text_label = (
        f"{class_name}: {confidence}%"
    )

    # Calculate label dimensions

    text_box = draw.textbbox(
        (0, 0),
        text_label,
        font=font
    )

    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    # Position label above bounding box
    # Keep the label inside the image

    label_x = max(0, min(x1, original_width - text_width - 4))

    label_y = max(0, y1 - text_height - 6)

    # Draw yellow label background

    draw.rectangle(
        [
            label_x,
            label_y,
            label_x + text_width + 6,
            label_y + text_height + 6
        ],
        fill="yellow"
    )

    # Draw label text

    draw.text(
        (label_x + 3, label_y + 3),
        text_label,
        fill="black",
        font=font
    )


# ==========================================================
# 10. DISPLAY DETECTION RESULTS
# ==========================================================

print("\n" + "=" * 60)

print("OBJECT DETECTION RESULTS")

print("=" * 60)

if detections:

    for item in detections:

        print(
            f"{item['Object']} - "
            f"{item['Confidence (%)']}%"
        )

else:

    print("No objects detected above the threshold.")


print("\nTotal detected objects:", len(detections))


# ==========================================================
# 11. OBJECT COUNT
# ==========================================================

object_counts = Counter(
    item["Object"] for item in detections
)

print("\nOBJECT COUNTS:")

if object_counts:

    for object_name, count in object_counts.items():

        print(object_name, ":", count)

else:

    print("No objects available for counting.")


# ==========================================================
# 12. DISPLAY FPN FEATURE MAP INFORMATION
# ==========================================================

print("\nFEATURE PYRAMID NETWORK OUTPUTS")

if feature_shapes:

    for name, shape in feature_shapes.items():

        print(
            "Feature level:",
            name,
            "| Shape:",
            shape
        )

else:

    print("FPN feature information not available.")


# ==========================================================
# 13. SAVE OUTPUT IMAGE
# ==========================================================

output_path = (
    RESULT_DIR / "detected_objects.jpg"
)

annotated_image.save(
    output_path,
    format="JPEG",
    quality=95
)

print("\nDetected image saved at:")
print(output_path)


# ==========================================================
# 14. SAVE DETECTION DETAILS
# ==========================================================

csv_path = (
    RESULT_DIR / "detection_results.csv"
)

columns = [
    "Object",
    "Confidence (%)",
    "X1",
    "Y1",
    "X2",
    "Y2"
]

df = pd.DataFrame(
    detections,
    columns=columns
)

df.to_csv(
    csv_path,
    index=False
)

print("\nDetection report saved at:")
print(csv_path)


# ==========================================================
# 15. DISPLAY FINAL IMAGE
# ==========================================================

plt.figure(figsize=(12, 8))

plt.imshow(annotated_image)

plt.title("Faster R-CNN Object Detection")

plt.axis("off")

plt.tight_layout()

visualization_path = (
    RESULT_DIR / "detection_visualization.png"
)

plt.savefig(
    visualization_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization saved at:")
print(visualization_path)


# ==========================================================
# FINAL STATUS
# ==========================================================

print("\n" + "=" * 60)

print("OBJECT DETECTION COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("\nGenerated Output Files:")

print("1. detected_objects.jpg")
print("2. detection_results.csv")
print("3. detection_visualization.png")

print("\nAll output files are stored inside the results folder.")
