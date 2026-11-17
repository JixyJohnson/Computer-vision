"""
PROGRAM 2: Compare YOLO26, Faster R-CNN and SSD using a labelled image set.

Metrics:
  - Mean matched bounding-box IoU
  - Bounding-box Dice (Dice calculated from rectangle regions)
  - mAP@0.50 and mAP@0.50:0.95 using TorchMetrics

Important:
  Detection models predict boxes, not pixel-accurate object masks.
  Therefore this program reports BOUNDING-BOX Dice, not segmentation-mask Dice.
  For true mask Dice, use a segmentation model and pixel-level ground-truth masks.

Dataset structure:
dataset/
  images/
    image1.jpg
    image2.jpg
  labels/
    image1.txt
    image2.txt
  classes.txt

classes.txt must contain one COCO class name per line, for example:
person
bicycle
car
dog

Each label TXT file uses YOLO format, one object per line:
class_id x_center_normalized y_center_normalized width_normalized height_normalized

The class names must match COCO names supported by all three pretrained models.
"""

from pathlib import Path
import time
import csv
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
)
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

DATASET = Path("dataset")
IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"
CLASSES_FILE = DATASET / "classes.txt"
CONFIDENCE = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not CLASSES_FILE.exists():
    raise FileNotFoundError("Create dataset/classes.txt first.")
if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
    raise FileNotFoundError("Create dataset/images and dataset/labels folders.")

class_names = [
    line.strip() for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
class_to_id = {name: i for i, name in enumerate(class_names)}

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_paths.extend(IMAGE_DIR.glob(ext))
image_paths = sorted(image_paths)

if not image_paths:
    raise FileNotFoundError("No images found in dataset/images.")

def load_ground_truth(image_path, width, height):
    """Read normalized YOLO labels and convert them to pixel xyxy boxes."""
    label_path = LABEL_DIR / (image_path.stem + ".txt")
    boxes, labels = [], []

    if not label_path.exists():
        raise FileNotFoundError(f"Missing annotation: {label_path}")

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue

        class_id = int(values[0])
        xc, yc, bw, bh = map(float, values[1:])

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        boxes.append([x1, y1, x2, y2])
        labels.append(class_id)

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }

# Load pretrained models
print("Loading models (first run may download model weights)...")
yolo = YOLO("yolo26n.pt")

frcnn_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
frcnn = fasterrcnn_resnet50_fpn(weights=frcnn_weights).to(DEVICE).eval()

ssd_weights = SSD300_VGG16_Weights.DEFAULT
ssd = ssd300_vgg16(weights=ssd_weights).to(DEVICE).eval()

frcnn_names = frcnn_weights.meta["categories"]
ssd_names = ssd_weights.meta["categories"]

def keep_known_classes(names, boxes, scores, labels):
    """Map model category names to the custom classes.txt IDs."""
    out_boxes, out_scores, out_labels = [], [], []
    for box, score, label in zip(boxes, scores, labels):
        name = names[int(label)]
        if name in class_to_id:
            out_boxes.append(box)
            out_scores.append(float(score))
            out_labels.append(class_to_id[name])

    return {
        "boxes": torch.tensor(out_boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(out_scores, dtype=torch.float32),
        "labels": torch.tensor(out_labels, dtype=torch.int64),
    }

def matched_iou_and_dice(pred, target):
    """Greedy same-class box matching; unmatched GT boxes score zero."""
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    pred_boxes = pred["boxes"]
    pred_labels = pred["labels"]
    pred_scores = pred["scores"]

    if len(gt_boxes) == 0:
        return [], []

    order = torch.argsort(pred_scores, descending=True)
    used = set()
    ious, dices = [], []

    for gi, gt_box in enumerate(gt_boxes):
        candidates = [
            pi for pi in order.tolist()
            if pi not in used and pred_labels[pi] == gt_labels[gi]
        ]

        if not candidates:
            ious.append(0.0)
            dices.append(0.0)
            continue

        candidate_boxes = pred_boxes[candidates]
        overlaps = box_iou(gt_box.reshape(1, 4), candidate_boxes)[0]
        best_local = int(torch.argmax(overlaps))
        best_index = candidates[best_local]
        used.add(best_index)

        iou = float(overlaps[best_local])
        dice = (2 * iou / (1 + iou)) if iou > 0 else 0.0
        ious.append(iou)
        dices.append(dice)

    return ious, dices

model_names = ["YOLO26", "Faster R-CNN", "SSD"]
metrics = {
    name: MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox"
    ) for name in model_names
}
summary = {name: {"ious": [], "dices": [], "seconds": []} for name in model_names}

for image_path in image_paths:
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    gt = load_ground_truth(image_path, width, height)
    image_tensor = to_tensor(pil_image).to(DEVICE)

    # YOLO26
    start = time.perf_counter()
    yolo_result = yolo.predict(
        source=str(image_path), conf=CONFIDENCE, verbose=False
    )[0]
    elapsed = time.perf_counter() - start

    y_boxes, y_scores, y_labels = [], [], []
    if yolo_result.boxes is not None:
        for box in yolo_result.boxes:
            name = yolo_result.names[int(box.cls[0])]
            if name in class_to_id:
                y_boxes.append(box.xyxy[0].cpu().tolist())
                y_scores.append(float(box.conf[0]))
                y_labels.append(class_to_id[name])

    y_pred = {
        "boxes": torch.tensor(y_boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(y_scores, dtype=torch.float32),
        "labels": torch.tensor(y_labels, dtype=torch.int64),
    }
    predictions = {"YOLO26": y_pred}
    summary["YOLO26"]["seconds"].append(elapsed)

    # Faster R-CNN
    start = time.perf_counter()
    with torch.no_grad():
        output = frcnn([image_tensor])[0]
    elapsed = time.perf_counter() - start
    f_pred = keep_known_classes(
        frcnn_names, output["boxes"].cpu(), output["scores"].cpu(),
        output["labels"].cpu()
    )
    keep = f_pred["scores"] >= CONFIDENCE
    f_pred = {k: v[keep] for k, v in f_pred.items()}
    predictions["Faster R-CNN"] = f_pred
    summary["Faster R-CNN"]["seconds"].append(elapsed)

    # SSD
    start = time.perf_counter()
    with torch.no_grad():
        output = ssd([image_tensor])[0]
    elapsed = time.perf_counter() - start
    s_pred = keep_known_classes(
        ssd_names, output["boxes"].cpu(), output["scores"].cpu(),
        output["labels"].cpu()
    )
    keep = s_pred["scores"] >= CONFIDENCE
    s_pred = {k: v[keep] for k, v in s_pred.items()}
    predictions["SSD"] = s_pred
    summary["SSD"]["seconds"].append(elapsed)

    # Update metrics and matched overlap scores
    target_for_metric = {
        "boxes": gt["boxes"], "labels": gt["labels"]
    }
    for name, pred in predictions.items():
        metrics[name].update([pred], [target_for_metric])
        ious, dices = matched_iou_and_dice(pred, gt)
        summary[name]["ious"].extend(ious)
        summary[name]["dices"].extend(dices)

# Save report
rows = []
print("\nMODEL COMPARISON")
print("-" * 82)
print(f"{'Model':<16} {'Mean IoU':>10} {'BBox Dice':>11} {'mAP50':>10} {'mAP50:95':>11} {'Time(s)':>10}")
print("-" * 82)

for name in model_names:
    result = metrics[name].compute()
    mean_iou = float(np.mean(summary[name]["ious"])) if summary[name]["ious"] else 0.0
    mean_dice = float(np.mean(summary[name]["dices"])) if summary[name]["dices"] else 0.0
    map50 = float(result["map_50"])
    map_all = float(result["map"])
    avg_time = float(np.mean(summary[name]["seconds"]))

    print(
        f"{name:<16} {mean_iou:>10.4f} {mean_dice:>11.4f} "
        f"{map50:>10.4f} {map_all:>11.4f} {avg_time:>10.4f}"
    )
    rows.append([name, mean_iou, mean_dice, map50, map_all, avg_time])

with open("model_comparison_metrics.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Model", "Mean_Bounding_Box_IoU", "Bounding_Box_Dice",
        "mAP50", "mAP50_95", "Average_Inference_Seconds"
    ])
    writer.writerows(rows)

print("\nSaved: model_comparison_metrics.csv")
print("Use the same images, annotations, class names and confidence threshold for all models.")
