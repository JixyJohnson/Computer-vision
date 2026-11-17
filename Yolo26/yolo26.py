"""
PROGRAM 1: YOLO26 object detection, instance segmentation and object-size measurement.

Input:
  Put one or more JPG/PNG images inside input_images/

Outputs:
  output/detection/  - bounding-box detection images
  output/segmentation/ - instance-segmentation images
  output/object_measurements.csv - width/height/area in pixels

For real-world centimetres, set PIXELS_PER_CM after camera calibration.
A single ordinary image cannot provide reliable physical size without a
known reference object, camera calibration, or depth information.
"""

from pathlib import Path
import csv
from ultralytics import YOLO

INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
CONFIDENCE = 0.40

# Example: if a calibration board shows that 100 pixels = 5 cm,
# PIXELS_PER_CM = 20.0. Leave as None if not calibrated.
PIXELS_PER_CM = None

INPUT_DIR.mkdir(exist_ok=True)
image_paths = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_paths.extend(INPUT_DIR.glob(pattern))

if not image_paths:
    raise FileNotFoundError(
        "No images found. Put JPG/PNG images inside the input_images folder."
    )

# Latest released Ultralytics family at the time this program was prepared.
# The first run downloads the pretrained weights automatically.
detection_model = YOLO("yolo26n.pt")
segmentation_model = YOLO("yolo26n-seg.pt")

detection_dir = OUTPUT_DIR / "detection"
segmentation_dir = OUTPUT_DIR / "segmentation"
detection_dir.mkdir(parents=True, exist_ok=True)
segmentation_dir.mkdir(parents=True, exist_ok=True)

rows = []

for image_path in image_paths:
    print(f"\nProcessing: {image_path.name}")

    # 1. Object detection with bounding boxes
    detection_result = detection_model.predict(
        source=str(image_path), conf=CONFIDENCE, verbose=False
    )[0]
    detection_result.save(filename=str(detection_dir / image_path.name))

    if detection_result.boxes is not None:
        for box in detection_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = detection_result.names[class_id]

            width_px = max(0.0, x2 - x1)
            height_px = max(0.0, y2 - y1)
            area_px = width_px * height_px

            width_cm = (
                width_px / PIXELS_PER_CM
                if PIXELS_PER_CM else ""
            )
            height_cm = (
                height_px / PIXELS_PER_CM
                if PIXELS_PER_CM else ""
            )

            rows.append([
                image_path.name, class_name, round(confidence, 4),
                round(width_px, 2), round(height_px, 2),
                round(area_px, 2), width_cm, height_cm
            ])

            print(
                f"{class_name}: width={width_px:.1f}px, "
                f"height={height_px:.1f}px, confidence={confidence:.2f}"
            )

    # 2. Instance segmentation with YOLO26-seg
    segmentation_result = segmentation_model.predict(
        source=str(image_path), conf=CONFIDENCE, verbose=False
    )[0]
    segmentation_result.save(
        filename=str(segmentation_dir / image_path.name)
    )

    # Pixel area of each predicted instance mask
    if segmentation_result.masks is not None:
        masks = segmentation_result.masks.data.cpu().numpy()
        for index, mask in enumerate(masks):
            mask_area = int(mask.sum())
            print(f"Segment {index + 1}: mask area={mask_area} pixels")

# 3. Save measurements
csv_path = OUTPUT_DIR / "object_measurements.csv"
with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "image", "object", "confidence", "width_pixels",
        "height_pixels", "bounding_box_area_pixels",
        "width_cm_if_calibrated", "height_cm_if_calibrated"
    ])
    writer.writerows(rows)

print("\nCompleted.")
print("Detection images:", detection_dir)
print("Segmentation images:", segmentation_dir)
print("Measurements CSV:", csv_path)
