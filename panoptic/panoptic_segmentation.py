
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms.functional import to_tensor

# 1. Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Load image
image = Image.open("test.jpg").convert("RGB")
image_tensor = to_tensor(image).to(device)

# 3. Load pretrained semantic segmentation model
semantic_weights = DeepLabV3_ResNet50_Weights.DEFAULT
semantic_model = deeplabv3_resnet50(
    weights=semantic_weights
).to(device).eval()

# 4. Load pretrained instance segmentation model
instance_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
instance_model = maskrcnn_resnet50_fpn(
    weights=instance_weights
).to(device).eval()

# 5. Semantic segmentation
with torch.no_grad():
    semantic_output = semantic_model(
        image_tensor.unsqueeze(0)
    )["out"]

semantic_mask = semantic_output.argmax(1)[0].cpu().numpy()

# 6. Instance segmentation
with torch.no_grad():
    instance_output = instance_model([image_tensor])[0]

# 7. Create panoptic-style map
panoptic_map = semantic_mask.copy()

# Keep only detections with confidence >= 0.5
scores = instance_output["scores"].cpu().numpy()
masks = instance_output["masks"].cpu().numpy()
labels = instance_output["labels"].cpu().numpy()

instance_id = 100

# Overlay each detected object with a unique ID
for mask, score, label in zip(masks, scores, labels):

    if score < 0.5:
        continue

    binary_mask = mask[0] > 0.5

    # Assign unique ID to detected instance
    panoptic_map[binary_mask] = instance_id
    print(
        "Instance ID:", instance_id,
        "Class:", instance_weights.meta["categories"][int(label)],
        "Confidence:", round(float(score), 3)
    )

    instance_id += 1

# 8. Display outputs
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(semantic_mask, cmap="tab20")
plt.title("Semantic Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(panoptic_map, cmap="nipy_spectral")
plt.title("Combined Panoptic-Style Map")
plt.axis("off")

plt.tight_layout()
plt.savefig("panoptic_output.png")
plt.show()

print("Panoptic-style segmentation completed.")
