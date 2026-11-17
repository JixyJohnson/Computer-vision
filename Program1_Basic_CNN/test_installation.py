
import torch
import torchvision
import matplotlib
import sklearn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

print("======================================")
print("     COMPUTER VISION SETUP TEST")
print("======================================")

# Check PyTorch
print("\nPyTorch:", torch.__version__)

# Check Torchvision
print("Torchvision:", torchvision.__version__)

# Check Matplotlib
print("Matplotlib:", matplotlib.__version__)

# Check Scikit-learn
print("Scikit-learn:", sklearn.__version__)

# Check GPU
print("GPU available:", torch.cuda.is_available())

# Check CPU
print("CPU available:", True)

# Test MobileNetV3 architecture
print("\nTesting MobileNetV3-Small...")

try:
    model = mobilenet_v3_small(weights=None)

    print("MobileNetV3-Small architecture loaded successfully!")

except Exception as e:
    print("MobileNetV3 loading failed:", e)

# Test pretrained weights availability
print("\nChecking pretrained weights...")

try:
    weights = MobileNet_V3_Small_Weights.DEFAULT

    print("Pretrained weights configuration available!")

except Exception as e:
    print("Pretrained weights check failed:", e)

print("\n======================================")
print("       SETUP TEST COMPLETED")
print("======================================")
