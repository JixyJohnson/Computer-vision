import cv2 as cv
import numpy as np

img = cv.imread('lowcontrast.jpeg', cv.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found!")
    exit()

print("Original image shape:", img.shape)

c = np.min(img)
d = np.max(img)
a = 0
b = 255

s = (b - a) / (d - c)

newimg = ((img - c) * s + a).astype('uint8')

print("Enhanced image shape:", newimg.shape)

cv.imshow('Original Image', img)
cv.imshow('Enhanced Image', newimg)

cv.waitKey(0)
cv.destroyAllWindows()