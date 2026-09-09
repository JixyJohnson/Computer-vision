import cv2 as cv
import numpy as np

img = cv.imread('bird.jpeg', cv.IMREAD_GRAYSCALE)

if img is None:
    print("Error: bird.jpg not found!")
    exit()

rows, cols = img.shape

M = np.float32([[1, 0, 100],
                [0, 1, 50]])

img_trans = cv.warpAffine(img, M, (cols, rows))

M = np.float32([[1, 0, 50],
                [0, 1, 50]])

img_trans1 = cv.warpAffine(img, M, (cols, rows))

M = np.float32([[1, 0, 50],
                [0, 1, 50]])

img_trans2 = cv.warpAffine(img, M, (200, 200))

cv.imshow('Original Image', img)

cv.imshow('Image Translated', img_trans)

cv.imshow('Image Translated Equal Size', img_trans1)

cv.imshow('Image Translated Clipped', img_trans2)


# Wait until a key is pressed
cv.waitKey(0)

# Close all windows
cv.destroyAllWindows()