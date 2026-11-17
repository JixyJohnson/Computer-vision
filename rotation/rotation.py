import cv2 as cv
img = cv.imread('bird.jpeg')

if img is None:
    print("Error: bird.jpeg not found!")
    exit()
rows, cols, d = img.shape

M = cv.getRotationMatrix2D(
    ((cols - 1) / 2.0, (rows - 1) / 2.0),
    90,
    1
)

img_rot1 = cv.warpAffine(img, M, (cols, rows))

M = cv.getRotationMatrix2D(
    ((cols - 1) / 2.0, (rows - 1) / 2.0),
    45,
    1
)

img_rot2 = cv.warpAffine(img, M, (cols, rows))
M = cv.getRotationMatrix2D(
    ((cols - 1) / 2.0, (rows - 1) / 2.0),
    90,
    0.5
)

img_rot3 = cv.warpAffine(img, M, (cols, rows))

cv.imshow('Original Image', img)

cv.imshow('Image Rotated 90 Degrees', img_rot1)

cv.imshow('Image Rotated 45 Degrees', img_rot2)

cv.imshow('Image Rotated and Scaled', img_rot3)
cv.waitKey(0)
cv.destroyAllWindows()