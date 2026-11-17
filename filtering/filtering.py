import cv2 as cv
import numpy as np

img = cv.imread('Pic1.png')

cv.imshow('Original Image', img)

noise = np.random.normal(0, 25, img.shape)

noisy_img = img.astype(np.float32) + noise

noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

cv.imshow('Additive Noise', noisy_img)

mean3 = cv.blur(noisy_img, (3, 3))
mean5 = cv.blur(noisy_img, (5, 5))
mean7 = cv.blur(noisy_img, (7, 7))

box3 = cv.boxFilter(noisy_img, -1, (3, 3))
box5 = cv.boxFilter(noisy_img, -1, (5, 5))
box7 = cv.boxFilter(noisy_img, -1, (7, 7))

cv.imshow('Mean Filter 3x3', mean3)
cv.imshow('Mean Filter 5x5', mean5)
cv.imshow('Mean Filter 7x7', mean7)

cv.imshow('Box Filter 3x3', box3)
cv.imshow('Box Filter 5x5', box5)
cv.imshow('Box Filter 7x7', box7)

cv.waitKey(0)
cv.destroyAllWindows()