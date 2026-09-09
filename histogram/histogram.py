import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Pic2.png', cv.IMREAD_GRAYSCALE)

cv.imshow('img', img)

cv.waitKey(0)
cv.destroyAllWindows()

h = cv.calcHist([img], [0], None, [256], [0, 256])

plt.plot(h)
plt.show()