import cv2 as cv

img = cv.imread('leaf.png')

print(img.shape)

w = 512
r = 512.0 / img.shape[1]

h = int(r * img.shape[0])

resize1 = cv.resize(img, (w, h),
                    interpolation=cv.INTER_LINEAR)

resize2 = cv.resize(img, (200, 200),
                    interpolation=cv.INTER_LINEAR)

resize3 = cv.resize(img, (0, 0),
                    fx=2, fy=0.5,
                    interpolation=cv.INTER_LINEAR)

resize4 = cv.resize(resize2, (1024, 1024),
                    interpolation=cv.INTER_LINEAR)

cv.imshow('Original', img)
cv.imshow('Resize 1 - width 512', resize1)
cv.imshow('Resize 2 - 200x200', resize2)
cv.imshow('Resize 3 - fx=2 fy=0.5', resize3)
cv.imshow('Resize 4 - 1024x1024', resize4)

cv.waitKey(0)
cv.destroyAllWindows()