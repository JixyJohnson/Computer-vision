import cv2 as cv

img = cv.imread('Pic3.png')

print(img.shape)   


w = 512

r = 512.0 / img.shape[1]
h = int(r * img.shape[0])

print(h)

resize1 = cv.resize(img, (w, h), interpolation=cv.INTER_NEAREST)

resize2 = cv.resize(img, (200, 200), interpolation=cv.INTER_NEAREST)

resize3 = cv.resize(img, (0, 0), fx=2, fy=0.5,
                    interpolation=cv.INTER_NEAREST)

resize4 = cv.resize(resize2, (1024, 1024),
                    interpolation=cv.INTER_NEAREST)

cv.imshow('original', img)
cv.imshow('resize w=512', resize1)
cv.imshow('resize dsize=(200,200)', resize2)
cv.imshow('resize fx=2 fy=0.5', resize3)
cv.imshow('resize resize2 to 1024x1024', resize4)

cv.waitKey(0)
cv.destroyAllWindows()