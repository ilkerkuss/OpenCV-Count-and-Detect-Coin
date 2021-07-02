import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import color

img = cv.imread(r"C:\Users\W10\Desktop\water_coins.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow("grey scale", gray)
cv.imshow("thresh", thresh)
cv.waitKey(0)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
cv.imshow("open", opening)
cv.waitKey(0)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
cv.imshow("dilate", sure_bg)
cv.waitKey(0)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv.imshow("dist", dist_transform)
cv.imshow("sure fg", sure_fg)
cv.waitKey(0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)



cv.imshow("unknown reg", unknown)

cv.waitKey(0)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
markers = markers.astype(np.int32)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0


markers = markers.astype(np.int32)
markers = cv.watershed(img,markers)
img[markers == -1] = [0,0,255]
img2=color.label2rgb(markers,bg_label=0)


cv.imshow('Segmentated Image',img2)
cv.imshow("thresh", sure_fg)
cv.imshow("contours", img)
cv.waitKey(0)