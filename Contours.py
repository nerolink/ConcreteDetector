import cv2 as cv
import numpy as np

#
# file = 'images/02.jpg'
# src = cv.imread(file)
# ROI = np.zeros(src.shape, np.uint8)
# pro_image = src.copy()
# pro_image = cv.cvtColor(pro_image, cv.COLOR_BGR2GRAY)
# pro_image = cv.Laplacian(pro_image, cv.CV_8U)
# pro_image, contours, hierarchy = cv.findContours(pro_image, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
# cv.imshow("pro_image", pro_image)
# cv.waitKey(0)
# cv.drawContours(ROI, contours, 1, (255, 255, 255), -1)
# ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
# ROI = cv.adaptiveThreshold(ROI, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 7)
# imgroi = cv.bitwise_and(ROI, pro_image)
# cv.waitKey(0)

file = 'images/02.jpg'
src = cv.imread(file)
empty = np.zeros(src.shape, np.uint8)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

lap = cv.Laplacian(gray, cv.CV_8U)
cv.imshow("Laplacian", lap)

threshold, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("binary", binary)

binary, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(empty, contours, -1, (0, 255, 0), 1)
cv.imshow("contours", empty)

cv.waitKey(0)
