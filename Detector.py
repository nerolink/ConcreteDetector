import cv2 as cv
import numpy as np
import itertools

kernel = np.ones((2, 2), np.uint8)


def isolated(mat, col, row):
    """
    去除单点
    :param mat:
    :param col:
    :param row:
    :return:
    """
    for i in range(col - 1, col + 2):
        for j in range(row - 1, row + 2):
            if i == col and j == row:
                continue
            if mat[j, i] != 255:
                return False
    return True


def wipe_off(mat):
    """
    遍历图片，去除单点
    :param mat:
    :return:
    """
    for i in range(1, mat.shape[0] - 1):
        for j in range(1, mat.shape[1] - 1):
            if mat[i, j] == 0 and isolated(mat, j, i):
                mat[i, j] = 255
    return mat


def to_rgb(mat):
    """
    把二值图转成rgb图
    :param mat:
    :return:
    """
    s = mat.shape
    mat = mat.reshape((-1, 1))
    g = np.ones(mat.shape) * 255
    b = np.ones(mat.shape) * 255
    mat = np.hstack((g, mat, b))
    mat = np.reshape(mat, newshape=(s[0], s[1], 3))
    return mat


def cover(mask, src):
    """
    把mask和原图合并
    :param mask:
    :param src:
    :return:
    """
    s = src.shape
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if mask[i, j, 1] == 0:
                src[i, j, 0] = 255
                src[i, j, 1] = 0
                src[i, j, 2] = 255
    return src


rgb_img = cv.imread("./images/05.jpg")
gray_rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)

# gray_rgb_img = cv.morphologyEx(gray_rgb_img, cv.MORPH_CLOSE, kernel=kernel)
cv.adaptiveThreshold(gray_rgb_img, dst=gray_rgb_img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                     thresholdType=cv.THRESH_BINARY, blockSize=5, C=10)
gray_rgb_img = wipe_off(gray_rgb_img)

canny = cv.Canny(rgb_img, 120, 180)
cv.bitwise_not(canny, canny)
canny = to_rgb(canny)

rgb_img = cover(canny, rgb_img)

closing = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel=kernel)
cv.imshow("canny", canny)
cv.imshow("prototype", rgb_img)
cv.imshow("gray", gray_rgb_img)
cv.imshow("closing", closing)
cv.waitKey(0)
cv.destroyAllWindows()
