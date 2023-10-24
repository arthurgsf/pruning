import cv2 as cv
import numpy as np

def norm(alpha=0, beta=1):
    def n(x, y):
        x = cv.normalize(x, None, alpha = alpha, beta = beta, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
        return x, y
    return n

def resize(target_shape):
    def r(image, mask):
        image = cv.resize(image, target_shape)
        mask = cv.resize(mask, target_shape, interpolation=cv.INTER_NEAREST)
        
        return image, mask
    return r

def expand_dims():
    def e(x, y):
        return np.expand_dims(x, -1), np.expand_dims(y, -1)
    return e

def windowing(lower_bound, upper_bound):
    def w(image, mask):
        return np.clip(image, lower_bound, upper_bound), mask
    return w