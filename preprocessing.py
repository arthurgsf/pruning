import cv2 as cv
import numpy as np
import albumentations as A
from typing import Callable

class Pipeline:
    def __init__(self, functions:list=[]):
        self.functions = functions

    def add(self, f:Callable):
        self.functions.append(f)
    
    def apply(self, x, y):
        for fn in self.functions:
            x, y = fn(x, y)
        return x, y

def norm(x, y):
    x = cv.normalize(x, None, alpha = 0, beta = 1, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    return x, y

def augment(image, mask):
    transform = A.Compose([
        A.Rotate(limit=(-45, 45), p = 0.5),
    ], p=1)

    data = {"image":image, "mask":mask}
    augmented = transform(**data)
    return augmented["image"], augmented["mask"]


def resize(target_shape):
    def r(image, mask):
        image = cv.resize(image, target_shape)
        mask = cv.resize(mask, target_shape, interpolation=cv.INTER_NEAREST)
        return image, mask
    return r

def expand_dims(x, y):
    return np.expand_dims(x, -1), np.expand_dims(y, -1)

def windowing(lower_bound, upper_bound):
    def w(image, mask):
        return np.clip(image, lower_bound, upper_bound), mask
    return w