import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from fruitfunc.general import blemishes_score, background_to_white, process_image
LOW = np.array([10, 100, 10])
UP = np.array([18, 255, 255])

def banana_score(path, display = False):

    # read image
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/' + path)
    img = process_image(img)

    img = background_to_white(img)

    # get blemish score
    blemish_score = blemishes_score(img, LOW, UP)
    blemish_score = min(blemish_score * 3, 1)

    # take blemish value add it to 1 and square it
    blemish_score = (blemish_score + 1) ** 2

    return min(blemish_score-1, 1)


# for i in range(1, 11):
#     path = f'Day{i}/banana2.JPG'
#     banana_score(path, False)
#     print()

