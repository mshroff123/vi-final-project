import cv2
import matplotlib.pyplot as plt
import numpy as np

from general import blemishes_score, background_to_white, process_image

LOW = np.array([10, 100, 10])
UP = np.array([18, 255, 255])

def banana_score(path, LOW, UP):

    score = 0

    # read image
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/' + path)

    # process image
    img = process_image(img)

    # convert background to white
    img = background_to_white(img)

    # get blemish score
    score += blemishes_score(img, LOW, UP)

    return score


banana_score('Day7/banana2.JPG', LOW, UP)

