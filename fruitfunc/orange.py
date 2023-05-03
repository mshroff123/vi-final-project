import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from fruitfunc.general import blemishes_score, background_to_white, process_image

def kiwi_score(path, display = False):

    # read image
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/' + path)
    img = process_image(img)
    img = background_to_white(img)

    reflection_score = reflection(img, display)
    print(reflection_score)

    return reflection_score


def reflection(img, display):

    # really high in fresher pears
    bright_lower = (1, 31, 67)
    bright_upper = (60, 95, 255)

    # calculate total number of pixels in object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = np.sum(gray != 255)


    # create masks
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bright_mask = cv2.inRange(hsv, bright_lower, bright_upper)

    if display:

        # display img
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # display images in masks but when displaying images show in color
        cv2.imshow('bright mask', cv2.bitwise_and(img, img, mask=bright_mask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    bright_pixels = cv2.countNonZero(bright_mask)

    bright_percent = bright_pixels / total_pixels

    return ((bright_percent+1)**2)-1


# for i in range(1, 11):
#     path = f'Day{i}/orange2.JPG'
#     kiwi_score(path, display = False)
#     print()