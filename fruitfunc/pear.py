import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from fruitfunc.general import blemishes_score, background_to_white, process_image

LOW = np.array([10, 100, 10])
UP = np.array([23, 255, 255])

def pear_score(path, display = False):

    score = 0

    # read image
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/' + path)
    img = process_image(img)
    img = background_to_white(img)
    
    # get hsv score
    hsv_score = hsv(img, display)

    # get blemish score
    blemish_score = blemishes_score(img, LOW, UP)
    blemish_score = min(blemish_score * 3, 1)


    # print(f'Blemish score: {blemish_score:.2f}')
    # print(f'HSV score: {hsv_score:.2f}')

    # return a weighted average of the scores and print it
    score = (blemish_score * 0.7) + (hsv_score * 0.2)
    # print(f'Score: {score:.2f}')

    return score



def hsv(img, display):

    # really high in fresher pears
    green_lower = (28, 0, 0)
    green_upper = (62, 170, 170)

    # really high in older pears
    yellow_lower = (20, 11, 14)
    yellow_upper = (26, 190, 240)

    # calculate total number of pixels in object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = np.sum(gray != 255)

    # create masks
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    if display:

        # display img
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # display images in masks but when displaying images show in color
        cv2.imshow('green mask', cv2.bitwise_and(img, img, mask=green_mask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('yellow mask', cv2.bitwise_and(img, img, mask=yellow_mask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    green_percent = green_pixels / total_pixels
    yellow_percent = yellow_pixels / total_pixels

    ripeness_score = yellow_percent / (green_percent + yellow_percent)
    ripeness_score = max(0, ripeness_score)
    ripeness_score = min(1, ripeness_score)

    return ripeness_score




# for i in range(1, 11):
#     path = f'Day{i}/pear2.JPG'
#     pear_score(path, False)
#     print()

