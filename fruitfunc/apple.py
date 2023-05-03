import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from fruitfunc.general import blemishes_score, background_to_white, process_image

LOW = np.array([18, 140, 150])
UP = np.array([24, 190, 205])

def apple_score(path, display = False):

    score = 0

    # read image
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/' + path)
    img = process_image(img)

    img = background_to_white(img)
    hsv_score = hsv(img, display)

    
    # get rgb score
    rgb_score = rgb(img, display)

    # get blemish score
    blemish_score = blemishes_score(img, LOW, UP)
    blemish_score = min(blemish_score * 3, 1)


    #print(f'Blemish score: {blemish_score:.2f}')
    #print(f'Hue score: {hsv_score:.2f}')

    # return a weighted average of the scores and print it
    score = (blemish_score * 0.7) + (hsv_score * 0.2) + (rgb_score * 0.1)
    #print(f'Score: {score:.2f}')

    return score


def rgb(img, display):

    # Split the image into its color channels
    b, g, r = cv2.split(img)

    # Compute the average green and yellow values
    avg_g = cv2.mean(g)[0]
    avg_y = cv2.mean(r + g)[0]

    # Compute the ripeness score
    score = abs(avg_g - avg_y) / 255.0

    # Normalize the score to be between 0 and 1
    score = max(0, min(score, 1))

    return ((score+1)**2)-1

def hsv(img, display):

    green_lower = (2, 114, 60)
    green_upper = (24, 182, 255)
    yellow_lower = (13, 180, 90)
    yellow_upper = (28, 241, 220)

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

    ripeness_score = yellow_pixels * 2 / (green_pixels + yellow_pixels)
    ripeness_score = max(0, ripeness_score)
    ripeness_score = min(1, ripeness_score)

    return ripeness_score



#iterate through all images by iterating the day number in the path
# for i in range(1, 11):
#     path = f'Day{i}/apple1.JPG'
#     apple_score(path, False)
#     print()




