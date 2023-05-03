# mango evolves from yellow with green spots to yellow with red undercolor
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from fruitfunc.general import blemishes_score, background_to_white, process_image

# update blemish values
LOW = np.array([10, 100, 10])
UP = np.array([16, 255, 255])

def mango_score(path, display = False):

    score = 0

    # read image
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/' + path)
    img = process_image(img)

    # get wrinkle score
    wrinkle_score = wrinkles(img, display)

    img = background_to_white(img)

    # get hsv score
    hsv_score = hsv(img, display)
    
    # get rgb score
    rgb_score = rgb(img, display)

    # get blemish score
    blemish_score = blemishes_score(img, LOW, UP)
    blemish_score = min(blemish_score * 3, 1)

    # print(f'Blemish score: {blemish_score:.2f}')
    # print(f'Hue score: {hsv_score:.2f}')
    # print(f'RGB score: {rgb_score:.2f}')
    # print(f'Wrinkle score: {wrinkle_score:.2f}')

    # return a weighted average of the scores and print it
    score = (blemish_score * 0.7) + (hsv_score * 0.3) + (rgb_score * 0.1) + (wrinkle_score * 0.1)
    #print(f'Score: {score:.2f}')

    return score



def rgb(img, display):

    b, g, r = cv2.split(img)

    avg_r = cv2.mean(r)[0]
    avg_g = cv2.mean(g)[0]
    avg_b = cv2.mean(b)[0]

    r_b_ratio = avg_b / avg_r
    r_g_ratio = avg_g / avg_r

    # print ratios
    # print(f'r_b_ratio: {r_b_ratio:.2f}')
    # print(f'r_g_ratio: {r_g_ratio:.2f}')
    # o_g_ratio = avg_o / avg_g


    # average the red-blue and red-green ratios
    # have the r_b_ratio have a higher weight than the r_g_ratio
    avg_ratio = 3*(1-((r_b_ratio + r_g_ratio) / 2))

    # Compute the histogram of the R, G, and B components
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    if display:
        plt.plot(hist_r, color='r')
        plt.plot(hist_g, color='g')
        plt.plot(hist_b, color='b')
        plt.xlim([0, 256])
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.show()

    return avg_ratio


def hsv(img, display):

    # fix thresholds
    green_lower = (2, 114, 60)
    green_upper = (24, 182, 255)
    yellow_lower = (13, 180, 90)
    yellow_upper = (28, 241, 255)

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


def wrinkles(img, display):

    #img = cv2.erode(img,kernel,iterations = 1)
    # apply gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray, 5, 80)

    if display:
        cv2.imshow('Original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Edges1', edges1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


    # dilate the mask
    # kernel = np.ones((3, 3), np.uint8)
    #mask = cv2.dilate(mask, kernel, iterations=1)



    edges1 = np.array(edges1)
    mask = np.array(mask)


    # Set elements of edges to zero where edges2 is nonzero
    edges1[mask != 0] = 0

    edges = edges1



    # Detect line segments using LSD
    lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(edges)

    # Draw the lines on the image
    line_img = lsd.drawSegments(img, lines)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # create a binary mask
    _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    total_length = 0

    # Iterate through the lines array
    for line in lines:
        # Get the coordinates of the start and end points of the line
        x1, y1, x2, y2 = line[0]

        # Calculate the length of the line using the distance formula
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        # Add the length to the total length
        total_length += length

    return total_length / 3000

    # Display the results
    if display:
    #     cv2.imshow('Mask', mask)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

        cv2.imshow('Edges', edges1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the result
        cv2.imshow('Lines', line_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #     cv2.imshow('Thres', thresh)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()




# # #iterate through all images by iterating the day number in the path
# for i in range(1, 11):
#     path = f'Day{i}/mango1.JPG'
#     mango_score(path, False)
#     print()

