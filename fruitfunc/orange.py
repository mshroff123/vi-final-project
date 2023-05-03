import cv2
import numpy as np
import matplotlib.pyplot as plt

from general import blemishes_score, background_to_white, process_image

def orange_score(path):

    score = 0

    # read image
    img = cv2.imread('../' + path)

    # process image
    img = process_image(img)

    # convert background to white
    img = background_to_white(img)

    # get saturation score
    sat_score = saturation(img)

    print(sat_score)
    return sat_score

def saturation(img):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract saturation channel
    saturation = hsv[:,:,1]

    # Compute average saturation value
    saturation_mean = np.mean(saturation)
    normalized_saturation = saturation_mean/255

    return 4*normalized_saturation

for i in range(1, 11):
    path = f'Day{i}/orange2.JPG'
    orange_score(path)
    print()