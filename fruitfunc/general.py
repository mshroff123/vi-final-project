import numpy as np
import math
import cv2
from PIL import Image
from collections import defaultdict
from scipy.ndimage import laplace

# MUST MODIFY THESE FUNCTIONS SO THAT THEY ISOLATE THE FRUITS OR JUST
# REWRITE ALL BACKGROUND PIXELS TO BLACK

"""
GENERAL CONTOUR TO USE FOR COLOR/TEXTURE: 
Reads in the image, gets a contour for the fruit
Returns the img, gray_img, and the mask for the contour
"""
def get_contour_data(img_name):
    # read the image and convert to grayscale
    img = cv2.imread(img_name)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding to create a binary mask for the fruit
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find contours in the binary mask and get the largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # create a mask for the fruit
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    return img, gray_img, mask

"""
COLOR: 
Returns a 1d numpy array with the RGB values within the contour
"""
def get_color_array(img_name):
    img, _, mask = get_contour_data(img_name)
    fruit_colors = img[np.where(mask == 255)]
    color_data = np.flip(fruit_colors, axis=1)
    return color_data

"""
TEXTURE: 
Returns a 1d numpy array with the grayscale values within the contour
"""
def get_gray_array(img_name):
    _, gray_img, mask = get_contour_data(img_name)
    gray_data = gray_img[np.where(mask == 255)]
    return gray_data

"""
GENERAL MAX VALUE: 
From color data/texture_data, it finds the color
that most frequently appears in the contour
FOR COLOR: format returned is [204 204 152] in RGB order, type = numpy.ndarray 
FOR TEXTURE: format returned is 185, type = numpy.uint8
"""
def get_max_value_from_contour(data):
    unique_values, val_counts = np.unique(data, axis=0, return_counts=True)
    most_frequent_val = unique_values[np.argmax(val_counts)]
    print(most_frequent_val)
    print(type(most_frequent_val))
    return most_frequent_val

# calls the PIL built in function to show a PIL image
def display_PIL(PIL_img):
    PIL_img.show()
    pass

if __name__ == "__main__":
    img_name = "../Day5/mango3.JPG"
    #color_hist = get_color_histogram(img_name, 4, 5, 2)
    #get_color_max(color_hist)
    # color_data = get_color_array(img_name)
    # get_max_value_from_contour(color_data)
    # get_hue_sat_histogram(img_name, 2**5)
    # get_texture_hist(img_name, 2**5)
    print("success")