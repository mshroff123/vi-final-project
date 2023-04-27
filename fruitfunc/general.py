import numpy as np
import math
import cv2
from PIL import Image
from collections import defaultdict
from scipy.ndimage import laplace

# MUST MODIFY THESE FUNCTIONS SO THAT THEY ISOLATE THE FRUITS OR JUST
# REWRITE ALL BACKGROUND PIXELS TO BLACK

"""
COLOR: 
Reads in the image, gets a contour for the fruit
Returns a 1d numpy array with the RGB values within the contour
"""
def extract_color_from_contour(img_name):
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

    # get the 1d np array of BGR values 
    # flip it to be in RGB order
    fruit_colors = img[np.where(mask == 255)]
    color_data = np.flip(fruit_colors, axis=1)
    return color_data

"""
COLOR: 
From color data returned by extract_color_from_contour, it finds the color
that most frequently appears in the contour
Format of color: [204 204 152] in RGB order
"""
def get_max_color_from_contour(color_data):
    unique_colors, color_counts = np.unique(color_data, axis=0, return_counts=True)
    most_frequent_color = unique_colors[np.argmax(color_counts)]
    return most_frequent_color

"""
COLOR: 
Reads in the image and converts it to a color histogram
For each fruit, specify the best bin numbers (based on preferred colors)
ISSUE: includes the background colors
"""
def get_color_histogram(img_name, r_bits, g_bits, b_bits):
    query_img = Image.open(img_name)
    query_np = np.array(query_img)
    
    # reshape and create the 3d rgb histogram
    color_hist, _ = np.histogramdd(query_np.reshape(-1,3), bins=(2**r_bits+ 2**g_bits+2**b_bits))
    return color_hist

"""
COLOR: 
From the color histogram, this should extract the RGB values of the color 
most frequently found in the image
ISSUE: includes the background so this skews the data
FIX: for the color hist function and everything follow, somehow use only the fruit contour
"""
def get_color_max(color_hist):
    # Find the bin with the highest count
    max_bin = np.unravel_index(np.argmax(color_hist, axis=None), color_hist.shape)

    # Convert the bin indices to a color value
    # color_value variable: tuple of three integers representing RGB of the color with highest count in the image
    # can apply thresholds to this value to see if it is mostly green/yellow etc
    color_value = tuple([int((max_bin[i] + 0.5) * 256 / (2**bits)) for i, bits in enumerate([4, 5, 2])])

    # Print the color value
    print(color_value)  
    return max_bin

"""
HUE AND SATURATION:
The function uses HSV image to get the hues for an image
Pretty similar to color image -- minimal difference
"""
def get_hue_sat_histogram(img_name, num_bins):
    query_img = Image.open(img_name)
    query_hsv = query_img.convert('HSV')
    query_np = np.array(query_hsv)
    
    # extract the hue and saturation channels and flatten
    hue_channel = query_np[:, :, 0].flatten()
    sat_channel = query_np[:, :, 1].flatten()
    
    # create the 1d histograms
    hue_hist, _ = np.histogram(hue_channel, bins=num_bins)
    sat_hist, _ = np.histogram(sat_channel, bins=num_bins)
    return hue_hist, sat_hist


"""
TEXTURE:
Reads in the image and converts it grayscale
Converts grayscale to Laplacian
Gets histogram of Laplacian numpy array with 2**5 bins
"""
def get_texture_hist(img_name, num_bins):
    query_img = Image.open(img_name)
    gray_img = query_img.convert('L')
    laplacian_np = np.array(laplace(np.array(gray_img))).reshape(-1,1)
    texture_hist, _ = np.histogramdd(np.absolute(laplacian_np), bins=num_bins)
    return texture_hist

# calls the PIL built in function to show a PIL image
def display_PIL(PIL_img):
    PIL_img.show()
    pass

if __name__ == "__main__":
    img_name = "../Day5/mango3.JPG"
    #color_hist = get_color_histogram(img_name, 4, 5, 2)
    #get_color_max(color_hist)
    fruit_mask = extract_color_from_contour(img_name)
    get_max_color_from_contour(fruit_mask)
    # get_hue_sat_histogram(img_name, 2**5)
    # get_texture_hist(img_name, 2**5)
    print("success")