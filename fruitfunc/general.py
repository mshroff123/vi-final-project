import numpy as np
import math
import cv2
from PIL import Image
from collections import defaultdict
from scipy.ndimage import laplace

"""
COLOR: 
Reads in the image and converts it to a color histogram
For each fruit, specify the best bin numbers (based on preferred colors)
"""
def get_color_histogram(img_name, r_bits, g_bits, b_bits):
    query_img = Image.open(img_name)
    query_np = np.array(query_img)
    
    # reshape and create the 3d rgb histogram
    hist, _ = np.histogramdd(query_np.reshape(-1,3), bins=(2**r_bits+ 2**g_bits+2**b_bits))
    return hist

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
    hist, _ = np.histogramdd(np.absolute(laplacian_np), bins=num_bins)
    return hist

# calls the PIL built in function to show a PIL image
def display_PIL(PIL_img):
    PIL_img.show()
    pass

if __name__ == "__main__":
    img_name = "../Day1/mango1.JPG"
    get_hue_histogram(img_name, 2**5)
    # get_texture_hist(img_name, 2**5)
    print("success")