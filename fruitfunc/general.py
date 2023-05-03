import numpy as np
import math
import cv2
from PIL import Image
from collections import defaultdict
from scipy.ndimage import laplace


"""
CONTOUR FOR COLOR/TEXTURE: 
Reads in the image, gets a contour for the fruit
Returns the img, gray_img, and the mask for the contour
"""
def get_color_texture_contour_data(img_name):
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
CONTOUR FOR HUE/SATURATION: 
Reads in the image, gets a contour for the fruit using HSV
Returns the img, hsv_img, and the mask for the contour
"""
def get_hue_saturation_contour_data(img_name):
    # read the image and convert to grayscale
    img = cv2.imread(img_name)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # apply thresholding to create a binary mask for the fruit
    _, thresh = cv2.threshold(hsv_img[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find contours in the binary mask and get the largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # create a mask for the fruit
    mask = np.zeros_like(hsv_img[..., 1])
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    return img, hsv_img, mask

"""
COLOR: 
Returns a 1d numpy array with the RGB values within the contour
"""
def get_color_array(img_name):
    img, _, mask = get_color_texture_contour_data(img_name)
    fruit_colors = img[np.where(mask == 255)]
    color_data = np.flip(fruit_colors, axis=1)
    return color_data

"""
TEXTURE: 
Returns a 1d numpy array with the grayscale values within the contour
"""
def get_gray_array(img_name):
    _, gray_img, mask = get_color_texture_contour_data(img_name)
    gray_data = gray_img[np.where(mask == 255)]
    return gray_data

"""
HUE: 
Returns a 1d numpy array with the hue values within the contour
"""
def get_hue_array(img_name):
    _, hsv_img, mask = get_hue_saturation_contour_data(img_name)
    hue_data = hsv_img[..., 0][mask > 0]
    return hue_data

"""
SATURATION: 
Returns a 1d numpy array with the saturation values within the contour
"""
def get_saturation_array(img_name):
    _, hsv_img, mask = get_hue_saturation_contour_data(img_name)
    sat_data = hsv_img[..., 1][mask > 0]
    return sat_data

"""
GENERAL MAX VALUE: 
It finds the color that most frequently appears in the contour
FOR COLOR: example format returned is [204 204 152] in RGB order, type = numpy.ndarray 
FOR TEXTURE: example format returned is 185, type = numpy.uint8
FOR HUE: example format returned is 23, type = numpy.uint8
FOR SATURATION: example format returned is 180, type = numpy.uint8
"""
def get_max_value_from_contour(data):
    unique_values, val_counts = np.unique(data, axis=0, return_counts=True)
    most_frequent_val = unique_values[np.argmax(val_counts)]
    return most_frequent_val

# calls the PIL built in function to show a PIL image
def display_PIL(PIL_img):
    PIL_img.show()
    pass

if __name__ == "__main__":
    img_name = "../Day5/mango3.JPG"
    print("success")


"""
PROCESS: 
Removes noise from image and cleans up the background
"""
def process_image(img):
# Process the image to remove noise
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img,kernel,iterations = 2)
    img = cv2.dilate(img,kernel,iterations = 1)

    return img

"""
CONVERT BACKGROUND TO WHITE:: 
Returns an image of the fruit with the background converted to white
"""
def background_to_white(img):

    # Split the image into its color channels
    b, g, r = cv2.split(img)

    # Convert the color channels to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the grayscale image to create a mask
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Invert the mask so that the object is black and the background is white
    mask = cv2.bitwise_not(thresh)

    # Apply the mask to each color channel
    b = cv2.bitwise_and(b, b, mask=mask)
    g = cv2.bitwise_and(g, g, mask=mask)
    r = cv2.bitwise_and(r, r, mask=mask)

    # Merge the color channels back into a single image
    result = cv2.merge((b, g, r))

    # Apply the mask to the background
    background = np.full(img.shape, 255, dtype=np.uint8)
    background = cv2.bitwise_and(background, background, mask=thresh)

    # Add the masked object to the masked background
    result = cv2.add(result, background)

    return result


"""
BLEMISHES: 
Returns a score between 0 and 1 calculated by the number of blemish pixels in object
"""

def blemishes_score(img, LOW, UP):
    
    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate total number of pixels within object
    total_pixels = np.sum(gray != 255)

    # threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # define range of brown/black color in HSV
    # Set minimum and maximum HSV values to display
    lower = np.array([LOW[0], LOW[1], LOW[2]])
    upper = np.array([UP[0], UP[1], UP[2]])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # # display image with mask using cv2
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # apply mask to the thresholded image
    blemish_pixels = cv2.countNonZero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))

    # calculate the blemish score as the ratio of blemish pixels to total pixels
    blemish_score = blemish_pixels / total_pixels


    return blemish_score

