import cv2
import matplotlib.pyplot as plt
import numpy as np

def banana_score():
    pass


# given a picture of a banana, return a ripeness score between 0 and 1
# img is against a white background
# count the number of blemish pixels and divide by the total number of pixels
def blemishes_score():
    # read image
    # convert background to white
    img = background_to_white('/Users/jay/Desktop/Project Fruit/Day4/banana2.JPG')
    
    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # display grayscale image using cv2
    cv2.imshow('image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    total_pixels = cv2.countNonZero(gray)

    # threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # define range of brown/black color in HSV
    # Set minimum and maximum HSV values to display
    lower = np.array([10, 100, 10])
    upper = np.array([18, 255, 255])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # display image with mask using cv2
    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # apply mask to the thresholded image
    blemish_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))

    # calculate the blemish score as the ratio of blemish pixels to total pixels
    blemish_score = blemish_pixels / total_pixels
    print(blemish_score)
    return blemish_score


# given a picture of object, against a black background, turn the background white
def background_to_white(path):
    img = cv2.imread(path)

    # Process the image to remove noise
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img,kernel,iterations = 2)
    img = cv2.dilate(img,kernel,iterations = 1)

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

blemishes_score()
