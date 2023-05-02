# mango evolves from yellow with green spots to yellow with red undercolor
import cv2
import numpy as np
import matplotlib.pyplot as plt

from general import blemishes_score, background_to_white, process_image

# update blemish values
LOW = np.array([10, 100, 10])
UP = np.array([16, 255, 255])

def mango_score(path, LOW, UP):

    score = 0

    # read image
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/' + path)

    # process image
    img = process_image(img)

    # convert background to white
    img = background_to_white(img)

    # get blemish score
    blemish_score += blemishes_score(img, LOW, UP)
    rgb_score = rgb(path)
    hue_score = hue(path)


    # return a weighted average of the scores
    score = (blemish_score * 0.5) + (rgb_score * 0.3) + (hue_score * 0.2)

    return score



def rgb(path):
    img = cv2.imread(path)

    # process image
    img = process_image(img)

    # convert background to white
    img = background_to_white(img)

    r, g, b = cv2.split(img)

    avg_r = cv2.mean(r)[0]
    avg_g = cv2.mean(g)[0]
    avg_b = cv2.mean(b)[0]

    r_b_ratio = avg_r / avg_b
    r_g_ratio = avg_r / avg_g

    # print(f'Average red: {avg_r:.2f}')
    # print(f'Average green: {avg_g:.2f}')
    # print(f'Average blue: {avg_b:.2f}')
    # print(f'Red-blue ratio: {r_b_ratio:.2f}')
    # print(f'Red-green ratio: {r_g_ratio:.2f}')

    # average the red-blue and red-green ratios
    # have the r_b_ratio have a higher weight than the r_g_ratio
    avg_ratio = (r_b_ratio + r_g_ratio) / 2
    print(f'Average ratio: {avg_ratio:.2f}')

    # Compute the histogram of the R, G, and B components
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    # # Plot the histograms
    # plt.plot(hist_r, color='r')
    # plt.plot(hist_g, color='g')
    # plt.plot(hist_b, color='b')
    # plt.xlim([0, 256])
    # plt.xlabel('Intensity')
    # plt.ylabel('Frequency')
    # plt.show()


def hsv():
    img = cv2.imread('/Users/jay/Desktop/Project Fruit/Day1/mango2.JPG')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # fix thresholds
    green_lower = (22, 154, 158)
    green_upper = (37, 178, 255)
    yellow_lower = (20, 100, 100)
    yellow_upper = (35, 255, 255)

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # display images in masks but when displaying images show in color
    cv2.imshow('green mask', cv2.bitwise_and(img, img, mask=green_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('yellow mask', cv2.bitwise_and(img, img, mask=yellow_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    total_pixels = img.shape[0] * img.shape[1]

    green_percent = green_pixels / total_pixels
    yellow_percent = yellow_pixels / total_pixels


    ripeness_score = yellow_percent / (green_percent + yellow_percent)
    ripeness_score = max(0, ripeness_score)
    ripeness_score = min(1, ripeness_score)

    print(f'Ripeness score: {ripeness_score:.2f}')

# need to not include border of image in the count
# need to apply minimal processing to the image to reduce noise but not remove wrinkles
def wrinkles2(img_path):
    img = cv2.imread(img_path)

    kernel = np.ones((3, 3), np.uint8)

    #img = cv2.erode(img,kernel,iterations = 1)
    # apply gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ww can use canny edge detection to identify edges
    edges1 = cv2.Canny(gray, 10, 100)
    edges2 = cv2.Canny(gray, 170, 200)
    edges = cv2.subtract(edges1, edges2)

    dilated = cv2.dilate(edges, kernel, iterations=1)

    # create a binary mask
    _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # create ripeness score based on the number of wrinkles
    num_wrinkles = cv2.countNonZero(thresh)
    max_wrinkles = 5000  # Set a maximum number of wrinkles for a fully ripe fruit
    ripeness_score = min(1, num_wrinkles / max_wrinkles)

    # Display the results
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Dilated', dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Mask', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'Number of wrinkles: {num_wrinkles}')
    print(f'Ripeness score: {ripeness_score:.2f}')


def wrinkles(path):

    img = cv2.imread(path)


    #img = cv2.erode(img,kernel,iterations = 1)
    # apply gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray, 20, 100)

    cv2.imshow('Original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Edges1', edges1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


    # dilate the mask
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)



    edges1 = np.array(edges1)
    mask = np.array(mask)


    # Set elements of edges to zero where edges2 is nonzero
    edges1[mask != 0] = 0

    edges = edges1

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # create a binary mask
    _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # create ripeness score based on the number of wrinkles
    num_wrinkles = cv2.countNonZero(thresh)
    max_wrinkles = 5000  # Set a maximum number of wrinkles for a fully ripe fruit
    ripeness_score = min(1, num_wrinkles / max_wrinkles)

    # Display the results

    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Thres', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'Number of wrinkles: {num_wrinkles}')
    print(f'Ripeness score: {ripeness_score:.2f}')







# # iterate through all images by iterating the day number in the path
# # path = '/Users/jay/Desktop/Project Fruit/Day1/mango2.JPG'
# for i in range(1, 6):
#     path = f'/Users/jay/Desktop/Project Fruit/Day{i}/mango2.JPG'
#     mango_score(path)
#     print()

#hue()

#mango_score('Day7/mango2.JPG', LOW, UP)
wrinkles('/Users/jay/Desktop/Project Fruit/Day1/mango3.JPG')


