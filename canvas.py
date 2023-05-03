import numpy as np
import cv2
from PIL import Image

"""
Reads dictionary of images and returns a canvas of the images
More specfically, produces a canvas of the test images of a fruit type that is passed in
Canvas is of size # of Days x 3
"""
def get_canvas(img_dict, fruit):
    # produce list of image keys
    img_keys = []
    for i in range(1,11):
        img_keys.append(str(i) + fruit + '3')

    # size of the thumbnails
    THUMBNAIL_SIZE = (100, 160)

    # create canvas
    num_cols = 3
    canvas_size = (num_cols * 100, len(img_keys) * 160)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # Iterate over rows
    for row in range(10):
        # Iterate over columns
        for col in range(num_cols):
            # Calculate index into list of image keys
            idx = row * num_cols + col

            # Load image
            path = img_dict[img_keys[row]][0]
            img = cv2.imread(path)
            img = cv2.resize(img, THUMBNAIL_SIZE)

            # Calculate position in canvas to insert image
            x = col * THUMBNAIL_SIZE[0]
            y = row * THUMBNAIL_SIZE[1]
            
            # Insert image into canvas
            canvas[y:y+THUMBNAIL_SIZE[1], x:x+THUMBNAIL_SIZE[0], :] = img

    cv2.imshow('Canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

