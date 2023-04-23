import numpy as np
import cv2
from PIL import Image


def get_canvas(img_dict, fruit):
    img_keys = []
    for i in range(1,8):
        img_keys.append(str(i) + fruit + '3')

    THUMBNAIL_SIZE = (100, 160)

    num_cols = 3
    canvas_size = (num_cols * 100, len(img_keys) * 160)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    for row in range(8):
        # Iterate over columns
        for col in range(num_cols):
            # Calculate index into list of image keys
            idx = row * num_cols + col
            # Load image
            path = img_dict[img_keys[idx]][0]
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

        


