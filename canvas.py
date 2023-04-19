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

    for i, key in enumerate(img_keys):
        path = img_dict[key]
        img = cv2.imread(path)
        img = cv2.resize(img, THUMBNAIL_SIZE)

        y_offset = i * 160

        canvas[y_offset:y_offset+100, 0:0+100, :] = img

