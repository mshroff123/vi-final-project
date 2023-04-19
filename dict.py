from collections import defaultdict
import os

image_dict = defaultdict(list)

def build_image_dict(folder):
    day = folder[-1]
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".JPG"):
                image_dict[day + file[:-4]].append(os.path.join(root, file))

def get_image_dict():
    return image_dict


