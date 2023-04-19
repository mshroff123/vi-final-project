from collections import defaultdict
import os


def get_image_dict(folder):
    day = folder[-1]
    image_dict = defaultdict(list)
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".JPG"):
                image_dict[day + file[:-4]].append(os.path.join(root, file))

    return image_dict

print(get_image_dict("Day1").keys())


