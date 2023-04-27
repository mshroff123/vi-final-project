from collections import defaultdict
import os

from fruitfunc/apple.py import apple_score

image_dict = defaultdict(list)
scores_dict = defaultdict(list)

apple_score()

"""
Given folder that contains pictures of fruit taken in a day, build a dictionary of images
Keys follow a structure of Day|FruitType|FruitNumber
Values are a path to the image (note that path strings are stored in list for flexibility)
"""
def build_image_dict(folder):
    day = folder[-1]
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".JPG"):
                image_dict[day + file[:-4]].append(os.path.join(root, file))


"""
Instead of calling build_image_dict for each day, use a for loop to iterate through the days
Return global varibal image_dict
"""
def get_image_dict():
    for i in range(1,8):
        build_image_dict('Day' + str(i))
    return image_dict

# # take in images_dict and for every image in the dict, calculate the score and add it to the scores_dict
# # note that each fruit type will use a different scoring method
# def build_scores_dict(images_dict):


# def get_scores_dict():
#     for i in range(1,8):
#         build_scores_dict('Day' + str(i))
#     return scores_dict


