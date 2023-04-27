from collections import defaultdict
import os

from fruitfunc.apple import apple_score
from fruitfunc.banana import banana_score
from fruitfunc.orange import orange_score
from fruitfunc.pear import pear_score
from fruitfunc.mango import mango_score
from fruitfunc.kiwi import kiwi_score


image_dict = defaultdict(list)
scores_dict = defaultdict(list)


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

# take in images_dict and for every image in the dict, calculate the score and add it to the scores_dict
# note that each fruit type will use a different scoring method
# specfically the fruit type will determine which function to call
def get_scores_dict(images_dict):
    for key in images_dict:
        if key[1] == 'a':
            scores_dict[key] = apple_score(images_dict[key])
        elif key[1] == 'b':
            scores_dict[key] = banana_score(images_dict[key])
        elif key[1] == 'o':
            scores_dict[key] = orange_score(images_dict[key])
        elif key[1] == 'p':
            scores_dict[key] = pear_score(images_dict[key])
        elif key[1] == 'm':
            scores_dict[key] = mango_score(images_dict[key])
        elif key[1] == 'k':
            scores_dict[key] = kiwi_score(images_dict[key])
        else:
            print("Error: Invalid fruit type")
    return scores_dict




