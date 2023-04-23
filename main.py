from dict import build_image_dict, get_image_dict
from canvas import get_canvas

build_image_dict("Day1")
build_image_dict("Day2")

img_dict = get_image_dict()

get_canvas(img_dict, 'apple')
