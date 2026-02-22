import cv2
import os
import numpy as np
from tqdm import tqdm

mask = cv2.imread("globe_info/country_colormap.png")

color_b_lst = [(94,199,129), (139,124,226), (207,116,154), (105,156,222), (15,204,65), (78,184,208), (224,61,151), (215,166,91), (200,132,55), 
               (143,230,50), (152,88,237), (220,106,117), (210,54,51), (205,67,69), (113,221,169), (225,237,59), (237,85,71), (213,106,121), (215,30,86), (89,182,225)] 
color_a_lst = [(28,145,56), (67,51,192), (157,45,82), (35,85,186), (1,154,14), (19,121,162), (192,12,79), (171,97,26), (147,60,10), 
               (70,202,8), (79,25,214), (182,37,46), (123,12,127), (157,15,15), (41,184,99), (192,216,12), (214,23,16), (186,241,139), (99,87,70), (26,120,192)]
country_names = ["Antarctica", "Russia", "Canada", "America", "China", "Greenland", "Australia", "Brazil", "Kazakhstan", 
                 "Argentina", "India", "Mongolia", "Algeria", "Congo", "Mexico", "Saudi Arabia", "Iran", "Sudan", "Libya", "Indonesia"]

colored_img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255

for color_b, color_a, country_name in tqdm(list(zip(color_b_lst, color_a_lst, country_names))):
    color_b, color_a = color_b[::-1], color_a[::-1]

    diff_map = np.abs(mask - color_b)
    diff_map = np.sum(diff_map, axis=2)#29B863
    activate_map = np.where(diff_map < 10, 1, 0)
    colored_img[activate_map == 1] = color_a

cv2.imwrite("color_adjust.png", colored_img)

# [(139,124,226)]
# []
# ["Russia"]