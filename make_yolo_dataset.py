import cv2
import pandas as pd
import os
import shutil
import yaml
import numpy as np
from collections import *
from tqdm import tqdm
from itertools import product

country_colormap = pd.read_excel("globe_info/country_colormap.xlsx")["color"][:20]
color2id = dict(zip(country_colormap, range(len(country_colormap))))

countries = [x.lower().strip() for x in pd.read_excel("globe_info/country_colormap.xlsx")["country"]][:20]
countries = [x[0].upper() + x[1:] for x in countries]

ids = list(map(str, range(len(countries))))
country_dct_r = dict(zip(ids, countries))


def get_bounding_box(mask_file):
    mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
    h, w, c = mask.shape
    rnt = []
    
    for color in color2id:
        color0 = np.array(list(map(int, color[1:-1].split(","))))

        diff_map = np.abs(mask - color0)
        diff_map = np.sum(diff_map, axis=2)

        activate_map = np.where(diff_map < 5, 1, 0)

        if np.count_nonzero(activate_map == 1) > 10 ** 3:
            country_id = color2id[color]
            indice = np.where(activate_map == 1)
            y1, y2 = indice[0].min() / h, indice[0].max() / h
            x1, x2 = indice[1].min() / w, indice[1].max() / w
            rnt.append(" ".join(map(str, [country_id, (x1+x2) / 2, x2-x1, (y1+y2) / 2, y2-y1, "\n"])))

    return rnt

def make_yolo(globe_dir="globe_synthetic", yolo_dir="globe_synthetic"):
    yolo_dir = "datasets/{}".format(yolo_dir)
    # 
    for data_dir, subset in product(["images", "labels"], ["train", "test", "val"]):
        os.makedirs("/".join([yolo_dir, data_dir, subset]), exist_ok=True)

    sample_ids = [x.split("_")[0] for x in sorted(os.listdir(globe_dir)) if "Mask" in x]
    print("sample_ids:", sample_ids)
    sz = len(sample_ids)

    def convert(sample_ids, subset):
        for sample_id in tqdm(sample_ids, desc=subset):
            shutil.copy("{}/{}_NoHand.png".format(globe_dir, sample_id), "{}/images/{}/{:0>6}.png".format(yolo_dir, subset, sample_id))
            bb = get_bounding_box("{}/{}_Mask.png".format(globe_dir, sample_id))
            with open("{}/labels/{}/{:0>6}.txt".format(yolo_dir, subset, sample_id), "w") as file:
                file.writelines(bb)

    convert(sample_ids[:int(0.9 * sz)], "train")    
    convert(sample_ids[int(0.9 * sz):], "val")
    
    yaml_config = {
        "path" : "../{}".format(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "test" : "",
        "names" : country_dct_r
    }

    with open('{}/{}.yaml'.format(yolo_dir, yolo_dir.split("/")[-1]), 'w') as yaml_file:
        yaml.dump(yaml_config, yaml_file, default_flow_style=False)

if __name__=="__main__":
    make_yolo(globe_dir="datasets_preprocess/synthetic_P_split")