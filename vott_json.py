import json
import os
import shutil
import yaml
from itertools import *
import pandas as pd

# make country dict
countries = [x.lower().strip() for x in pd.read_excel("globe_info/country_colormap.xlsx")["country"]][:20]
countries = [x[0].upper() + x[1:] for x in countries]
country_dct = dict(zip(countries, range(len(countries))))

country_colormap = pd.read_excel("globe_info/country_colormap.xlsx")["color"][:20]
color2id = dict(zip(country_colormap, range(len(country_colormap))))


ids = list(map(str, range(len(countries))))
country_dct_r = dict(zip(ids, countries))

def convert_vott_to_yolo(vott_dir="globe_vott", yolo_dir="globe_real"):
    yolo_dir = "datasets/{}".format(yolo_dir)

    for data_type, subset in product(["images","labels"], ["train","val"]):
        os.makedirs("{}/{}/{}".format(yolo_dir, data_type, subset), exist_ok=True)

    label_dir = "{}/labels".format(vott_dir)
    image_dir = "{}/images".format(vott_dir)

    json_files = sorted([x for x in os.listdir(label_dir) if ".json" in x])
    image_files = sorted(os.listdir(image_dir))

    sz = len(json_files)

    # training set:
    def convert(json_files, subset):
        for json_file in json_files:
            with open("{}/{}".format(label_dir, json_file), 'r') as file: data = json.load(file)

            image_name = data["asset"]["name"]

            # print("image_name:", image_name)
            if image_name in image_files:
                h0, w0 = data["asset"]["size"]["height"], data["asset"]["size"]["width"]
                shutil.copyfile("{}/images/{}".format(vott_dir, image_name), "{}/images/{}/{}".format(yolo_dir, subset, image_name))

                label_lst = []

                for region in data["regions"]:
                    name = region["tags"][0].lower().strip()
                    name = name[0].upper() + name[1:]

                    h, w, l, t = map(lambda x:region["boundingBox"][x], ["height", "width", "left", "top"])
                    h, t, w, l = h / h0, t / h0, w / w0, l / w0

                    label_lst.append(" ".join(map(str, [country_dct[name], l + w / 2, t + h / 2, w, h])) + "\n") # # xc, yc, w, h
                
                label_name = image_name.replace(".png", ".txt").replace(".JPG", ".txt").replace(".jpg", ".txt").replace(".JPG", ".txt")
                label_path = "{}/labels/{}/{}".format(yolo_dir, subset, label_name)
                with open(label_path, "w") as label_file: label_file.writelines(label_lst)
    
    convert(json_files[:int(0.9 * sz)], "train")    
    convert(json_files[int(0.9 * sz):], "val")

    yaml_config = {
        "path" : "../{}".format(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "test" : "",
        "names" : country_dct_r
    }

    with open('{}/{}.yaml'.format(yolo_dir, yolo_dir.split("/")[-1]), 'w') as yaml_file:
        yaml.dump(yaml_config, yaml_file, default_flow_style=False)

def extract_from_subdir(src_dir="", dst_dir="globe_vott"):
    image_names = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_path = root + "/" + file
            dst_path = dst_dir + "/images/" + file
            if file.endswith(".jpg") or file.endswith(".JPG"):
                image_names.append(file[:-4])
                shutil.copyfile(src_path, dst_path)

    for root, _, files in os.walk(src_dir):
        for file in files:
            src_path = root + "/" + file
            dst_path = dst_dir + "/labels/" + file
            if file.endswith(".json"):
                for image_name in image_names:
                    if image_name in src_path:
                        with open(src_path, "r") as json_file:
                            data = json.load(json_file)
                        data["asset"]["name"] = image_name + "." + data["asset"]["format"]
                        with open(src_path, "w") as json_file:
                            json.dump(data, json_file)
                        break
                shutil.copyfile(src_path, dst_path)
    
    


    

if __name__=="__main__":
    pass
    convert_vott_to_yolo("globe_vott", "globe_real")

    # extract_from_subdir("globe_photo")