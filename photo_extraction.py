import os
import cv2
from tqdm import tqdm
import json
import random
import string
import shutil
import yaml
from itertools import *
import pandas as pd
from ultralytics import YOLO

class PhotoAnnotation:
    def __init__(self):
        self.yolo_dir = "datasets/globe_real"
        self.image_dir = "globe_vott/images"
        self.label_dir = "globe_vott/labels"

        self.image_names = []

        for image_file in sorted(os.listdir(self.image_dir)):
            self.image_names.append(image_file[:-4])

        
        countries = [x.lower().strip() for x in pd.read_excel("globe_info/country_colormap.xlsx")["country"]][:20]
        countries = [x[0].upper() + x[1:] for x in countries]
        self.country_dct = dict(zip(countries, range(len(countries))))

        country_colormap = pd.read_excel("globe_info/country_colormap.xlsx")["color"][:20]
        self.color2id = dict(zip(country_colormap, range(len(country_colormap))))


        ids = list(map(str, range(len(countries))))
        self.country_dct_r = dict(zip(ids, countries))

    def predict_images(self, image_names=[], weight_path="checkpoints/yolov9_globe_P_400.pt", is_save=False):
        model = YOLO(weight_path, verbose=True)

        rnt = {}

        for image_name in tqdm(image_names):
            image_path = "{}/{}.png".format(self.image_dir, image_name)
            result = model([image_path])[0]
            countries = [result.names[int(x)] for x in result.boxes.cls.cpu().numpy()]

            rnt[image_name] = {
                "countries": countries,
                "xyxy": result.boxes.xyxy.cpu().numpy().astype("float"),
                "xywh": result.boxes.xywh.cpu().numpy().astype("float")
            }

            if is_save:
                save_dir = "prediction/{}".format(weight_path.split("/")[-1][:-3])
                os.makedirs(save_dir, exist_ok=True)
                result.save(filename="{}/{}.png".format(save_dir, image_name))  # save to disk
        
        return rnt
    
    def convert_vott_dict(self, yolo_prediction):
        bounding_box_dct = {}

        for image_name, preds in yolo_prediction.items():
            bounding_box_dct[image_name] = []
            for country, xyxy, xywh in zip(preds["countries"], preds["xyxy"], preds["xywh"]):
                region_dct = {}
                region_dct["id"] = self.get_random_str(len("H5DI7R1lx"))
                region_dct["type"] = 'RECTANGLE'
                region_dct["tags"] = [country]
                region_dct["boundingBox"] = {'height': xywh[3], 'width': xywh[2], 'left': xywh[0] - xywh[2] / 2, 'top': xywh[1] - xywh[3] / 2}
                region_dct["points"] = [
                    {'x':xyxy[0], 'y':xyxy[1]}, 
                    {'x':xyxy[0], 'y':xyxy[3]}, 
                    {'x':xyxy[2], 'y':xyxy[1]},
                    {'x':xyxy[2], 'y':xyxy[3]}
                ]
                bounding_box_dct[image_name].append(region_dct)
        return bounding_box_dct

    def photo_extraction(self, src_dir="C:/Users/gaomany/Downloads/globe_3", save_dir="globe_photo"):
        os.makedirs(save_dir, exist_ok=True)
        image_id = 0
        for root, directories, files in tqdm(list(os.walk(src_dir))):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png"):
                    image = cv2.imread("{}/{}".format(root, file), cv2.IMREAD_COLOR)
                    if image is not None:
                        image = cv2.resize(image, (1280, 720))
                        cv2.imwrite("{}/{:0>4}.png".format(save_dir, image_id), image)
                        image_id += 1

    def get_random_str(self, str_len):
        letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
        return "".join([letters[random.randint(0, len(letters)-1)] for _ in range(str_len)])

    def save_label_as_vott_json(self, bounding_box_dct):
        with open('globe_vott/labels/globe_annotation.vott', 'r') as file:
            vott_config = json.load(file)

        for image_name, bounding_box_lst in bounding_box_dct.items():
            image_file = image_name + ".png"
            target_sample_id = None

            for sample_id in vott_config["assets"]:
                if image_file in vott_config["assets"][sample_id]["name"]:
                    target_sample_id = sample_id
                    break
            
            sample_json_path = "globe_vott/labels/{}-asset.json".format(target_sample_id)
            if os.path.exists(sample_json_path):
                with open(sample_json_path, 'r') as file:
                    sample_json = json.load(file)
                sample_json["regions"] = bounding_box_lst
                with open(sample_json_path, 'w') as file:
                    json.dump(sample_json, file)


        # with open('globe_vott/labels/{}.json'.format(json_data["asset"]["id"]), 'w') as file:
        #     json.dump(json_data, file)
    
    def convert_vott_to_yolo(self):
        for data_type, subset in product(["images","labels"], ["train","val","test"]):
            os.makedirs("{}/{}/{}".format(self.yolo_dir, data_type, subset), exist_ok=True)


        json_files = sorted([x for x in os.listdir(self.label_dir) if ".json" in x])


        image_files = sorted(os.listdir(self.image_dir))

        sz = len(json_files)

        # training set:
        def convert(json_files, subset):
            for json_file in tqdm(json_files):
                with open("{}/{}".format(self.label_dir, json_file), 'r') as file: data = json.load(file)
                image_name = data["asset"]["name"]
                if image_name in image_files:
                    h0, w0 = data["asset"]["size"]["height"], data["asset"]["size"]["width"]
                    shutil.copyfile("{}/{}".format(self.image_dir, image_name), "{}/images/{}/{}".format(self.yolo_dir, subset, image_name))

                    label_lst = []

                    for region in data["regions"]:
                        name = region["tags"][0].lower().strip()
                        name = name[0].upper() + name[1:]

                        if name == "Index": continue

                        h, w, l, t = map(lambda x:region["boundingBox"][x], ["height", "width", "left", "top"])
                        h, t, w, l = h / h0, t / h0, w / w0, l / w0

                        label_lst.append(" ".join(map(str, [self.country_dct[name], l + w / 2, t + h / 2, w, h])) + "\n") # # xc, yc, w, h
                    
                    label_name = image_name.replace(".png", ".txt").replace(".JPG", ".txt").replace(".jpg", ".txt").replace(".JPG", ".txt")
                    label_path = "{}/labels/{}/{}".format(self.yolo_dir, subset, label_name)
                    with open(label_path, "w") as label_file: label_file.writelines(label_lst)
        
        convert(json_files, "train")    
        convert(json_files[int(0.8 * sz):int(0.9 * sz)], "val")
        convert(json_files[int(0.9 * sz):], "test")

        yaml_config = {
            "path" : "../{}".format(self.yolo_dir),
            "train": "images/train",
            "val": "images/val",
            "test" : "images/test",
            "names" : self.country_dct_r
        }

        with open('{}/{}.yaml'.format(self.yolo_dir, self.yolo_dir.split("/")[-1]), 'w') as yaml_file:
            yaml.dump(yaml_config, yaml_file, default_flow_style=False)

    def extract_from_subdir(self, src_dir="", dst_dir="globe_vott"):
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
    pa = PhotoAnnotation()

    # pa.photo_extraction("D:/globe_photos")

    weight_path="checkpoints/yolov9c_fused2.pt"

    predictions = pa.predict_images(pa.image_names, weight_path=weight_path, is_save=True)
    # bb_dct = pa.convert_vott_dict(predictions)
    # pa.save_label_as_vott_json(bb_dct)

    # pa.convert_vott_to_yolo()