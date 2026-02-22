import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from itertools import product
import shutil
import yaml
import random
from scipy import ndimage
from collections import defaultdict

class IdentityDefaultDict(defaultdict):
    '''
    This class is utilited for identity mapping.
    '''
    def __missing__(self, key):
        return key

class UnionFind:
    def __init__(self, n=None):
        self.parent = IdentityDefaultDict()
        if n: self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def merge(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y: self.parent[x] = y
    def get_group_count(self):
        return len(set(self.find(x) for x in self.parent))
    
class SyntheticAugmentation:
    def __init__(self):
        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

        country_colormap = pd.read_excel("globe_info/country_colormap.xlsx")["color"][:20]
        countries = [x.lower().strip() for x in pd.read_excel("globe_info/country_colormap.xlsx")["country"]][:20]
        self.countries = [x[0].upper() + x[1:] for x in countries]

        self.color2id = dict(zip(country_colormap, range(len(country_colormap))))

        ids = list(map(str, range(len(countries))))
        self.country_dct_r = dict(zip(ids, countries))
        self.country_dct_r['20'] = "Index"

        self.min_area_limit = 100

        self.res_w, self.res_h = 1280, 720

    def image_shift(self, image, dx, dy):
        h, w = image.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, M, (w, h))
    
    def get_connected_components_by_distance(self, mask, globe_mask):
        structure = ndimage.generate_binary_structure(2, 2)
        dilated_kernel = np.ones((3, 3), np.uint8)
        ex_kernel = np.ones((5, 5), np.uint8)

        sphere_mask = cv2.cvtColor(globe_mask, cv2.COLOR_BGR2GRAY)
        sphere_mask[sphere_mask>0] = 1
        bounding = cv2.dilate(sphere_mask.astype(np.uint8), dilated_kernel, iterations=1) - sphere_mask
        indices = np.argwhere(bounding == 1)

        max_distance = 0
        for x1, y1 in indices:
            for x2, y2 in indices:
                max_distance = max(max_distance, abs(x1-x2) + abs(x2-y2))

        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ex_kernel)
        labelmaps, connected_num = ndimage.label(mask, structure=structure)

        uf = UnionFind()
        cc_lst = []
        cc_coords = []

        inf = 10 ** 20

        # get region boundings
        for i in range(1, connected_num+1):
            if np.count_nonzero(labelmaps == i) > self.min_area_limit:
                cc = np.zeros_like(labelmaps)
                cc[labelmaps==i] = 1

                bounding = cv2.dilate(cc.astype(np.uint8), dilated_kernel, iterations=1) - cc
                indices = np.argwhere(bounding == 1)
                
                cc_lst.append(cc)
                cc_coords.append(indices)
        
        # make union-find relationship
        coord_sz = len(cc_coords)
        for i in range(coord_sz):
            for j in range(i+1, coord_sz):
                min_distance = inf
                for x1, y1 in cc_coords[i]:
                    for x2, y2 in cc_coords[j]:
                        min_distance = min(min_distance, abs(x1-x2) + abs(y1-y2))
                if min_distance * 15 < max_distance: uf.merge(i, j)

        # fulfill labels
        label_dct = {}
        for i in range(coord_sz):
            if uf.find(i) not in label_dct:
                label_dct[uf.find(i)] = np.zeros_like(mask).astype(np.uint8)
            label_dct[uf.find(i)] += cc_lst[i].astype(np.uint8)
        
        rnt = [x for x in label_dct.values()]
        for i in range(len(rnt)): rnt[i][rnt[i] > 0] = 1

        return rnt

    def augment_synthetic(self, globe_image, globe_mask, globe_shape, hand_image, index_coord):
        x, y = index_coord
        h, w = globe_mask.shape[:2]

        index_sz = 20
        max_color_diff = 10

        augmented_countires, augmented_images, augmented_labels, augmented_globes = [], [], [], []
        label_text = ""

        kernel_size = 20
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        augmented_lmt = 5


        for color, country_id in self.color2id.items():
            if augmented_lmt == 0: break

            color0 = np.array(list(map(int, color[1:-1].split(","))))

            diff_map = np.abs(globe_mask - color0)
            diff_map = np.sum(diff_map, axis=2)
            
            activate_map_0 = np.where(diff_map < max_color_diff, 1, 0).astype(np.uint8)
            # do closing...
            activate_map_0 = cv2.morphologyEx(activate_map_0, cv2.MORPH_CLOSE, kernel)
            
            indices = np.argwhere(activate_map_0 == 1)

            if np.count_nonzero(activate_map_0 == 1) > self.min_area_limit:
                for region_i, activate_map in enumerate(self.get_connected_components_by_distance(activate_map_0, globe_mask)):
                    # 1. Move finger
                    globe_mask_b = globe_mask[:, :, 0]
                    globe_mask_all = np.zeros_like(globe_mask_b)
                    globe_mask_all[(globe_mask_b > 0) & (globe_mask_b < 255)] = 1

                    indices = np.argwhere(activate_map == 1)
                    random_index = np.random.choice(indices.shape[0])
                    ny, nx = indices[random_index]
                    dx, dy = nx - x, ny - y
                    shifted_hand = self.image_shift(hand_image, dx, dy)
                    
                    hand_mask_r = np.ones_like(globe_mask)
                    hand_mask_r[shifted_hand > 0] = 0
                    
                    augmented_image = globe_image * hand_mask_r + shifted_hand
                    augmented_images.append(augmented_image)
                    foreground_shape = np.where((globe_shape + shifted_hand) > 0, 1, 0)
                    augmented_globes.append(augmented_image * foreground_shape)
                    # 2. Fill label text
                    indices = np.where(activate_map == 1)

                    y1, y2 = indices[0].min() / h, indices[0].max() / h
                    x1, x2 = indices[1].min() / w, indices[1].max() / w
                    
                    label_text += " ".join(map(str, [country_id, (x1+x2) / 2, (y1+y2) / 2, x2-x1, y2-y1, "\n"]))

                    augmented_labels.append(" ".join(map(str, [20, nx / w, ny / h, index_sz / w, index_sz / h, "\n"]))) # x0, y0, w0, h0

                    augmented_countires.append("{}-{:0>2}".format(self.countries[country_id], region_i))

                    augmented_lmt -= 1
        
        for i in range(len(augmented_labels)):
            augmented_labels[i] += label_text
        
        return augmented_countires, augmented_images, augmented_labels, augmented_globes

    def split_elements(self, image_dir="globe_synthetic"):
        sorted_key = lambda x: int(x.split("_")[0])
        sample_ids = [x.split("_")[0] for x in sorted(os.listdir(image_dir), key=sorted_key) if "MaskGlobeHandle" in x]
        coordinates_files = [x for x in sorted(os.listdir(image_dir), key=sorted_key) if ")_Hand" in x]
        hand_files = [x for x in sorted(os.listdir(image_dir), key=sorted_key) if "_HandOnly" in x]

        indexs = list(range(len(sample_ids)))
        for i in tqdm(indexs):
            sample_id, hand_file, coord_file = sample_ids[i], hand_files[i], coordinates_files[i]

            dst_dir = image_dir + "_split/" + sample_id

            os.makedirs(dst_dir, exist_ok=True)

            # 1. binary mask
            mask_globe_handle = cv2.imread("{}/{}_MaskGlobeHandle.png".format(image_dir, sample_id), cv2.IMREAD_GRAYSCALE)
            mask_globe_handle[mask_globe_handle > 0] = 1
            mask_hand = cv2.imread("{}/{}_MaskHand.png".format(image_dir, sample_id), cv2.IMREAD_GRAYSCALE)
            mask_hand[mask_hand > 0] = 1


            # 2. save the extracted masks
            cv2.imwrite("{}/MaskGlobeHandle.png".format(dst_dir), mask_globe_handle * 255)
            cv2.imwrite("{}/MaskHand.png".format(dst_dir), mask_hand * 255)

            # 3. crop out the key element
            image_hand_only = cv2.imread("{}/{}".format(image_dir, hand_file), cv2.IMREAD_COLOR)

            split_hand = image_hand_only * mask_hand[:,:,np.newaxis]
            cv2.imwrite("{}/SplitHand.png".format(dst_dir), split_hand)

            l, r = coord_file.find("("), coord_file.find(")")
            index_coord = tuple(map(int, coord_file[l+1:r].split(","))) # the coordinate of the index finger

            image_globe_handle = cv2.imread("{}/{}_NoHand.png".format(image_dir, sample_id), cv2.IMREAD_COLOR) # fusion with the real dataset (COCO 2017)
            split_globe_handle = image_globe_handle * mask_globe_handle[:,:,np.newaxis]
            cv2.imwrite("{}/SplitGlobeHandle.png".format(dst_dir), split_globe_handle)

            # country mask
            mask_globe = cv2.imread("{}/{}_MaskGlobe.png".format(image_dir, sample_id), cv2.IMREAD_COLOR)
            globe_shape = cv2.imread("{}/{}_MaskGlobeHandle.png".format(image_dir, sample_id), cv2.IMREAD_COLOR)

            # 4. augment a patch of images
            augmented_items = self.augment_synthetic(image_globe_handle, mask_globe, globe_shape, split_hand, index_coord)

            augmented_lst = list(zip(*augmented_items))
            random.shuffle(augmented_lst)

            for country, image, label, globe in augmented_lst:
                cv2.imwrite("{}/{}_Hand.png".format(dst_dir, country), image)
                cv2.imwrite("{}/{}_Foreground.png".format(dst_dir, country), globe)
                with open("{}/{}_Hand.txt".format(dst_dir, country), "w") as file:
                    file.write(label)
            
            augmented_labels = augmented_items[2]
            # globe only
            for label in augmented_labels:
                image_globe = cv2.imread("{}/{}_NoHand.png".format(image_dir, sample_id), cv2.IMREAD_COLOR)
                cv2.imwrite("{}/Globe_NoHand.png".format(dst_dir), image_globe)
                label = "\n".join(label.split("\n")[1:])
                with open("{}/Globe_NoHand.txt".format(dst_dir), "w") as file:
                    file.write(label)
                break

    def make_yolo_from_augmentated(self, 
                                   sample_dir="globe_synthetic_split", 
                                   subset_radio={"train":0.9, "val":0.1}):
        dst_dct = {}

        dataset_dir = sample_dir.split("/")[-1]

        flag_dct = {"train":0, "val":1, "test":2, "images":".png", "labels":".txt"}

        for data_type, subset in product(["images", "labels"], ["train", "val", "test"]):
            dst_dir = "datasets/{}/{}/{}".format(dataset_dir, data_type, subset)
            dst_dct[(flag_dct[data_type], subset)] = dst_dir
            os.makedirs(dst_dir, exist_ok=True)

        
        sample_ids = os.listdir(sample_dir)

        sample_cnt = len(sample_ids)

        random.shuffle(sample_ids)

        subset_dct = {}

        # setting with ratio:
        train_lmt = int(subset_radio["train"] * sample_cnt)
        val_lmt = train_lmt + int(subset_radio["val"] * sample_cnt)

        for i in range(len(sample_ids)):
            if i < train_lmt: subset_dct[i] = "train"
            elif train_lmt <= i < val_lmt: subset_dct[i] = "val"
            else: subset_dct[i] = "test"

        for i, sample_id in tqdm(list(enumerate(sample_ids))):
            for file_name in os.listdir("{}/{}".format(sample_dir, sample_id)):
                src_file_path = "{}/{}/{}".format(sample_dir, sample_id, file_name)

                suffix = file_name[-4:]
                if "_" in file_name and "Foreground" not in file_name: # _ is the special identification
                    shutil.copy(src_file_path, "{}/{}_{}".format(dst_dct[(suffix, subset_dct[i])], sample_id, file_name))
        
        
        yaml_config = {
            "path" : "../datasets/{}".format(dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "test" : "images/test",
            "names" : self.country_dct_r
        }

        with open('datasets/{}/{}.yaml'.format(dataset_dir, dataset_dir), 'w') as yaml_file:
            yaml.dump(yaml_config, yaml_file, default_flow_style=False)

    def make_fused_dataset(self, COCO2017_dir="COCO2017_test", synthetic_dir="datasets_preprocess/synthetic_P_split"):
        def get_a_coco_image():
            coco_image_file = random.sample(os.listdir(COCO2017_dir), 1)[0]
            coco_image = cv2.imread("{}/{}".format(COCO2017_dir, coco_image_file), cv2.IMREAD_COLOR)
            return cv2.resize(coco_image, (self.res_w, self.res_h))

        for sample_dir in tqdm(sorted(os.listdir(synthetic_dir))):
            src_dir = "{}/{}".format(synthetic_dir, sample_dir)
            save_dir = src_dir.replace("synthetic", "fused")
            
            os.makedirs(save_dir, exist_ok=True)
            for image_file in os.listdir(src_dir):
                if "Foreground" in image_file:
                    foreground_image = cv2.imread("{}/{}".format(src_dir, image_file), cv2.IMREAD_COLOR)
                    background_mask = np.where(foreground_image == 0, 1, 0)
                    fused_image = get_a_coco_image() * background_mask + foreground_image
                    cv2.imwrite("{}/{}".format(save_dir, image_file), fused_image)
                else:
                    shutil.copyfile("{}/{}".format(src_dir, image_file), "{}/{}".format(save_dir, image_file))


    def get_valid_countries(self):
        train_image_dir = "datasets/globe_synthetic_split/images/train"
        s = set(sorted([x.split("_")[1] for x in os.listdir(train_image_dir)]))
        print(len(s),s)

if __name__=="__main__":
    sa = SyntheticAugmentation()

    # fused:
    # sa.make_fused_dataset()

    # data augmentation:
    dataset_dir = "datasets_preprocess/synthetic_P"
    sa.split_elements(image_dir=dataset_dir)

    sa.make_yolo_from_augmentated(sample_dir=dataset_dir+"_split")

    # sa.make_yolo_from_augmentated(sample_dir=dataset_dir+"_split")
    # check
    # sa.get_valid_countries()
    # fused_image = get_fused_image()
    # cv2.imwrite("temp.png", fused_image)
    pass

# TODO: country color randomization