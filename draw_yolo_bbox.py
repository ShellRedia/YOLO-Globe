import cv2
import os
import random
import pandas as pd
from tqdm import tqdm


class Draw_YOLO_Bounding_Box:
    def __init__(self, 
        label_dir="datasets/globe_real/labels/train",
        image_dir="datasets/globe_real/images/train",
        output_dir="yolo_bbox_output"
        ):

        self.label_dir, self.image_dir, self.output_dir = label_dir, image_dir, output_dir

        os.makedirs(output_dir, exist_ok=True)

        countries = [x.lower().strip() for x in pd.read_excel("globe_info/country_colormap.xlsx")["country"]][:20]
        self.countries = [x[0].upper() + x[1:] for x in countries]
        self.country_dct = dict(zip(self.countries, range(len(self.countries))))

        country_colormap = pd.read_excel("globe_info/country_colormap.xlsx")["color"][:20]
        self.color2id = dict(zip(country_colormap, range(len(country_colormap))))

        ids = list(map(str, range(len(self.countries))))
        self.country_dct_r = dict(zip(ids, self.countries))


    def process(self):
        random.seed(42)
        colors = [[random.randint(0, 255) for _ in range(3)]
                for _ in range(len(self.countries))]

        image_files = os.listdir(self.image_dir)

        box_total = 0
        image_total = 0
        for image_file in tqdm(image_files):
            box_num = self.draw_box_on_image(
                image_file, self.countries, colors, self.label_dir, self.image_dir, self.output_dir)  # 对图片画框
            box_total += box_num
            image_total += 1
            # print('Box number:', box_total, 'Image number:', image_total)


    def plot_one_box(self, x, image, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    



    def draw_box_on_image(self, image_file, classes, colors, LABEL_FOLDER, RAW_IMAGE_FOLDER, OUTPUT_IMAGE_FOLDER):
        """
        This function will add rectangle boxes on the images.
        """
        txt_path = os.path.join(LABEL_FOLDER, '%s.txt' % (image_file[:-4]))

        image_path = os.path.join(RAW_IMAGE_FOLDER, image_file)

        save_file_path = os.path.join(OUTPUT_IMAGE_FOLDER, image_file)  

        source_file = open(txt_path) if os.path.exists(txt_path) else []
        image = cv2.imread(image_path)
        try:
            height, width, channels = image.shape
        except:
            print('no shape info.')
            return 0

        box_number = 0
        for line in source_file:  
            staff = line.split()  
            class_idx = int(staff[0])

            x_center, y_center, w, h = float(
                staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
            x1 = round(x_center-w/2)
            y1 = round(y_center-h/2)
            x2 = round(x_center+w/2)
            y2 = round(y_center+h/2)

            self.plot_one_box([x1, y1, x2, y2], image, color=colors[class_idx],
                        label=classes[class_idx], line_thickness=None)

            cv2.imwrite(save_file_path, image)

            box_number += 1
        return box_number



if __name__ == '__main__':
    dybb = Draw_YOLO_Bounding_Box()
    dybb.process()
    pass