import geopandas as gpd
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from tqdm import tqdm

alpha = 0.5

random.seed(42)

overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)

# 灰度图像->单通道图像, Grayscale image -> single-channel image
to_blue = lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_green = lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_light_green = lambda x: np.array([np.zeros_like(x), x / 2, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1,2,0)).astype(dtype=np.uint8)

to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

# 读取国家边界数据

used_color = set()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



# fig.patch.set_facecolor('black')



def random_hex_color():
    """Generate a random hexadecimal color."""
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color

for country_name in tqdm(world.name):
    fig, ax = plt.subplots(figsize=(135, 77))
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    country = world[world.name == country_name]
    selected_color = random_hex_color()

    while selected_color in used_color:
        selected_color = random_hex_color()
    used_color.add(selected_color)

    country.plot(ax=ax, color="black")

    plt.savefig('countries/{}.png'.format(country_name))

    plt.close()

image = cv2.imread("Political_Map_Pat.PNG", cv2.IMREAD_COLOR)
h, w, c = image.shape

mask = cv2.imread("Colormap.png", cv2.IMREAD_COLOR)
mask = cv2.resize(mask, (w, h))


overlay_map = overlay(image, mask)
cv2.imwrite("overlay.png", overlay_map)


