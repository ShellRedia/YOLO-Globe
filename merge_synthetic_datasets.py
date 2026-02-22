import os
import shutil
from tqdm import tqdm

src_dirs = [
    "D:/UnrealProjects/GlobeSynthetic_v2/Saved/2024-7-17_22-21-50",
    "D:/UnrealProjects/GlobeSynthetic_v2/Saved/2024-7-18_0-56-40",
    "D:/UnrealProjects/GlobeSynthetic_v2/Saved/2024-7-18_10-6-9",
    "D:/UnrealProjects/GlobeSynthetic_v2/Saved/2024-7-18_12-25-25"
    ]
dst_dir = "datasets_preprocess/synthetic_default"

os.makedirs(dst_dir, exist_ok=True)

sample_count = 1 * 10 ** 3

for i, src_dir in enumerate(src_dirs):
    base_i = i * sample_count
    for file in tqdm(list(os.listdir(src_dir))):
        seg = file.split("_")
        sample_id = int(seg[0])
        if sample_id <= sample_count:
            file_c = "_".join([str(sample_id + base_i)] + seg[1:])
            shutil.copyfile(src_dir+"/"+file, dst_dir+"/"+file_c)