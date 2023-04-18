# \ICCV2019-LearningToPaint\data\generate_merged_mask.py
from PIL import Image
import numpy as np
import glob
import os

# 创建一个空白的黑白图像，大小为 (width, height)
width, height = 512, 512
# final_img = np.zeros((width, height, 3))

for i in range(15):
    path = "CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(i) + "/"
    files = glob.glob(path + "*.png")

    # 创建一个字典用于存储文件名相同前缀的文件
    file_dict = {}
    for file in files:
        # 获取文件名的前缀
        prefix = os.path.basename(file)[:5]
        if prefix in file_dict:
            file_dict[prefix].append(file)
        else:
            file_dict[prefix] = [file]

    for prefix, files in file_dict.items():
        final_img = np.zeros((width, height, 3))
        for filename in files:
            img = Image.open(filename).convert('1')  # 读取并将图像转换为黑白二值图像
            img_arr = np.array(img)
            img_arr[img_arr == 255] = 1
            img_arr = np.expand_dims(img_arr, axis=-1)
            final_img = np.maximum(final_img, img_arr)

        final_img = final_img * 255
        final_img = final_img.astype(np.uint8)
        final_img = Image.fromarray(final_img)

        final_img.save(f"merged_mask/{prefix}.png")




