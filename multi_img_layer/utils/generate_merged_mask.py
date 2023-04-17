# from PIL import Image
# import numpy as np
# import glob
# import os

# # 创建一个空白的黑白图像，大小为 (width, height)
# width, height = 512, 512
# final_img = np.zeros((width, height, 3))

# path = "test/"
# files = glob.glob(path + "*.png")

# # 创建一个字典用于存储文件名相同前缀的文件
# file_dict = {}
# for file in files:
#     # 获取文件名的前缀
#     prefix = os.path.basename(file)[:5]
#     if prefix in file_dict:
#         file_dict[prefix].append(file)
#     else:
#         file_dict[prefix] = [file]

# for prefix, files in file_dict.items():
#     # 创建一个空白的黑白图像，大小为 (width, height)
#     combined_image = np.zeros((width, height, 3))
#     for file in files:
#         img = Image.open(file).convert('1')  # 读取并将图像转换为黑白二值图像
#         img_arr = np.array(img)
#         img_arr[img_arr == 255] = 1
#         img_arr = np.expand_dims(img_arr, axis=-1)

#         final_img = np.maximum(combined_image, img_arr) * 255
#         # final_img = final_img * 255
#         # final_img = final_img.astype(np.uint8)
#         final_img = Image.fromarray(final_img.astype(np.uint8))

#         # combined_image = Image.fromarray(np.maximum(np.array(combined_image), img_arr.astype(np.uint8)*255).astype(np.uint8))
#     final_img.save(f"{prefix}_merged.png")

from PIL import Image
import numpy as np
import glob
import os

# 创建一个空白的黑白图像，大小为 (width, height)
width, height = 512, 512
# final_img = np.zeros((width, height, 3))

path = "CelebAMask-HQ-mask-anno/0/"
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



# import cv2
# import numpy as np

# # 加载图片和掩码
# img = cv2.imread('test/00000_hair.png')
# mask = cv2.imread('test/00000_skin.png', 0)

# # 使用掩码将图片分成两部分
# face = cv2.bitwise_and(img, img, mask=mask)
# background = cv2.bitwise_and(img, img, mask=~mask)

# # 显示结果
# cv2.imshow('Face', face)
# cv2.imshow('Background', background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 在上面的代码中，我们首先使用OpenCV的imread函数加载原始图片和语义分割掩码。然后，我们使用bitwise_and函数将原始图片和掩码进行按位与运算，得到分割后的人脸部分和背景部分。

# # 注意，这个示例假设掩码文件是一个黑白二值图像，其中白色表示人脸区域，黑色表示背景区域。如果掩码文件不是这种格式，您可能需要在加载后对其进行预处理。





