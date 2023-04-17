import os
import shutil

# 源文件夹和目标文件夹路径
src_dir = 'CelebAMask-HQ/CelebA-HQ-img/'
dst_dir = 'origin-img/'

# 遍历源文件夹中的前2000张图片
for i in range(2000):
    # 拼接出图片文件名
    filename = f"{i}.jpg"
    # 源文件路径
    src_path = os.path.join(src_dir, filename)
    # 目标文件路径
    dst_path = os.path.join(dst_dir, filename)
    # 将源文件复制到目标文件夹
    shutil.copy(src_path, dst_path)

# import torch
# def custom_loss(output, target):
#     # Define boundaries of flat region
#     lower_bound = 0.4
#     upper_bound = 0.6
#     width = upper_bound - lower_bound
#     # Calculate distance from target to flat region
#     distance = torch.abs(output - target)
#     dist_to_lower = torch.abs(distance - width/2) + torch.min(distance, torch.tensor(width/2.0))
#     # Calculate MSE loss
#     mse_loss = torch.mean(torch.pow(output - target, 2))
#     # Combine the two losses with a weighting factor
#     lambda_factor = 0.5
#     loss = lambda_factor * mse_loss + (1 - lambda_factor) * torch.mean(torch.pow(dist_to_lower, 2))
#     return loss

# for i in range (12):
#     a = i/10
#     print(a, custom_loss(torch.tensor(a), torch.tensor(0.5)))
# # print(custom_loss(0.8,0.5))
# # print(custom_loss(0.8,0.5))
# # print(custom_loss(0.8,0.5))

# loss = KLDiv(z | | target)
# KLDiv(p | | q) = sum(p(x) * log(p(x)/q(x)))