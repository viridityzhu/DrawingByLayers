import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F

from DRL.actor import *
from Renderer.stroke_gen import *
from Renderer.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='image/test.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
args = parser.parse_args()

canvas_cnt = args.divide * args.divide
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
img = cv2.imread(args.img, cv2.IMREAD_COLOR)

mask_file = args.img.replace('origin_img', 'merged_mask').replace('.jpg', '.png')
img_id = eval(mask_file.split('/')[-1].split('.')[0])
mask_file = mask_file.replace(str(img_id), f'{img_id:05d}')
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
mask = mask[..., np.newaxis]
mask_opposite = torch.logical_not(torch.tensor(mask)).float()
mask_opposite = np.array(mask_opposite)

origin_shape = (img.shape[1], img.shape[0])

coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device) # Coordconv

Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer))

def decode(x, canvas): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res

def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)    
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x

def large2small(x, is_rgb=True):
    # (d * width, d * width) -> (d * d, width, width)
    if is_rgb:
        last_dim = 3
    else:
        last_dim = 1
    x = x.reshape(args.divide, width, args.divide, width, last_dim)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, last_dim)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, imgid, divide=False, name='generated'):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    cv2.imwrite('output/' + name + str(imgid) + '.png', output)

def draw_on_canvas(step, actions, canvas, img, imgids_no, name, divide=False):
    global imgids
    canvas, res = decode(actions, canvas)
    print('{} step {}, L2Loss = {}'.format(name, step, ((canvas - img) ** 2).mean()))
    for j in range(5):
        save_img(res[j], imgids[imgids_no], divide, name)
        imgids[imgids_no] += 1
    return canvas

actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()
Decoder = Decoder.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)
'''
0. 远景，粗笔画
1. 近景，粗笔画
2. 远景，细笔画
3. 近景，细笔画
风格画：
4. 背景虚化 (1+3+4) style_blur_background
5. 前景虚化 (2+3+4) style_blur_foreground
6. 前景 (2+4) style_only_foreground
7. 背景 (1+3) style_only_background
8. 粗画 (1+2) style_only_coarse
9. 细画 (3+4) style_only_fine
'''
canvas_layers = [torch.zeros([1, 3, width, width]).to(device) for _ in range(10)]
imgids = [0 for _ in range(10 + 1)]
imgids[0] = args.imgid


patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.

patch_mask = cv2.resize(mask, (width * args.divide, width * args.divide))
patch_mask = large2small(patch_mask, is_rgb=False)
patch_mask = np.transpose(patch_mask, (0, 3, 1, 2))
patch_mask = torch.tensor(patch_mask).to(device).float()

patch_mask_opposite = cv2.resize(mask_opposite, (width * args.divide, width * args.divide))
patch_mask_opposite = large2small(patch_mask_opposite, is_rgb=False)
patch_mask_opposite = np.transpose(patch_mask_opposite, (0, 3, 1, 2))
patch_mask_opposite = torch.tensor(patch_mask_opposite).to(device).float()

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img).to(device).float() / 255.

mask = cv2.resize(mask, (width, width))
mask = mask.reshape(1, width, width, 1)
mask = np.transpose(mask, (0, 3, 1, 2))
mask = torch.tensor(mask).to(device)

mask_opposite = cv2.resize(mask_opposite, (width, width))
mask_opposite = mask_opposite.reshape(1, width, width, 1)
mask_opposite = np.transpose(mask_opposite, (0, 3, 1, 2))
mask_opposite = torch.tensor(mask_opposite).to(device)

img_background = img * mask_opposite
img_foreground = img * mask
patch_img_background = patch_img * patch_mask_opposite
patch_img_foreground = patch_img * patch_mask

os.system('mkdir output')

with torch.no_grad():
    if args.divide != 1:
        args.max_step = args.max_step // 2
    # 1. 远景，粗笔画
    for i in range(args.max_step // 2):
        stepnum = T * i / args.max_step
        actions = actor(torch.cat([canvas, img_background, stepnum, coord], 1))

        canvas = draw_on_canvas(i, actions, canvas, img_background, 0, 'full')
        canvas_layers[0] = draw_on_canvas(i, actions, canvas_layers[0], img_background, 1, 'background_rough')
        canvas_layers[4] = draw_on_canvas(i, actions, canvas_layers[4], img_background, 5, 'style_blur_background')
        canvas_layers[7] = draw_on_canvas(i, actions, canvas_layers[7], img_background, 8, 'style_only_background')
        canvas_layers[8] = draw_on_canvas(i, actions, canvas_layers[8], img_background, 9, 'style_only_coarse')
        
    # 2. 近景，粗笔画
    for i in range(args.max_step // 2, args.max_step):
        stepnum = T * i / args.max_step
        actions = actor(torch.cat([canvas, img_foreground, stepnum, coord], 1))
        canvas = draw_on_canvas(i, actions, canvas, img_foreground, 0, 'full')
        canvas_layers[1] = draw_on_canvas(i, actions, canvas_layers[1], img_foreground, 2, 'foreground_rough')
        canvas_layers[5] = draw_on_canvas(i, actions, canvas_layers[5], img_foreground, 6, 'style_blur_foreground')
        canvas_layers[6] = draw_on_canvas(i, actions, canvas_layers[6], img_foreground, 7, 'style_only_foreground')
        canvas_layers[8] = draw_on_canvas(i, actions, canvas_layers[8], img_foreground, 9, 'style_only_coarse')

    if args.divide != 1:
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))    
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        coord = coord.expand(canvas_cnt, 2, width, width)
        T = T.expand(canvas_cnt, 1, width, width)
        # 3. 远景，细笔画
        for i in range(args.max_step // 2):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, patch_img_background, stepnum, coord], 1))
            canvas = draw_on_canvas(i, actions, canvas, patch_img_background, 0, 'full', divide=True)
            canvas_layers[2] = draw_on_canvas(i, actions, canvas_layers[2], patch_img_background, 3, 'background_fine', divide=True)
            canvas_layers[4] = draw_on_canvas(i, actions, canvas_layers[4], patch_img_background, 5, 'style_blur_background', divide=True)
            canvas_layers[5] = draw_on_canvas(i, actions, canvas_layers[5], patch_img_background, 6, 'style_blur_foreground', divide=True)
            canvas_layers[7] = draw_on_canvas(i, actions, canvas_layers[7], patch_img_background, 8, 'style_only_background', divide=True)
            canvas_layers[9] = draw_on_canvas(i, actions, canvas_layers[9], patch_img_background, 10, 'style_only_fine', divide=True)
        # 4. 近景，细笔画
        for i in range(args.max_step // 2, args.max_step):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, patch_img_foreground, stepnum, coord], 1))
            canvas = draw_on_canvas(i, actions, canvas, patch_img_foreground, 0, 'full', divide=True)
            canvas_layers[3] = draw_on_canvas(i, actions, canvas_layers[3], patch_img_foreground, 4, 'foreground_fine', divide=True)
            canvas_layers[4] = draw_on_canvas(i, actions, canvas_layers[4], patch_img_foreground, 5, 'style_blur_background', divide=True)
            canvas_layers[5] = draw_on_canvas(i, actions, canvas_layers[5], patch_img_foreground, 6, 'style_blur_foreground', divide=True)
            canvas_layers[6] = draw_on_canvas(i, actions, canvas_layers[6], patch_img_foreground, 7, 'style_only_foreground', divide=True)
            canvas_layers[9] = draw_on_canvas(i, actions, canvas_layers[9], patch_img_foreground, 10, 'style_only_fine', divide=True)
