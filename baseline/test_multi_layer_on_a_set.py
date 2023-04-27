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
parser.add_argument('--img_num', default=200, type=int, help='test set size')
parser.add_argument('--test_task', default='whole', type=str, help='whole, foreground, or background')
parser.add_argument('--save_dir', default='ttt200', type=str, help='save folder.')

parser.add_argument('--actor0', default='final_model/Paint-run41/actor_1.pkl', type=str, help='Background Actor model')
parser.add_argument('--actor1', default='final_model/Paint-run41/actor_0.pkl', type=str, help='Foreground Actor model')
parser.add_argument('--actor2', default='final_model/Paint-run32/actor_1.pkl', type=str, help='Full Actor model')
# parser.add_argument('--actor2', default='./actor.pkl', type=str, help='Full Actor model')
# parser.add_argument('--actor', default='./final_model/baseline-Paint-run4/actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
args = parser.parse_args()

img_test = []
msk_test = []
test_num = 0
origin_shape = None
def load_data():
    '''
    loads data from CelebA dataset as testing sets.
    '''
    global test_num, origin_shape
    for i in range(args.img_num):
        img_id0 = str(i)
        img_id = '%05d' % i

        img = cv2.imread('./data/origin_img/' + img_id0 + '.jpg', cv2.IMREAD_UNCHANGED)
        msk = cv2.imread('./data/merged_mask/' + img_id + '.png', cv2.IMREAD_GRAYSCALE)
        _, msk = cv2.threshold(msk, 127, 1, cv2.THRESH_BINARY)
        img = cv2.resize(img, (width, width))
        msk = cv2.resize(msk, (width, width))
        origin_shape = (img.shape[1], img.shape[0])
        if args.test_task == 'foreground':
            img = img * np.expand_dims(msk, axis=2)
        elif args.test_task == 'background':
            img = img * np.expand_dims(1 - msk, axis=2)
        elif args.test_task != 'whole':
            raise Exception('invalid test_task')

        test_num += 1
        img_test.append(img)
        msk_test.append(msk)
    print('finish loading data, {} testing images and masks'.format(str(test_num)))
        


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

# def save_img(res, imgid, divide=False, name='generated'):
#     output = res.detach().cpu().numpy() # d * d, 3, width, width    
#     output = np.transpose(output, (0, 2, 3, 1))
#     if divide:
#         output = small2large(output)
#         output = smooth(output)
#     else:
#         output = output[0]
#     output = (output * 255).astype('uint8')
#     output = cv2.resize(output, origin_shape)
#     cv2.imwrite('output/' + name + str(imgid) + '.png', output)
def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    cv2.imwrite(f'output_final/{args.save_dir}/generated' + str(imgid) + '.png', output)

def get_blank_canvas():
    canvas_full = torch.zeros([1, 3, width, width]).to(device)
    canvas_foreground = torch.zeros([1, 3, width, width]).to(device)
    canvas_background = torch.zeros([1, 3, width, width]).to(device)
    '''
    0. 远景，粗笔画
    1. 近景，粗笔画
    2. 远景，细笔画
    3. 近景，细笔画
    风格画：
    4. 背景虚化 (1+2+4) style_blur_background
    5. 前景虚化 (1+2+3) style_blur_foreground
    6. 前景 (2+4) style_only_foreground
    7. 背景 (1+3) style_only_background
    8. 粗画 (1+2) style_only_coarse
    9. 细画 (3+4) style_only_fine
    '''
    canvas_layers = [torch.zeros([1, 3, width, width]).to(device) for _ in range(10)]
    canvas_packed = canvas_full, canvas_foreground, canvas_background, canvas_layers
    return canvas_packed


def prepare_img(img, mask):
    mask_opposite = 1 - mask
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
    img_packed = img, mask, mask_opposite, img_background, img_foreground, patch_img, patch_mask, patch_mask_opposite, patch_img_background, patch_img_foreground
    return img_packed

def draw_on_canvas(step, actions, canvas, img, mask, imgids_no, name, divide=False, img_id=0):
    global imgids
    old_canvas = canvas.clone()
    canvas, ress = decode(actions, canvas)
    if name is not None:
        print('{}: step {}, L2Loss = {}'.format(img_id, step, ((canvas - img) ** 2).mean()))
    #     print('{} step {}, L2Loss = {}'.format(name, step, ((canvas - img) ** 2).mean()))
    #     for j in range(5):
    #         res = ress[j] * mask + old_canvas * (1 - mask)
    #         save_img(res, imgids[imgids_no], divide, name)
    #         imgids[imgids_no] += 1
    return canvas * mask + old_canvas * (1 - mask)


def test(img_packed, canvas_packed, img_id):
    global T, coord
    with torch.no_grad():
        img, mask, mask_opposite, img_background, img_foreground, patch_img, patch_mask, patch_mask_opposite, patch_img_background, patch_img_foreground = img_packed 
        canvas_full, canvas_foreground, canvas_background, canvas_layers = canvas_packed 
        step_div = [args.max_step // 8 * 1, args.max_step // 8 * 2, args.max_step // 8 * 5, args.max_step]
        # 1. 远景，粗笔画
        for i in range(step_div[0]):
            stepnum = T * i / args.max_step
            actions = actor1(torch.cat([canvas_background, img_background, stepnum, coord], 1))

            canvas_background = draw_on_canvas(i, actions, canvas_background, img_background, mask_opposite, 0, None, img_id=img_id)
            canvas_full = draw_on_canvas(i, actions, canvas_full, img_background, mask_opposite, 0, 'full', img_id=img_id)
            # canvas_layers[0] = draw_on_canvas(i, actions, canvas_layers[0], img_background, mask_opposite, 1, 'background_rough')
            # canvas_layers[4] = draw_on_canvas(i, actions, canvas_layers[4], img_background, mask_opposite, 5, 'style_blur_background')
            # canvas_layers[5] = draw_on_canvas(i, actions, canvas_layers[5], img_background, mask_opposite, 6, 'style_blur_foreground')
            # canvas_layers[7] = draw_on_canvas(i, actions, canvas_layers[7], img_background, mask_opposite, 8, 'style_only_background')
            # canvas_layers[8] = draw_on_canvas(i, actions, canvas_layers[8], img_background, mask_opposite, 9, 'style_only_coarse')
            
        # 2. 近景，粗笔画
        for i in range(step_div[0], step_div[1]):
            stepnum = T * i / args.max_step
            actions = actor0(torch.cat([canvas_foreground, img_foreground, stepnum, coord], 1))
            canvas_foreground = draw_on_canvas(i, actions, canvas_foreground, img_foreground, mask, 0, None, img_id=img_id)
            canvas_full = draw_on_canvas(i, actions, canvas_full, img_foreground, mask, 0, 'full', img_id=img_id)
            # canvas_layers[1] = draw_on_canvas(i, actions, canvas_layers[1], img_foreground, mask, 2, 'foreground_rough')
            # canvas_layers[4] = draw_on_canvas(i, actions, canvas_layers[4], img_foreground, mask, 5, 'style_blur_background')
            # canvas_layers[5] = draw_on_canvas(i, actions, canvas_layers[5], img_foreground, mask, 6, 'style_blur_foreground')
            # canvas_layers[6] = draw_on_canvas(i, actions, canvas_layers[6], img_foreground, mask, 7, 'style_only_foreground')
            # canvas_layers[8] = draw_on_canvas(i, actions, canvas_layers[8], img_foreground, mask, 9, 'style_only_coarse')

        if args.divide != 1:
            canvas_full = canvas_full[0].detach().cpu().numpy()
            canvas_full = np.transpose(canvas_full, (1, 2, 0))    
            canvas_full = cv2.resize(canvas_full, (width * args.divide, width * args.divide))
            canvas_full = large2small(canvas_full)
            canvas_full = np.transpose(canvas_full, (0, 3, 1, 2))
            canvas_full = torch.tensor(canvas_full).to(device).float()
            canvas_foreground = canvas_foreground[0].detach().cpu().numpy()
            canvas_foreground = np.transpose(canvas_foreground, (1, 2, 0))    
            canvas_foreground = cv2.resize(canvas_foreground, (width * args.divide, width * args.divide))
            canvas_foreground = large2small(canvas_foreground)
            canvas_foreground = np.transpose(canvas_foreground, (0, 3, 1, 2))
            canvas_foreground = torch.tensor(canvas_foreground).to(device).float()
            canvas_background = canvas_background[0].detach().cpu().numpy()
            canvas_background = np.transpose(canvas_background, (1, 2, 0))    
            canvas_background = cv2.resize(canvas_background, (width * args.divide, width * args.divide))
            canvas_background = large2small(canvas_background)
            canvas_background = np.transpose(canvas_background, (0, 3, 1, 2))
            canvas_background = torch.tensor(canvas_background).to(device).float()
            # layer_list = [4, 5, 6, 7, 9]
            # for l in layer_list:
            #     canvas_layers[l] = canvas_layers[l][0].detach().cpu().numpy()
            #     canvas_layers[l] = np.transpose(canvas_layers[l], (1, 2, 0))    
            #     canvas_layers[l] = cv2.resize(canvas_layers[l], (width * args.divide, width * args.divide))
            #     canvas_layers[l] = large2small(canvas_layers[l])
            #     canvas_layers[l] = np.transpose(canvas_layers[l], (0, 3, 1, 2))
            #     canvas_layers[l] = torch.tensor(canvas_layers[l]).to(device).float()

                
            coord_patch = coord.expand(canvas_cnt, 2, width, width)
            T_patch = T.expand(canvas_cnt, 1, width, width)
            # 3. 远景，细笔画
            for i in range(step_div[1], step_div[2]):
                stepnum = T_patch * i / args.max_step
                actions = actor2(torch.cat([canvas_background, patch_img_background, stepnum, coord_patch], 1))
                canvas_background = draw_on_canvas(i, actions, canvas_background, patch_img_background, patch_mask_opposite, 0, None, divide=True, img_id=img_id)
                canvas_full = draw_on_canvas(i, actions, canvas_full, patch_img_background, patch_mask_opposite, 0, 'full', divide=True, img_id=img_id)
                # canvas_layers[2] = draw_on_canvas(i, actions, canvas_layers[2], patch_img_background, patch_mask_opposite, 3, 'background_fine', divide=True)
                # canvas_layers[5] = draw_on_canvas(i, actions, canvas_layers[5], patch_img_background, patch_mask_opposite, 6, 'style_blur_foreground', divide=True)
                # canvas_layers[7] = draw_on_canvas(i, actions, canvas_layers[7], patch_img_background, patch_mask_opposite, 8, 'style_only_background', divide=True)
                # canvas_layers[9] = draw_on_canvas(i, actions, canvas_layers[9], patch_img_background, patch_mask_opposite, 10, 'style_only_fine', divide=True)
            # 4. 近景，细笔画
            for i in range(step_div[2], step_div[3]):
                stepnum = T_patch * i / args.max_step
                actions = actor2(torch.cat([canvas_foreground, patch_img_foreground, stepnum, coord_patch], 1))
                canvas_foreground = draw_on_canvas(i, actions, canvas_foreground, patch_img_foreground, patch_mask, 0, None, divide=True, img_id=img_id)
                canvas_full = draw_on_canvas(i, actions, canvas_full, patch_img_foreground, patch_mask, 0, 'full', divide=True, img_id=img_id)
                # canvas_layers[3] = draw_on_canvas(i, actions, canvas_layers[3], patch_img_foreground, patch_mask, 4, 'foreground_fine', divide=True)
                # canvas_layers[4] = draw_on_canvas(i, actions, canvas_layers[4], patch_img_foreground, patch_mask, 5, 'style_blur_background', divide=True)
                # canvas_layers[6] = draw_on_canvas(i, actions, canvas_layers[6], patch_img_foreground, patch_mask, 7, 'style_only_foreground', divide=True)
                # canvas_layers[9] = draw_on_canvas(i, actions, canvas_layers[9], patch_img_foreground, patch_mask, 10, 'style_only_fine', divide=True)
        save_img(canvas_full, img_id, True)
        l2 = ((canvas_full - patch_img) ** 2).mean()
        return l2

def main():
    load_data()
    total_L2_loss = 0
    for img_id in range(len(img_test)):
        i, m = img_test[img_id], msk_test[img_id]
        img_packed = prepare_img(i, m)
        canvas_packed = get_blank_canvas()
        l2 = test(img_packed, canvas_packed, img_id)
        total_L2_loss += l2
    print('Testing on {}, Mean L2 loss = {}'.format(args.test_task, total_L2_loss / len(img_test)))

if __name__ == '__main__':
    canvas_cnt = args.divide * args.divide
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    # img = cv2.imread(args.img, cv2.IMREAD_COLOR)

    # mask_file = args.img.replace('origin_img', 'merged_mask').replace('.jpg', '.png')
    # img_id = eval(mask_file.split('/')[-1].split('.')[0])
    # mask_file = mask_file.replace(str(img_id), f'{img_id:05d}')
    # mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    # _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    # mask = mask[..., np.newaxis]
    # mask_opposite = torch.logical_not(torch.tensor(mask)).float()
    # mask_opposite = np.array(mask_opposite)


    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width - 1.)
            coord[0, 1, i, j] = j / (width - 1.)
    coord = coord.to(device) # Coordconv

    Decoder = FCN()
    Decoder.load_state_dict(torch.load(args.renderer))

    actor0 = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
    actor0.load_state_dict(torch.load(args.actor0))
    actor0 = actor0.to(device).eval()
    actor1 = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
    actor1.load_state_dict(torch.load(args.actor1))
    actor1 = actor1.to(device).eval()
    actor2 = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
    actor2.load_state_dict(torch.load(args.actor2))
    actor2 = actor2.to(device).eval()
    Decoder = Decoder.to(device).eval()

    os.system(f'mkdir output_final')
    os.system(f'mkdir output_final/{args.save_dir}')

    main()