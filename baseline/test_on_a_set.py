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
parser.add_argument('--img_num', default=25, type=int, help='test set size')
parser.add_argument('--test_task', default='background', type=str, help='whole, foreground, or background')
# ---------------------- model ----------------------
# baseline
# parser.add_argument('--actor', default='./actor.pkl', type=str, help='Actor model')
# parser.add_argument('--save_dir', default='baseline', type=str, help='save folder.')
# baseline transferred
# parser.add_argument('--actor', default='final_model/baseline-Paint-run4/actor.pkl', type=str, help='Actor model')
# parser.add_argument('--save_dir', default='transferred_back', type=str, help='save folder.')
# background actor
parser.add_argument('--actor', default='final_model/Paint-run41/actor_1.pkl', type=str, help='Actor model')
parser.add_argument('--save_dir', default='background', type=str, help='save folder.')
# foreground actor
# parser.add_argument('--actor', default='final_model/Paint-run41/actor_0.pkl', type=str, help='Actor model')
# parser.add_argument('--save_dir', default='foreground', type=str, help='save folder.')
# 全能 actor 0, 50w
# parser.add_argument('--actor', default='final_model/Paint-run32/actor_0.pkl', type=str, help='Actor model')
# parser.add_argument('--save_dir', default='ttt200', type=str, help='save folder.')
# 全能 actor 1, 50w
# parser.add_argument('--actor', default='final_model/Paint-run32/actor_1.pkl', type=str, help='Actor model')
# parser.add_argument('--save_dir', default='hybrid_fore', type=str, help='save folder.')
# 全能 actor 0, 25w
# parser.add_argument('--actor', default='final_model/Paint-run32-425/actor_0.pkl', type=str, help='Actor model')
# parser.add_argument('--save_dir', default='ttt200', type=str, help='save folder.')
# 全能 actor 1, 25w
# parser.add_argument('--actor', default='final_model/Paint-run32-425/actor_1.pkl', type=str, help='Actor model')
# parser.add_argument('--save_dir', default='hybrid_back', type=str, help='save folder.')
# ---------------------- model ----------------------
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=1, type=int, help='divide the target image to get better resolution')
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

def large2small(x):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
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
    cv2.imwrite(f'output_final/{args.save_dir}/generated'  + str(imgid) + '.png', output)



def get_blank_canvas():
    canvas = torch.zeros([1, 3, width, width]).to(device)
    return canvas


def prepare_img(img):
    patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
    patch_img = large2small(patch_img)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.

    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.
    return img, patch_img


def test(img, patch_img, canvas, img_id):
    global T, coord
    with torch.no_grad():
        if args.divide != 1:
            args.max_step = args.max_step // 2
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
            canvas, res = decode(actions, canvas)
            l2 = ((canvas - img) ** 2).mean()
            print('{}: canvas step {}, L2Loss = {}'.format(img_id, i, l2))
            # Since we are testing, no need to save images
            # for j in range(5):
            #     save_img(res[j], args.imgid)
            #     args.imgid += 1
        if args.divide != 1:
            canvas = canvas[0].detach().cpu().numpy()
            canvas = np.transpose(canvas, (1, 2, 0))    
            canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
            canvas = large2small(canvas)
            canvas = np.transpose(canvas, (0, 3, 1, 2))
            canvas = torch.tensor(canvas).to(device).float()
            coord = coord.expand(canvas_cnt, 2, width, width)
            T = T.expand(canvas_cnt, 1, width, width)
            for i in range(args.max_step):
                stepnum = T * i / args.max_step
                actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
                canvas, res = decode(actions, canvas)
                l2 = ((canvas - img) ** 2).mean()
                print('{}: divided canvas step {}, L2Loss = {}'.format(img_id, i, l2))
                # Since we are testing, no need to save images
                # for j in range(5):
                #     save_img(res[j], args.imgid, True)
                #     args.imgid += 1
        # save the final canvas
        save_img(canvas, img_id)
    return l2


def main():
    load_data()
    total_L2_loss = 0
    for img_id, i in enumerate(img_test):
        img, patch_img = prepare_img(i)
        canvas = get_blank_canvas()
        l2 = test(img, patch_img, canvas, img_id)
        total_L2_loss += l2
    print('Testing on {}, Mean L2 loss = {}'.format(args.test_task, total_L2_loss / len(img_test)))

if __name__ == '__main__':
    canvas_cnt = args.divide * args.divide
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)

    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width - 1.)
            coord[0, 1, i, j] = j / (width - 1.)
    coord = coord.to(device) # Coordconv

    Decoder = FCN()
    Decoder.load_state_dict(torch.load(args.renderer))
    actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
    actor.load_state_dict(torch.load(args.actor))
    actor = actor.to(device).eval()
    Decoder = Decoder.to(device).eval()

    os.system(f'mkdir output_final')
    os.system(f'mkdir output_final/{args.save_dir}')

    main()
