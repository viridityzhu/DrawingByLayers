import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
from DRL.ddpg import decode
from utils.util import *
from PIL import Image
from torchvision import transforms, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aug = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             ])

width = 128
convas_area = width * width

img_train = []
img_test = []
train_num = 0
test_num = 0

class Paint:
    '''
    The Paint class represents a painting environment with a canvas and a target image to be painted. 
    The environment can be reset, and actions can be taken to modify the canvas.
    Methods:
        - load_data(): loads data from a dataset and separates it into training and testing sets.
        - _pre_data(id, test): preprocesses an image by applying augmentations and transposing it.
        - reset(test=False, begin_num=False): resets the environment to its initial state and returns the initial observation.
        - observation(): returns the current observation of the environment.
        - step(action): performs an action on the environment and returns the next observation, reward, done flag, and additional information.
        - _cal_trans(s, t): calculates the transformation of a source image given a target image.
        - _cal_dis(): calculates the distance between the current canvas and the target image.
        - _cal_reward(): calculates the reward obtained for the current step.
    '''
    def __init__(self, batch_size, max_step):
        '''
        Args:
            - batch_size (int): the number of images in each batch
            - max_step (int): the maximum number of steps allowed for each episode
        '''
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        self.observation_space = (self.batch_size, width, width, 7)
        self.test = False # whether the environment is in test mode or not
        
    def load_data(self):
        '''
        loads data from CelebA dataset and separates it into training and testing sets.
        '''
        global train_num, test_num
        for i in range(200000):
            img_id = '%06d' % (i + 1)
            try:
                img = cv2.imread('./data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                if i > 2000:                
                    train_num += 1
                    img_train.append(img)
                else:
                    test_num += 1
                    img_test.append(img)
            finally:
                if (i + 1) % 10000 == 0:                    
                    print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))
        
    def _pre_data(self, id, test):
        '''
        preprocesses an image by applying augmentations and transposing it.
        '''
        if test:
            img = img_test[id]
        else:
            img = img_train[id]
        if not test:
            img = aug(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))
    
    def reset(self, test=False, begin_num=False):
        '''
        resets the environment to its initial state and returns the initial observation.
        '''
        self.test = test
        self.imgid = [0] * self.batch_size # a list containing the index of the current image in the batch
        self.gt = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num)  % test_num
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self._pre_data(id, test))
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        self.lastdis = self.ini_dis = self._cal_dis()
        return self.observation()
    
    def observation(self):
        '''
        returns the current observation (state) of the environment.
        Current observation (state) includes the current canvas, the target image, and the number of steps taken so far.
        - canvas B * 3 * width * width
        - gt B * 3 * width * width
        - T B * 1 * width * width
        '''
        ob = []
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(device)), 1) # canvas, img, T

    def _cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action):
        '''
        performs an action on the environment and returns:
            - the next observation
            - reward
            - done flag
            - additional information (None)
        '''
        self.canvas = (decode(action, self.canvas.float() / 255) * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self._cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def _cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    
    def _cal_reward(self):
        dis = self._cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)
