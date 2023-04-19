import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from Renderer.model import *
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *
from utils.util import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.
coord = coord.to(device)

criterion = nn.MSELoss()

Decoder = FCN()
Decoder.load_state_dict(torch.load('../renderer.pkl'))

def decode(x, canvas): # b * 5 * (10 + 3)
    '''
    The differentiable renderer that has been trained in stage 1. 
    Takes in x containing stroke information and RGB color values, 
    as well as the current canvas. 
    It decodes the stroke information and modifies the canvas accordingly.
    Note: the input x is a batch of 5 strokes, each stroke is a 13-dim vector.
    '''
    x = x.view(-1, 10 + 3) # [bx5, 13], 10: circle, 3: rgb values
    stroke = 1 - Decoder(x[:, :10]) # [bx5, 128, 128] 128x128
    stroke = stroke.view(-1, 128, 128, 1) # [bx5, 128, 128, 1]
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3) # [bx5, 128, 128, 3]
    stroke = stroke.permute(0, 3, 1, 2) # [bx5, 1, 128, 128]
    color_stroke = color_stroke.permute(0, 3, 1, 2) # [bx5, 3, 128, 128]
    stroke = stroke.view(-1, 5, 1, 128, 128) # [b, 5, 1, 128, 128]
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas

def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)
    
class DDPG(object):
    '''
    It initializes the actor and critic neural networks and their corresponding target networks, 
    creates a replay buffer for experience replay, 
    and defines methods for playing, updating, observing, and saving the agent's state.


    Methods:
    - evaluate(state, action, target=False): Given a state and an action, evaluates the expected reward by passing the state-action pair through the critic neural network. If target=True, uses the target critic network instead.
    - update_policy(lr): Updates the actor and critic neural networks by performing gradient descent on the loss functions. Returns the policy loss and value loss.
    - observe(reward, state, done, step): Adds a new experience to the replay buffer and update state.
    - select_action(state, return_fix=False, noise_factor=0): Given a state, returns an action. If return_fix=True, returns the action without adding noise.
    - reset(obs, factor): Resets the environment with a new observation and noise factor.
    - _play(state, target=False): returns an action by passing the state through the actor neural network. If target=True, uses the target actor network instead.
    - load_weights(path): Loads the weights of a previously saved checkpoint.
    - save_model(path): Saves the current model to the specified path.
    - eval(): Sets the neural networks to evaluation mode.
    - train(): Sets the neural networks to training mode.
    - _noise_action(noise_factor, state, action): Adds Gaussian noise to the action to encourage exploration.
    - _update_gan(state): Updates the generative adversarial network (GAN) using the current state.
    - _choose_device(): Sends the neural networks to the specified device (e.g. CPU or GPU).

    '''
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None, lambda_stroke_size_reg=0.001, actor_num=4):
        '''
        Args:
        - batch_size (int): the batch size for training the neural networks
        - env_batch (int): the number of environments for running the RL algorithm in parallel
        - max_step (int): the maximum number of steps per episode
        - tau (float): the target network update rate
        - discount (float): the discount factor for future rewards
        - rmsize (int): the size of the replay buffer
        - writer: the TensorBoard object for logging training progress
        - resume (str): the file path to a previously saved checkpoint for resuming training
        - output_path (str): the file path to save trained models and results
        '''

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size        
        self.lambda_stroke_size_reg = lambda_stroke_size_reg
        self.ACTOR_NUM = actor_num

        self.current_actor_num = 0
        self.actors = [ResNet(10, 18, 65) for _ in range(self.ACTOR_NUM)] # target, canvas, stepnum, coordconv, mask 3 + 3 + 1 + 2 + 1
        self.actor_targets = [ResNet(10, 18, 65) for _ in range(self.ACTOR_NUM)]
        self.actor_optims  = [Adam(actor.parameters(), lr=1e-2) for actor in self.actors]
        self.stroke_sizes = [
            # (lower bound, upper bound)
            (0.1, 99.0), # actor1. 用较粗的笔画画出图像中的远景 
            (0.1, 99.0), # actor2. 用较粗的笔画画出图像中的近景
            (0.0, 0.2), # actor3. 用较细的笔画画出图像中的远景
            (0.0, 0.2), # actor4. 用较细的笔画画出图像中的近景
        ]

        self.critic = ResNet_wobn(3 + 9, 18, 1) # add the last canvas for better prediction
        self.critic_target = ResNet_wobn(3 + 9, 18, 1) 
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-2)

        if (resume != None):
            self.load_weights(resume)

        for i in range(self.ACTOR_NUM):
            hard_update(self.actor_targets[i], self.actors[i])
        hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.memory = [rpm(rmsize * max_step // 10), rpm(rmsize * max_step // 10), \
                       rpm(rmsize * max_step // 10 * 4), rpm(rmsize * max_step // 10 * 4), rpm(rmsize * 1)] 

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0
        
        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action
        self._choose_device()        

    def _play(self, state, target=False):
        '''
        Passing the state through the actor neural network and return an action. 
        If target==True, uses the target actor network instead.
        '''
        state = torch.cat(( state[:, :6].float() / 255, # canvas and gt image
                            state[:, 6:7].float() / self.max_step, # T stepnum / max_step
                            state[:, 7:8], # mask
                            coord.expand(state.shape[0], 2, 128, 128) # coord encoding
                            ), 1)
        if target:
            actor_target = self.actor_targets[self.current_actor_num]
            action = actor_target(state)
        else:
            actor= self.actors[self.current_actor_num]
            action = actor(state) # state ---actor---> action
        return action

    def _update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3 : 6]
        mask = state[:, 7 : 8]

        # Apply the mask to the canvases and ground truth images
        masked_canvas = canvas * mask
        masked_gt = gt * mask
        fake, real, penal = update(masked_canvas.float() / 255, masked_gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)       
        
    def evaluate(self, state, action, target=0):
        '''
        Evaluates the expected reward by passing the state-action pair through the critic neural network. 
        Return Q value, gan_reward.
        If target=True, uses the target critic network instead.
        '''
        T = state[:, 6 : 7]
        gt = state[:, 3 : 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = decode(action, canvas0)
        mask = state[:, 7 : 8]

        # Apply the mask to the canvases and ground truth images
        masked_canvas0 = canvas0 * mask
        masked_canvas1 = canvas1 * mask
        masked_gt = gt * mask

        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        # L2_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)        
        coord_ = coord.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord_], 1)
        # canvas0 is not necessarily added
        if (target == 1):
            Q = self.critic_target(merged_state)
            return (Q + gan_reward), gan_reward
        elif (target == 2):
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
            return (Q + gan_reward), gan_reward
        else:
            gan_reward = cal_reward(masked_canvas1, masked_gt) - cal_reward(masked_canvas0, masked_gt)
            return gan_reward
    
    def update_policy(self, lr):
        '''
        Updates the actor and critic neural networks using loss functions. 
        Returns the policy loss and value loss.
        '''
        self.log += 1
        reg_stroke_size_sum = 0
        # pre_critic_output_sum = 0
        # pre_gan_loss_sum = 0
        policy_loss_sum = 0
        # value_loss_sum = 0
        tol_Q_sum = 0
        
        policy_loss_actors = [0, 0, 0, 0]
        stroke_size_actors = [0, 0, 0, 0]
        
        # batch_1_4 = self.batch_size // 4
        # buffer_for_critic = []
        
        for i in range(self.ACTOR_NUM):
            # train each actor
            self.current_actor_num = i

            # Sample batch from replay buffer of the current actor
            state, action, reward, next_state, terminal = self.memory[i].sample_batch(self.batch_size, device)
            # terminal is a bool flag indicating whether the episode has ended.
            # buffer_for_critic.append([state[batch_1_4 * i : batch_1_4 * i + batch_1_4], 
            #                           action, 
            #                           reward, 
            #                           next_state, 
            #                           terminal
            #                           ])

            # self._select_current_actor(step)
            # for param_group in self.critic_optim.param_groups:
            #     param_group['lr'] = lr[0]
            for param_group in self.actor_optims[i].param_groups:
                param_group['lr'] = lr[1]

            self._update_gan(next_state)
            

            action = self._play(state)
            pre_gan_loss = self.evaluate(state.detach(), action)
            policy_loss = -pre_gan_loss.mean() # -Q(s, a) --- pre_critic_output + pre_gan_loss
            
            # stroke size regularization
            stroke_size = self._compute_stroke_size(action) # [b*5*2, 1]
            lower_bound, upper_bound = self.stroke_sizes[i]
            # 将upper_bound扩展到指定形状
            upper_bound = torch.tensor(upper_bound)
            lower_bound = torch.tensor(lower_bound)
            upper_bound = upper_bound.expand((self.batch_size*5*2, 1)).to(stroke_size.device)
            lower_bound = lower_bound.expand((self.batch_size*5*2, 1)).to(stroke_size.device)

            reg_stroke_size = torch.max(torch.zeros_like(stroke_size), stroke_size - upper_bound)**2 + torch.max(torch.zeros_like(stroke_size), lower_bound - stroke_size)**2

            actor_total_loss = policy_loss - self.lambda_stroke_size_reg * reg_stroke_size.mean()
            # actor_total_loss = pre_critic_output + pre_gan_loss + a * reg_stroke_size
            self.actors[i].zero_grad()
            actor_total_loss.backward(retain_graph=True)
            self.actor_optims[i].step()
            
            # Target update
            soft_update(self.actor_targets[i], self.actors[i], self.tau)

            reg_stroke_size_sum += reg_stroke_size.mean()
            # pre_critic_output_sum += pre_critic_output 
            # pre_gan_loss_sum += pre_gan_loss
            policy_loss_sum += policy_loss
            # value_loss_sum += value_loss
            tol_Q_sum += actor_total_loss

            policy_loss_actors[i] = policy_loss
            stroke_size_actors[i] = reg_stroke_size.mean()

        state, action, reward, next_state, terminal = self.memory[4].sample_batch(self.batch_size, device)

        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]

        with torch.no_grad():
            next_action = self._play(next_state, True)
            target_q, target_gan_loss = self.evaluate(next_state, next_action, target=1)
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q
                
        cur_q, step_reward = self.evaluate(state, action, target=2)
        target_q += step_reward.detach()
        
        value_loss = criterion(cur_q, target_q) # L2(current_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)

         # Q, critic loss
        return policy_loss_sum, reg_stroke_size_sum, tol_Q_sum, value_loss, \
            policy_loss_actors, stroke_size_actors

    def _compute_stroke_size(self, action):
        '''
        Action contains stroke parameters (10) and rgb values (3).
        This function computes the stroke size (radius) for each stroke.
        Note that stroke sizes are linearly interpolated between the initial and final values.
        So I concatenate the initial and final values and return a tensor of size [batch_size * 5 * 2, 1].
        There is a 5, because there are 5 strokes per batch.
        '''
        action = action.view(-1, 10 + 3) # [bx5, 13], 10: circle, 3: rgb values
        z0 = action[:, 6 : 7] # radius, [bx5, 1]
        z2 = action[:, 7 : 8] # radius
        tmp = 1. / 100
        i1, i2 = 0, 99
        t1, t2 = i1 * tmp, i2 * tmp
        size1 = (1-t1) * z0 + t1 * z2 # radius at t1 (initial), [bx5, 1]
        size2 = (1-t2) * z0 + t2 * z2 # radius at t2 (final)
        sizes = torch.cat([size1, size2], 0) # [bx5x2, 1]
        return sizes

    def observe(self, reward, state, done, step):
        '''
        Adds a new experience to the replay buffer.
        Includes [the most recent state, action, reward, next state, done flag].
        Then update the most recent state.
        '''
        self._select_current_actor(step)
        s0 = self.state.to(device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = state.to(device='cpu')
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.memory[self.current_actor_num].append([s0[i], a[i], r[i], s1[i], d[i]])
            if (step == self.max_step):
                self.memory[4].append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def _noise_action(self, noise_factor, state, action):
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)
    
    def _select_current_actor(self, step):
        actor_step_num = self.max_step // 10
        if step <= actor_step_num:
            self.current_actor_num = 0
        elif step <= actor_step_num * 2:
            self.current_actor_num = 1
        elif step <= actor_step_num * 6:
            self.current_actor_num = 2
        else:
            self.current_actor_num = 3
    
    def select_action(self, state, step, return_fix=False, noise_factor=0):
        '''
        Given a state, returns an action. 
        If return_fix=True, returns the action without adding noise.
        '''
        self.eval()

        self._select_current_actor(step)

        with torch.no_grad():
            action = self._play(state)
            action = to_numpy(action)
        if noise_factor > 0:        
            action = self._noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        '''
        Resets the environment with a new observation and noise factor.
        '''
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load('{}/actor_{}.pkl'.format(path, i)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        load_gan(path)
        
    def save_model(self, path):
        self.critic.cpu()
        for i, actor in enumerate(self.actors):
            actor.cpu()
            torch.save(actor.state_dict(),'{}/actor_{}.pkl'.format(path, i))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
        save_gan(path)
        self._choose_device()

    def eval(self):
        '''
        Set eval mode
        '''
        for actor in self.actors:
            actor.eval()
        for actor_target in self.actor_targets:
            actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def train(self):
        '''
        Set train mode
        '''
        for actor in self.actors:
            actor.train()
        for actor_target in self.actor_targets:
            actor_target.train()
        self.critic.train()
        self.critic_target.train()
    
    def _choose_device(self):
        Decoder.to(device)
        for actor in self.actors:
            actor.to(device)
        for actor_target in self.actor_targets:
            actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
