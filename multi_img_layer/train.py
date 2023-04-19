#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from DRL.ddpg import DDPG
from DRL.multi import fastenv
from utils.util import *
from utils.tensorboard import TensorBoard
import time

# exp = os.path.abspath('.').split('/')[-1]
exp = str(os.path.basename(os.getcwd())) + '_' + str(time.time())
# log_dir = os.path.join('../train_log', exp)
writer = TensorBoard('../train_log/{}'.format(exp))
os.system('ln -sf ../train_log/{} ./log'.format(exp))
os.system('mkdir ./model')

def train(agent: DDPG, env: fastenv, evaluate: Evaluator):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor
    # training loop
    while step <= train_times: # 2000000
        step += 1
        episode_steps += 1
        # reset the environment if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)    
        # action: 5 * strokes
        action = agent.select_action(observation, episode_steps, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action, episode_steps) # reward not used
        agent.observe(reward, observation, done, episode_steps)

        if step % 200 == 0:
            print('step: {}, episode: {}, episode_steps: {}, reward: {}'.format(step, episode, episode_steps, reward.mean()))

        # every 40 steps, update policy and reset the environment
        if (episode_steps >= max_step and max_step):
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                # if True:
                    # in test mode, the agent acts for a whole episode, and returns the total reward
                    reward, dist = evaluate(env, policy=agent.select_action, debug=debug)
                    # print, log, and save model.
                    if debug: prRed('eval - Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            tot_policy_loss = 0.
            tot_stroke_size = 0.
            # tot_critic_output = 0.
            # tot_gan_loss = 0.
            policy_loss_actor_sum = [0., 0., 0., 0.]
            stroke_size_actor_sum = [0., 0., 0., 0.]
            if step > args.warmup:
                # adjust learning rate
                if step < 10000 * max_step: # 10000 * 40, 0 - 400000
                    lr = (3e-4, 1e-3) # lr for critic, lr for actor
                elif step < 20000 * max_step: # 20000 * 40, 400000 - 800000
                    lr = (1e-4, 3e-4)
                else: # 800000 - 2000000
                    lr = (3e-5, 1e-4)
                # update policy
                for _ in range(episode_train_times):
                    policy_loss_sum, reg_stroke_size_sum, Q, value_loss, \
                        policy_loss_actors, stroke_size_actors = agent.update_policy(lr)
                    # __(policy_loss_sum) = pre_critic_output_sum + pre_gan_loss_sum
                    # Q, value_loss = agent.update_policy(lr)
                    tot_policy_loss += policy_loss_sum.data.cpu().numpy()
                    tot_stroke_size += reg_stroke_size_sum.data.cpu().numpy()
                    # tot_critic_output += pre_critic_output_sum.data.cpu().numpy()
                    # tot_gan_loss += pre_gan_loss_sum.data.cpu().numpy()
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                    for i in range(4):
                        policy_loss_actor_sum[i] += policy_loss_actors[i].data.cpu().numpy()
                        stroke_size_actor_sum[i] += stroke_size_actors[i].data.cpu().numpy()
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/policy_loss', tot_policy_loss / episode_train_times, step)
                writer.add_scalar('train/stroke_size', tot_stroke_size / episode_train_times, step)
                # writer.add_scalar('train/critic_output', tot_critic_output / episode_train_times, step)
                # writer.add_scalar('train/gan_loss', tot_gan_loss / episode_train_times, step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
                
                writer.add_scalar('train/policy_loss_actor0', policy_loss_actor_sum[0] / episode_train_times, step)
                writer.add_scalar('train/policy_loss_actor1', policy_loss_actor_sum[1] / episode_train_times, step)
                writer.add_scalar('train/policy_loss_actor2', policy_loss_actor_sum[2] / episode_train_times, step)
                writer.add_scalar('train/policy_loss_actor3', policy_loss_actor_sum[3] / episode_train_times, step)
                writer.add_scalar('train/stroke_size_actor0', stroke_size_actor_sum[0] / episode_train_times, step)
                writer.add_scalar('train/stroke_size_actor1', stroke_size_actor_sum[1] / episode_train_times, step)
                writer.add_scalar('train/stroke_size_actor2', stroke_size_actor_sum[2] / episode_train_times, step)
                writer.add_scalar('train/stroke_size_actor3', stroke_size_actor_sum[3] / episode_train_times, step)

                if debug: prRed(f'Update policy - Q:{tot_Q / episode_train_times:.3f}, critic_loss:{tot_value_loss / episode_train_times:.3f}')
            if debug: prBlack('episode{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset the environment and episode
            observation = None
            episode_steps = 0
            episode += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')

    # hyper-parameter
    parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95**5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--lambda_stroke_size_reg', default=1, type=float, help='weigh of stroke size regularization')
    parser.add_argument('--actor_num', default=4, type=int, help='how many actors to use.')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=1, type=int, help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes') # 2000000
    parser.add_argument('--episode_train_times', default=5, type=int, help='train times for each episode') # 10
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    
    args = parser.parse_args()    
    args.output = get_output_folder(args.output, "Paint")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Initialize environment, agent, and evaluator
    fenv = fastenv(args.max_step, args.env_batch, writer, args.actor_num)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, \
                 writer, args.resume, args.output, args.lambda_stroke_size_reg, args.actor_num)
    evaluate = Evaluator(args, writer)

    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
    # train
    train(agent, fenv, evaluate)
