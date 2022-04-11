# # from models.wrn import Wide_ResNet
# #
# import torch
#
# # from dataset.cifar_dataset import CIFAR10, CIFAR100
# # import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# input_size = 32
# dataroot = "./adv/data/cifar10"
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(
#         root=dataroot,
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [
#                 transforms.Resize((input_size, input_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ]
#         ),
#     ),
#     batch_size=256,
#     shuffle=True,
#     num_workers=0,
#     drop_last=True,
# )

# import gym
# import time
# from gym.envs.registration import register
# import argparse
# import gym
# import time
def set_seed(seed=None):
    import torch
    import numpy as np
    import random

    def set_torch_deterministic():
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = False
        cudnn.deterministic = True

    set_torch_deterministic()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


set_seed(40)
import gym

gym.logger.set_level(40)
# parser = argparse.ArgumentParser(description=None)
# parser.add_argument("-e", "--env", default="soccer", type=str)
#
# args = parser.parse_args()

# from rarl.envs.adversarial.mujoco.ant_heel import AntHeelEnv
# def main():
#     print()
# from tvt_psro.envs.gym_multigrid.envs import SoccerGame4HEnv10x15N2
#
# env = SoccerGame4HEnv10x15N2()
# print(env.state_space)
# print(env.observation_space)
# print(env.agents)
# print(env.get_state().shape)
# print(env.agents)
# if args.env == 'soccer':
#     register(
#         id='multigrid-soccer-v0',
#         entry_point='tvt_psro.envs.gym_multigrid.envs:SoccerGame4HEnv10x15N2',
#     )
#     env = gym.make('multigrid-soccer-v0')

# register(
#     id='hopper-heel-adv',
#     entry_point='rarl.envs.adversarial.mujoco.hopper_heel:HopperHeelEnv',
#     reward_threshold=6000.0,
#     max_episode_steps=1000,
# )
#
# env = gym.make('hopper-heel-adv')
# # print(env)
# # obs = env.reset()
# # print(obs)
# #
# # print(env.observation_space)
# # print(env.action_space)
# #
# # print(env.adv_action_space)

# from envs.adversarial.mujoco.hopper_heel import HopperHeelEnv
# import ctypes
#
# env = HopperHeelEnv()

env = gym.make("Hopper-v4")
obs = env.reset()
print(obs)

print(env.action_space)

env_name = "Hopper-v4"
has_continuous_action_space = True

# has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = 1000  # max timesteps in one episode
max_training_timesteps = int(
    3e6
)  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)  # save model frequency (in num timesteps)

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = (
    0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
)
min_action_std = (
    0.1  # minimum action_std (stop decay after action_std <= min_action_std)
)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

#####################################################

#  Note : print/log frequencies should be > than max_ep_len

# ############### PPO hyperparameters ################

update_timestep = max_ep_len * 4  # update policy every n timesteps
K_epochs = 80  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

random_seed = 0  # set random seed if required (0 = no random seed)

#####################################################

print("training environment name : " + env_name)

env = gym.make(env_name)

# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

import os

directory = "./rarl/PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + "/" + env_name + "/"
if not os.path.exists(directory):
    os.makedirs(directory)
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, 0, 0)
from rarl.ppo.PPO import PPO

ppo_agent = PPO(
    state_dim,
    action_dim,
    lr_actor,
    lr_critic,
    gamma,
    K_epochs,
    eps_clip,
    has_continuous_action_space,
    action_std,
)

ppo_agent.load(checkpoint_path)

print(ppo_agent)

# steps = 0
# while not done:
#     action = {}
#     action["pro"] = env.action_space.sample()
#     action["adv"] = env.adv_action_space.sample()
#     action['adv'] = np.array([5.0, 5.0], dtype=np.float32)
#     print(action)
#     obs, reward, done, info = env.step(action)
#     print(reward)
#     steps += 1
#     time.sleep(2)
#     # env.render()
#     if steps > 500:
#         break
from rarl.envs.adversarial.mujoco.hopper_heel import HopperHeelEnv
from rarl.envs.adversarial.mujoco.hopper_torso_6 import HopperTorso6Env

env = HopperTorso6Env()
reward_list = []
for _ in range(500):
    done = False
    obs = env.reset()
    sum_reward = 0.0
    while not done:
        # action = {}
        action = ppo_agent.select_action(obs)
        # action["adv"] = env.adv_action_space.sample()
        # action = ppo_agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        # print(reward)
        sum_reward += reward
        # env.render()
        # time.sleep(0.05)
        if done:
            print(sum_reward)
            break

    reward_list.append(sum_reward)

print()
print(sum(reward_list))

# print(env.adv_action_space)
#
# print(env.model.names)
import numpy as np

# done = False
#
# steps = 0
# while not done:
#     # action = {}
#     action = env.action_space.sample()
#     # action["adv"] = env.adv_action_space.sample()
#     # action['adv'] = np.array([5.0, 5.0], dtype=np.float32)
#     print(action)
#     obs, reward, done, info = env.step(action)
#     print(reward)
#     steps += 1
#     time.sleep(2)
#     env.render()
#     if steps > 500:
#         break


# print(env.model.name_bodyadr.flatten())
# start_addr = ctypes.addressof(env.model.names)
# body_names = [ctypes.string_at(start_addr + int(inc))
#                 for inc in env.model.name_bodyadr.flatten()]

# print(body_names)
# print(env.model.body(env._adv_f_bname))

# print(env.model.body)
# for name in (dir(env.model)):
#     print(name)
# print(env.model.names)
# print(dir(env.model))
