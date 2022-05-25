import numpy as np
import torch
import random

import gym

gym.logger.set_level(40)
from rarl.envs.adversarial.mujoco.walker2d import Walker2dEnv

env = Walker2dEnv()

print(env.pro_action_space)
print(env.adv_action_space)

seed = 10


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed=seed)
env.pro_action_space.seed(seed=seed)
env.adv_action_space.seed(seed=seed)

for _ in range(1):
    env.reset(seed=seed)
    done = False
    while not done:
        action = env.sample_action()
        print(action)
        # action = env.pro_action_space.sample()
        obs, reward, done, _ = env.step(action)
        # env.render()
        print("{}-{}".format(obs, reward))
    print("end of episode")
