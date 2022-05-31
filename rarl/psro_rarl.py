import sys

sys.path.append("../")

from rarl.agent import Agent
from rarl.envs.adversarial.mujoco.hopper_torso_6 import HopperTorso6Env

# from rarl.envs.adversarial.mujoco.walker2d import Walker2dEnv
from rarl.envs.adversarial.mujoco.hopper_heel import HopperHeelEnv
from rarl.envs.adversarial.mujoco.walker2d_torso import Walker2dTorsoEnv
from rarl.envs.adversarial.mujoco.walker2d_heel import Walker2dHeelEnv
import numpy as np
from meta_solvers.prd_solver import projected_replicator_dynamics
import argparse
import torch
import os.path as osp


# from rarl.utils import setup_seed


def setup_seed(seed=42):
    import numpy as np
    import torch
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def soft_update(online, target, tau=0.9):
    for param1, param2 in zip(target.parameters(), online.parameters()):
        param1.data *= 1.0 - tau
        param1.data += param2.data * tau


class psro_rarl:
    def __init__(self, args):
        self.args = args

        self.args = args
        self.max_loop = args.max_loop
        self.solution = args.solution
        self.train_max_episode = args.train_max_episode
        self.eval_max_episode = args.eval_max_episode

        self.train_env = self.get_env()
        self.eval_env = self.get_env()

        self.agent_str = ["pro", "adv"]

        self.pro_list = []
        self.adv_list = []
        self.meta_games = [
            np.array([[]], dtype=np.float32),
            np.array([[]], dtype=np.float32),
        ]

        self.meta_strategies = [
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

        setup_seed(args.seed)
        self.result_dir = "./results"
        self.result_dict = {}

    def get_env(self):
        import gym

        gym.logger.set_level(40)
        # env_name = self.args.env_name
        env_name = self.args.env
        if env_name == "walker_heel":
            env = Walker2dHeelEnv()
        elif env_name == "walker_torso":
            env = Walker2dTorsoEnv()
        elif env_name == "hopper_torso":
            env = HopperTorso6Env()
        else:
            env = HopperHeelEnv()
        # env = HopperTorso6Env()
        # env = Walker2dTorsoEnv()
        # env = Walker2dEnv()
        seed = self.args.seed
        env.seed(seed=seed)
        env.pro_action_space.seed(seed=seed)
        env.adv_action_space.seed(seed=seed)
        return env

    def get_agent(self, agent_idx):
        config = {
            "train_env": self.train_env,
            "eval_env": self.eval_env,
            "agent_config": {
                "agent_idx": agent_idx,
                "agent_str": self.agent_str,
                "max_ep_len": self.args.max_ep_len,
                "action_std": 0.6,
                "action_std_decay_rate": 0.05,
                "action_std_decay_freq": 2.5e5,
                "min_action_std": 0.1,
                "K_epochs": 80,
                "eps_clip": 0.2,
                "gamma": 0.99,
                "lr_actor": 0.003,
                "lr_critic": 0.001
                # action_std = 0.6  # starting std for action distribution (Multivariate Normal)
                # action_std_decay_rate = 0.05
                # linearly decay action_std (action_std = action_std - action_std_decay_rate)
                # min_action_std = (
                #     0.1  # minimum action_std (stop decay after action_std <= min_action_std)
                # )
                # action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
                #
                # #####################################################
                #
                # #  Note : print/log frequencies should be > than max_ep_len
                #
                # # ############### PPO hyperparameters ################
                #
                # update_timestep = max_ep_len * 4  # update policy every n timesteps
                # K_epochs = 80  # update policy for K epochs in one PPO update
                #
                # eps_clip = 0.2  # clip parameter for PPO
                # gamma = 0.99  # discount factor
                #
                # lr_actor = 0.0003  # learning rate for actor network
                # lr_critic = 0.001  # learning rate for critic network
                #
                # random_seed = 0  # set random seed if required (0 = no random seed)
            },
        }

        return Agent(config)

    def rarl_init(self, agent, adversary):
        max_steps = self.args.rarl_init_max_episode * self.args.max_ep_len
        while True:
            agent.train(o_agent=adversary)
            adversary.train(o_agent=agent)
            if adversary.current_step > max_steps and agent.current_step > max_steps:
                break

    def init(self):
        agent = self.get_agent(agent_idx=0)
        adversary = self.get_agent(agent_idx=1)

        rarl_init = True
        if rarl_init:
            self.rarl_init(agent, adversary)
        # pro_reward = 0.0
        # adv_reward = 0.0

        reward = agent.eval(o_agent=adversary)
        pro_reward = reward
        adv_reward = -reward

        self.pro_list.append(agent)
        self.adv_list.append(adversary)
        r = len(self.pro_list)
        c = len(self.adv_list)
        self.meta_games = [
            np.full([r, c], fill_value=pro_reward),
            np.full([r, c], fill_value=adv_reward),
        ]
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]
        print(self.meta_games)
        print(self.meta_strategies)

    def solve(self):
        for loop in range(self.max_loop):
            agent = self.get_agent(agent_idx=0)
            adversary = self.get_agent(agent_idx=1)

            use_soft_update = True
            if use_soft_update:
                soft_update(agent.ppo_agent.policy, self.pro_list[-1].ppo_agent.policy)
                soft_update(
                    adversary.ppo_agent.policy, self.adv_list[-1].ppo_agent.policy
                )
                agent.ppo_agent.update_policy_old()
                adversary.ppo_agent.update_policy_old()

            max_steps = self.train_max_episode * self.args.max_ep_len
            while True:
                adv_idx = np.random.choice(
                    range(len(self.adv_list)), p=self.meta_strategies[1]
                )
                adv = self.adv_list[adv_idx]
                agent.train(o_agent=adv)
                if agent.current_step > max_steps:
                    break
            while True:
                pro_idx = np.random.choice(
                    range(len(self.pro_list)), p=self.meta_strategies[0]
                )
                pro = self.pro_list[pro_idx]
                adversary.train(o_agent=pro)
                if adversary.current_step > max_steps:
                    break

            print("augment the game")
            self.pro_list.append(agent)
            self.adv_list.append(adversary)
            r = len(self.pro_list)
            c = len(self.adv_list)
            meta_games = [
                np.full([r, c], fill_value=np.nan),
                np.full([r, c], fill_value=np.nan),
            ]
            (o_r, o_c) = self.meta_games[0].shape
            for i in [0, 1]:
                for t_r in range(o_r):
                    for t_c in range(o_c):
                        meta_games[i][t_r][t_c] = self.meta_games[i][t_r][t_c]
            for t_r in range(r):
                for t_c in range(c):
                    if np.isnan(meta_games[0][t_r][t_c]):
                        pro = self.pro_list[t_r]
                        adv = self.adv_list[t_c]

                        reward = pro.eval(
                            o_agent=adv, episode_per_eval=self.args.eval_max_episode
                        )
                        meta_games[0][t_r][t_c] = reward
                        meta_games[1][t_r][t_c] = -reward

            self.meta_games = meta_games
            if self.args.solution == "nash":
                self.meta_strategies = projected_replicator_dynamics(self.meta_games)
            else:
                self.meta_strategies = [
                    np.array([1.0 for _ in range(len(self.pro_list))])
                    / len(self.pro_list),
                    np.array([1.0 for _ in range(len(self.adv_list))])
                    / len(self.adv_list),
                ]
            print(self.meta_games)
            print(self.meta_strategies)

            results = {
                "meta_games": self.meta_games,
                "meta_strategies": self.meta_strategies,
                "pro_list": self.pro_list,
                "adv_list": self.adv_list,
            }

            self.result_dict[loop] = results
            torch.save(
                self.result_dict,
                osp.join(
                    self.result_dir,
                    "seed_{}_env_{}_solution_{}".format(
                        self.args.seed, self.args.env, self.args.solution
                    ),
                ),
            )

    # def eval(self):
    #     adversary = self.get_agent(agent_idx=1)
    #     max_steps = self.train_max_episode * self.args.max_ep_len
    #     while True:
    #         pro_idx = np.random.choice(
    #             range(len(self.pro_list)), p=self.meta_strategies[0]
    #         )
    #         pro = self.pro_list[pro_idx]
    #         adversary.train(o_agent=pro)
    #         if adversary.current_step > max_steps:
    #             break
    #     reward_list = []
    #     for pro in self.pro_list:
    #         reward = pro.eval(o_agent=adversary, episode_per_eval=self.args.eval_max_episode)
    #         reward_list.append(reward)
    #
    #     mean_reward = 0.0
    #     for i, val in enumerate(reward_list):
    #         mean_reward += self.meta_strategies[0][i] * val
    #     return reward_list, mean_reward


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="hopper_heel")
    parser.add_argument("--max_loop", type=int, default=4)
    parser.add_argument("--solution", type=str, default="nash")

    parser.add_argument("--train_max_episode", type=int, default=500)
    parser.add_argument("--rarl_init_max_episode", type=int, default=1000)
    parser.add_argument("--eval_max_episode", type=int, default=100)

    parser.add_argument("--max_ep_len", type=int, default=1000)

    args = parser.parse_args()
    psro_rarl_agent = psro_rarl(args=args)

    psro_rarl_agent.init()
    psro_rarl_agent.solve()
