from rarl.agent import Agent
from rarl.envs.adversarial.mujoco.hopper_torso_6 import HopperTorso6Env
import numpy as np
from meta_solvers.prd_solver import projected_replicator_dynamics
import argparse


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

    def get_env(self):
        import gym

        gym.logger.set_level(40)
        # env_name = self.args.env_name
        env = HopperTorso6Env()
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

    def init(self):
        agent = self.get_agent(agent_idx=0)
        adversary = self.get_agent(agent_idx=1)

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
            self.meta_strategies = projected_replicator_dynamics(self.meta_games)
            print(self.meta_games)
            print(self.meta_strategies)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_loop", type=int, default=4)
    parser.add_argument(
        "--solution", type=str, default="the solution for the meta game"
    )
    parser.add_argument("--train_max_episode", type=int, default=100)
    parser.add_argument("--eval_max_episode", type=int, default=100)

    parser.add_argument("--max_ep_len", type=int, default=1000)
    args = parser.parse_args()
    psro_rarl_agent = psro_rarl(args=args)

    psro_rarl_agent.init()
    psro_rarl_agent.solve()
