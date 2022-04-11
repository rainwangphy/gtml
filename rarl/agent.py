from rarl.ppo.PPO_new import PPO


class Agent:
    def __init__(self, config):
        self.config = config
        self.agent_idx = config["agent_config"]["agent_idx"]
        self.agent_str = config["agent_config"]["agent_str"]
        self.train_env = config["train_env"]
        self.eval_env = config["eval_env"]

        # max_ep_len = config["agent_config"][
        #     "max_ep_len"
        # ]  # max timesteps in one episode

        # max_ep_len = self.max_ep_len
        # max_training_timesteps = int(
        #     3e6
        # )  # break training loop if timeteps > max_training_timesteps
        #
        # print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
        # log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
        # save_model_freq = int(1e5)  # save model frequency (in num timesteps)

        # action_std = self.action_std  # starting std for action distribution (Multivariate Normal)
        # action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        # min_action_std = (
        #     0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        # )
        # action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

        #####################################################

        #  Note : print/log frequencies should be > than max_ep_len

        # ############### PPO hyperparameters ################

        # self.update_timestep = max_ep_len * 4  # update policy every n timesteps
        action_std = config["agent_config"]["action_std"]
        K_epochs = config["agent_config"][
            "K_epochs"
        ]  # update policy for K epochs in one PPO update

        eps_clip = config["agent_config"]["eps_clip"]  # clip parameter for PPO
        gamma = config["agent_config"]["gamma"]  # discount factor

        lr_actor = config["agent_config"]["lr_actor"]  # learning rate for actor network
        lr_critic = config["agent_config"][
            "lr_critic"
        ]  # learning rate for critic network

        env = self.train_env
        self.has_continuous_action_space = True
        has_continuous_action_space = self.has_continuous_action_space
        # state space dimension
        state_dim = env.observation_space.shape[0]

        # action space dimension
        if self.agent_str[self.agent_idx] == "pro":
            if has_continuous_action_space:
                action_dim = env.action_space.shape[0]
            else:
                action_dim = env.action_space.n
        else:
            action_dim = env.adv_action_space.shape[0]

        self.ppo_agent = PPO(
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
        self.current_step = 0

    def train(self, o_agent, episode_per_train=10):

        env = self.train_env
        ppo_agent = self.ppo_agent
        has_continuous_action_space = self.has_continuous_action_space

        max_ep_len = self.config["agent_config"]["max_ep_len"]
        update_timestep = max_ep_len * 4
        action_std_decay_rate = self.config["agent_config"][
            "action_std_decay_rate"
        ]  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = (
            self.config["agent_config"]["min_action_std"]
            # minimum action_std (stop decay after action_std <= min_action_std)
        )
        action_std_decay_freq = int(
            self.config["agent_config"]["action_std_decay_freq"]
        )  # action_std decay frequency (in num timesteps)
        assert o_agent.agent_idx == 1 - self.agent_idx
        for _ in range(episode_per_train):
            state = env.reset()
            for t in range(1, max_ep_len + 1):
                # select action with policy
                # print(state.shape)
                action_a = ppo_agent.select_action(state, train=True)
                action_o = o_agent.ppo_agent.select_action(state, train=False)
                action = {
                    self.agent_str[self.agent_idx]: action_a,
                    self.agent_str[o_agent.agent_idx]: action_o,
                }
                state, reward, done, _ = env.step(action)
                # saving reward and is_terminals
                if self.agent_idx == 1:
                    reward = -reward
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)
                self.current_step += 1
                time_step = self.current_step
                # time_step += 1
                # current_ep_reward += reward
                # update PPO agent
                if time_step % update_timestep == 0:
                    print("update")
                    ppo_agent.update()
                # if continuous action space; then decay action std of ouput action distribution
                if (
                        has_continuous_action_space
                        and time_step % action_std_decay_freq == 0
                ):
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                # break; if the episode is over
                if done:
                    break

    def eval(self, o_agent, episode_per_eval=10):
        # print()
        assert o_agent.agent_idx == 1 - self.agent_idx
        assert self.agent_idx == 0
        env = self.eval_env
        max_ep_len = self.config["agent_config"]["max_ep_len"]
        ppo_agent = self.ppo_agent

        sum_reward = 0.0
        for _ in range(episode_per_eval):
            state = env.reset()
            for t in range(1, max_ep_len + 1):
                # select action with policy
                # print(state.shape)
                action_a = ppo_agent.select_action(state, train=False)
                action_o = o_agent.ppo_agent.select_action(state, train=False)
                action = {
                    self.agent_str[self.agent_idx]: action_a,
                    self.agent_str[o_agent.agent_idx]: action_o,
                }
                state, reward, done, _ = env.step(action)
                sum_reward += reward
                if done:
                    break
        return sum_reward / episode_per_eval
