import numpy as np
from gym import utils, spaces
from envs.adversarial.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)
        ## Adversarial setup
        self._adv_f_bname = "torso"  # Byte String name of body on which the adversary force will be applied
        bnames = (
            "world",
            "torso",
            "bthigh",
            "bshin",
            "bfoot",
            "fthigh",
            "fshin",
            "ffoot",
        )
        # bnames = self.model.body_names
        self._adv_bindex = bnames.index(
            self._adv_f_bname
        )  # Index of the body on which the adversary force will be applied
        adv_max_force = 5.0
        high_adv = np.ones(2) * adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _adv_to_xfrc(self, adv_act):
        adv_act = np.clip(
            adv_act, a_max=self.adv_action_space.high, a_min=self.adv_action_space.low
        )
        new_xfrc = self.data.xfrc_applied * 0.0
        new_xfrc[self._adv_bindex] = np.array(
            [adv_act[0], 0.0, adv_act[1], 0.0, 0.0, 0.0]
        )
        self.data.xfrc_applied[:] = new_xfrc

    def sample_action(self):
        act = {}
        act["pro"] = self.pro_action_space.sample()
        act["adv"] = self.adv_action_space.sample()
        sa = act
        # class act(object):
        # def __init__(self,pro=None,adv=None):
        # self.pro=pro
        # self.adv=adv
        # sa = act(self.pro_action_space.sample(), self.adv_action_space.sample())
        return sa

    def step(self, action):
        if hasattr(action, "__dict__"):
            self._adv_to_xfrc(action.adv)
            a = action.pro
        elif type(action) == dict:
            self._adv_to_xfrc(action["adv"])
            a = action["pro"]
        else:
            a = action

        xposbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(a).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[1:],
                self.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
