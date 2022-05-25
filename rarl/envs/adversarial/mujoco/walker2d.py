import numpy as np
from gym import utils, spaces
from envs.adversarial.mujoco import mujoco_env


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        ## Adversarial setup
        self._adv_f_bname = "torso"  # Byte String name of body on which the adversary force will be applied
        # bnames = self.model.body_names
        bnames = (
            "world",
            "torso",
            "thigh",
            "leg",
            "foot",
            "thigh_left",
            "leg_left",
            "foot_left",
        )

        self._adv_bindex = bnames.index(
            self._adv_f_bname
        )  # Index of the body on which the adversary force will be applied
        adv_max_force = 5.0
        high_adv = np.ones(2) * adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _adv_to_xfrc(self, adv_act):
        # print("pre")
        # print(self.data.xfrc_applied)
        adv_act = np.clip(
            adv_act, a_max=self.adv_action_space.high, a_min=self.adv_action_space.low
        )
        new_xfrc = self.data.xfrc_applied * 0.0
        new_xfrc[self._adv_bindex] = np.array(
            [adv_act[0], 0.0, adv_act[1], 0.0, 0.0, 0.0]
        )
        self.data.xfrc_applied[:] = new_xfrc
        # print(self.data.xfrc_applied)

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
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def reset_model_zero(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += 0.8
        self.viewer.cam.elevation = -20
