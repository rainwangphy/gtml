import numpy as np
from gym import utils, spaces
from envs.adversarial.mujoco import mujoco_env


class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "inverted_double_pendulum.xml", 5)
        utils.EzPickle.__init__(self)
        ## Adversarial setup
        self._adv_f_bname = "pole2"  # Byte String name of body on which the adversary force will be applied
        # bnames = self.model.body_names
        bnames = ("world", "cart", "pole", "pole2")
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
        new_xfrc = self.sim.data.xfrc_applied * 0.0
        new_xfrc[self._adv_bindex] = np.array(
            [adv_act[0], 0.0, adv_act[1], 0.0, 0.0, 0.0]
        )
        self.sim.data.xfrc_applied[:] = new_xfrc

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

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        site_name = "tip"
        site_index = self.model.site_names.index(site_name)
        x, _, y = self.sim.data.site_xpos[site_index]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos[:1],  # cart x pos
                np.sin(self.sim.data.qpos[1:]),  # link angles
                np.cos(self.sim.data.qpos[1:]),
                np.clip(self.sim.data.qvel, -10, 10),
                np.clip(self.sim.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent * 0.5
        v.cam.lookat[2] += 3  # v.model.stat.center[2]
