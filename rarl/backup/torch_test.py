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

import gym
import time
from gym.envs.registration import register
import argparse
import gym
import time

gym.logger.set_level(40)
parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env", default="soccer", type=str)

args = parser.parse_args()

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
#
# obs = env.reset()
# print(obs)
#
# print(env.action_space)
# print(env.adv_action_space)
#
# print(env.model.names)
# import numpy as np
# done = False
#
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


with_adv_list = [
    69.55677858533937,
    642.6367221460473,
    71.72543483927754,
    598.5426373703403,
    289.62873397546394,
    874.6905332053858,
    305.0211309204869,
    66.05086673087631,
    915.9201447128172,
    96.72925471785537,
    779.72714691923,
    759.8915690313211,
    59.85489349297059,
    991.3159135926139,
    990.9067283221075,
    66.54792617962808,
    453.5579032825074,
    938.5925711102942,
    68.00500303234847,
    625.2689091624472,
    310.64628251380674,
    668.1050281018306,
    271.1293670958654,
    928.9950478382104,
    611.3208391768544,
    317.57066618226156,
    49.73360392676168,
    337.85744053371207,
    1104.501498810969,
    456.4494474962506,
    329.9661822404739,
    1173.5999662288255,
    289.74209155113687,
    66.42464834913825,
    303.53120831873025,
    791.3625289797508,
    456.05346546828457,
    59.105547662852075,
    291.28886522504587,
    60.12002764964969,
    270.7938468441323,
    292.1469487818232,
    48.28426105225476,
    540.1326048702244,
    606.221403615699,
    305.34905733088345,
    277.6420817753126,
    703.7949790920113,
    645.4258348385932,
    699.2329278608245,
    323.89882403098125,
    538.1961742832804,
    650.2215749860245,
    951.7421900828641,
    301.2430221025921,
    703.0486904600682,
    324.82375724074535,
    632.2468479057808,
    75.83079111588444,
    332.08589251313197,
    81.03708860953317,
    71.61423266188277,
    524.6049744651215,
    449.1092773079362,
    675.3319398960857,
    684.4103164295,
    809.4784661750017,
    662.3186440658276,
    58.37866496159487,
    282.4735131021347,
    392.6369479974879,
    311.77301902936523,
    435.98774842918164,
    531.2137386596505,
    600.7087122316793,
    630.6673718306533,
    57.37437975033162,
    72.94719822712037,
    91.04659108128199,
    72.82929398487133,
    114.24286995026996,
    316.3925267267263,
    47.69765902781759,
    91.83508861953248,
    490.7348792142701,
    793.0622259515916,
    272.42440054976265,
    572.9813214677138,
    64.80306151641832,
    748.2727122456807,
    914.5166467262494,
    49.447631419674344,
    330.1391158129329,
    312.15617008005637,
    313.3361712926758,
    74.23627421201523,
    294.98047898847574,
    773.0299689404262,
    462.6138859419484,
    512.6527738037571,
]
withoud_adv_list = [
    55.790365140554265,
    651.86553200377,
    1244.0890495769906,
    572.9385010093712,
    910.9120669805859,
    59.94909656552389,
    66.59811066952558,
    59.79381783226228,
    55.19937820460588,
    51.93995968602729,
    816.3433901929552,
    82.77133410614833,
    679.6255580214797,
    685.3580762745659,
    105.68025417216525,
    950.0116680045156,
    324.6248090683584,
    685.1228215510941,
    95.02471918028597,
    930.562656158578,
    623.5716297343437,
    661.5564879101155,
    335.8059009586996,
    856.7390472142137,
    72.39344202877119,
    70.3277865395568,
    599.5771049846785,
    87.61798732277256,
    412.14965918603684,
    583.9457301646577,
    681.2470133097281,
    537.6601122151392,
    691.9712218618203,
    812.4726948763537,
    53.63438644763518,
    328.7624054311821,
    56.402879259622054,
    717.1976812603858,
    546.700433019214,
    57.02916419021036,
    307.8645961044666,
    307.101246662981,
    947.4185498698088,
    70.22511209507022,
    339.61672395903906,
    920.5258649150365,
    535.7112808024693,
    78.76175809765984,
    74.05060245843288,
    318.85933830208495,
    311.6583593274992,
    777.5613018898351,
    56.79778119417547,
    68.52218528889297,
    64.00432901546776,
    257.280920131698,
    670.1871124377365,
    749.153351120725,
    1325.444940727806,
    56.982047683920015,
    586.6376346313044,
    830.4262520672685,
    351.6645904421245,
    775.1629907797192,
    803.561553996547,
    333.53400615632563,
    576.8144805724185,
    860.218202589289,
    469.969991049761,
    316.97300698703555,
    652.021892244267,
    1183.8003177474902,
    88.75872258379918,
    1161.584069504786,
    299.62486115425344,
    789.527995607674,
    898.895796565189,
    281.98757792262,
    559.7405811782178,
    597.5048191746472,
    60.687601769307435,
    1153.9283480516888,
    57.43686239014171,
    550.4911186102091,
    247.63062965776263,
    802.7257600562941,
    46.527024234783255,
    1170.8617932314478,
    686.1532504012067,
    52.52273356260428,
    54.032344954995814,
    77.33311147192197,
    351.75293527076354,
    1056.8572497602358,
    831.3785857907781,
    721.8604515986808,
    69.80434267125443,
    52.26128130710423,
    63.80591527562466,
    563.8016269679267,
]

print(sum(with_adv_list))
print(sum(withoud_adv_list))
