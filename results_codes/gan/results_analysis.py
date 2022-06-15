# parser.add_argument("--gan_name", type=str, default="gan")
# parser.add_argument("--dataset", type=str, default="cifar10")

gan_name_list = ["gan", "wgan", "wgan_gp"]
solution_list = ["nash", "uniform"]
dataset_list = ["cifar10", "stl10"]

# cuda_devices_list = [1, 2, 3]
#
# cuda_str = "CUDA_VISIBLE_DEVICES"
# filename = "run_exp.sh"
#
# f = open(file=filename, mode="w")
# pare_folder = "./results/outputs"
# f.write("mkdir -p {}\n\n".format(pare_folder))
import torch
import os.path as osp
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
result_dir = "./results"
figure_dir = '../../plot_figures'
# device_number = 0

# import os.path as osp


figure_dir += '/gan'
if not osp.exists(figure_dir):
    os.mkdir(figure_dir)
seed = 0
for gan_name in gan_name_list:
    for dataset in dataset_list:
        res_dict = {}
        for solution in solution_list:
            res = torch.load(
                osp.join(
                    result_dir,
                    "seed_{}_gan_name_{}_dataset_{}_solution_{}".format(
                        seed, gan_name, dataset, solution
                    ),
                ),
            )
            print(res)

            # sim_matrix = res[5]['meta_game'][0]
            # heat_map_lin = sns.heatmap(sim_matrix, vmin=0, vmax=1, center=0,
            #                            annot=True, cmap="magma", linewidths=.25)
            # heat_map_lin.invert_yaxis()
            # plt.show()

        #     score_list = []
        #     var_list = []
        #     for i in range(5):
        #         idx = i + 1
        #         score_list.append(res[idx]["score"]["inception_score"][0])
        #         var_list.append(res[idx]["score"]["inception_score"][1])
        #     # print(score_list)
        #     # print(var_list)
        #     res_dict[solution] = {"score": score_list, "var": var_list}
        # print(res_dict)
        # x = np.array([i + 1 for i in range(5)])
        #
        # for solution in solution_list:
        #     mean = np.array(res_dict[solution]["score"])
        #     std = np.array(res_dict[solution]["var"])
        #     if solution == 'nash':
        #         plt.errorbar(x, mean, yerr=std, fmt="o-", label='Nash', capsize=4)
        #     else:
        #         plt.errorbar(x, mean, yerr=std, fmt="o-", label='Uniform', capsize=4)
        #     # plt.fill_between(
        #     #     x,
        #     #     mean + std,
        #     #     mean - std,
        #     #     alpha=0.2,
        #     # )
        # plt.legend()
        # plt.savefig(
        #     osp.join(figure_dir, "{}_{}.pdf".format(dataset, gan_name)), bbox_inches="tight", pad_inches=0.0
        # )
        # plt.show()
