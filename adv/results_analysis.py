import torch

nb_iter_list = [5, 10]
solution_list = ["nash", 'uniform']
dataset_list = ["cifar10", "cifar100"]

# cuda_devices_list = [4, 5, 6, 7]
#
# cuda_str = "CUDA_VISIBLE_DEVICES"
# filename = "run_exp.sh"
#
# f = open(file=filename, mode="w")
# pare_folder = "./results/outputs"
# f.write("mkdir -p {}\n\n".format(pare_folder))

result_dir = "./results"
figure_dir = '../plot_figures'


#                 "seed_{}_dataset_{}_solution_{}_nb_iter_{}.pth".format(
#                     self.args.seed,
#                     self.args.dataset,
#                     self.args.solution,
#                     self.args.nb_iter,
#                 ),

# device_number = 0
import os.path as osp
import os

figure_dir += '/adv'
if not osp.exists(figure_dir):
    os.mkdir(figure_dir)

import numpy as np
import matplotlib.pyplot as plt
for nb_iter in nb_iter_list:
    for dataset in dataset_list:
        for solution in solution_list:
            res = torch.load(osp.join(result_dir, "seed_{}_dataset_{}_solution_{}_nb_iter_{}.pth".format(
                    0,
                    dataset,
                    solution,
                    nb_iter,
                ),))

            # print(res)
            acc_list = []
            for i in range(5):
                idx = i+1
                acc_list.append(
                    res[idx]['final_accuracy']
                )
            print(acc_list)
            mean = np.array(acc_list)
            x = np.array([i+1 for i in range(len(mean))])
            plt.plot(x, mean, 'o-', label=solution)
        plt.legend()
        plt.savefig(
            osp.join(figure_dir, "{}_{}.pdf".format(dataset, nb_iter)), bbox_inches="tight", pad_inches=0.0
        )
        plt.show()
