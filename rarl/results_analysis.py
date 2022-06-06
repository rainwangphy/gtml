import torch
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
result_dir = "./results"
figure_dir = '../plot_figures'

figure_dir += '/rarl'
if not osp.exists(figure_dir):
    os.mkdir(figure_dir)

env_list = ["walker_heel", "walker_torso", "hopper_torso", "hopper_heel"]
# seed_list = [123, 231, 312]
seed_list = [123, 231, 312, 456, 564]
solution_list = ['nash', 'uniform']

for env in env_list:
    solution_results = {}
    for solution in solution_list:
        utility_over_seed = []
        for seed in seed_list:
            results = torch.load(
                osp.join(
                    result_dir,
                    "seed_{}_env_{}_solution_{}_simple".format(
                        seed, env, solution
                    ),
                ),
            )
            # print(results)
            utility_list = []
            for i in range(5):
                if i == 0:
                    pre_strategy = np.array([1.0])
                else:
                    pre_strategy = results[i - 1]['meta_strategies'][0]
                meta_game = results[i]['meta_games'][0]
                rewards = meta_game[:-1, -1]
                # print(rewards)
                assert len(pre_strategy) == len(rewards)
                utility_list.append(
                    np.sum(rewards * pre_strategy)
                )
            # print(utility_list)
            utility_over_seed.append(utility_list)
        print(utility_over_seed)
        utility_over_seed = np.array(utility_over_seed)
        # utility_over_seed = np.transpose(utility_over_seed)
        # print(utility_over_seed)
        mean = np.mean(utility_over_seed, axis=0)
        std = np.std(utility_over_seed, axis=0)
        solution_results[solution] = {
            'mean': mean,
            'std': std
        }
    x = np.array([i + 1 for i in range(5)])
    for solution in solution_list:
        # for solution in solution_list:
        mean = solution_results[solution]["mean"]
        std = solution_results[solution]["std"]
        plt.errorbar(x, mean, yerr=std, fmt="o-", label=solution)
        # plt.fill_between(
        #     x,
        #     mean + std,
        #     mean - std,
        #     alpha=0.2,
        # )
    plt.legend()
    plt.savefig(
        osp.join(figure_dir, "{}.pdf".format(env)), bbox_inches="tight", pad_inches=0.0
    )
    plt.show()
# sim_result = {}
# for i in range(5):
#     sim_result[i] = {
#         "meta_games": results[i]['meta_games'],
#         "meta_strategies": results[i]['meta_strategies'],
#     }
# torch.save(sim_result, osp.join(
#     result_dir,
#     "seed_{}_env_{}_solution_{}_simple".format(
#         seed, env, solution
#     ),
# ))
# print(results)
