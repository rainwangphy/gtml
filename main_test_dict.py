# import argparse
#
# config = {
#     "task": "biased_SST_95",
#     "data_dir": "datasets",
#     "device": "cuda",
#     "model_config": argparse.Namespace(
#         **{
#             "architecture": "bilstm",
#             "input_format": "bert-base-uncased",
#             "task": None,
#             "update_every": 1,
#             "l2_reg": 0,
#             "clip_grad": 10,
#             "optimizer": "sgd",
#             "lr_scheduler": "linear_decay",
#             "lr": 2e-5,
#             "weight_decay": 0,
#         }
#     ),
#     "adv_config": argparse.Namespace(
#         **{
#             "input_format": "bert-base-uncased",
#             "adv_tasks": None,
#             "adv_architecture": "small_transformer_generative",
#             "adv_filename": "pretrained_models"
#             "/biased_SST_95_gen_LM_small_transformer_generative_wikitext103_model.pt",
#             "adv_output_size": None,
#             "ratio_model": False,
#             "adv_threshold": 2.302585,
#             "adv_obj": "exp_kl",
#             "alpha": 1.0,
#             "beta": 1.0,
#             "tau": 0.01,
#             "self-norm-lambda": 0,
#             "adv-optimizer": "sgd",
#             "adv-mom": 0,
#             "lr_scheduler": "linear_decay",
#             "adv_lr": 1e-4,
#             "weight_decay": 0,
#             "clip_grad_adv": None,
#             "adv_update_every": 1,
#             "norm_k_model": None,
#             "norm_k_adv": 5,
#             "norm_model_only": False,
#             "norm_adv_only": False,
#             "renorm_ratios": None,
#             "joint": True,
#             "class_conditional": False,
#             "adv_on_acc": False,
#         }
#     ),
# }
#
# # args = argparse.Namespace(**config)
# # print(args.adv_config.input_format)
# #
# # print(args)

import numpy as np

res = np.random.rand(30)
print(res)
print(res[-1])
print(res[:-1])
print(res[-1] - res[:-1])
