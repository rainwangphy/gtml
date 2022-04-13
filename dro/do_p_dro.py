import argparse
import numpy as np
from dro.p_dro import p_dro_model, p_dro_adversary
from src.tasks import task_list, prepare_task
from src.tasks import LanguageModelingTask, CCLanguageModelingTask
from src.utils import cacheable, get_loader, get_group_dro_loader
import os
import torch
from dro import utils
from meta_solvers.prd_solver import projected_replicator_dynamics


class do_p_dro:
    def __init__(self, args):
        self.args = args
        self.max_loop = args.max_loop
        self.solution = args.solution
        self.train_max_epoch = args.train_max_epoch
        self.eval_max_epoch = args.eval_max_epoch
        self.device = args.device

        self.config = argparse.Namespace(
            **{
                "task": "biased_SST_95",
                "data_dir": "datasets",
                "batch_size": 64,
                "max_tokens_per_batch": 2500,
                "num_workers": 1,
                # optim_args.batch_size,
                # max_tokens_per_batch=optim_args.max_tokens_per_batch,
                # shuffle=True,
                # collate_fn=task.collate_fn,
                # num_workers=optim_args.num_workers,
                "device": self.device,
                "model_config": argparse.Namespace(
                    **{
                        "architecture": "bilstm",
                        "input_format": "bert-base-uncased",
                        "task": None,
                        "input_shape": None,
                        "update_every": 1,
                        "l2_reg": 0,
                        "clip_grad": 10,
                        "optimizer": "sgd",
                        "lr_scheduler": "linear_decay",
                        "lr": 2e-5,
                        "weight_decay": 0,
                        "n_steps": 500,
                    }
                ),
                "adv_config": argparse.Namespace(
                    **{
                        "input_format": "bert-base-uncased",
                        "non_param": False,
                        "adv_task": None,
                        "adv_architecture": "small_transformer_generative",
                        "adv_filename": "pretrained_models"
                        "/biased_SST_95_gen_LM_small_transformer_generative_wikitext103_model.pt",
                        "input_shape": None,
                        "adv_output_size": None,
                        "ratio_model": False,
                        "adv_threshold": 2.302585,
                        "adv_obj": "exp_kl",
                        "alpha": 1.0,
                        "beta": 1.0,
                        "tau": 0.01,
                        "self_norm_lambda": 0,
                        "adv_optimizer": "sgd",
                        "adv_mom": 0,
                        "lr_scheduler": "linear_decay",
                        "adv_lr": 1e-4,
                        "weight_decay": 0,
                        "clip_grad_adv": None,
                        "adv_update_every": 1,
                        "norm_k_model": None,
                        "norm_k_adv": 5,
                        "norm_model_only": False,
                        "norm_adv_only": False,
                        "renorm_ratios": None,
                        "joint": True,
                        "class_conditional": False,
                        "adv_on_acc": False,
                    }
                ),
            }
        )

        self.task = None
        self.adv_task = None
        self.sampler = None
        self.loader = None
        self.get_dataset()

        self.model_list = []
        self.adversary_list = []

        self.meta_games = [
            np.array([[]], dtype=np.float32),
            np.array([[]], dtype=np.float32),
        ]

        self.meta_strategies = [
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

    def get_dataset(self):
        task, input_shape, output_size = prepare_task(
            self.config.task,
            path=self.config.data_dir,
            model_name=self.config.model_config.input_format,
        )
        self.task = task
        self.config.model_config.input_shape = input_shape
        self.config.adv_config.input_shape = input_shape
        self.config.model_config.task = self.task
        # In this case the adversary is a proper generative model
        # adv_output_size = None
        # If the adversary models x, y or x | y, we need to specify the number
        # of classes
        if self.config.adv_config.joint or self.config.adv_config.class_conditional:
            self.config.adv_config.adv_output_size = self.task.n_classes
        elif self.config.adv_config.ratio_model:
            self.config.adv_config.adv_output_size = 1

        if self.config.adv_config.ratio_model:
            adv_task = task
        elif self.config.adv_config.joint or self.config.adv_config.class_conditional:
            adv_task = CCLanguageModelingTask.from_text_task(
                task, generative=not self.config.adv_config.class_conditional
            )
        else:
            adv_task = LanguageModelingTask.from_text_task(task)
        self.adv_task = adv_task
        self.config.adv_config.adv_task = self.adv_task

        sampler, loader = get_loader(
            task.train_data,
            self.config.batch_size,
            max_tokens_per_batch=self.config.max_tokens_per_batch,
            shuffle=True,
            collate_fn=task.collate_fn,
            num_workers=self.config.num_workers,
        )

        self.sampler = sampler
        self.loader = loader

        # TODO: compute the log_p
        self.compute_log_p()

    def compute_log_p(self):
        print("compute log p")
        task = self.task
        adv_args = self.config.adv_config
        adv = self.get_adversary().adversary
        # Create adversary
        # adv = make_adversary(
        #     adv_args.adv_architecture,
        #     filename=adv_args.adv_filename,
        #     input_shape=input_shape,
        #     output_size=adv_output_size,
        #     device=device,
        #     ratio_model=adv_args.ratio_model,
        # )
        # Pre-compute baseline LM scores
        if adv_args.ratio_model:
            # This is a hack: the "baseline" log probilities are not used in
            # this scenario
            all_log_probs = torch.zeros(len(task.train_data))
        else:
            # Pre-compute language modeling scores
            adv_type = "_cc" if adv_args.class_conditional else ""
            if adv_args.adv_filename is not None:
                adv_filename = os.path.basename(adv_args.adv_filename)
            else:
                adv_filename = adv_args.adv_architecture
            all_log_probs_filename = os.path.join(
                "results",
                f"lm_{adv_filename}{adv_type}_train",
            )
            # Pre-compute the log probabilities of the training samples
            all_log_probs = utils.compute_dataset_log_probs(
                adv,
                task,
                "train",
                batch_size=self.config.batch_size,
                max_tokens_per_batch=self.config.max_tokens_per_batch,
                joint=adv_args.joint,
                class_conditional=adv_args.class_conditional,
                cached_filename=all_log_probs_filename,
            )
        # Add the baseline log p as an attribute to the train data
        # First initialize attributes (FIXME: this should be removed)
        if (
            not hasattr(task.train_data, "attributes")
            or task.train_data.attributes is None
        ):
            task.train_data.attributes = [{} for _ in range(len(task.train_data))]
        # Sanity check
        if not len(all_log_probs) == len(task.train_data):
            raise ValueError(f"{len(all_log_probs)} != {len(task.train_data)}")
        for idx, log_p in enumerate(all_log_probs):
            task.train_data.attributes[idx]["log_p"] = log_p

    def get_model(self):
        config = self.config
        return p_dro_model(config)

    def get_adversary(self):
        config = self.config
        return p_dro_adversary(config)

    def init(self):
        # print()
        model = self.get_model()
        adversary = self.get_adversary()
        # print(len(self.loader))
        sum_loss = 0.0
        for i, data in enumerate(self.loader):
            # print("{}-{}".format(i, len(self.loader)))
            loss = model.eval(data, adversary)
            # adversary.eval(data, model)
            sum_loss += loss
        sum_loss /= len(self.loader)
        self.model_list.append(model)
        self.adversary_list.append(adversary)
        r = len(self.model_list)
        c = len(self.adversary_list)
        self.meta_games = [
            np.full([r, c], fill_value=-sum_loss),
            np.full([r, c], fill_value=sum_loss),
        ]
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]
        print(self.meta_games)
        print(self.meta_strategies)

    def solve(self):
        for loop in range(self.max_loop):
            # classifier = self.get_classifier()
            print("get attacker")
            # attacker = self.get_attacker()
            print("train classifier")
            model = self.get_model()
            adversary = self.get_adversary()
            for _ in range(self.train_max_epoch):
                for i, data in enumerate(self.loader):
                    adv_idx = np.random.choice(
                        len(self.adversary_list), p=self.meta_strategies[1]
                    )
                    model.train(data, self.adversary_list[adv_idx])
            for _ in range(self.train_max_epoch):
                for i, data in enumerate(self.loader):
                    model_idx = np.random.choice(
                        len(self.model_list), p=self.meta_strategies[0]
                    )
                    model.train(data, self.model_list[model_idx])
                    # batch_idx = np.random.choice(len(self.train_data_loader))
                    # attacker_idx = np.random.choice(
                    #     len(self.attacker_list), p=self.meta_strategies[1]
                    # )
                    # (imgs, labels) = self.attacker_list[
                    #     attacker_idx
                    # ].perturbed_train_dataloader[batch_idx]
                    # # for i, (imgs, labels) in enumerate(attacker.perturbed_train_dataloader):
                    # end_epoch = True if i == len(self.train_data_loader) else False
                    # classifier.train(imgs, labels, end_epoch)
            # accuracy = 0.0
            # for i, (imgs, labels) in enumerate(attacker.perturbed_train_dataloader):
            #     accuracy += classifier.eval(imgs, labels)
            # accuracy /= len(self.train_data_loader)
            # self.generator_list.append(generator)
            # self.discriminator_list.append(discriminator)

            # self.classifier_list.append(classifier)
            # self.attacker_list.append(attacker)
            self.model_list.append(model)
            self.adversary_list.append(adversary)
            r = len(self.model_list)
            c = len(self.adversary_list)
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
                        sum_loss = 0.0
                        model = self.model_list[t_r]
                        adversary = self.adversary_list[t_c]
                        for i, data in enumerate(self.loader):
                            # print("{}-{}".format(i, len(self.loader)))

                            loss = model.eval(data, adversary)
                            # adversary.eval(data, model)
                            sum_loss += loss
                        sum_loss /= len(self.loader)
                        # generator = self.generator_list[t_r]
                        # discriminator = self.discriminator_list[t_c]
                        # gen_loss = 0.0
                        # dis_loss = 0.0
                        # sum_idx = 0
                        # for _ in range(self.eval_max_epoch):
                        #     for i, (imgs, _) in enumerate(self.data):
                        #         sum_idx += 1
                        #         data = {"real_imgs": imgs}
                        #         g_loss = generator.eval(data, discriminator)
                        #         gen_loss += g_loss["g_loss"]
                        #
                        #         d_loss = discriminator.eval(data, generator)
                        #         dis_loss += d_loss["d_loss"]
                        # gen_loss /= sum_idx
                        # dis_loss /= sum_idx
                        meta_games[0][t_r][t_c] = -sum_loss
                        meta_games[1][t_r][t_c] = sum_loss

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
    parser.add_argument("--train_max_epoch", type=int, default=100)
    parser.add_argument("--eval_max_epoch", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    # print()
    do_dro = do_p_dro(args)
    do_dro.init()
    # do_dro.solve()
