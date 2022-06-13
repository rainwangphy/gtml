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
                        "adv_lr": 2e-5,
                        "weight_decay": 0,
                        "clip_grad_adv": 0,
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
        model = self.get_model()
        adversary = self.get_adversary()

        best_model_state_dict = torch.load('biased_sst_p_dro_run_0_robust_model.pt')
        model.model.load_state_dict(best_model_state_dict)
        best_adv_state_dict = torch.load('biased_sst_p_dro_run_0_robust_lm.pt')
        adversary.adversary.load_state_dict(best_adv_state_dict)

        model_file = os.path.join("results", 'init' + '_model.pt')
        lm_file = os.path.join("results", 'init' + '_lm.pt')
        torch.save(model.model.state_dict(), model_file)
        torch.save(adversary.adversary.state_dict(), lm_file)

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
        def parse_domain(domain_descriptor):
            domain_attributes = {}
            if len(domain_descriptor) == 0:
                return lambda x: True
            for k_v in domain_descriptor.split(","):
                k, v = k_v.split("=")
                domain_attributes[k] = v

            def filtering(attr):
                for k, v in domain_attributes.items():
                    if k not in attr or str(attr[k]) != v:
                        return False

                return True

            return filtering

        eval_on_domains = ['biased=True,label=0', 'biased=True,label=1', 'biased=False,label=0', 'biased=False,label=1']
        eval_domain_filters = {domain: parse_domain(domain) for domain in eval_on_domains}
        group_idxs = {domain: [idx for idx, x in enumerate(self.task.test_data.attributes) if domain_filter(x)] for
                      domain, domain_filter in eval_domain_filters.items()}

        average_ss = []
        robust_ss = []

        for loop in range(self.max_loop):
            # classifier = self.get_classifier()
            print(str(loop), "get attacker")
            # attacker = self.get_attacker()
            print("train classifier")
            model = self.get_model()
            adversary = self.get_adversary()

            best_model_state_dict = torch.load('biased_sst_p_dro_run_0_robust_model.pt')
            model.model.load_state_dict(best_model_state_dict)
            best_adv_state_dict = torch.load('biased_sst_p_dro_run_0_robust_lm.pt')
            adversary.adversary.load_state_dict(best_adv_state_dict)
            if loop == 0:
                (test_examples_scores, test_losses) = self.task.eval_model(self.model_list[0].model, data='test',
                                                                           by_example=True, nll=True)
                test_examples_scores = test_examples_scores.numpy()
                average_ss.append(test_examples_scores.mean())
                print(loop, 'Before train Avg score: ', test_examples_scores.mean())
                group_scores = np.asarray([test_examples_scores[g_idxs].mean() for g_idxs in group_idxs.values()])
                lower_is_better = False
                if lower_is_better:
                    current_robust_score = group_scores.max()
                else:
                    current_robust_score = group_scores.min()
                print(loop, 'Before train Robust score: ', current_robust_score)
                robust_ss.append(current_robust_score)

            for _ in range(int(self.train_max_epoch * (loop + 1))):
                for i, data in enumerate(self.loader):
                    adv_idx = np.random.choice(
                        len(self.adversary_list), p=self.meta_strategies[1]
                    )
                    model.train(data, self.adversary_list[adv_idx])
            for _ in range(1):
                for i, data in enumerate(self.loader):
                    model_idx = np.random.choice(
                        len(self.model_list), p=self.meta_strategies[0]
                    )
                    adversary.train(data, self.model_list[model_idx])

            model_file = os.path.join("results", str(loop) + '_model.pt')
            lm_file = os.path.join("results", str(loop) + '_lm.pt')

            self.model_list.append(model)
            torch.save(model.model.state_dict(), model_file)
            self.adversary_list.append(adversary)
            torch.save(adversary.adversary.state_dict(), lm_file)

            save_file = str(loop) + '_model.pt'
            best_model_state_dict = torch.load(os.path.join("results", save_file))
            model.model.load_state_dict(best_model_state_dict)
            # Evaluate on general domain
            (test_examples_scores, test_losses) = self.task.eval_model(model.model, data='test', by_example=True,
                                                                       nll=True)
            test_examples_scores = test_examples_scores.numpy()
            test_losses = test_losses.numpy()
            average_ss.append(test_examples_scores.mean())

            group_scores = np.asarray([test_examples_scores[g_idxs].mean() for g_idxs in group_idxs.values()])
            lower_is_better = False
            if lower_is_better:
                current_robust_score = group_scores.max()
            else:
                current_robust_score = group_scores.min()

            robust_ss.append(current_robust_score)

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
                        print(sum_loss)

                        meta_games[0][t_r][t_c] = -sum_loss
                        meta_games[1][t_r][t_c] = sum_loss

            self.meta_games = meta_games

            # solution = "nash"
            solution = "uniform"

            print(solution)

            if solution == "nash":
                self.meta_strategies = projected_replicator_dynamics(self.meta_games)
            else:
                self.meta_strategies = [np.array([1.0 for _ in range(len(self.model_list))]) / len(self.model_list),
                                        np.array([1.0 for _ in range(len(self.adversary_list))]) / len(
                                            self.adversary_list)]
            print(self.meta_games)
            print(self.meta_strategies)

            average_score_d = np.average(a=average_ss, weights=self.meta_strategies[0])
            average_robust_score = np.average(a=robust_ss, weights=self.meta_strategies[0])

            print('Average', average_score_d)
            print('Robust', average_robust_score)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_loop", type=int, default=4)
    parser.add_argument(
        "--solution", type=str, default="the solution for the meta game"
    )
    parser.add_argument("--train_max_epoch", type=int, default=5)
    parser.add_argument("--eval_max_epoch", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    # print()
    do_dro = do_p_dro(args)
    do_dro.init()
    do_dro.solve()