from adv.nlp.configs import DATASET_CONFIGS
import datasets
import os
import textattack
import transformers

import torch


class do_at_nlp:
    def __init__(self, args):
        self.args = args

        self.model_wrapper = None
        self.attacker = None

        self.task_type = "regression"
        if self.task_type == "regression":
            self.loss_fct = torch.nn.MSELoss(reduction="none")
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self.train_dataset = None
        self.eval_dataset = None

    def get_datasets(self):
        args = self.args
        dataset_config = DATASET_CONFIGS[args.train]

        if "local_path" in dataset_config:
            train_dataset = datasets.load_dataset(
                "csv",
                data_files=os.path.join(dataset_config["local_path"], "train.tsv"),
                delimiter="\t",
            )["train"]
        else:
            train_dataset = datasets.load_dataset(
                dataset_config["remote_name"], split="train"
            )

        if "local_path" in dataset_config:
            eval_dataset = datasets.load_dataset(
                "csv",
                data_files=os.path.join(dataset_config["local_path"], "val.tsv"),
                delimiter="\t",
            )["train"]
        else:
            eval_dataset = datasets.load_dataset(
                dataset_config["remote_name"], split="validation"
            )
        train_dataset = textattack.datasets.HuggingFaceDataset(
            train_dataset,
            dataset_columns=dataset_config["dataset_columns"],
            label_names=dataset_config["label_names"],
        )

        eval_dataset = textattack.datasets.HuggingFaceDataset(
            eval_dataset,
            dataset_columns=dataset_config["dataset_columns"],
            label_names=dataset_config["label_names"],
        )
        train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
        eval_dataset = eval_dataset.filter(lambda x: x["label"] != -1)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def get_model(self):
        args = self.args
        dataset_config = DATASET_CONFIGS[args.train]
        if args.model_type == "bert":
            pretrained_name = "bert-base-uncased"
        elif args.model_type == "roberta":
            pretrained_name = "roberta-base"
        else:
            raise NotImplementedError

        if args.model_chkpt_path:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                args.model_chkpt_path
            )
        else:
            num_labels = dataset_config["labels"]
            config = transformers.AutoConfig.from_pretrained(
                pretrained_name, num_labels=num_labels
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_name, config=config
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_name, use_fast=True
        )
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer
        )

        self.model_wrapper = model_wrapper

    def get_attacker(self):
        print()
        args = self.args
        model_wrapper = self.model_wrapper
        if args.attack == "a2t":
            attack = textattack.attack_recipes.A2TYoo2021.build(
                model_wrapper, mlm=False
            )
        elif args.attack == "a2t_mlm":
            attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper, mlm=True)
        else:
            raise ValueError(f"Unknown attack {args.attack}.")

        self.attacker = attack

    def generate_adversarial_examples(self):
        print()
        # attack_args = AttackArgs(
        #     num_successful_examples=num_train_adv_examples,
        #     num_examples_offset=0,
        #     query_budget=self.training_args.query_budget_train,
        #     shuffle=True,
        #     parallel=self.training_args.parallel,
        #     num_workers_per_device=self.training_args.attack_num_workers_per_device,
        #     disable_stdout=True,
        #     silent=True,
        #     log_to_txt=log_file_name + ".txt",
        #     log_to_csv=log_file_name + ".csv",
        # )
        #
        # attacker = Attacker(self.attack, self.train_dataset, attack_args=attack_args)
        # results = attacker.attack_dataset()
