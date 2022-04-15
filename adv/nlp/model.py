from adv.nlp.configs import DATASET_CONFIGS
import datasets
import os
import textattack
import transformers

import torch


class a2t_model:
    def __init__(self, config):
        self.config = config
        self.args = config["args"]

        self.model = None

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

        self.model = model_wrapper
