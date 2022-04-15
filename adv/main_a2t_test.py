import textattack
import os
import argparse
import random
import math
import datetime

import textattack
import transformers
import datasets
import pandas as pd

from adv.nlp.configs import DATASET_CONFIGS


def int_or_float(v):
    try:
        return int(v)
    except ValueError:
        return float(v)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    type=str,
    # required=True,
    default="imdb",
    choices=sorted(list(DATASET_CONFIGS.keys())),
    help="Name of dataset for training.",
)
parser.add_argument(
    "--augmented-data",
    type=str,
    required=False,
    default=None,
    help="Path of augmented data (in TSV).",
)
parser.add_argument(
    "--pct-of-augmented",
    type=float,
    required=False,
    default=0.2,
    help="Percentage of augmented data to use.",
)
parser.add_argument(
    "--eval",
    type=str,
    # required=True,
    default="imdb",
    choices=sorted(list(DATASET_CONFIGS.keys())),
    help="Name of huggingface dataset for validation",
)
parser.add_argument(
    "--parallel", action="store_true", help="Run training with multiple GPUs."
)
parser.add_argument(
    "--model-type",
    type=str,
    # required=True,
    default="bert",
    choices=["bert", "roberta"],
    help='Type of model (e.g. "bert", "roberta").',
)
parser.add_argument(
    "--model-save-path",
    type=str,
    default="./saved_model",
    help="Directory to save model checkpoint.",
)
parser.add_argument(
    "--model-chkpt-path",
    type=str,
    default=None,
    help="Directory of model checkpoint to resume from.",
)
parser.add_argument(
    "--num-epochs", type=int, default=4, help="Number of epochs to train."
)
parser.add_argument(
    "--num-clean-epochs", type=int, default=1, help="Number of clean epochs"
)
parser.add_argument(
    "--num-adv-examples",
    type=int_or_float,
    default=0.2,
    help="Number (or percentage) of adversarial examples for training.",
)
parser.add_argument(
    "--attack-epoch-interval",
    type=int,
    default=1,
    help="Attack model to generate adversarial examples every N epochs.",
)
parser.add_argument(
    "--attack",
    type=str,
    default="a2t",
    choices=["a2t", "a2t_mlm"],
    help="Name of attack.",
)
parser.add_argument(
    "--per-device-train-batch-size",
    type=int,
    default=8,
    help="Train batch size (per GPU device).",
)
parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
parser.add_argument(
    "--num-warmup-steps", type=int, default=500, help="Number of warmup steps."
)
parser.add_argument(
    "--grad-accumu-steps",
    type=int,
    default=1,
    help="Number of gradient accumulation steps.",
)
parser.add_argument(
    "--checkpoint-interval-epochs",
    type=int,
    default=None,
    help="If set, save model checkpoint after every `N` epochs.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()
