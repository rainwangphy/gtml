import OpenAttack as oa
import datasets  # use the Hugging Face's datasets library
from adv.nlp.attacks.fd import FDAttacker
from adv.nlp.attacks.hotflip import HotFlipAttacker
from adv.nlp.attacks.uat import UATAttacker


# change the SST dataset into 2-class
def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }


# choose a trained victim classification model
victim = oa.DataManager.loadVictim("BERT.SST")
# choose 20 examples from SST-2 as the evaluation data
dataset = datasets.load_dataset("sst", split="train[:20]", script_version="master").map(function=dataset_mapping)
# choose PWWS as the attacker and initialize it with default parameters
attacker = oa.attackers.BAEAttacker()
# attacker = FDAttacker()
# prepare for attacking
attack_eval = oa.AttackEval(attacker, victim)
# launch attacks and print attack results
attack_eval.eval(dataset, visualize=True)
#
# attacker = UATAttacker()
# # prepare for attacking
# attack_eval = oa.AttackEval(attacker, victim)
# # launch attacks and print attack results
# attack_eval.eval(dataset, visualize=True)
#
# attacker = HotFlipAttacker()
# # prepare for attacking
# attack_eval = oa.AttackEval(attacker, victim)
# # launch attacks and print attack results
# attack_eval.eval(dataset, visualize=True)
