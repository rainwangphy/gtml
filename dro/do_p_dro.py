import argparse
import numpy as np

class do_p_dro:
    def __init__(self, args):
        self.args = args
        self.max_loop = args.max_loop
        self.solution = args.solution
        self.train_max_epoch = args.train_max_epoch
        self.eval_max_epoch = args.eval_max_epoch
        self.device = args.device

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

    def init(self):
        print()

    def solve(self):
        print()


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
    do_dro.solve()
