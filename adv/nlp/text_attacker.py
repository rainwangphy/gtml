class text_attacker:
    def __init__(self, config):
        self.config = config

        self.models = config["models"]
        self.models_dis = config["models_dis"]

        self.clean_dataset = config["clean_dataset"]

    def get_adversarial_dataset(self):
        print()
