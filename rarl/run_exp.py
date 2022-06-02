# if env_name == "walker_heel":
#     env = Walker2dHeelEnv()
# elif env_name == "walker_torso":
#     env = Walker2dTorsoEnv()
# elif env_name == "hopper_torso":
#     env = HopperTorso6Env()
# else:
#     env = HopperHeelEnv()

env_list = ["walker_heel", "walker_torso", "hopper_torso", "hopper_heel"]
# seed_list = [123, 231, 312]
seed_list = [456, 564]
solution_list = ["uniform", 'nash']


# parser.add_argument("--gan_name", type=str, default="gan")
# parser.add_argument("--dataset", type=str, default="cifar10")

# gan_name_list = ["gan", "wgan"]
# solution_list = ["nash", "uniform"]
# dataset_list = ["cifar10", "stl10"]


cuda_devices_list = [2, 3]

cuda_str = "CUDA_VISIBLE_DEVICES"
filename = "run_exp.sh"

f = open(file=filename, mode="w")
pare_folder = "./results/outputs"
f.write("mkdir -p {}\n\n".format(pare_folder))

device_number = 0
for env in env_list:
    for solution in solution_list:
        for seed in seed_list:
            config = ""
            config += "--env {} ".format(env)
            config += "--solution {} ".format(solution)
            config += "--seed {} ".format(seed)
            # config += "--dueling_or_not {} ".format(dueling_or_not)
            # config += "--double_or_not {} ".format(double_or_not)

            config_l = ""
            config_l += "env_{}_".format(env)
            config_l += "solution_{}_".format(solution)
            config_l += "seed_{}".format(seed)
            # config_l += "dueling_or_not_{}_".format(dueling_or_not)
            # config_l += "double_or_not_{}".format(double_or_not)
            f.write(cuda_str + "={} python3 -u psro_rarl.py ".format(cuda_devices_list[device_number]))
            f.write(config)
            f.write(" > " + pare_folder)
            f.write("/log_" + config_l + ".txt")
            f.write("&\n")
            device_number += 1
            device_number = device_number % len(cuda_devices_list)
        f.write("\n")
