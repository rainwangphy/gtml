# parser.add_argument("--gan_name", type=str, default="gan")
# parser.add_argument("--dataset", type=str, default="cifar10")

gan_name_list = ["gan", "wgan"]
solution_list = ["nash", "uniform"]
dataset_list = ["cifar10", "stl10"]


cuda_devices_list = [2, 3]

cuda_str = "CUDA_VISIBLE_DEVICES"
filename = "run_exp.sh"

f = open(file=filename, mode="w")
pare_folder = "./results/outputs"
f.write("mkdir -p {}\n\n".format(pare_folder))

device_number = 0
for gan_name in gan_name_list:
    for solution in solution_list:
        for dataset in dataset_list:
            config = ""
            config += "--gan_name {} ".format(gan_name)
            config += "--solution {} ".format(solution)
            config += "--dataset {} ".format(dataset)
            # config += "--dueling_or_not {} ".format(dueling_or_not)
            # config += "--double_or_not {} ".format(double_or_not)

            config_l = ""
            config_l += "gan_name_{}_".format(gan_name)
            config_l += "solution_{}_".format(solution)
            config_l += "dataset_{}_".format(dataset)
            # config_l += "dueling_or_not_{}_".format(dueling_or_not)
            # config_l += "double_or_not_{}".format(double_or_not)
            f.write(cuda_str + "={} python3 -u do_gan.py ".format(cuda_devices_list[device_number]))
            f.write(config)
            f.write(" > " + pare_folder)
            f.write("/log_" + config_l + ".txt")
            f.write("&\n")
            device_number += 1
            device_number = device_number % len(cuda_devices_list)
        f.write("\n")
