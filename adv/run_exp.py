nb_iter_list = [5, 10]
solution_list = ["nash", "uniform"]
dataset_list = ["cifar10", "cifar100"]

cuda_devices_list = [4, 5, 6, 7]

cuda_str = "CUDA_VISIBLE_DEVICES"
filename = "run_exp.sh"

f = open(file=filename, mode="w")
pare_folder = "./results/outputs"
f.write("mkdir -p {}\n\n".format(pare_folder))

device_number = 0
for nb_iter in nb_iter_list:
    for solution in solution_list:
        for dataset in dataset_list:
            config = ""
            config += "--nb_iter {} ".format(nb_iter)
            config += "--solution {} ".format(solution)
            config += "--dataset {} ".format(dataset)
            # config += "--dueling_or_not {} ".format(dueling_or_not)
            # config += "--double_or_not {} ".format(double_or_not)

            config_l = ""
            config_l += "nb_iter_{}_".format(nb_iter)
            config_l += "solution_{}_".format(solution)
            config_l += "dataset_{}_".format(dataset)
            # config_l += "dueling_or_not_{}_".format(dueling_or_not)
            # config_l += "double_or_not_{}".format(double_or_not)
            f.write(cuda_str + "={} python3 -u do_at_cv.py ".format(cuda_devices_list[device_number]))
            f.write(config)
            f.write(" > " + pare_folder)
            f.write("/log_" + config_l + ".txt")
            f.write("&\n")
            device_number += 1
            device_number = device_number % len(cuda_devices_list)
        f.write("\n")
