import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(48,16))
for logs in os.listdir(args.path):
    if "server" in logs:
        path = args.path+logs
        file = open(path, "r")
        acc_list = []
        loss_list = []
        time_cost_list = []
        for line in file:
            if "loss" in line:
                acc_list.append(int(line.split(" ")[10].strip("\n").strip("(").strip(")").strip("%")))
                loss_list.append(float(line.split(" ")[7].strip(",")))
            if "cost" in line:
                time_cost_list.append(float(line.split(" ")[-1].strip("\n")))
        if time_cost_list != []:
            print("%s time costs: %s, avg time cost: %.4f" % (logs, time_cost_list, sum(time_cost_list)/len(time_cost_list)))
        x = range(len(loss_list))
        axs[1].plot(x, acc_list, label=logs)
        axs[2].plot(x, loss_list, label=logs)
    if "client" in logs:
        path = args.path+logs
        file = open(path, "r")
        loss_list = []
        for line in file:
            if "Loss" in line:
                loss  = line.split(" ")[8]
                loss = loss.split(":")[1]
                loss = loss.strip("\n")
                loss_list.append(float(loss))
        x = range(len(loss_list))
        axs[0].plot(x, loss_list, label=logs)
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[0].set_ylabel("client loss")
axs[0].set_xlabel("client iteration")
axs[1].set_ylabel("server acc")
axs[1].set_xlabel("server round")
axs[2].set_ylabel("server loss")
axs[2].set_xlabel("server round")
plt.savefig("./result.png")