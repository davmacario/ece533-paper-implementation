import matplotlib.pyplot as plt
import numpy as np
import os

num_to_plot = 40

list_of_fednova_paths = [n for n in os.listdir("./fednova/") if n[-3:] == "npy"]
fednova_data = [np.load("./fednova/" + n) for n in list_of_fednova_paths]
num_fednova = len(fednova_data[0])
fednova_mse = []
for i in range(num_fednova):
    fednova_mse.append((1/num_fednova)*sum([n[i] for n in fednova_data]))

plt.plot(fednova_mse[0:num_to_plot], c='r')
#plt.scatter(range(num_to_plot), fednova_mse[0:num_to_plot], s=4, c='r')

list_of_fedavg_paths = [n for n in os.listdir("./fedavg/") if n[-3:] == "npy"]
fedavg_data = [np.load("./fedavg/" + n) for n in list_of_fedavg_paths]
num_fedavg = len(fedavg_data[0])
fedavg_mse = []
for i in range(num_fedavg):
    fedavg_mse.append((1/num_fedavg)*sum([n[i] for n in fedavg_data]))

plt.plot(fedavg_mse[0:num_to_plot], c='b')
#plt.scatter(range(num_to_plot), fedavg_mse[0:num_to_plot], s=4, c='b')

plt.show()