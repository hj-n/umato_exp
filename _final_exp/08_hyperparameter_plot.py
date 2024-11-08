import matplotlib.pyplot as plt

import numpy as np
import json

n_neighbors_list = [2, 5, 10, 20, 30, 40, 50, 60]
min_dist_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
hub_num_list = [10, 20, 30, 50, 70, 100, 150, 200]

lenlen = len(n_neighbors_list)

default_n_neighbors = 50
default_min_dist = 0.1
default_hub_num = 150

fig, axs = plt.subplots(3, lenlen, figsize=(20, 8))

with open("../synthetic/mammoth_umap.json", "r") as f:
	labels = np.array(json.load(f)["labels"])

for i, (param_list, param_name) in enumerate(zip([n_neighbors_list, min_dist_list, hub_num_list], ["n_neighbors", "min_dist", "hub_num"])):
	for j, param in enumerate(param_list):
		emb = np.load(f"./08_hyperparameter_exp/results_hptest/mammoth_umato_{param_name}_{param}.npy")
		axs[i, j].scatter(emb[:, 0], -emb[:, 1], c=labels, cmap="tab10", s=1)
		axs[i, j].set_title(f"{param_name}_{param}")
		axs[i, j].set_xticks([])
		axs[i, j].set_yticks([])
		axs[i, j].spines['top'].set_visible(False)
		axs[i, j].spines['right'].set_visible(False)
		axs[i, j].spines['bottom'].set_visible(False)
		axs[i, j].spines['left'].set_visible(False)


plt.tight_layout()
plt.savefig(f"./08_hyperparameter_exp/plot/mammoth_umato_hp.png")

plt.savefig(f"./08_hyperparameter_exp/plot/mammoth_umato_hp.svg")
