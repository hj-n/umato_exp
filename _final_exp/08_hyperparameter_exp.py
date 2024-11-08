import numpy as np

import _dr_provider as drp
import json
from tqdm import tqdm

def mammoth_generator():
	with open("../synthetic/mammoth_umap.json", "r") as f:
		data = json.load(f)
		X = np.array(data["3d"])
		labels = np.array(data["labels"])

	return X, labels


X, labels = mammoth_generator()

n_neighbors_list = [2, 5, 10, 20, 30, 40, 50, 60]
min_dist_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
hub_num_list = [10, 20, 30, 50, 70, 100, 150, 200]

default_n_neighbors = 50
default_min_dist = 0.1
default_hub_num = 150

for n_neighbors in tqdm(n_neighbors_list):
	emb = drp.run_umato(X, n_neighbors=n_neighbors, min_dist=default_min_dist, hub_num=default_hub_num)
	np.save(f"./08_hyperparameter_exp/results_hptest/mammoth_umato_n_neighbors_{n_neighbors}.npy", emb)

for min_dist in tqdm(min_dist_list):
	emb = drp.run_umato(X, n_neighbors=default_n_neighbors, min_dist=min_dist, hub_num=default_hub_num)
	np.save(f"./08_hyperparameter_exp/results_hptest/mammoth_umato_min_dist_{min_dist}.npy", emb)

for hub_num in tqdm(hub_num_list):
	emb = drp.run_umato(X, n_neighbors=default_n_neighbors, min_dist=default_min_dist, hub_num=hub_num)
	np.save(f"./08_hyperparameter_exp/results_hptest/mammoth_umato_hub_num_{hub_num}.npy", emb)
