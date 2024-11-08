import _dr_provider as drp


import os, json
import numpy as np
from zadu import zadu
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

DATASETS = os.listdir("../datasets_candidate/npy/")




for dataset in DATASETS:
	print(f"{dataset} computing...")
	
	raw = np.load(f"../datasets_candidate/npy/{dataset}/data.npy")

	size = raw.shape[0]

	if size > 10000:
		continue
	
	zadu_obj = zadu.ZADU([
		{ "id": "tnc", "params": { "k": 10 } },
		{ "id": "kl_div", "params": { "sigma": 0.1 } }
	], raw)


	hub_num_end = size / 2
	hub_nums = [ 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
	hub_nums = [hub_num for hub_num in hub_nums if hub_num <= hub_num_end]
	
	for hub_num in tqdm(hub_nums):
		if os.path.exists(f"./08_hyperparameter_exp/results_hubnum/{dataset}_umato_hub_num_{hub_num}.npy"):
			continue
		emb = drp.run_umato(raw, n_neighbors=75, min_dist=0.1, hub_num=hub_num)
		np.save(f"./08_hyperparameter_exp/results_hubnum/{dataset}_umato_hub_num_{hub_num}.npy", emb)

		## compute accuracy
		acc = zadu_obj.measure(emb)
		tnc = 2 * acc[0]["trustworthiness"] * acc[0]["continuity"] / (acc[0]["trustworthiness"] + acc[0]["continuity"])
		kl_div = acc[1]["kl_divergence"]

		with open(f"./08_hyperparameter_exp/results_hubnum/{dataset}_umato_hub_num_{hub_num}_results.json", "w") as f:
			json.dump({
				"tnc": tnc,
				"kl_divergence": kl_div,
			}, f)
	