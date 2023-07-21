import _dr_provider as drp 

import numpy as np

import os, json
from tqdm import tqdm

from scipy.spatial import procrustes as proc
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from zadu import zadu

import _permutation as perm

DATASETS = os.listdir("../datasets_candidate/npy")
METADATA = json.load(open("./_metadata.json", "r"))

# SAMPLING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DR_TECHNIQUES = list(METADATA.keys())




for dataset in DATASETS:

	if os.path.exists(f"./03_01_stability_subsampling/results/{dataset}.csv"):
		continue

	X = np.load(f"../datasets_candidate/npy/{dataset}/data.npy")
	size = X.shape[0]

	if size < 9999 or size > 10001:
		continue

	dr_technique_list = []
	sampling_list = []
	disparity_list = []
		
	print(dataset)
	for dr_technique in DR_TECHNIQUES:
		print("-", dr_technique)
		runner_function_name = f"run_{dr_technique}"


		arguments = METADATA[dr_technique]["default"]

		X_emb = getattr(drp, runner_function_name)(X, **arguments)

		# for sampling in tqdm(SAMPLING):
		for i in tqdm(range(50)):

			sampling = np.random.uniform(0.1, 0.99)

			sampler = np.random.choice(size, int(size*sampling), replace=False)
			subsampled_X = X[sampler]
			subsampled_X_emb = getattr(drp, runner_function_name)(subsampled_X, **arguments)
			X_emb_subampled = X_emb[sampler]

			# plt.scatter(X_emb_subampled[:, 0], X_emb_subampled[:, 1])
			# plt.savefig(f"./03_01_stability_subsampling/results_plot/{dataset}_{dr_technique}_{sampling}_{i}_original.png")

			# plt.clf()
			# plt.scatter(subsampled_X_emb[:, 0], subsampled_X_emb[:, 1])
			# plt.savefig(f"./03_01_stability_subsampling/results_plot/{dataset}_{dr_technique}_{sampling}_{i}_subsampled.png")

			plt.clf()

			results = perm.permutation(X_emb_subampled.T, subsampled_X_emb.T)

			X1, X2 = results.new_a, results.new_b

			X1, X2, disparity = proc(X1, X2)

			# scores = zadu.ZADU([{ "id": "tnc", "params": {"k": 30}}], X_emb_subampled).measure(subsampled_X_emb)

			# trust = scores[0]["trustworthiness"]
			# conti = scores[0]["continuity"]

			# disaprity_tnc = trust * conti / (trust + conti)

			# print(disaprity_tnc)





			dr_technique_list.append(dr_technique)
			sampling_list.append(sampling)
			disparity_list.append(disparity)

			print(disparity)
		
		## save result
	result = {
		"dr_technique": dr_technique_list,
		"sampling": sampling_list,
		"disparity": disparity_list
	}

	df = pd.DataFrame(result)

	df.to_csv(f"./03_01_stability_subsampling/results/{dataset}.csv", index=False)

			



			





