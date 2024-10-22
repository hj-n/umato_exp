import _dr_provider as drp


import os, json
import numpy as np
from zadu import zadu
import pandas as pd
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

DATASETS = os.listdir("../datasets_candidate/npy/")
METADATA = json.load(open("./_metadata_umato.json", "r"))
SPEC = json.load(open("./_spec.json", "r"))



for dataset in DATASETS:
	if dataset != "optical_recognition_of_handwritten_digits":
		continue

	print(f"{dataset} computing...")
	if os.path.exists(f"./01_accuracy/umato_results/{dataset}.csv"):
		continue
	
	## load file and initialize zadu object
	raw = np.load(f"../datasets_candidate/npy/{dataset}/data.npy")	
	label = np.load(f"../datasets_candidate/npy/{dataset}/label.npy")

	size = raw.shape[0]

	if size > 10000:
		continue

	zadu_obj = zadu.ZADU([{ "id": "tnc", "params": { "k": 10 } }], raw)
	# zadu_obj = zadu.ZADU([{ "id": "kl_div", "params": { "sigma": 0.1 } }], raw)


	## phase 1: find optimal embedding based on KL divergence (0.1)
	params_dict = {}
	for dr_technique in tqdm(METADATA.keys()):
		if len(METADATA[dr_technique]["bounds"]) > 0:
			bound = METADATA[dr_technique]["bounds"]

			if "n_neighbors" in bound.keys() and bound["n_neighbors"][1] > size-2:
				bound["n_neighbors"] = (2, size-2)
			if "n_inliers" in bound.keys() and bound["n_inliers"][1] > size-2:
				bound["n_inliers"] = (2, size-2)
			if "n_outliers" in bound.keys() and bound["n_outliers"][1] > size-2:
				bound["n_outliers"] = (2, size-2)
			if "hub_num" in bound.keys() and bound["hub_num"][1] > size / 4:
				bound["hub_num"] = (2, size / 4)


			runner_function_name = f"run_{dr_technique}"
			def f(**kwargs):
				try:
					start = time.time()
					emb = getattr(drp, runner_function_name)(raw, **kwargs)
					end = time.time()
					print("Generating embedding:", end - start)
					start = time.time()
					score = (2 * zadu_obj.measure(emb)[0]["trustworthiness"] * zadu_obj.measure(emb)[0]["continuity"]) / (zadu_obj.measure(emb)[0]["trustworthiness"] + zadu_obj.measure(emb)[0]["continuity"])
					score = - zadu_obj.measure(emb)[0]["kl_divergence"]
					end = time.time()
					print("Computing score:", end - start)
				except:
					score = 0

				print("Score:", score)
				return score
			
			optimizer = BayesianOptimization(f=f, pbounds=bound, verbose=0, allow_duplicate_points=True)
			optimizer.maximize(init_points=10, n_iter=20)
			params = optimizer.max["params"]
		else:
			params = {}
	
		params_dict[dr_technique] = params


		
	## phase 2: run all metrics
	spec = SPEC["spec"]
	retreiver = SPEC["retreiver"]
	zadu_obj = zadu.ZADU(spec, raw)

	## saver
	dr_technique_list = []
	metric_list = []
	value_list = []

	for dr_technique in tqdm(METADATA.keys()):
		params = params_dict[dr_technique]
		runner_function_name = f"run_{dr_technique}"
		emb = getattr(drp, runner_function_name)(raw, **params)
		values = zadu_obj.measure(emb)

		for i, ret_list in enumerate(retreiver):
			for ret in ret_list:
				if "params" in spec[i].keys():
					ret_hp = spec[i]["params"]
					metric = f"{ret}_{list(ret_hp.keys())[0]}_{list(ret_hp.values())[0]}"
				else:
					metric = ret
				value = values[i][ret]

				dr_technique_list.append(dr_technique)
				metric_list.append(metric)
				value_list.append(value)
		
		## save emb as plot as matplotlib
		plt.figure(figsize=(10, 10))

		plt.scatter(emb[:, 0], emb[:, 1], c=label, cmap="tab10")
		plt.title(f"{dr_technique}")
		plt.savefig(f"./01_accuracy/embeddings/{dataset}_{dr_technique}.png")
	
	## save
	df = pd.DataFrame({
		"dr_technique": dr_technique_list,
		"metric": metric_list,
		"value": value_list
	})






	df.to_csv(f"./01_accuracy/umato_results/{dataset}.csv", index=False)







			


	