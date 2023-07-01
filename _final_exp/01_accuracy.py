import _dr_provider as drp


import os, json
import numpy as np
from zadu import zadu
import pandas as pd

from bayes_opt import BayesianOptimization
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

DATASETS = os.listdir("../datasets/npy/")
METADATA = json.load(open("./_metadata.json", "r"))
SPEC = json.load(open("./_spec.json", "r"))



for dataset in DATASETS:
	print(f"{dataset} computing...")
	if os.path.exists(f"./01_accuracy/results/{dataset}.csv"):
		continue
	
	## load file and initialize zadu object
	raw = np.load(f"../datasets/npy/{dataset}/data.npy")	
	zadu_obj = zadu.ZADU([{ "id": "kl_div", "params": { "sigma": 0.1 } }], raw)
	size = raw.shape[0]


	## phase 1: find optimal embedding based on KL divergence (0.1)
	params_dict = {}
	for dr_technique in tqdm(METADATA.keys()):
		if len(METADATA[dr_technique]["bounds"]) > 0:
			bound = METADATA[dr_technique]["bounds"]

			if "n_neighbors" in bound.keys():
				bound["n_neighbors"] = (2, size-1)
			if "n_inliers" in bound.keys():
				bound["n_inliers"] = (2, size-1)
			if "n_outliers" in bound.keys():
				bound["n_outliers"] = (2, size-1)
			if "hub_num" in bound.keys():
				bound["hub_num"] = (2, size-1)


			runner_function_name = f"run_{dr_technique}"
			def f(**kwargs):
				emb = getattr(drp, runner_function_name)(raw, **kwargs)
				return zadu_obj.measure(emb)[0]["kl_divergence"] * -1
			
			optimizer = BayesianOptimization(f=f, pbounds=bound, verbose=0)
			optimizer.maximize(init_points=1, n_iter=2)
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
	
	## save
	df = pd.DataFrame({
		"dr_technique": dr_technique_list,
		"metric": metric_list,
		"value": value_list
	})

	df.to_csv(f"./01_accuracy/results/{dataset}.csv", index=False)







			


	