from zadu import zadu 
from zaduvis import zaduvis
import matplotlib.pyplot as plt

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

import _dr_provider as drp


dataset = "human_stress_detection"

raw = np.load(f"../datasets/npy/{dataset}/data.npy")
size = raw.shape[0]

METADATA = json.load(open("./_metadata_umato.json", "r"))
SPEC = json.load(open("./_spec.json", "r"))

zadu_obj = zadu.ZADU([{
			"id": "snc",
			"params": {"k": 50}
	}], raw)

for dr_technique in tqdm(METADATA.keys()):

	if os.path.exists(f"./04_reliabilitymap/vis/{dataset}_{dr_technique}.png"):
		continue

	if dr_technique != "umato":
		continue

	runner_function_name = f"run_{dr_technique}"
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


		
		def f(**kwargs):
			try:
				start = time.time()
				emb = getattr(drp, runner_function_name)(raw, **kwargs)
				end = time.time()
				print("Generating embedding:", end - start)
				start = time.time()
				measured = zadu_obj.measure(emb)[0]
				score = (2 * measured["steadiness"] * measured["cohesiveness"]) / (measured["steadiness"] + measured["cohesiveness"])
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


	## phase 2: run  again with params
	emb = getattr(drp, runner_function_name)(raw, **params)

	
	spec = [{
    "id": "tnc",
    "params": {"k": 25}
	},{
			"id": "snc",
			"params": {"k": 50}
	}]
	zadu_obj = zadu.ZADU(spec, raw, return_local=True)
	scores, local_list = zadu_obj.measure(emb)

	tnc_local = local_list[0]
	snc_local = local_list[1]

	local_trustworthiness = tnc_local["local_trustworthiness"]
	local_continuity = tnc_local["local_continuity"]
	local_steadiness = snc_local["local_steadiness"]
	local_cohesiveness = snc_local["local_cohesiveness"]


	## save
	np.save(f"./04_reliabilitymap/embeddings/{dataset}_{dr_technique}.npy", emb)

	fig, ax = plt.subplots(1, 2, figsize=(25, 12.5))
	zaduvis.reliability_map(emb, local_trustworthiness, local_continuity, k=10, ax=ax[0])
	zaduvis.reliability_map(emb, local_steadiness, local_cohesiveness, k=10, ax=ax[1], point_s=0.6)


	plt.tight_layout()
	plt.savefig(f"./04_reliabilitymap/vis/{dataset}_{dr_technique}.png", dpi=300)
	plt.clf()
	## 