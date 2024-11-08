import numpy as np

import _dr_provider as drp

from sklearn.datasets import make_swiss_roll, make_s_curve

import json

from zadu import zadu
import pandas as pd

import time
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import os



METADATA = json.load(open("./_metadata.json", "r"))

np.random.seed(42)

def swissroll_generator():
	X, labels = make_swiss_roll(n_samples=5000, noise=0.0)
	return X, labels


def swissroll_generator_larger_width():
	X, labels = make_swiss_roll(n_samples=5000, noise=0.0)
	X[:, 2] *= 3
	X[:, 1] *= 3
	return X, labels


def scurve_generator():
	X, labels = make_s_curve(n_samples=5000, noise=0.0)
	return X, labels

def mammoth_generator():
	with open("../synthetic/mammoth_umap.json", "r") as f:
		data = json.load(f)
		X = np.array(data["3d"])
		labels = np.array(data["labels"])

	return X, labels

def spheres_generator():
	df = pd.read_csv("../synthetic/spheres.csv")
	## columns 0 to 100 to numpy
	X = df.iloc[:, :100].values
	## column called "label" to numpy
	labels = df["label"].values
	return X, labels

generator_names = ["swissroll", "scurve", "mammoth", "spheres"]

"""

for i, generator in enumerate([swissroll_generator, scurve_generator, mammoth_generator, spheres_generator]):
	X, labels = generator()

	with open(f"./05_synthetic_embeddings/labels_{generator_names[i]}.json", "w") as f:
		json.dump(labels.tolist(), f)
	generator_name = generator_names[i]

	zadu_obj = zadu.ZADU([{ "id": "tnc", "params": { "k": 10 } }], X)

	size = X.shape[0]
	params_dict = {}
	for dr_technique in tqdm(METADATA.keys()):
		print("-", dr_technique)
		runner_function_name = f"run_{dr_technique}"

		if os.path.exists(f"./05_synthetic_embeddings/{dr_technique}_{generator_name}.json"):
			continue

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

			if dr_technique == "lamp":
				emb = getattr(drp, runner_function_name)(X, {})
				with open(f"./05_synthetic_embeddings/{dr_technique}_{generator_name}.json", "w") as f:
					json.dump(emb.tolist(), f)
				continue

			def f(**kwargs):
				try:
					start = time.time()
					emb = getattr(drp, runner_function_name)(X, **kwargs)
					end = time.time()
					print("Generating embedding:", end - start)
					start = time.time()
					score = (2 * zadu_obj.measure(emb)[0]["trustworthiness"] * zadu_obj.measure(emb)[0]["continuity"]) / (zadu_obj.measure(emb)[0]["trustworthiness"] + zadu_obj.measure(emb)[0]["continuity"])
					end = time.time()
					print("Computing score:", end - start)
				except:
					score = 0

				print("Score:", score)
				return score
			
			optimizer = BayesianOptimization(f=f, pbounds=bound, verbose=0, allow_duplicate_points=True)
			optimizer.maximize(init_points=5, n_iter=10)
			params = optimizer.max["params"]
		else:
			params = {}
	
		params_dict[dr_technique] = params

		## get final embedding
		emb = getattr(drp, runner_function_name)(X, **params)

		## save synthetic embeddings
		with open(f"./05_synthetic_embeddings/{dr_technique}_{generator_name}.json", "w") as f:
			json.dump(emb.tolist(), f)

"""

X, labels = spheres_generator()

emb = drp.run_umato(X, n_neighbors=50, min_dist=0.1, hub_num=300)

with open(f"./05_synthetic_embeddings/umatomy_spheres.json", "w") as f:
	json.dump(emb.tolist(), f)


