import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll, make_s_curve

import os , json
import numpy as np
import pandas as pd

np.random.seed(42)

def swissroll_generator():
	with open("./05_synthetic_embeddings/labels_swissroll.json", "r") as f:
		labels = np.array(json.load(f))
	return None, labels


def scurve_generator():
	with open("./05_synthetic_embeddings/labels_scurve.json", "r") as f:
		labels = np.array(json.load(f))
	return None, labels

def mammoth_generator():
	with open("./05_synthetic_embeddings/labels_mammoth.json", "r") as f:
		labels = np.array(json.load(f))
	return None, labels

def spheres_generator():
	with open("./05_synthetic_embeddings/labels_spheres.json", "r") as f:
		labels = np.array(json.load(f))
	return None, labels

datasets = ["swissroll", "scurve", "spheres", "mammoth"]
generators = [swissroll_generator, scurve_generator, spheres_generator, mammoth_generator]

techniques = ["umato", "umap", "tsne", "trimap", "pacmap", "lle", "isomap", "pca", "lmds", "lamp"]

fig, ax = plt.subplots(len(datasets), len(techniques), figsize=(28, 10))

for i, dataset in enumerate(datasets):
	try:
		_, labels = generators[i]()
	except:
		continue
	for j, technique in enumerate(techniques):
		print(dataset, technique)

		if dataset == "swissroll" and technique == "umato":
			technique = "umatomy"
		
		if dataset == "spheres" and technique == "umato":
			technique = "umatomy"

		if not os.path.exists(f"./05_synthetic_embeddings/{technique}_{dataset}.json"):
			continue
		
		with open(f"./05_synthetic_embeddings/{technique}_{dataset}.json", "r") as f:
			X = np.array(json.load(f))

		
		ax[i, j].scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=1)

		if i == 3:
			ax[i, j].set_xlabel(technique)
		
		## remove border
		ax[i, j].spines['top'].set_visible(False)
		ax[i, j].spines['right'].set_visible(False)
		ax[i, j].spines['bottom'].set_visible(False)
		ax[i, j].spines['left'].set_visible(False)

		## remove ticks
		ax[i, j].set_xticks([])
		ax[i, j].set_yticks([])



plt.tight_layout()
plt.savefig("./05_synthetic_embeddings_plot/result.png", dpi=300)


