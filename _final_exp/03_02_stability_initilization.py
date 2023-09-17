import numpy as np

import pandas as pd

import os, json

import _dr_provider as drp 

from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

from tqdm import tqdm

import _permutation as perm
from scipy.spatial import procrustes as proc

DATASETS = os.listdir("../datasets_candidate/npy")
METADATA = json.load(open("./_metadata.json", "r"))


DR_TECHNIQUES = ["umap", "umato", "pacmap", "tsne"]

for dataset in DATASETS:
	if os.path.exists(f"./03_02_stability_init/results/{dataset}.csv"):
		continue
  
	X = np.load(f"../datasets_candidate/npy/{dataset}/data.npy")
	size = X.shape[0]

	if size <=  10000:
		continue


	print(dataset)

	disparity_list = []
	dr_technique_list = []
	for dr_technique in DR_TECHNIQUES:
		print(dr_technique)
		runner_function_name = f"run_{dr_technique}"
		arguments = METADATA[dr_technique]["default"]

		embeddings = []
		for initialization in tqdm(["pca", "spectral", "random", "random", "random"]):
			if initialization == "pca":
				embedding = PCA(n_components=2).fit_transform(X)
			elif initialization == "spectral":
				embedding = SpectralEmbedding(n_components=2).fit_transform(X)
			elif initialization == "random":
				embedding = np.random.uniform(0, 1, (size, 2))
			
			arguments["init"] = embedding
			final_emb = getattr(drp, runner_function_name)(X, **arguments)

			embeddings.append(final_emb)
	

		for i, emb1 in tqdm(enumerate(embeddings)):
			for j, emb2 in enumerate(embeddings):
				if i <= j:
					continue

				results = perm.permutation(emb1.T, emb2.T)

				_, _, disparity = proc(results.new_a, results.new_b)
				disparity_list.append(disparity)
				dr_technique_list.append(dr_technique)
		

	df = pd.DataFrame({
		"dr_technique": dr_technique_list,
		"disparity": disparity_list
	})

	df.to_csv(f"./03_02_stability_init/results/{dataset}.csv", index=False)
	

			



