import numpy as np
import os, json

from tqdm import tqdm
import _dr_provider as drp

import time

from sklearn.datasets import fetch_covtype, fetch_rcv1, fetch_kddcup99

from sklearn.decomposition import PCA, TruncatedSVD

import scipy.sparse as sp


DATASETS = os.listdir("../datasets_candidate/npy/")
METADATA = json.load(open("./_metadata.json", "r"))

SIZE_RANGE = (300, 50000)

result = {}
for dataset in DATASETS:


	print(dataset)
	if os.path.exists(f"./02_scalability/results/{dataset}.json"):
		continue

	X = np.load(f"../datasets_candidate/npy/{dataset}/data.npy")



	for dr_technique in METADATA.keys():
		if "default" not in METADATA[dr_technique]:
			continue
		print("-", dr_technique)
		times = []
		for i in tqdm(range(5)):

			runner_function_name = f"run_{dr_technique}"

			start = time.time()
			emb = getattr(drp, runner_function_name)(X, **METADATA[dr_technique]["default"])
			end = time.time()

			times.append(end-start)
		
		result[dr_technique] = np.mean(times)
			
		## save result 
	with open(f"./02_scalability/results/{dataset}.json", "w") as f:
		json.dump(result, f)



DR_TECHNIQUES = ["pca", "umap", "pacmap", "trimap"]			
DR_TECHNIQUES = ["pca"]

for dataset in [ "covertype", "kddcup99"]:
	print(dataset)

	if os.path.exists(f"./02_scalability/results_big/{dataset}.json"):
		continue

	if dataset == "covertype":
		X = fetch_covtype().data
	elif dataset == "rcv1":
		X = fetch_rcv1().data
		X = TruncatedSVD(n_components=500).fit_transform(X)
	elif dataset == "kddcup99":
		X = fetch_kddcup99().data
		X = X[:, np.array([type(x) != bytes for x in X[0]])]
		# X = sp.csr_matrix(X, dtype=np.float32)
		# X = X.toarray()

	## remove columns with str
	


	for dr_technique in DR_TECHNIQUES:
		if "default" not in METADATA[dr_technique]:
			continue
		print("-", dr_technique)
		runner_function_name = f"run_{dr_technique}"

		start = time.time()
		emb = getattr(drp, runner_function_name)(X, **METADATA[dr_technique]["default"])
		end = time.time()

		
		result[dr_technique] = end - start
		print(dr_technique, "time:", end-start)
	

			
	## save result
	with open(f"./02_scalability/results_big/{dataset}.json", "w") as f:
		json.dump(result, f)

		

		
