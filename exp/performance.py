import umap ## umap, densmap ## conda install umap
import pacmap ## pacmap ## pip install pacmap
import umato
import trimap ## trimap ## pip install trimap
from MulticoreTSNE import MulticoreTSNE as TSNE ## pip install MulticoreTSNE
from sklearn.decomposition import PCA ## pca
from sklearn.manifold import Isomap	## isomap ## conda install scikit-learn

import os
import numpy as np

import performance_helper as ph
import pandas as pd

from tqdm import tqdm

DATASETS = os.listdir("../datasets/npy/")

dr_id_list = ["pca", "pacmap","umap", "trimap", "tsne", "densmap", "umato", "isomap"]	
metric_id_list = (
	["trustworthiness"] +
	["continuity"] + 
	["kl_divergence"] * 3+
	["distance_to_measure"] * 3 +
	["steadiness", "cohesiveness"] +
	["stress", "pearson_r", "spearman_rho"]
)
params_list = (
    [{"k": x} for x in [20]] * 2  +
    [{"sigma": x} for x in [0.01, 0.1, 1] ] * 2 +
    [{}] * 5
)

for dataset in DATASETS[72:]:

	print(f"{dataset} computing...")
	if os.path.exists(f"./performace/{dataset}.csv"):
		continue

	data = np.load(f"../datasets/npy/{dataset}/data.npy")


	data_size = data.shape[0]
  
	if data_size > 10000:
		data = data[np.random.choice(data_size, 10000, replace=False)]

	metric_list = []
	dr_list = []
	score_list = []

	
	for i, metric_id in enumerate(metric_id_list):
		print("    ..", metric_id)
		for dr_id in tqdm(dr_id_list):
			print("       ..", dr_id, end=": ")
			try:
				result = ph.run(data, dr_id, metric_id, params_list[i])

				metric_list.append(metric_id)
				dr_list.append(dr_id)
				score_list.append(result)
				print(result)
			except:
				pass
	df = pd.DataFrame({
		"metric": metric_list,
		"dr": dr_list,
		"score": score_list
	})
	

	df.to_csv(f"./performance/{dataset}.csv")

	