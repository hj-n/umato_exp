import _dr_provider as drp
import _cvm_provider as cvmp

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

DATASETS = os.listdir("../datasets/npy/")
METADATA = json.load(open("./_metadata_umato.json", "r"))

DATASETS = ["date_fruit"]

for dataset in DATASETS:
	print(f"{dataset} computing...")
	
	raw = np.load(f"../datasets/npy/{dataset}/data.npy")
	label = np.load(f"../datasets/npy/{dataset}/label.npy")

	size = raw.shape[0]

	cvm_names = ["btw_ch", "dsc"]

	for jj, cvm in enumerate([cvmp.dsc_normalize, cvmp.btw_ch]):

		def run_cvm(raw, emb, label):
			unique_labels = np.unique(label)
			label_dict = {}
			for i, label_single in enumerate(unique_labels):
				label_dict[label_single] = i

			int_labels = np.zeros(label.shape[0])
			for i in range(label.shape[0]):
				int_labels[i] = label_dict[label[i]]

			label_num = len(np.unique(label))
			raw_cvm_mat = np.zeros((label_num, label_num))
			emb_cvm_mat = np.zeros((label_num, label_num))

			for label_i in range(label_num):
				for label_j in range(label_i + 1, label_num):
					## raw data of a pair of labels
					filter_label = np.logical_or(label == label_i, label == label_j)
					raw_pair = raw[filter_label]
					emb_pair = emb[filter_label]
					## label of the raw data of a pair of labels
					raw_pair_label = int_labels[filter_label]
					emb_pair_label = int_labels[filter_label]

					## change the label to 0 and 1
					raw_pair_label[raw_pair_label == label_i] = 0
					raw_pair_label[raw_pair_label == label_j] = 1
					emb_pair_label[emb_pair_label == label_i] = 0
					emb_pair_label[emb_pair_label == label_j] = 1

					## compute cvm
					raw_cvm_mat[label_i, label_j] = cvm(raw_pair, raw_pair_label)
					emb_cvm_mat[label_i, label_j] = cvm(emb_pair, emb_pair_label)
			
			## compute the label-trustworthiness and label-continuity score
			mat = raw_cvm_mat - emb_cvm_mat
			## convert NaN to 0
			mat[np.isnan(mat)] = 0
			lt_mat = raw_cvm_mat - emb_cvm_mat
			lt_mat[lt_mat < 0] = 0
			## convert NaN to 0
			lt_mat[np.isnan(lt_mat)] = 0
			lt = 1 - np.sum(lt_mat) / (label_num * (label_num - 1) / 2)

			lc_mat = emb_cvm_mat - raw_cvm_mat
			lc_mat[lc_mat < 0] = 0
			## convert NaN to 0
			lc_mat[np.isnan(lc_mat)] = 0
			lc = 1 - np.sum(lc_mat) / (label_num * (label_num - 1) / 2)

			return raw_cvm_mat, emb_cvm_mat, mat, 2 * lt * lc / (lt + lc)

		for dr_technique in tqdm(METADATA.keys()):

			if os.path.exists(f"./06_pairwise_test/results/{dataset}_{dr_technique}_{cvm_names[jj]}_score.json"):
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
						__, ___, _, score = run_cvm(raw, emb, label)
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

			print(params)
			## get final embedding
			emb = getattr(drp, runner_function_name)(raw, **params)

			## save synthetic embeddings
			with open(f"./06_pairwise_test/embeddings/{dataset}_{dr_technique}_{cvm_names[jj]}.json", "w") as f:
				json.dump(emb.tolist(), f)

			## compute cvm
			raw_matrix, emb_matrix, matrix, score = run_cvm(raw, emb, label)

			## save cvm matrix
			with open(f"./06_pairwise_test/results/{dataset}_{dr_technique}_{cvm_names[jj]}_matrix.json", "w") as f:
				json.dump({
					"raw": raw_matrix.tolist(),
					"emb": emb_matrix.tolist(),
					"mat": matrix.tolist()
				}, f)

			## save cvm score
			with open(f"./06_pairwise_test/results/{dataset}_{dr_technique}_{cvm_names[jj]}_score.json", "w") as f:
				json.dump([score], f)