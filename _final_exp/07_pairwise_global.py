import numpy as np 

import _dr_provider as drp

import os, json
from scipy.spatial.distance import cdist

import pandas as pd
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")


def distance_matrix_to_density(distance_matrix, sigma):
	"""
	Compute the density of each point based on the pairwise distance matrix
	INPUT:
		ndarray: distance_matrix: pairwise distance matrix
		float: sigma: sigma parameter for the Gaussian kernel
	OUTPUT:
		ndarry: density
	"""

	normalized_distance_matrix = distance_matrix / np.max(distance_matrix)
	density = np.sum(np.exp(- (normalized_distance_matrix ** 2) / sigma), axis=-1)
	density = density / np.sum(density)
	return density


def pairwise_distance_matrix(point, distance_function="euclidean"):
	"""
	Compute the pairwise distance matrix of the point list
	You can use any distance function from scipy.spatial.distance.cdist or specify a callable function
	INPUT:
		ndarray: point: list of points
		str or callable: distance_function: distance function to use
	OUTPUT:
		ndarry: pairwise distance matrix 
	"""
	if callable(distance_function):
		distance_matrix = cdist(point, point, distance_function)
	elif distance_function == "snn":
		## TODO
		pass
	else:
		distance_matrix = cdist(point, point, distance_function)
	return distance_matrix


def kl_div(orig, emb, sigma=0.1, distance_matrices=None):
  """
  Compute Kullback-Leibler divergence of the embedding
  INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
  OUTPUT:
		dict: Kullback-Leibler divergence (kl)
	"""
  
  if distance_matrices is None:
    orig_distance_matrix = pairwise_distance_matrix(orig)
    emb_distance_matrix = pairwise_distance_matrix(emb)
  else:
    orig_distance_matrix, emb_distance_matrix = distance_matrices

  density_orig = distance_matrix_to_density(orig_distance_matrix, sigma)
  density_emb = distance_matrix_to_density(emb_distance_matrix, sigma)
  
  matrix = density_orig * np.log(density_orig / density_emb)
  kl = np.sum(matrix)
  return {
    "kl_divergence": kl,
		"bypoint": matrix
	}



dataset = "dry_bean"

METADATA = json.load(open("./_metadata_umato.json", "r"))

raw = np.load(f"../datasets/npy/{dataset}/data.npy")[::5]
label = np.load(f"../datasets/npy/{dataset}/label.npy")[::5]
size = raw.shape[0]


for dr_technique in tqdm(METADATA.keys()):

	# if os.path.exists(f"./07_pairwise_global/embeddings/{dataset}_{dr_technique}.png"):
	# 	continue


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
				# print("Generating embedding:", end - start)
				start = time.time()
				measured = kl_div(raw, emb)
				score = - measured["kl_divergence"]
				end = time.time()
				print(score)
				# print("Computing score:", end - start)
			except:
				score = 0

			# print("Score:", score)
			return score
		
		optimizer = BayesianOptimization(f=f, pbounds=bound, verbose=0, allow_duplicate_points=True)
		optimizer.maximize(init_points=5, n_iter=10)
		params = optimizer.max["params"]
	else:
		params = {}


	## phase 2: run  again with params
	emb = getattr(drp, runner_function_name)(raw, **params)

	kl_div_result = kl_div(raw, emb)
	print(kl_div_result["kl_divergence"])

	sum_result = 0
	pairwise_matrix = []

	unique_labels = np.unique(label)
	for li in unique_labels:
		pairwise_row = []
		for lj in unique_labels:
			if li == lj:
				continue
			filterer_li = label == li
			filterer_lj = label == lj

			emb_filtered = emb[np.logical_or(filterer_li, filterer_lj)]
			raw_filtered = raw[np.logical_or(filterer_li, filterer_lj)]

			kl_div_result = kl_div(raw_filtered, emb_filtered)
			kl_div_result["kl_divergence"] = round(kl_div_result["kl_divergence"], 3)
			# print(kl_div_result["kl_divergence"], end=" ")
			sum_result += kl_div_result["kl_divergence"]
			pairwise_row.append(kl_div_result["kl_divergence"])
		
		pairwise_matrix.append(pairwise_row)


		# print()

	with open(f"./07_pairwise_global/results/{dataset}_{dr_technique}.json", "w") as f:
		json.dump({
			"kl_divergence": sum_result,
			"pairwise_matrix": pairwise_matrix
		}, f)
	
	np.save(f"./07_pairwise_global/embeddings/{dataset}_{dr_technique}.npy", emb)
	