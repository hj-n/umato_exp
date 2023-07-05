import matplotlib.pyplot as plt

import seaborn as sns

import os
import pandas as pd
import numpy as np

from scipy import stats

FILES = os.listdir("./01_accuracy/cherry/")
METRICS = ["trustworthiness", "continuity", "mrre_false", "mrre_missing", "kl_divergence", "distance_to_measure", "stress"]

# METRICS = ["kl_divergence_sigma_1", "kl_divergence_sigma_0.1", "kl_divergence_sigma_0.01"]

# METRICS = ["steadiness", "cohesiveness"]

DR_TECHNIQUE = ["pca", "lle", "umap", "trimap", "pacmap", "tsne", "umato", "lamp", "lmds"]

# METRICS = ["distance_to_measure_sigma_1", "distance_to_measure_sigma_0.1", "distance_to_measure_sigma_0.01"]


METRICS = ["trustworthiness_k_10", "trustworthiness_k_20", "trustworthiness_k_30", "trustworthiness_k_40", "trustworthiness_k_50"]
METRICS = ["continuity_k_10", "continuity_k_20", "continuity_k_30", "continuity_k_40", "continuity_k_50"]

METRICS = ["mrre_false_k_10", "mrre_false_k_20", "mrre_false_k_30", "mrre_false_k_40", "mrre_false_k_50"]
# METRICS = ["mrre_missing_k_10", "mrre_missing_k_20", "mrre_missing_k_30", "mrre_missing_k_40", "mrre_missing_k_50"]

METRICS = ["trustworthiness_k_10",  "trustworthiness_k_50", "continuity_k_10", "continuity_k_50", 
	   				"mrre_false_k_10",  "mrre_false_k_50", "mrre_missing_k_10",  "mrre_missing_k_50",
						"steadiness", "cohesiveness", "kl_divergence_sigma_1", "kl_divergence_sigma_0.1", "distance_to_measure_sigma_1", "distance_to_measure_sigma_0.1", "stress"]

METRIC_DIRECTION = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]




dr_techniuqe_full = []
metric_full = []
value_full = []
dataset_full = []
for file in FILES:
	result = pd.read_csv("./01_accuracy/results/" + file)
	dr_technique_list = result["dr_technique"].to_numpy().tolist()
	metric_list = result["metric"].to_numpy().tolist()
	value_list = result["value"].to_numpy().tolist()
  
	dr_techniuqe_full += dr_technique_list
	metric_full += metric_list
	value_full += value_list
	dataset_full += [file.split(".")[0]] * len(dr_technique_list)
  
  
df = pd.DataFrame({
	"dr_technique": dr_techniuqe_full,
	"metric": metric_full,
	"value": value_full,
	"dataset": dataset_full
})

dr_technique_unique = DR_TECHNIQUE


for i, metric_prefix in enumerate(METRICS):
	## check if metric_prefix is in metric_full
	filtering = []
	for metric in metric_full:
		if metric.startswith(metric_prefix):
			filtering.append(True)
		else:
			filtering.append(False)
	
	metric_df = df[filtering]

	print(metric_prefix)
	mean_val_list = []
	for dr_technique in dr_technique_unique:
		filtering = metric_df["dr_technique"] == dr_technique
		metric_technique_df = metric_df[filtering]




		mean_val = metric_technique_df["value"].mean()
		std_val = metric_technique_df["value"].std()

		mean_val_list.append(mean_val)

		print(dr_technique, mean_val, std_val)
	
	## compute ranking of techniques
	ranking = np.argsort(mean_val_list)
	technique_ranking = np.array(dr_technique_unique)[ranking]
	if METRIC_DIRECTION[i] == -1:
		technique_ranking = technique_ranking[::-1]
	print(technique_ranking)
	print("-------")




