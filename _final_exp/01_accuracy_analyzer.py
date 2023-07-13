import pandas as pd

import numpy as np


import os


FILES = os.listdir("./01_accuracy/results/")

METRICS = ["trustworthiness", "continuity", "kl_divergence", "distance_to_measure", "stress"]

METRICS = ["kl_divergence_sigma_1", "kl_divergence_sigma_0.1", "kl_divergence_sigma_0.01", "stress", "trustworthiness", "continuity",]


for file in FILES:
	result = pd.read_csv("./01_accuracy/results/" + file)
	dr_technique_list = result["dr_technique"].to_numpy().tolist()

	unique_dr_technique_list = list(set(dr_technique_list))

	for metric in METRICS:
		filtering = []
		for metric_name in result["metric"].to_numpy().tolist():
			if metric_name.startswith(metric):
				filtering.append(True)
			else:
				filtering.append(False)
		
		metric_df = result[filtering]

		average_score = []
		for dr_technique in unique_dr_technique_list:
			filtering = metric_df["dr_technique"] == dr_technique
			metric_technique_df = metric_df[filtering]

			mean_val = metric_technique_df["value"].mean()
			std_val = metric_technique_df["value"].std()

			average_score.append(mean_val)

		## compute ranking of techniques
		ranking = np.argsort(average_score)
		technique_ranking = np.array(unique_dr_technique_list)[ranking]

		umato_index = np.where(technique_ranking == "umato")[0][0]

		print(file, metric, "umato ranking is ", umato_index + 1, "out of", len(technique_ranking))
	