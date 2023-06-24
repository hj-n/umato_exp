import numpy as np
import pandas as pd
import os

DATASET_LIST = os.listdir("../embeddings/")

DATASET_LIST.remove(".gitignore")


size_arr = []
trustworthiness_arr = []
continuity_arr = []
mrre_false_arr = []
mrre_missing_arr = []
steadiness_arr = []
cohesiveness_arr = []
kl_div_arr = []
pearson_r_arr = []
spearman_rho_arr = []
stress_arr = []

n_neighbors_arr = []
min_dist_arr = []
hub_num_arr = []

relative_n_neighbors_arr = []
relative_hub_num_arr = []

for dataset in DATASET_LIST:
	files = os.listdir(f"../embeddings/{dataset}/")
	scores = [file for file in files if file.endswith("_score.npy")]
	embeddings = [f"{file[:-10]}.npy" for file in scores]
	hps = [f"{file[:-10]}_hp.npy" for file in scores]

	for idx in range(len(scores)):
		score = np.load(f"../embeddings/{dataset}/{scores[idx]}", allow_pickle=True)
		embedding = np.load(f"../embeddings/{dataset}/{embeddings[idx]}")
		hp = np.load(f"../embeddings/{dataset}/{hps[idx]}")

		size = embedding.shape[0]
		trustworthiness = score[0]["trustworthiness"]
		continuity = score[0]["continuity"]
		mrre_false = score[2]["mrre_false"]
		mrre_missing = score[2]["mrre_missing"]
		steadiness = score[4]["steadiness"]
		cohesiveness = score[4]["cohesiveness"]
		kl_div = score[5]["kl_divergence"]
		pearson_r = score[7]["pearson_r"]
		spearman_rho = score[8]["spearman_rho"]
		stress = score[9]["stress"]

		n_neighbors = hp[0]
		min_dist = hp[1]
		hub_num = hp[2]

		relative_n_neighbors = n_neighbors / size
		relative_hub_num = hub_num / size

		size_arr.append(size)
		trustworthiness_arr.append(trustworthiness)
		continuity_arr.append(continuity)
		mrre_false_arr.append(mrre_false)
		mrre_missing_arr.append(mrre_missing)
		steadiness_arr.append(steadiness)
		cohesiveness_arr.append(cohesiveness)
		kl_div_arr.append(kl_div)
		pearson_r_arr.append(pearson_r)
		spearman_rho_arr.append(spearman_rho)
		stress_arr.append(stress)

		n_neighbors_arr.append(n_neighbors)
		min_dist_arr.append(min_dist)
		hub_num_arr.append(hub_num)

		relative_n_neighbors_arr.append(relative_n_neighbors)
		relative_hub_num_arr.append(relative_hub_num)


df = pd.DataFrame({
	"size": size_arr,
	"trustworthiness": trustworthiness_arr,
	"continuity": continuity_arr,
	"mrre_false": mrre_false_arr,
	"mrre_missing": mrre_missing_arr,
	"steadiness": steadiness_arr,
	"cohesiveness": cohesiveness_arr,
	"kl_div": kl_div_arr,
	"pearson_r": pearson_r_arr,
	"spearman_rho": spearman_rho_arr,
	"stress": stress_arr,
	"n_neighbors": n_neighbors_arr,
	"min_dist": min_dist_arr,
	"hub_num": hub_num_arr,
	"relative_n_neighbors": relative_n_neighbors_arr,
	"relative_hub_num": relative_hub_num_arr
})

df.to_csv("./aggregate_hp_result.csv", index=False)



