import numpy as np
import pandas as pd
import json
import os

import seaborn as sns
import matplotlib.pyplot as plt

DATASETS = os.listdir("../datasets_candidate/npy/")


DR_TECHNIQUES = ["umato", "umap", "tsne", "pca", "lle", "pacmap", "trimap", "lmds"]


size_list = []
dr_list = []
time_list = []

for dataset in DATASETS:
	X = np.load(f"../datasets_candidate/npy/{dataset}/data.npy")
	size = X.shape[0]

	performance = json.load(open(f"./02_scalability/results/{dataset}.json", "r"))

	for dr_technique in DR_TECHNIQUES:
		size_list.append(size)
		dr_list.append(dr_technique)
		time_list.append(performance[dr_technique])



df = pd.DataFrame({"size": size_list, "dr": dr_list, "time": time_list})


sns.set_theme(style="whitegrid")

## set size
plt.figure(figsize=(6, 2.4))

sns.pointplot(data=df, y="dr", x="time", hue="dr", join=False)

## remove legend
plt.legend([],[], frameon=False)

plt.xlabel("Time (s)")
plt.ylabel("")

## set ticks
yticks = ["UMATO", "UMAP", "t-SNE", "PCA", "LLE", "PaCMAP", "Trimap", "L-MDS"]
plt.yticks(range(len(yticks)), yticks)

plt.tight_layout()

plt.savefig("./02_scalability/plot/scalability.png", dpi=300)
plt.savefig("./02_scalability/plot/scalability.pdf", dpi=300)


	

