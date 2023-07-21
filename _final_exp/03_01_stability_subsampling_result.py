

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc, rcParams

result_files = os.listdir("./03_01_stability_subsampling/results/")


dataset_list = []
dr_technique_list = []
sampling_list = []
disparity_list = []

# result_files = ["epileptic_seizure_recognition.csv"]

result_files.remove("sentiment_labeld_sentences.csv")
# result_files = ["optical_recognition_of_handwritten_digits.csv", "paris_housing_classification.csv", "predicting_pulsar_star.csv"]

for result_file in result_files:
	result_data = pd.read_csv(f"./03_01_stability_subsampling/results/{result_file}")

	dataset = result_file.split(".")[0]
	dataset_list += [dataset]*len(result_data)
	dr_technique_list += list(result_data["dr_technique"])
	sampling_list += list(result_data["sampling"])
	disparity_list += list(result_data["disparity"])

	


final_result = {
	"dataset": dataset_list,
	"dr_technique": dr_technique_list,
	"sampling": sampling_list,
	"disparity": disparity_list
}


final_result = pd.DataFrame(final_result)



## pointplot
plt.figure(figsize=(6.5, 2.7))
sns.set(style="whitegrid")
sns.pointplot(
	y="dr_technique", x="disparity", 
	data=final_result, join=False, hue="dr_technique",
	order=["pca", "lmds", "umato", "lle", "tsne", "pacmap","trimap", "umap", "lamp"]
)
y_ticks = [
	"PCA", "L-MDS", "$\mathbf{UMATO}$", "LLE", "$t$-SNE", "PaCMAP", "TriMap", "UMAP", "LAMP"
]
plt.yticks(np.arange(len(y_ticks)), y_ticks)
plt.xscale("log")
plt.legend([],[], frameon=False)


plt.xlabel("Disparity")
plt.ylabel("")

plt.tight_layout()



plt.savefig("./03_01_stability_subsampling/plots/stability_subsampling.png", dpi=300)
plt.savefig("./03_01_stability_subsampling/plots/stability_subsampling.pdf", dpi=300)

plt.clf()

