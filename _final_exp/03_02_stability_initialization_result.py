

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc, rcParams

result_files = os.listdir("./03_02_stability_init/results/")


dataset_list = []
dr_technique_list = []
disparity_list = []

# result_files =
for result_file in result_files:
	result_data = pd.read_csv(f"./03_02_stability_init/results/{result_file}")

	dataset = result_file.split(".")[0]
	dataset_list += [dataset]*len(result_data)
	dr_technique_list += list(result_data["dr_technique"])
	disparity_list += list(result_data["disparity"])

	


final_result = {
	"dataset": dataset_list,
	"dr_technique": dr_technique_list,
	"disparity": disparity_list
}


final_result = pd.DataFrame(final_result)



## pointplot
plt.figure(figsize=(6.5, 2.4))
sns.set(style="whitegrid")
sns.pointplot(
	y="dr_technique", x="disparity", 
	data=final_result, join=False, hue="dr_technique",
	order=["umato", "tsne", "pacmap", "umap"]
)
y_ticks = [
	"$\mathbf{UMATO}$", "$t$-SNE", "PaCMAP", "UMAP"
]
plt.yticks(np.arange(len(y_ticks)), y_ticks)
plt.xscale("log")
plt.legend([],[], frameon=False)


plt.xlabel("Disparity")
plt.ylabel("")

plt.tight_layout()



plt.savefig("./03_02_stability_init/plots/stability_init.png", dpi=300)
plt.savefig("./03_02_stability_init/plots/stability_init.pdf", dpi=300)

plt.clf()

