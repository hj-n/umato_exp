import matplotlib.pyplot as plt
import numpy as np
import json


dataset = ["cnae9", "flickr_material_database", "fashion_mnist"]


techniques = ["umato", "umap", "tsne", "pca"]


## plot embeddings


fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for technique_name in techniques:
	dataset_name = "fashion_mnist"
	emb = np.load(f"./07_pairwise_global/embeddings/{dataset_name}_{technique_name}.npy")
	label = np.load(f"../datasets/npy/{dataset_name}/label.npy")
	axs[techniques.index(technique_name)].scatter(emb[:, 0], emb[:, 1], c=label, cmap="tab10")
	axs[techniques.index(technique_name)].set_title(f"{dataset_name}_{technique_name}")
	## remove tick
	axs[techniques.index(technique_name)].set_xticks([])
	axs[techniques.index(technique_name)].set_yticks([])

	## remove border
	axs[techniques.index(technique_name)].spines['top'].set_visible(False)
	axs[techniques.index(technique_name)].spines['right'].set_visible(False)
	axs[techniques.index(technique_name)].spines['bottom'].set_visible(False)
	axs[techniques.index(technique_name)].spines['left'].set_visible(False)

plt.tight_layout()


plt.savefig(f"./07_pairwise_global/plot/embeddings.png")
## svg save
plt.savefig(f"./07_pairwise_global/plot/embeddings.svg")

plt.clf()

## plot kl divergence as heatmap
fig, axs = plt.subplots(len(dataset), len(techniques), figsize=(16, 10))

for dataset_name in dataset:
	range_matrix = [0, 0]
	for technique_name in techniques:
		with open(f"./07_pairwise_global/results/{dataset_name}_{technique_name}.json", "r") as f:
			data = json.load(f)
		
		matrix = np.array(data["pairwise_matrix"])
		range_matrix[0] = min(range_matrix[0], np.min(matrix))
		range_matrix[1] = max(range_matrix[1], np.max(matrix))


	for technique_name in techniques:
		with open(f"./07_pairwise_global/results/{dataset_name}_{technique_name}.json", "r") as f:
			data = json.load(f)
		
		matrix = np.array(data["pairwise_matrix"])

		newmatrix = np.zeros((matrix.shape[0], matrix.shape[0]))
		for i in range(matrix.shape[0]):
			## insert i, j value as 0 (currently have no)
			currindex = 0
			for j in range(len(newmatrix)):
				if j == i:
					continue
				newmatrix[i][j] = matrix[i][currindex]
				currindex += 1
			
		matrix = newmatrix.tolist()
		score = round(data["kl_divergence"], 3)
		axs[dataset.index(dataset_name), techniques.index(technique_name)].imshow(matrix, cmap="Blues", vmin=range_matrix[0], vmax=range_matrix[1], aspect=100, extent=[0,100,0,1])
		axs[dataset.index(dataset_name), techniques.index(technique_name)].set_title(f"{dataset_name}_{technique_name}_{score}")
		## remove tick
		axs[dataset.index(dataset_name), techniques.index(technique_name)].set_xticks([])
		axs[dataset.index(dataset_name), techniques.index(technique_name)].set_yticks([])
		## remove border
		axs[dataset.index(dataset_name), techniques.index(technique_name)].spines['top'].set_visible(False)
		axs[dataset.index(dataset_name), techniques.index(technique_name)].spines['right'].set_visible(False)
		axs[dataset.index(dataset_name), techniques.index(technique_name)].spines['bottom'].set_visible(False)
		axs[dataset.index(dataset_name), techniques.index(technique_name)].spines['left'].set_visible(False)
		if techniques.index(technique_name) == len(techniques) - 1:
			cbar = axs[dataset.index(dataset_name), techniques.index(technique_name)].figure.colorbar(axs[dataset.index(dataset_name), techniques.index(technique_name)].imshow(matrix, cmap="Blues", vmin=range_matrix[0], vmax=range_matrix[1]), ax=axs[dataset.index(dataset_name), techniques.index(technique_name)])


plt.tight_layout()
plt.savefig(f"./07_pairwise_global/plot/kl.png")
## svg save
plt.savefig(f"./07_pairwise_global/plot/kl.svg")