
import os
import numpy as np
from tqdm import tqdm
import umato
import warnings
warnings.filterwarnings("ignore")

n_neighbors_range = (0.025, 0.25)
min_dist_range    = (0.001, 0.99)
hub_num_ratio_range = (0.025, 0.25)

embedding_num = 100

import signal


# def sig_handler(signum, frame):
# 	print("segfault")
    
# signal.signal(signal.SIGSEGV, sig_handler)


DATASETS = os.listdir("../datasets/npy/")

for dataset_i, dataset in enumerate(DATASETS):
	print(f"Dataset {dataset_i+1}/{len(DATASETS)}: {dataset}")
	data = np.load(f"../datasets/npy/{dataset}/data.npy")
	for i in tqdm(range(embedding_num)):
		try:
			if not os.path.exists(f"../embeddings/{dataset}/{i}.npy"):
				n_neighbors_ratio = np.random.uniform(n_neighbors_range[0], n_neighbors_range[1])
				n_neighbors = int(n_neighbors_ratio * data.shape[0])
				min_dist    = np.random.uniform(min_dist_range[0], min_dist_range[1])
				hub_num_ratio = np.random.uniform(hub_num_ratio_range[0], hub_num_ratio_range[1])
				hub_num = int(hub_num_ratio * data.shape[0])
				emb = umato.UMATO(n_neighbors=n_neighbors, min_dist=min_dist, hub_num=hub_num).fit_transform(data)
				if not os.path.exists(f"../embeddings/{dataset}/"):
					os.makedirs(f"../embeddings/{dataset}/")

				np.save(f"../embeddings/{dataset}/{i}_hp.npy", np.array([n_neighbors, min_dist, hub_num]))
				np.save(f"../embeddings/{dataset}/{i}.npy", emb)
		except:
			print("error")

		
