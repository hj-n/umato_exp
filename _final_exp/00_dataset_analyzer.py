import numpy as np

import os

from sklearn.decomposition import PCA
import pandas as pd

from tabulate import tabulate

FILES = os.listdir("../datasets_candidate/npy")


dataset_list = []
size_list = []
size_class_list = []
dim_list = []
dim_class_list = []
intrinsic_dim_list = []
intrinsic_dim_class_list = []
sparsity_ratio_list = []
sparsity_ratio_class_list = []



for file in FILES:
	dataset = np.load("../datasets_candidate/npy/" + file + "/data.npy")


	### size trait
	size = dataset.shape[0]

	if size <= 1000:
		size_class = "small"
	elif size <= 3000:
		size_class = "medium"
	else:
		size_class = "large"
	
	## dimensionality trait
	dim = dataset.shape[1]

	if dim < 100:
		dim_class = "low"
	elif dim < 500:
		dim_class = "medium"
	else:
		dim_class = "high"

	## intrinsic dimensionality trait (percentage of principal components need to exaplin 95% of variance)
	pca_result = PCA().fit(dataset)
	intrinsic_dim = np.where(np.cumsum(pca_result.explained_variance_ratio_) >= 0.95)[0][0] + 1
	intrinsic_dim = intrinsic_dim / dim

	if intrinsic_dim <= 0.1:
		intrinsic_dim_class = "low"
	elif intrinsic_dim <= 0.5:
		intrinsic_dim_class = "medium"
	else:
		intrinsic_dim_class = "high"
	

	## sparsity ratio (ratio of zero values in the dataset
	sparsity_ratio = 1 - np.count_nonzero(dataset) / (dataset.shape[0] * dataset.shape[1])

	if sparsity_ratio <= 0.2:
		sparsity_ratio_class = "dense"
	elif sparsity_ratio <= 0.8:
		sparsity_ratio_class = "medium"
	else:
		sparsity_ratio_class = "sparse"

	


	dataset_list.append(file)
	size_list.append(size)
	size_class_list.append(size_class)
	dim_list.append(dim)
	dim_class_list.append(dim_class)
	intrinsic_dim_list.append(intrinsic_dim)
	intrinsic_dim_class_list.append(intrinsic_dim_class)
	sparsity_ratio_list.append(sparsity_ratio)
	sparsity_ratio_class_list.append(sparsity_ratio_class)




	
df = pd.DataFrame({
	"dataset": dataset_list,
	"size": size_list,
	"size_class": size_class_list,
	"dim": dim_list,
	"dim_class": dim_class_list,
	"intrinsic_dim": intrinsic_dim_list,
	"intrinsic_dim_class": intrinsic_dim_class_list,
	"sparsity_ratio": sparsity_ratio_list,
	"sparsity_ratio_class": sparsity_ratio_class_list

})

## print df pretty
# print(tabulate(df, headers='keys', tablefmt='psql'))
print(df.to_latex(index=False))