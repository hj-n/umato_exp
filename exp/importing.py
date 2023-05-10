import umap ## umap, densmap ## conda install umap
import pacmap ## pacmap ## pip install pacmap
import umato
import trimap ## trimap ## pip install trimap
from MulticoreTSNE import MulticoreTSNE as TSNE ## pip install MulticoreTSNE
from sklearn.decomposition import PCA ## pca
from sklearn.manifold import Isomap	## isomap ## conda install scikit-learn

import os
import numpy as np
from tqdm import tqdm

import timeit


def run_umap(X):
	reducer = umap.UMAP(n_neighbors=15, n_components=2)
	return reducer.fit_transform(X)

def run_pacmap(X):
	reducer = pacmap.PaCMAP(n_neighbors=15)
	return reducer.fit_transform(X)

def run_trimap(X):
	reducer = trimap.TRIMAP(n_inliers=15, n_outliers=15)
	return reducer.fit_transform(X)

def run_tsne(X):
	reducer = TSNE()
	return reducer.fit_transform(X)

def run_densmap(X):
	reducer = umap.UMAP(n_neighbors=15, n_components=2, densmap=True)
	return reducer.fit_transform(X)

def run_umato(X):
	hub_num = int(X.shape[0] * 0.1)
	reducer = umato.UMATO(n_neighbors=15, hub_num=hub_num)
	return reducer.fit_transform(X)

def run_pca(X):
	reducer = PCA(n_components=2)
	return reducer.fit_transform(X)

def run_isomap(X):
	reducer = Isomap(n_neighbors=15, n_components=2)
	return reducer.fit_transform(X)

