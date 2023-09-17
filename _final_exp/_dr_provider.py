import umap
import pacmap
import umato
import trimap
from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding


from sklearn.datasets import load_iris
import _lamp as lamp
import _lmds as lmds
import numpy as np

from scipy.spatial.distance import cdist


def run_umap(X, n_neighbors, min_dist, init="spectral"):
	n_neighbors = int(n_neighbors)
	min_dist = float(min_dist)
	reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, init=init)
	return reducer.fit_transform(X)

def run_pacmap(X, n_neighbors, MN_ratio, FP_ratio, init="random"):
	n_neighbors = int(n_neighbors)
	MN_ratio = float(MN_ratio)
	FP_ratio = float(FP_ratio)

	if n_neighbors * MN_ratio < 1:
		MN_ratio = 1 / n_neighbors
	if n_neighbors * FP_ratio < 1:
		FP_ratio = 1 / n_neighbors

	reducer = pacmap.PaCMAP(n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)
	return reducer.fit_transform(X, init=init)


def run_trimap(X, n_inliers, n_outliers, init=None):
	n_inliers = int(n_inliers)
	n_outliers = int(n_outliers)
	reducer = trimap.TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers)
	return reducer.fit_transform(X, init=init)

def run_tsne(X, perplexity, init="pca"):
	perplexity = float(perplexity)
	reducer = TSNE(perplexity=perplexity, init=init)
	return reducer.fit_transform(X)

def run_umato(X, n_neighbors, min_dist, hub_num, init="pca"):
	n_neighbors = int(n_neighbors)
	min_dist = float(min_dist)
	hub_num = int(hub_num)

	reducer = umato.UMATO(n_neighbors=n_neighbors, min_dist=min_dist, hub_num=hub_num, init="pca")
	return reducer.fit_transform(X)

def run_pca(X):
	reducer = PCA(n_components=2)
	return reducer.fit_transform(X)

def run_isomap(X, n_neighbors):
	n_neighbors = int(n_neighbors)
	reducer = Isomap(n_neighbors=n_neighbors, n_components=2)
	return reducer.fit_transform(X)


def run_lle(X, n_neighbors):
	n_neighbors = int(n_neighbors)
	reducer = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2)
	return reducer.fit_transform(X)


def run_lamp(X):
	reducer = lamp.Lamp(Xdata = X)
	return reducer.fit()

def run_lmds(X, hub_num):
	hub_num = int(hub_num)

	emb = []
	while len(emb) == 0:
		hub_num = np.random.randint(20, X.shape[0]-2)
		hubs = np.random.choice(X.shape[0], hub_num, replace=False)
		DI = cdist(X[hubs, :], X, "euclidean")
		emb = lmds.landmark_MDS(DI, hubs, 2)
	return emb

