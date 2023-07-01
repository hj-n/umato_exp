import umap ## umap, densmap ## conda install umap
import pacmap ## pacmap ## pip install pacmap
import umato
import trimap ## trimap ## pip install trimap
from MulticoreTSNE import MulticoreTSNE as TSNE ## pip install MulticoreTSNE
from sklearn.decomposition import PCA ## pca
from sklearn.manifold import Isomap	## isomap ## conda install scikit-learn

from bayes_opt import BayesianOptimization

from zadu import zadu

def run(raw, dr_id, metric_id, metric_params):


	if metric_id == "trustworthiness" or metric_id == "continuity":
		spec_id = "tnc"
	
	if metric_id == "steadiness" or metric_id == "cohesiveness":
		spec_id = "snc"

	if metric_id == "neighbor_dissimilarity":
		spec_id = "nd"
	
	if metric_id == "kl_divergence":
		spec_id = "kl_div"
	
	if metric_id == "distance_to_measure":
		spec_id = "dtm"
	
	if metric_id == "stress":
		spec_id = "stress"
	
	# if metric_id == "topographic_product":
	# 	spec_id = "topo"
	
	if metric_id == "pearson_r":
		spec_id = "pr"
	
	if metric_id == "spearman_rho":
		spec_id = "srho"

	
	if metric_id in {"neighbor_dissimilarity", "kl_divergence", "distance_to_measuere"}:
		multiplier = -1
	else:
		multiplier = 1


	spec = [{
		"id" : spec_id,
		"params" : metric_params
	}]

	if dr_id == "umap":
		bound = {"n_neighbors": (2, 100), "min_dist": (0.01, 0.99)}
		def f(n_neighbors, min_dist):
			reducer = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist))
			emb = reducer.fit_transform(raw)
			return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier
	
	if dr_id == "pacmap":
		bound = {"n_neighbors": (2, 100), "MN_ratio": (0.1, 5), "FP_ratio": (0.1, 5)}

		def f(n_neighbors, MN_ratio, FP_ratio):
			if n_neighbors * MN_ratio < 1:
				MN_ratio = 1 / n_neighbors
			if n_neighbors * FP_ratio < 1:
				FP_ratio = 1 / n_neighbors

			reducer = pacmap.PaCMAP(n_neighbors=int(n_neighbors), MN_ratio=float(MN_ratio), FP_ratio=float(FP_ratio))
			emb = reducer.fit_transform(raw)
			return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier
		
	if dr_id == "trimap":
		bound = {"n_inliers": (2, 100), "n_outliers": (2, 100), "n_random": (2, 100)}

		def f(n_inliers, n_outliers, n_random):
			reducer = trimap.TRIMAP(n_inliers=int(n_inliers), n_outliers=int(n_outliers), n_random=int(n_random))
			emb = reducer.fit_transform(raw)
			return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier
	
	if dr_id == "tsne":
		bound = {"perplexity": (2, 500)}

		def f(perplexity):
			reducer = TSNE(perplexity=int(perplexity))
			emb = reducer.fit_transform(raw)
			return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier
	
	if dr_id == "densmap":
		bound = {"n_neighbors": (2, 100), "min_dist": (0.01, 0.99)}

		def f(n_neighbors, min_dist):
			reducer = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist), densmap=True)
			emb = reducer.fit_transform(raw)
			return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier
		
	if dr_id == "umato":
		bound = {"n_neighbors": (2, 100), "min_dist": (0.01, 0.99), "hub_num": (20, raw.shape[0] - 1 if raw.shape[0] - 1 < 400 else 400)}

		def f(n_neighbors, min_dist, hub_num):
			reducer = umato.UMATO(n_neighbors=int(n_neighbors), min_dist=float(min_dist), hub_num=int(hub_num))
			emb = reducer.fit_transform(raw)
			return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier
	
	# if dr_id == "pca":

	# 	bound = {}

	# 	def f():
	# 		reducer = PCA()
	# 		emb = reducer.fit_transform(raw)
	# 		return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier
	
	if dr_id == "isomap":
		bound = {"n_neighbors": (2, 100)}

		def f(n_neighbors):
			reducer = Isomap(n_neighbors=int(n_neighbors))
			emb = reducer.fit_transform(raw)
			return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier

	if dr_id != "pca":
		optimizer = BayesianOptimization(
			f=f,
			pbounds=bound,
			random_state=1,
			verbose=0
		)

		optimizer.maximize(
			init_points=5,
			n_iter=10,
		)	
		return optimizer.max["target"] * multiplier
	else:
		emb = PCA(n_components=2).fit_transform(raw)
		return zadu.ZADU(spec, raw).measure(emb)[0][metric_id] * multiplier

	
