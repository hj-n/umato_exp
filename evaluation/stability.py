from scipy.spatial import procrustes
import numpy as np
from datasets.loader import load_all_datasets
import umato
import umap
import MulticoreTSNE
import pacmap
import sklearn
import trimap
import json
import warnings

warnings.simplefilter('ignore')

THRESHOLD_DATASET_SIZE = 5000

if __name__ == "__main__":
    datasets = [dataset for dataset in load_all_datasets() if len(dataset) >= THRESHOLD_DATASET_SIZE]
    sample_sizes = [1, 2, 5, 10, 20, 30, 50, 60, 80, 100]
    algorithms = [('UMATO//10', 20, lambda data: umato.UMATO(hub_num=len(data) // 10,
                                                             n_neighbors=15).fit_transform(data)),
                  ('UMATO//20', 40, lambda data: umato.UMATO(hub_num=len(data) // 20,
                                                             n_neighbors=15).fit_transform(data)),
                  ('UMATO//40', 80, lambda data: umato.UMATO(hub_num=len(data) // 40,
                                                             n_neighbors=15).fit_transform(data)),
                  ('UMATO//80', 160, lambda data: umato.UMATO(hub_num=len(data) // 80,
                                                              n_neighbors=15).fit_transform(data)),
                  ('UMATO//160', 320, lambda data: umato.UMATO(hub_num=len(data) // 160,
                                                               n_neighbors=15).fit_transform(data)),
                  ('UMAP', 0, lambda data: umap.UMAP(n_neighbors=15).fit_transform(data)),
                  ('t-SNE', 0, lambda data: MulticoreTSNE.MulticoreTSNE().fit_transform(data)),
                  ('PacMAP', 0, lambda data: pacmap.PaCMAP().fit_transform(data)),
                  ('PCA', 0, lambda data: sklearn.decomposition.PCA().fit_transform(data)),
                  ('Isomap', 0, lambda data: sklearn.manifold.Isomap().fit_transform(data)),
                  ('TriMap', 0, lambda data: trimap.TRIMAP().fit_transform(data)),
                  ('DensMAP', 0, lambda data: umap.UMAP(densmap=True).fit_transform(data))
                  ]

    bench_data = {}
    for i, dataset in enumerate(datasets[:1]):
        bench_data[dataset.name] = {}

        print(f"({i + 1}/{len(datasets)}) {dataset}")

        dataset_embeddings = {}
        for algorithm in algorithms:
            alg_name, limit, alg_func = algorithm
            assert len(dataset) >= limit
            dataset_embeddings[alg_name] = alg_func(dataset.data)

        for sample_size in sample_sizes:
            bench_data[dataset.name][sample_size] = {}
            sample_size_bench = []
            subset_size = len(dataset) * sample_size // 100
            print(f"\tSampling {sample_size}% ({subset_size})")
            subset_indexes = np.random.choice(np.arange(len(dataset)), size=subset_size, replace=False)
            subset_data = dataset.data[subset_indexes]
            for algorithm in algorithms:
                alg_name, limit, alg_func = algorithm
                if subset_size < limit:
                    continue

                subset_embeddings = alg_func(subset_data)

                _, _, disparity = procrustes(subset_embeddings, dataset_embeddings[alg_name][subset_indexes])

                sample_size_bench.append({alg_name: disparity})
                bench_data[dataset.name][sample_size][alg_name] = disparity

                print(f"\t\t{alg_name:15}: {disparity:.4f}")

    with open('stability.json', 'w') as f:
        json.dump(bench_data, f, indent=4)
