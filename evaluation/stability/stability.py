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
import pickle

warnings.simplefilter('ignore')

THRESHOLD_DATASET_SIZE = 5000
embeddings_dump_dir = 'embeddings'

if __name__ == "__main__":
    datasets = [dataset for dataset in load_all_datasets() if len(dataset) >= THRESHOLD_DATASET_SIZE]
    sample_sizes = [1, 2, 5, 10, 20, 30, 50, 60, 80, 100]
    algorithms = [('UMATO_N//30(15)', -1, lambda data, superset_size, *args: umato.UMATO(
                      hub_num=superset_size // 30,
                      n_neighbors=15).fit_transform(data)),
                  ('UMATO-75(50)', 75, lambda data, *args: umato.UMATO(hub_num=75,
                                                                       n_neighbors=50).fit_transform(data)),
                  ('UMATO-75(15)', 75, lambda data, *args: umato.UMATO(hub_num=75,
                                                                       n_neighbors=15).fit_transform(data)),
                  ('UMAP', 0, lambda data, *args: umap.UMAP(n_neighbors=15).fit_transform(data)),
                  ('t-SNE', 0, lambda data, *args: MulticoreTSNE.MulticoreTSNE().fit_transform(data)),
                  ('PaCMAP', 0, lambda data, *args: pacmap.PaCMAP(n_neighbors=15).fit_transform(data)),
                  ('Isomap', 0, lambda data, *args: sklearn.manifold.Isomap(n_neighbors=15).fit_transform(data)),
                  ('TriMap', 0, lambda data, *args: trimap.TRIMAP().fit_transform(data)),
                  ('DensMAP', 0, lambda data, *args: umap.UMAP(densmap=True).fit_transform(data))
                  ]

    bench_data = {}
    for i, dataset in enumerate(datasets):
        bench_data[dataset.name] = {}

        print(f"({i + 1}/{len(datasets)}) {dataset}")

        dataset_embeddings = {}
        for algorithm in algorithms:
            alg_name, limit, alg_func = algorithm
            assert len(dataset) >= limit
            dataset_embeddings[alg_name] = alg_func(dataset.data, len(dataset))
        with open(f"{embeddings_dump_dir}/{dataset.name}.embeddings", "wb") as f:
            pickle.dump(dataset_embeddings, f)

        subset_embeddings = {}
        for sample_size in sample_sizes:
            bench_data[dataset.name][sample_size] = {}
            subset_embeddings[sample_size] = {}
            subset_size = len(dataset) * sample_size // 100
            print(f"\tSampling {sample_size}% ({subset_size})")
            subset_indexes = np.random.choice(np.arange(len(dataset)), size=subset_size, replace=False)
            subset_embeddings[sample_size]['index'] = subset_indexes
            subset_data = dataset.data[subset_indexes]
            for algorithm in algorithms:
                alg_name, limit, alg_func = algorithm
                if limit == -1:
                    limit = len(dataset) // 30
                if subset_size < limit:
                    continue

                subset_embeddings[sample_size][alg_name] = alg_func(subset_data, len(dataset))

                _, _, disparity = procrustes(subset_embeddings[sample_size][alg_name],
                                             dataset_embeddings[alg_name][subset_indexes])

                bench_data[dataset.name][sample_size][alg_name] = disparity

                print(f"\t\t{alg_name:15}: {disparity:.4f}")
        with open(f"{embeddings_dump_dir}/{dataset.name}.subsets.embeddings", "wb") as f:
            pickle.dump(subset_embeddings, f)

        with open('stability.json', 'w') as f:
            json.dump(bench_data, f, indent=4)
