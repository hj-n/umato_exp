from scipy.spatial import procrustes
import numpy as np
from datasets.loader import load_all_datasets
import umato
import umap
import MulticoreTSNE

if __name__ == "__main__":
    datasets = load_all_datasets()
    sample_sizes = [1, 2, 5, 10, 20, 30, 50, 60, 80, 100]
    algorithms = [('umato', lambda data: umato.UMATO(hub_num=min(300, len(data) - 1),
                                                     n_neighbors=min(15, len(data) - 1)).fit_transform(data)),
                  ('umap', lambda data: umap.UMAP(n_neighbors=min(15, len(data) - 1)).fit_transform(data)),
                  ('tsne', lambda data: MulticoreTSNE.MulticoreTSNE().fit_transform(data)),
                  ]

    for dataset in datasets:
        print(dataset)

        dataset_embeddings = {}
        for algorithm in algorithms:
            alg_name, alg_func = algorithm
            dataset_embeddings[alg_name] = alg_func(dataset.data)

        for sample_size in sample_sizes:
            print(f"\tSampling {sample_size}%")
            subset_size = len(dataset) * sample_size // 100
            subset_indexes = np.random.choice(np.arange(len(dataset)), size=subset_size, replace=False)
            subset_data = dataset.data[subset_indexes]
            for algorithm in algorithms:
                alg_name, alg_func = algorithm
                if subset_size < 16:
                    continue

                subset_embeddings = alg_func(subset_data)

                _, _, disparity = procrustes(subset_embeddings, dataset_embeddings[alg_name][subset_indexes])

                print(f"\t\t{alg_name:15}: {disparity:.4f}")
