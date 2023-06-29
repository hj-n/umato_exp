import json
from pprint import pprint
from matplotlib import pyplot as plt

source = 'results.json'

with open(source, 'r') as f:
    data = json.load(f)

algorithms_to_compare = ['DensMAP', 'Isomap', 'PaCMAP', 'TriMap', 'UMAP', 't-SNE', 'UMATO-75(15)']

filtered_data = {}

for dataset in data:
    filtered_data[dataset] = {}
    for sample_size in data[dataset]:
        filtered_data[dataset][sample_size] = {}
        for alg_name in algorithms_to_compare:
            if alg_name in data[dataset][sample_size]:
                filtered_data[dataset][sample_size][alg_name] = data[dataset][sample_size][alg_name]

pprint(filtered_data)

results_per_sample_size = {}
for dataset in filtered_data:
    for sample_size in filtered_data[dataset]:
        for alg_name in algorithms_to_compare:
            if alg_name in filtered_data[dataset][sample_size]:
                if alg_name not in results_per_sample_size:
                    results_per_sample_size[alg_name] = {}
                sample_size_int = int(sample_size)
                if sample_size_int not in results_per_sample_size[alg_name]:
                    results_per_sample_size[alg_name][sample_size_int] = []
                results_per_sample_size[alg_name][sample_size_int].append(filtered_data[dataset][sample_size][alg_name])

pprint(results_per_sample_size)

fig, ax = plt.subplots()

target_sample_size = 10

data = [results_per_sample_size[alg_name][target_sample_size] for alg_name in results_per_sample_size.keys() if
        target_sample_size in results_per_sample_size[alg_name]]

# Create box plots for each algorithm
ax.boxplot(data)

# Set the x-ticks labels to be the algorithms names
ax.set_xticklabels(map(lambda x: 'UMATO' if 'UMATO' in x else x, results_per_sample_size.keys()))

plt.show()
