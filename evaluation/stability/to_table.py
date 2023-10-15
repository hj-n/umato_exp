import json
import pandas as pd
import numpy as np

source = 'results.json'

with open(source, 'r') as f:
    data = json.load(f)

algorithms_to_compare = ['DensMAP', 'Isomap', 'PaCMAP', 'TriMap', 'UMAP', 't-SNE', 'UMATO-75(50)', 'UMATO-300(50)']

filtered_data = {}

for dataset in data:
    filtered_data[dataset] = {}
    for sample_size in data[dataset]:
        filtered_data[dataset][sample_size] = {}
        for alg_name in algorithms_to_compare:
            if alg_name in data[dataset][sample_size]:
                filtered_data[dataset][sample_size][alg_name] = data[dataset][sample_size][alg_name]

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
                results_per_sample_size[alg_name][sample_size_int].append(
                    filtered_data[dataset][sample_size][alg_name])

# Create a DataFrame for the combined values
combined_table = pd.DataFrame(columns=["Algorithm", "Sample Size", "Avg & Std"])

# Loop over the results_per_sample_size dictionary to populate the DataFrame
for alg_name in results_per_sample_size:
    for sample_size in results_per_sample_size[alg_name]:
        if len(results_per_sample_size[alg_name][sample_size]) == 14:
            avg = np.mean(results_per_sample_size[alg_name][sample_size])
            std = np.std(results_per_sample_size[alg_name][sample_size])
            combined = f"{avg:.2f}, {std:.2f}"
            new_row = pd.DataFrame({"Algorithm": [alg_name], "Sample Size": [sample_size], "Avg & Std": [combined]})
        else:
            new_row = pd.DataFrame({"Algorithm": [alg_name], "Sample Size": [sample_size], "Avg & Std": ["N/A"]})
        combined_table = pd.concat([combined_table, new_row], ignore_index=True)

# Pivot the DataFrame
combined_table = combined_table.pivot(index='Algorithm', columns='Sample Size', values='Avg & Std')

print(combined_table)

# Print LaTeX table
print(combined_table.to_latex(escape=False))
