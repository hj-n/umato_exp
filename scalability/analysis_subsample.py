import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

df = pd.read_csv("scalability_subsample.csv", index_col='index')
picked_dataset_list = ["paris_housing_classification", "dry_bean", "htru2", "rice_seed_gonen_jasmine", "magic_gamma_telescope", "letter_recognition", "fraud_detection_bank"]
sample_lengths = [100, 200, 500, 1000, 2000, 5000, 10000]

data_lengths = {}
data_dims = {}

# remove columns with inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# add averages column for each subsample size
for length in sample_lengths:
    df['avg_' + str(length)] = df.loc[:, df.columns.str.endswith('.'+str(length))].mean(axis=1)


# generate a new dataframe with only the averages
df_avg = df.loc[:, df.columns.str.contains('avg_')]
for index in df_avg.index:
    df_avg.loc[index] = df_avg.loc[index] / df_avg.loc[index, 'avg_100']

print(df_avg)

# plot the averages with a connectig line for each dataset
# x axis: subsample size, y axis: average time
alg_label = ['UMATO', 'UMAP', 'PaCMAP', 'densMAP', 'triMAP', 'MulticoreTSNE']
for i in range(len(df)):
    plt.plot(sample_lengths, df_avg.loc[i], label=f"{alg_label[i]}", marker=".")

plt.xlabel('subsample size')
plt.ylabel('runtime / runtime with sample size 100')
plt.xscale('log', base=10)
plt.gca().set_ylim([0.8, 5])
plt.minorticks_off()
plt.xticks(sample_lengths, sample_lengths)
plt.title('Runtime Ratio-Subsample Size Scalability')
plt.legend()
plt.savefig("scalability_subsample.png")
plt.show()