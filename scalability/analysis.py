import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

df = pd.read_csv("scalability.csv", index_col='index')
dict = {}
picked_dataset_list = []

# get dataset length and dimension
rootdir = "../umato_exp/datasets/npy"
for rootdir, dirs, files in os.walk(rootdir):
    dataset_list = dirs
    break

data_lengths = {}
data_dims = {}

for name in df['name']:
    dict[name] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 'inf': 0}

# remove columns with inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)
names = df['name']
df = df.dropna(axis=1).drop(['name', 'repeat_num'], axis=1)

for i, datadir in enumerate(df.columns):
    if datadir not in df.columns:
        continue
    x = np.load(f"../datasets/npy/{datadir}/data.npy")
    label = np.load(f"../datasets/npy/{datadir}/label.npy")
    data_lengths[datadir] = x.shape[0]
    data_dims[datadir] = x.shape[1]
    # print the shape of datasets with length >= 10000
    # if x.shape[0] >= 10000:
    #    picked_dataset_list.append(datadir)
    #    print(f"{datadir}: {x.shape[0]} {x.shape[1]}")

# df = df[picked_dataset_list]

#print(df)

# draw a boxplot of each rows
boxplot = plt.figure()
bplot = df.T.rename(columns=(lambda x: names.loc[x])).boxplot()
bplot.set_ylabel('elapsed time (s)')
boxplot.savefig('boxplot.png')
plt.clf()

# add statistics column
df['mean'] = df.mean(axis=1)
df['std'] = df.std(axis=1)
df['min'] = df.min(axis=1) 
df['median'] = df.median(axis=1)
df['max'] = df.max(axis=1)
df['q1'] = df.quantile(0.25, axis=1)
df['q3'] = df.quantile(0.75, axis=1)
df['name'] = names
print(df[['name', 'mean', 'std', 'min', 'median', 'q1', 'q3', 'max']])

'''
# count number of infs and number of times each algorithm is in top 5
for column in df.columns:
    if column in ['index', 'name', 'repeat_num']:
        continue
    sorted_df = df.sort_values(by=column)
    for (i, row) in enumerate(sorted_df.iterrows(), 1):
        row = row[1]
        if row[column] == np.inf:
            dict[row['name']]['inf'] += 1
        else:
            if row[column] - row['mean'] > row['std'] and row['name'] == 'UMAP':
                print(f"outlier: {row['name']} {column} {row[column]}")
            if row['name'] == 'UMAP' and column == 'cifar10':
                print(f"UMAP: {row[column]}")
            dict[row['name']][i] += 1

print()
for alg, valdict in dict.items():
    print(alg)
    for key, value in valdict.items():
        if key == 'inf':
            print("number of infs: " + str(value))
        else:
            print(str(key) + "th place: " + str(value))
'''
            

# generate a bar chart of statistics

df_stat = df[['name', 'mean', 'median', 'std']]
X_axis = np.arange(len(df_stat))
plt.bar(X_axis - 0.2, df_stat['mean'], 0.2, label = 'mean')
plt.bar(X_axis, df_stat['median'], 0.2, label = 'median')
plt.bar(X_axis + 0.2, df_stat['std'], 0.2, label = 'std')
plt.xticks(X_axis, df_stat['name'])
plt.xlabel("algorithm")
plt.ylabel("time (s)")
plt.legend()
plt.savefig('barchart.png')

# generate a length-time plot and dimension-time plot of data from UMATO and UMAP
# x = data_lengths.values()
# y = data_dims.values()
# z = df.loc[df['name'] == 'UMAP'].drop(['name', 'mean', 'std', 'min', 'median', 'max'], axis=1)

# save the plot
'''
plt.xscale('log', base=10)
plt.xlabel('length')
plt.ylabel('elapsed time')
plt.scatter(x, z)
plt.savefig('length_time_umap.png')
plt.clf()

plt.xscale('log', base=10)
plt.xlabel('dimension')
plt.ylabel('elapsed time')
plt.scatter(y, z)
plt.savefig('dim_time_umap.png')
plt.clf()

z = df.loc[df['name'] == 'UMATO'].drop(['name', 'mean', 'std', 'min', 'median', 'max'], axis=1)

# save the plot
plt.xscale('log', base=10)
plt.xlabel('length')
plt.ylabel('elapsed time')
plt.scatter(x, z)
plt.savefig('length_time_umato.png')
plt.clf()

plt.xscale('log', base=10)
plt.xlabel('dimension')
plt.ylabel('elapsed time')
plt.scatter(y, z)
plt.savefig('dim_time_umato.png')
plt.clf()
'''