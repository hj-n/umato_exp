
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASETS = os.listdir("../datasets_candidate/npy/")

hub_nums = [ 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]

tnc_list = []
kl_div_list = []
hub_num_list = []
normalized_hubnum_list = []
size_list = []
dataset_list = []

for dataset in DATASETS:

	X = np.load(f"../datasets_candidate/npy/{dataset}/data.npy")
	size = X.shape[0]

	tnc_list_curr = []
	kl_div_list_curr = []
	for hub_num in hub_nums:
		if not os.path.exists(f"./08_hyperparameter_exp/results_hubnum/{dataset}_umato_hub_num_{hub_num}_results.json"):
			continue
		
		with open(f"./08_hyperparameter_exp/results_hubnum/{dataset}_umato_hub_num_{hub_num}_results.json", "r") as f:
			results = json.load(f)
			tnc = results["tnc"]
			kl_div = results["kl_divergence"]

			tnc_list_curr.append(tnc)
			kl_div_list_curr.append(1 - kl_div)
			hub_num_list.append(hub_num)
			normalized_hubnum_list.append(hub_num / size)
			size_list.append(size)
			dataset_list.append(dataset)
	tnc_list_curr = np.array(tnc_list_curr)
	if tnc_list_curr.size == 0:
		continue
	tnc_list_curr = tnc_list_curr / np.max(tnc_list_curr) 
	tnc_list += tnc_list_curr.tolist()

	kl_div_list_curr = np.array(kl_div_list_curr)
	if kl_div_list_curr.size == 0:
		continue
	kl_div_list_curr = kl_div_list_curr / np.max(kl_div_list_curr)
	kl_div_list += kl_div_list_curr.tolist()


# kl_div_list = 1 - np.array(kl_div_list)
# tnc_list = np.log(tnc_list)
# hub_num_list = np.log(1 / np.array(hub_num_list))

df = pd.DataFrame({
	"tnc": tnc_list,
	"kl_div": kl_div_list,
	"hub_num": hub_num_list,
	"size": size_list,
	"n_hub_num": normalized_hubnum_list,
	"dataset": dataset_list
})


		

## draw scatterplot tnc vs hub_num with figsize (10, 10)
fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(df["hub_num"], df["tnc"], s=20, c="blue", alpha=0.2)




# sns.lmplot(data=df, x="hub_num", y="tnc", logistic=True, scatter_kws={"s": 0.7})

## connect by dataset
# for dataset in DATASETS:
	# df_dataset = df[df["dataset"] == dataset]
	# plt.plot(df_dataset["hub_num"], df_dataset["tnc"], label=dataset, color="black", alpha=0.3)

plt.xlabel("hub_num")
plt.ylabel("tnc")
# plt.colorbar()
# plt.xlim(0, 500)
# plt.xscale("log")

plt.savefig("./08_hyperparameter_exp/plot/tnc_vs_hub_num.png")


## draw scatterplot kl_div vs hub_num
fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(df["hub_num"], df["kl_div"], s=20, c="red", alpha=0.2)



plt.xlabel("hub_num")
plt.ylabel("kl_div")
# plt.colorbar()
# plt.xlim(0, 500)
plt.ylim(0.907, 1.01)

## log scale
# plt.xscale("log")

plt.savefig("./08_hyperparameter_exp/plot/kl_div_vs_hub_num.png")

df.to_csv("./08_hyperparameter_exp/plot/data.csv", index=False)

