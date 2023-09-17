import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os, json


FILES = os.listdir("./02_scalability/results/")


umap_results = []
umato_results = []
pacmap_results = []
tsne_results = []
trimap_results = []
pca_results = []
lle_results = []
lmds_results = []

for file in FILES:
  content = json.load(open(f"./02_scalability/results/{file}", "r"))
  umap_results.append(content["umap"])
  umato_results.append(content["umato"])
  pacmap_results.append(content["pacmap"])
  tsne_results.append(content["tsne"])
  trimap_results.append(content["trimap"])
  pca_results.append(content["pca"])
  lle_results.append(content["lle"])
  lmds_results.append(content["lmds"])


df = pd.DataFrame({
  "Time (s)": umap_results + umato_results + pacmap_results + tsne_results + trimap_results + pca_results + lle_results + lmds_results,
  "DR Technique": ["UMAP"]*len(umap_results) + ["$\mathbf{UMATO}$"]*len(umato_results) + ["PacMAP"]*len(pacmap_results) + ["t-SNE"]*len(tsne_results) + ["Trimap"]*len(trimap_results) + ["PCA"]*len(pca_results) + ["LLE"]*len(lle_results) + ["L-MDS"]*len(lmds_results)
})

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6.5, 3.5))
sns.pointplot(
  y="DR Technique", x="Time (s)",
  data=df, join=False, hue="DR Technique",
  order=["PCA", "$\mathbf{UMATO}$", "PacMAP", "Trimap", "UMAP", "t-SNE", "L-MDS", "LLE" ]
)

## remove legend
plt.legend([],[], frameon=False)

plt.ylabel("")


plt.tight_layout()

plt.savefig("./02_scalability/plot/scalability.pdf", dpi=300)
plt.savefig("./02_scalability/plot/scalability.png", dpi=300)

