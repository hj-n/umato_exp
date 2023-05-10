from zadu import zadu
import os
import numpy as np
from tqdm import tqdm





spec = [{
    "id" : "tnc", "params": { "k": 20 },
}, 
{
		"id" : "lcmc", "params": { "k": 20 },
},{
    "id" : "mrre", "params": { "k": 20 },
}, {
    "id" : "nd", "params": { "k": 20 },
}, {
    "id" : "snc", "params": { "k": 20 },
}, {
    "id" : "kl_div", "params": {},
},  {
    "id" : "dtm", "params": {},
},{
    "id" : "pr", "params": {},
}, {
    "id" : "srho", "params": {},
}, {
		"id" : "stress", "params": {},
}
]


datasets_list = os.listdir("../embeddings/")

print(datasets_list)
for dataset in tqdm(datasets_list):
  embeddings = os.listdir(f"../embeddings/{dataset}/")
  embeddings = [file for file in embeddings if not file.endswith("_hp.npy") and not file.endswith("_score.npy")]
  
  data = np.load(f"../datasets/npy/{dataset}/data.npy")
  zadu_obj = None
  for embedding in embeddings:
		
    if os.path.exists(f"../embeddings/{dataset}/{embedding[:-4]}_score.npy"):
      continue
    if zadu_obj is None:
      zadu_obj = zadu.ZADU(spec, data)
    emb_data = np.load(f"../embeddings/{dataset}/{embedding}")  
    scores = zadu_obj.measure(emb_data)
    
    np.save(f"../embeddings/{dataset}/{embedding[:-4]}_score.npy", scores)