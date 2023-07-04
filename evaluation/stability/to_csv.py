import json
import pandas as pd

with open("stability.json", 'r') as f:
    data = json.load(f)

print(len(data))

with open('out.csv', 'w') as f:
    for dataset_name in data:
        f.write(dataset_name + '\n')
        df = pd.DataFrame(data[dataset_name])
        df.to_csv(f)
        f.write('\n')
