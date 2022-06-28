import pandas as pd
import pickle
from tqdm import tqdm
filename="../covid_dataset_shuffled.csv"
df = pd.read_csv(filename)

pages = list(set(list(df["page_name"].values)))
print("Num of page names", len(pages))
#print(pages)
sponsors = list(set(list(df["funding_entity"].values)))
print("Num of sponsors", len(sponsors))
#print(sponsors)
hypergraph = dict()
for page in pages:
    hypergraph[page] = []
for page in tqdm(pages):
    for i, row in df.iterrows():
        if page == row["page_name"]:
            hypergraph[page].append(i)

keys = list(hypergraph.keys())
for sponsor in sponsors:
    if sponsor not in keys:
        hypergraph[sponsor] = []
for sponsor in tqdm(sponsors):
    for i, row in df.iterrows():
        if sponsor == row["funding_entity"]:
            hypergraph[sponsor].append(i)

result = dict()
for key, hyperedge in hypergraph.items():
    hyperedge = list(set(hyperedge))
    if len(hyperedge) >= 2:
        result[key] = hyperedge
print(result)
with open("../covid_advertiser.pkl", "wb") as f:
    pickle.dump(result, f)



    


