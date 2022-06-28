import torch
import pickle
import pandas as pd
from torch.nn import MaxPool1d, AvgPool1d
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

filename = "covid_keyword.pkl"
with open(filename, "rb") as f:
    data = pickle.load(f)

labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]

labels=['Forecasting']
print(labels)
for label in labels:
    filename = "covid_" + label + "_edge.pt"
    edge = torch.load(filename, map_location="cpu")
    dim = edge.size(dim=1)
    pool = AvgPool1d(dim)
    # pool = MaxPool1d(dim)
    attention = pool(edge).reshape(-1)
    name = list(data.keys())
    df = pd.DataFrame()
    num_nodes = [len(hyperedge) for hyperedge in data.values()]
    docs = [hyperedge for hyperedge in data.values()]
    df["attention"] = abs(attention.numpy().astype(float))
    df["name"] =name
    df["num_nodes"] = num_nodes
    df["docs"] = docs
    min_att = df["attention"].min()
    max_att = df["attention"].max()
    df = df.sort_values(by=['attention'], ascending=False)
    print(label)
    num_samples = int(0.1*len(df))
    num_samples = 80
    print(num_samples)
    print(df.head(num_samples))
    #print(df.head(num_samples).sort_values(by=['num_nodes'], ascending=False))
    #print(df.tail(num_samples))
    print("-" * 50)

    df.to_csv(label+"_attns.csv")
