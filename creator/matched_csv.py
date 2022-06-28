import pandas as pd
from tqdm import tqdm
labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]
df1 = pd.read_csv("org_proc.csv")
df2 = pd.read_csv("covid_dataset_shuffled.csv")
df2 = df2.sample(frac=1).reset_index(drop=True)
print(len(df2))
print(df1["processed_text"])
result = []
print(df2.columns)
cols = dict()
cols["processed_text"] = []
cols["ad_creative_body"] = []
for label in labels:
    cols[label] = []
for i, row_i in tqdm(df2.iterrows()):
    proc1 = row_i["processed_text"]
    result = df1.loc[df1["processed_text"] == proc1.strip()]
    if result.empty:
        continue
    cols["processed_text"].append(result["processed_text"].values[0].strip())
    cols["ad_creative_body"].append(result["ad_creative_body"].values[0])
    for label in labels:
        cols[label].append(row_i[label])
df3 = pd.DataFrame(cols)
print(df3)
print(len(df3))
df3.to_csv("sbp_dataset.csv")
