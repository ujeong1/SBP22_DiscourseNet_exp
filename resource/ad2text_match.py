import pandas as pd
from tqdm import tqdm
#filename = "./62716ca40bf06e5252bc3d19,covid,US,ALL,POLITICAL_AND_ISSUE_ADS,FACEBOOK.csv"
filename = "./61f5dba6bba118ed7d70e78f,covid,US,BG,GB,ALL,POLITICAL_AND_ISSUE_ADS,FACEBOOK.csv"
filename = "./sbp_covid_data.csv"
df = pd.read_csv(filename)
df = df.drop(["Unnamed: 0"], axis=1)
filename = "processed_text/covid_dataset.txt"
line2idx = dict()
i=0
ids = []
ex_df = pd.DataFrame(columns=df.columns)
with open(filename, "rb") as f:
    lines = f.readlines()
    for line in tqdm(lines[:1050]):
        line = line.decode('latin1')
        for i, row in df.iterrows():
            text = row["ad_creative_body"]
            if line.lower().strip() == text.lower().strip():
                ids.append(i)
                ex_df.append(row, ignore_index=True)
print(len(ids))
print(len(set(ids)))

for id in ids:
    ex_df = ex_df.append(df.iloc[id])
ex_df.to_csv("for_annotation.csv")
