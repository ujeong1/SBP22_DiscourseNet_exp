import pandas as pd
from tqdm import tqdm
filename = "covid_dataset_shuffled.csv"
df = pd.read_csv(filename)

filename = "org.csv"#61f5dba6bba118ed7d70e78f,covid,US,BG,GB,ALL,POLITICAL_AND_ISSUE_ADS,FACEBOOK.csv"
ad_df = pd.read_csv(filename)
rows = []
for i, row_i in tqdm(df.iterrows()):
    id_i = row_i["id"]
    row = ad_df.loc[ad_df['id'].astype(float) == id_i]
    rows.append(row["ad_creative_body"].values)
print(len(df))
print(len(rows))
df["ad_creative_body"] = pd.DataFrame(rows)
print(df["ad_creative_body"])
print(df["id"])
print("**********")
print(df["processed_text"])
df.to_csv("sbp_dataset.csv")
