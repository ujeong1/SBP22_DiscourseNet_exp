import pandas as pd
labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]
df = pd.read_csv("covid_dataset_shuffled.csv")
rows = []
org = pd.read_csv("org.csv")
result = []
for i, row_i in df.iterrows():
    id_i = row_i["id"]#.astype(float)
    for j, row_j in org.iterrows():
        id_j = row_i["id"]#.astype(float)
        if id_i == id_j:
            value = org.loc[org['id'] == id]
            text = value["ad_creative_body"].to_string()
            print(row_i["processed_text"])
            print(text)
            exit(0)
    result.append(text)

df["ad_creative_body"] = pd.DataFrame(result)
#df = df[df['processed_text'].notna()]
df = df[df['ad_creative_body'].notna()]
df = df.drop("Unnamed: 0", axis=1)
df = df.drop("Unnamed: 0.1", axis=1)
print(len(df))
for label in labels:
    df[label] = df[label].fillna(0)
    df[label] = df[label].astype(int)
print(df)
print(len(df))

df.to_csv("sbp_dataset.csv")
