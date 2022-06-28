import pandas as pd
labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]
df = pd.read_csv("sbp_dataset.csv")

df = df[df['ad_creative_body'].notna()]
df = df.drop("Unnamed: 0", axis=1)
for label in labels:
    df[label] = df[label].fillna(0)
for label in labels:
    df[label] = df[label].astype(int)
print(df)
print(len(df))
df.to_csv("sbp_dataset.csv")
