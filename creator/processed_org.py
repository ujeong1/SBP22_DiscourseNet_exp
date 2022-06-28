import re
import pandas as pd
pd.set_option('display.max_rows', None)

df1 = pd.read_csv("org.csv")
df1 = df1[df1['ad_creative_body'].notna()]#df1.dropna("ad_creative_body", axis=1)
def filter_content_regex(df):
    df["processed_text"] = df["ad_creative_body"]
    df.processed_text = df.apply(lambda row: re.sub(r"http\S+", "", row.processed_text), 1)
    df.processed_text = df.apply(lambda row: " ".join(filter(lambda x: x[0] != "@", row.processed_text.split())), 1)
    df.processed_text = df.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.processed_text).split()), 1)
    return df
print(df1["ad_creative_body"].head(100))
df2 = filter_content_regex(df1)
print(df2["processed_text"].head(100))
df2.to_csv("org_proc.csv")
