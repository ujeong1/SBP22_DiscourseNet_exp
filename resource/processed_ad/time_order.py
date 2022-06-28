import pandas as pd
filename = "62716ca40bf06e5252bc3d19,covid,US,ALL,POLITICAL_AND_ISSUE_ADS,FACEBOOK.csv"
df = pd.read_csv(filename)
df = df.sort_values(by='ad_creation_time', ascending=False)
print(df["ad_creation_time"])
