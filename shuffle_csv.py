import pandas as pd
df = pd.read_csv("dataset/covid_dataset.csv")
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("covid_dataset_shuffled.csv")
