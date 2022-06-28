import pandas as pd
df = pd.read_csv("../dataset/covid_final.csv")
docs = df["text"].values
with open("../processed_text/covid_dataset.txt", "wb") as f:
    for doc in docs:
        f.write(doc.encode('latin1'))

