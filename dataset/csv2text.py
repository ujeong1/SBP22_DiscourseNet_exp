import pandas as pd
df = pd.read_csv("sbp_dataset.csv")

text = df["processed_text"].values.tolist()
#text = df["ad_creative_body"].values.tolist()
print(len(text))
with open("covid_corpus.txt", "w") as f:
    for line in text:
        line = " ".join(line.split()).strip()+"\n"
        f.write(line)#.encode('latin1'))
