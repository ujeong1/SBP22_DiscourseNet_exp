import pandas as pd
df = pd.read_csv("../dataset/covid_dataset.csv")
#print(df)

cls = dict()
labels = ["Prevention","Treatment","Diagnosis","Mechanism","Case Report","Transmission","Forecasting","General"]
for label in labels:
    cls[label] = 0
for i, row in df.iterrows():
    for label in labels:
        if row[label] == 1:
            cls[label]+=1

print(cls)
    


