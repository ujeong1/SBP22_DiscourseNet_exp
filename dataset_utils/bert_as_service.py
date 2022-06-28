from bert_serving.client import BertClient
import pandas as pd
import math
import numpy as np
#df = pd.read_csv("../covid_dataset_shuffled.csv")
df = pd.read_csv("../dataset/sbp_dataset.csv")
labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]
for label in labels:
    df[label] = df[label].fillna(0)
for i, row in df.iterrows():
    result = 0
    for label in labels:
        value = row[label]
        result += value
    if result == 0:
        print(i, row)
        exit(0)

#docs = list(df["processed_text"].values)
docs = list(df["ad_creative_body"].values)
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')                    
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
'''
bc = BertClient()
vectors = bc.encode(docs)
print(vectors.shape)

with open('covid_bert.npy', 'wb') as f:
    np.save(f, vectors)
