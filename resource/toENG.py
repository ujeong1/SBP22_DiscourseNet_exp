from langdetect import detect
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
#mypath = "processed_ad/"
mypath = "./"
onlyfiles = ["62716ca40bf06e5252bc3d19,covid,US,ALL,POLITICAL_AND_ISSUE_ADS,FACEBOOK.csv"]#[f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = ["61f5dba6bba118ed7d70e78f,covid,US,BG,GB,ALL,POLITICAL_AND_ISSUE_ADS,FACEBOOK.csv"]
csvs = dict()
for filename in onlyfiles:
    data = pd.read_csv(mypath+filename)
    keyword = filename.split(",")[1]
    csvs[keyword] = data
sentences = list(data["ad_creative_body"])
mypath = "processed_text/"
for keyword, data in tqdm(csvs.items()):
    sentences = list(data["ad_creative_body"])
    result = []
    for sentence in sentences:
        print(sentence)
        print("*"*10)
        #if sentence.strip() == "," or sentence.strip() == ", ,":
        #    continue
        lang = detect(sentence) # outputs 'en', 'es', 'en'
        if lang == "en":
            if 'covid' in sentence or 'virus' in sentence:
                result.append(sentence)
        else:
            continue
print(len(result))
    # print(len(result))
    # with open(mypath+keyword+".txt", "w") as f:
    #     for text in result:
    #         text = text.strip("\n")
    #         f.write(text+"\n")
 
