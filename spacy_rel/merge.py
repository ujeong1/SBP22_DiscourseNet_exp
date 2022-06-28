from os import listdir
from os.path import isfile, join
mypath = "processed_text/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

result = []
for filename in onlyfiles:
    with open(mypath+filename, "r") as f:
        for line in f.readlines():
            result.append(line)

with open("FBAD.txt", "w") as f:
    for line in result:
        f.write(line)
