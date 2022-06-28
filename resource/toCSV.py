total = 0
filename = "processed_text/covid.txt" 
with open(filename, "r") as f:
    for line in f.readlines():
        if 'covid' in line.lower() or 'virus' in line.lower() :
            total+=1
print(total)
