import pickle

filename = "covid_entity.pkl"

with open(filename, "rb") as f:
    data = pickle.load(f)
print(data)
