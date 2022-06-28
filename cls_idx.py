import pandas as pd
import numpy as np

label_list = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting",
              "General"]


def import_targets(df):
    targets = []
    for i, row in df.iterrows():
        target = []
        for label in label_list:
            if row[label] == 1:
                target.append(1)
            else:
                target.append(0)
        targets.append(target)

    return targets
def class_idx(df):
    num_nodes = len(df)
    targets = import_targets(df)
    # origin_idx = np.arange(num_nodes)
    class_dict = dict()
    for i in range(len(label_list)):
        class_dict[i] = []
    for i in range(len(label_list)):
        for idx, target in enumerate(targets):
            if target[i]:
                class_dict[i].append(idx)
    print(class_dict)
    return class_dict
