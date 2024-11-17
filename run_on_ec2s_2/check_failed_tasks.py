import os
import numpy as np
# directory = os.fsencode("../pyscf/failed_train_set")
directory = os.fsencode("../pyscf/failed_test_set")

outs = []
logs = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    if filename.endswith(".log"):
        logs.append(int(filename.split(".")[0].split("_")[-1]))
    elif filename.endswith(".out"):
        outs.append(int(filename.split(".")[0].split("_")[-1]))
    else:
        print(filename)

print([len(logs), len(outs)])

outs = list(np.sort(outs))
logs = list(np.sort(logs))

print(outs)
print(logs)