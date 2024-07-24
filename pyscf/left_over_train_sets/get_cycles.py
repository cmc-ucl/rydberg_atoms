

# filename = "everything/maolinml/ap-northeast-1-18-183-164-170/graphene_mol_r_6_6_h_194.out"

def get_cycle_for_file(filename):

    with open(filename, "r") as f:
        data = f.readlines()

    cycle = 0
    for line in data:
        if line.startswith('cycle'):
            cycle = int(line.split(":")[0].split(" ")[1])

    return cycle

# print(get_cycle_for_file(filename))

import os
import numpy as np

accounts = ['maolinml', 'maolinml2', 'maolinml3']

data = {}
for account in accounts:
    folder = account
    for subfolder in os.listdir(folder):
        # print(subfolder)

        files = os.listdir(f'{folder}/{subfolder}')
        # if not (len(files) == 3):
        #     # print(files)
        #     print(f'{folder}/{subfolder}')
        assert len(files) == 3
        check_if_train_set = 0
        
        for file in files:
            if file.startswith("train_set"):
                check_if_train_set = 1
            if '.out' in file:
                filename = f'{folder}/{subfolder}/{file}'
                ind = int(file.split(".")[0].split("_")[-1])
                cycle = get_cycle_for_file(filename)
                # print(filename, ind, cycle)
                data[ind] = cycle
    
        assert check_if_train_set == 1
#     # print()

# print(data)
# print(list(np.sort(list(data.keys()))))
sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}


print("Left over train set as {task_id: cycles optimized}")
print(sorted_data)
# print()
# print(sorted_data.keys())