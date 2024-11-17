import os
import numpy as np
############
# directory = os.fsencode("../run_on_ec2s/all_tasks")
# directory2 = os.fsencode("../run_on_ec2s/all_tasks/finishe_ones")


    
# jsons = []
# xyzs = []
# outs = []
# pys = []
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)

#     if filename.endswith(".out"):
#         idx = filename.split(".")[0].split("_")[-1]

#         jsons.append(int(idx))
        
#         # f1 = f"../run_on_ec2s/all_tasks/graphene_mol_r_6_6_h_{idx}.json"
#         # f2 = f"../run_on_ec2s/all_tasks/graphene_mol_r_6_6_h_{idx}.py"
#         # f3 = f"../run_on_ec2s/all_tasks/graphene_mol_r_6_6_h_{idx}.xyz"
#         # f4 = f"../run_on_ec2s/all_tasks/graphene_mol_r_6_6_h_{idx}.out"

#         # g1 = f"../run_on_ec2s/all_tasks/finishe_ones/graphene_mol_r_6_6_h_{idx}.json"
#         # g2 = f"../run_on_ec2s/all_tasks/finishe_ones/graphene_mol_r_6_6_h_{idx}.py"
#         # g3 = f"../run_on_ec2s/all_tasks/finishe_ones/graphene_mol_r_6_6_h_{idx}.xyz"
#         # g4 = f"../run_on_ec2s/all_tasks/finishe_ones/graphene_mol_r_6_6_h_{idx}.out"

#         # os.rename(f1, g1)
#         # os.rename(f2, g2)
#         # os.rename(f3, g3)
#         # os.rename(f4, g4)
        

# print(len(jsons))
# print(list(np.sort(jsons)))

############
# directory = os.fsencode("../pyscf/failed_tasks")
    
# to_runs = []
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)

#     if filename.endswith(".out"):
#         to_runs.append(int(filename.split(".")[0].split("_")[-1]))
#     else:
#         print(filename)

# print(to_runs)

############
# directory = os.fsencode("../pyscf/test_set")
directory = os.fsencode("../pyscf/train_set")
    
jsons = []
xyzs = []
outs = []
pys = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    if filename.endswith(".json"):
        jsons.append(int(filename.split(".")[0].split("_")[-1]))
    elif filename.endswith(".xyz"):
        xyzs.append(int(filename.split(".")[0].split("_")[-1]))
    elif filename.endswith(".out"):
        outs.append(int(filename.split(".")[0].split("_")[-1]))
    elif filename.endswith(".py"):
        if 'graphene' in filename:
            pys.append(int(filename.split(".")[0].split("_")[-1]))
        else:
            print(filename)
    else:
        print(filename)

print([len(jsons), len(xyzs), len(outs), len(pys)])

jsons = np.sort(jsons)
xyzs = np.sort(xyzs)
outs = np.sort(outs)

assert (jsons == xyzs).all()
assert (jsons == outs).all()
print(list(jsons))


# outs2 = []

# for out in outs:
#     if out in jsons:
#         assert out in xyzs
#     else:
#         assert out not in xyzs
#         outs2.append(out)
#         # os.remove(f"{os.fsdecode(directory)}/{out}.out")

# print(outs2)

# need_to_run = []
# for py in pys:
#     if py in jsons:
#         assert py in xyzs
#     else:
#         need_to_run.append(int(py.split("_")[-1]))

# need_to_run = list(np.sort(need_to_run))
# print(need_to_run)