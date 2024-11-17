import os

folder = "../train_set_2/"

for file in os.listdir(folder):
    os.rename(f"{folder}/{file}", f"{folder}/train_set_{file}")


for file in os.listdir(folder):
    print(file)
