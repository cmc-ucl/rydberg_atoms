import os

# directory = os.fsencode("usage_456882910219")
directory = os.fsencode("usage_545821822555")
directoryname = os.fsdecode(directory)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    with open(f"{directoryname}/{filename}") as f:
        # data = f.readlines()
        data = f.read()

    print(data)

    # # # print(data)
    # print(data[1])
    # print(data[2])
    # # # print(type(data[2]))
    # # print(data[2].split(" "))
    
    # lastfile = data[2].split(" ")[-1]
    # idx = lastfile.split(".")[0].split("_")[-1]

    # if "out" in lastfile:
    #     print(f"{idx} is either running or dead")
    # else:
    #     print(f"{idx} is done")
        
    # print()