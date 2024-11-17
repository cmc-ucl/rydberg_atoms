import subprocess

instances = {
    # 'us-east-1': ['3-236-21-14', '3-236-6-0', '44-213-67-67'],
    # 'us-west-2': ['52-40-191-127', '54-189-90-24'],
    # 'ap-northeast-2': ['3-39-245-155', '3-37-15-60'],
    # 'ap-northeast-1': ['3-112-206-163'],
    # 'eu-west-1': ['34-254-201-161', ],
    # 'eu-north-1': ['16-16-67-29', ],
}

# task_ids = [12, 13, 16, 17, 24,25,26,27,28,29,30,31,32,33,34]

k = 0
for region in instances.keys():
    ips = instances[region]
    for ip in ips:
        # print(region, ip, k)
        subprocess.run(['sh', 'run_test_set.sh', f"{region}", f"{ip}", f"{task_ids[k]}"], stdout=subprocess.DEVNULL)
        k += 1
        print()