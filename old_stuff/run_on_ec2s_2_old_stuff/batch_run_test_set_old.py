import boto3
from datetime import datetime, timedelta

from utils import get_ec2_cpu_utilization, print_cpu_utilization

from utils import all_regions as regions
from utils import all_task_ids as task_ids
import subprocess

import os
import sys
which_account = sys.argv[1]


if which_account == '1':
    os.environ['AWS_PROFILE'] = 'maolinml'
    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
elif which_account == '2':
    os.environ['AWS_PROFILE'] = 'maolinml2'
    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
else:
    os.environ['AWS_PROFILE'] = 'maolinml3'
    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]

# print(regions)
# print(task_ids)
# print(os.environ['AWS_PROFILE'])
# print(account)

k = 0

for region in regions:
    print(region)

    ec2 = boto3.resource('ec2', region_name=region)
    filters = [
        {
            'Name': 'instance-state-name', 
            'Values': ['running']
        }
    ]
    instances = ec2.instances.filter(Filters=filters)
    public_ips = []
    ids = []
    for instance in instances:
        public_ip = instance.meta.data['PublicIpAddress']
        public_ip = public_ip.replace(".", "-")
        public_ips.append(public_ip)
        ids.append(instance.id)

    period = 600

    for ip, instance_id in zip(public_ips, ids):
        
        end_time = datetime.now()
        # start_time = end_time - timedelta(hours=3)
        start_time = end_time - timedelta(minutes=5)

        # Get CPU utilization data
        cpu_utilization_data = get_ec2_cpu_utilization(region, instance_id, start_time, end_time, period=period)

        # Print CPU utilization data
        usage = print_cpu_utilization(cpu_utilization_data)
        
        if usage < 5: # 100: # 5:
            try:
                subprocess.run(['sh', f"run_test_set.sh", f"{account}", f"{region}", f"{ip}", f"{task_ids[k]}"], stdout=subprocess.DEVNULL)
                # subprocess.run(['sh', f"run_test_set.sh", f"{account}", f"{region}", f"{ip}", f"{task_ids[k]}"])
                print(instance_id, usage, k, task_ids[k])
                k += 1
            except:
                print("submission failed")
        else:
            print(instance_id, usage)

    print()

