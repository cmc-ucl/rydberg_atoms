import subprocess
import boto3
from datetime import datetime, timedelta
from operator import itemgetter
import numpy as np
from utils import get_ec2_cpu_utilization, print_cpu_utilization
from utils import all_regions as regions

import os
import sys
which_account = sys.argv[1]

if which_account == '1':
    os.environ['AWS_PROFILE'] = 'maolinml'
elif which_account == '2':
    os.environ['AWS_PROFILE'] = 'maolinml2'
else:
    os.environ['AWS_PROFILE'] = 'maolinml3'


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
    
    if os.environ['AWS_PROFILE'] == 'maolinml':
        if region == "us-east-1":
            key_file = "../../../../Braket/keys/graphene.pem"
        elif region == "us-east-2":
            region2 = region.replace("-", "_")
            key_file = f"../../../../Braket/keys/graphene_{region2}.pem"
        elif region == "us-west-1":
            key_file = "../../../../Braket/keys/docker-west-1.pem"
        elif region == "us-west-2":
            key_file = "../../../../Braket/keys/trydocker.pem"
        elif region == "sa-east-1":
            key_file = f"../../../../Braket/keys/graphene-{region}.pem"    
        else:
            key_file = f"../../../../Braket/keys/graphene_{region}.pem"    
    else:
        key_file = f"../../../../Braket/keys/maolinml2/{region}.pem"

    if region == "us-east-1":
        ec2_addresses = [f"ec2-user@ec2-{public_ip}.compute-1.amazonaws.com:" for public_ip in public_ips]
    else:
        ec2_addresses = [f"ec2-user@ec2-{public_ip}.{region}.compute.amazonaws.com:" for public_ip in public_ips]
    
    period = 600
    for ip, instance_id, ec2_address in zip(public_ips, ids, ec2_addresses):
        
        end_time = datetime.now()
        # start_time = end_time - timedelta(hours=3) # two datapoints
        start_time = end_time - timedelta(minutes=5)

        # Get CPU utilization data
        cpu_utilization_data = get_ec2_cpu_utilization(region, instance_id, start_time, end_time, period=period)

        # Print CPU utilization data
        usage = print_cpu_utilization(cpu_utilization_data)
        
        if usage < 5:
            try:
                command = [
                    "scp", 
                    "-i", 
                    key_file,
                    f"{ec2_address}/home/ec2-user/graphene_*",
                    "."
                ]
                subprocess.run(command)
            except:
                print("download failed")
        
        print(instance_id, usage)

    print()