import boto3
from datetime import datetime, timedelta

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

# print(regions)
# print(os.environ['AWS_PROFILE'])

minutes = 30
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
    ids = [instance.id for instance in instances]
    period = 600

    for instance_id in ids:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)

        # Get CPU utilization data
        cpu_utilization_data = get_ec2_cpu_utilization(region, instance_id, start_time, end_time, period=period)

        # Print CPU utilization data
        usage = print_cpu_utilization(cpu_utilization_data)
        print(instance_id, usage)

    print()

