import os
os.environ['AWS_PROFILE'] = 'maolinml2'

import boto3

regions = [
    # 'us-east-1'
    # 'us-east-2',
    # 'us-west-1',
    # 'us-west-2',

    'ap-south-1',
    'ap-northeast-3',
    'ap-northeast-2',
    'ap-northeast-1',
    'ap-southeast-2',
    'ap-southeast-1',
    'ca-central-1',
    'eu-central-1',
    'eu-west-1',
    'eu-west-2',
    'eu-west-3',
    'eu-north-1',
    'sa-east-1',
]

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
    ids = []
    for instance in instances:
        ids.append(instance.id)
        instance.terminate()

    print(ids)
