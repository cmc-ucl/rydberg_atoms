import os
os.environ['AWS_PROFILE'] = 'maolinml2'


regions = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ap-south-1",
    "ca-central-1",
    "eu-north-1",
    "eu-west-2",
    "ap-northeast-3",
    "ap-northeast-2",
    "ap-northeast-1",
    "ap-southeast-2",
    "ap-southeast-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-3",
    "sa-east-1",
]

import subprocess
import boto3
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


    for ip in public_ips:
        subprocess.run(['sh', 'check_usage_456882910219.sh', f"{region}", f"{ip}"], stdout=subprocess.DEVNULL)
        with open(f"usage_456882910219/{region}_{ip}") as my_file:
            usage = my_file.read()
        
        print(region, ip, usage)
