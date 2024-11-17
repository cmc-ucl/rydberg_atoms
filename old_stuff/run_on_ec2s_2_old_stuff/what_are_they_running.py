import boto3
from datetime import datetime

from utils import all_regions as regions
import subprocess

import os
log_file = "logfiles/what_are_they_running.log"


while True:
    for which_account in [1,2,3]:
        if which_account == 1:
            os.environ['AWS_PROFILE'] = 'maolinml'
            boto3.setup_default_session(profile_name='maolinml')
        elif which_account == 2:
            os.environ['AWS_PROFILE'] = 'maolinml2'
            boto3.setup_default_session(profile_name='maolinml2')
        elif which_account == 3:
            os.environ['AWS_PROFILE'] = 'maolinml3'
            boto3.setup_default_session(profile_name='maolinml3')

        log = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.environ['AWS_PROFILE']
        print(log)
        with open(log_file, "a") as f:
            print(log, file=f)

        for region in regions:
            print(region)
            with open(log_file, "a") as f:
                print(region, file=f)

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
                key_file = f"../../../../Braket/keys/{os.environ['AWS_PROFILE']}/{region}.pem"

            subprocess.run(['chmod', '400', key_file])

            if region == "us-east-1":
                ec2_addresses = [f"ec2-user@ec2-{public_ip}.compute-1.amazonaws.com" for public_ip in public_ips]
            else:
                ec2_addresses = [f"ec2-user@ec2-{public_ip}.{region}.compute.amazonaws.com" for public_ip in public_ips]        

            for ip, instance_id, ec2_address in zip(public_ips, ids, ec2_addresses):
                
                command = [
                    'ssh',
                    '-T',
                    '-i',
                    key_file,
                    ec2_address,
                    "ls -lt > logfile"
                ]
                subprocess.run(command)
                
                command = [
                    "scp", 
                    "-i", 
                    key_file,
                    f"{ec2_address}:/home/ec2-user/logfile",
                    f"logfiles/{os.environ['AWS_PROFILE']}/{region}-{ip}"
                ]
                subprocess.run(command)                


