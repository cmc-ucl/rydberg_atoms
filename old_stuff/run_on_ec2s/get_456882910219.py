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

    key_file = f"../../../../Braket/keys/maolinml2/{region}.pem"

    # Note the ec2_addresses are different from those in the run file
    if region == "us-east-1":
        ec2_addresses = [f"ec2-user@ec2-{public_ip}.compute-1.amazonaws.com:" for public_ip in public_ips]
    else:
        ec2_addresses = [f"ec2-user@ec2-{public_ip}.{region}.compute.amazonaws.com:" for public_ip in public_ips]        


    ssm_client = boto3.client('ssm', region_name=region)
    
    response = ssm_client.list_commands()
    command = response['Commands'][0]
    # print(command)
    command_id = command['CommandId']

    for (instance_id, ec2_address, public_ip) in zip(ids, ec2_addresses, public_ips):
        response = ssm_client.list_command_invocations(
            CommandId = command_id,
            InstanceId = instance_id
            )
        command = response['CommandInvocations'][0]
        status = command['Status']
        print(instance_id, status, public_ip)

        # command = [
        #     "scp", 
        #     "-i", 
        #     key_file,
        #     f"{ec2_address}/home/ec2-user/graphene_*",
        #     "all_tasks"
        # ]
        # subprocess.run(command)

        if status == "Success":
            # print(ec2_address)
            command = [
                "scp", 
                "-i", 
                key_file,
                f"{ec2_address}/home/ec2-user/graphene_*.out",
                "."
            ]
            subprocess.run(command)
        # elif status in ["TimeOut", "Failed"]:
        #     command = [
        #         "scp", 
        #         "-i", 
        #         key_file,
        #         f"{ec2_address}/home/ec2-user/graphene_*",
        #         "failed_tasks"
        #     ]
        #     subprocess.run(command)

    print()
