import boto3

import os
import sys
which_account = sys.argv[1]


if which_account == '1':
    os.environ['AWS_PROFILE'] = 'maolinml'
    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    boto3.setup_default_session(profile_name='maolinml')
elif which_account == '2':
    os.environ['AWS_PROFILE'] = 'maolinml2'
    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    boto3.setup_default_session(profile_name='maolinml2')
else:
    os.environ['AWS_PROFILE'] = 'maolinml3'
    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    boto3.setup_default_session(profile_name='maolinml3')

region = 'us-east-2'

ec2 = boto3.client('ec2', region_name=region)

# Specify the instance ID of the EC2 instance you want to terminate
instance_id = 'i-0882659c1b4de5260'

# Terminate the instance
try:
    response = ec2.terminate_instances(InstanceIds=[instance_id])
    print(f"Instance {instance_id} is terminating...")
    print(response)
except Exception as e:
    print(f"Error terminating instance {instance_id}: {str(e)}")
