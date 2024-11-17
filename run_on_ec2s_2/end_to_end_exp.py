# determine account and region
import boto3

# from utils import imageids, instancetypes


import os
import sys
# which_account = sys.argv[1]
which_account = 2

if which_account == 1:
    os.environ['AWS_PROFILE'] = 'maolinml'
    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    boto3.setup_default_session(profile_name='maolinml')
elif which_account == 2:
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

# create ec2
imageids = {'us-east-2': 'ami-0649bea3443ede307'}
instancetypes = {'us-east-2': 't2.micro'}

num_instances = 2
EbsOptimized = False #True,

ec2 = boto3.client('ec2', region_name=region)

try:
    response = ec2.create_security_group(
        Description=f'launch-wizard-{region}',
        GroupName=f'launch-wizard-{region}',
    )
    security_group_id = response['GroupId']
except:
    security_groups = ec2.describe_security_groups(
        Filters=[
            {
                'Name': 'group-name',
                'Values': [f"launch-wizard-{region}"]
            }
        ]
    )
    security_group = security_groups['SecurityGroups'][0]
    security_group_id = security_group['GroupId']


print(security_group_id)

try:
    response = ec2.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions = [
            {
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [
                {
                "CidrIp": "0.0.0.0/0"
                }
            ]
            }
        ]
    )
    print(response)
except:
    pass



response = ec2.run_instances(
    ImageId=imageids[region],
    InstanceType=instancetypes[region],
    KeyName = region,
    MaxCount=num_instances,
    MinCount=num_instances,
    EbsOptimized = EbsOptimized,
    NetworkInterfaces = [
        {
        "AssociatePublicIpAddress": True,
        "DeviceIndex": 0,
        "Groups": [security_group_id]
        }
    ],
    TagSpecifications = [
        {
        "ResourceType": "instance",
        "Tags": [
            {
            "Key": "Name",
            "Value": "graphene"
            }
        ]
        }
    ],
    IamInstanceProfile = {
        f"Arn": f"arn:aws:iam::{account}:instance-profile/EnablesEC2ToAccessSystemsManagerRole"
    },
    MetadataOptions = {
        "HttpEndpoint": "enabled",
        "HttpPutResponseHopLimit": 2,
        "HttpTokens": "required"
    },
    PrivateDnsNameOptions = {
        "HostnameType": "ip-name",
        "EnableResourceNameDnsARecord": True,
        "EnableResourceNameDnsAAAARecord": False
    }
)

instance_ids = [instance['InstanceId'] for instance in response['Instances']]

response2 = ec2.describe_instances(InstanceIds=instance_ids)

ec2_addresses = [f"ec2-user@{Instance['PublicDnsName']}" for Instance in response2['Reservations'][0]['Instances']]

public_ips = [Instance['NetworkInterfaces'][0]['Association']['PublicIp'] for Instance in response2['Reservations'][0]['Instances']]

public_ips = [ip.replace(".", "-") for ip in public_ips]

print(instance_ids)
print(ec2_addresses)
print(public_ips)

# run a command to a file
import subprocess
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

for ip, instance_id, ec2_address in zip(public_ips, instance_ids, ec2_addresses):
    command = [
        'ssh',
        '-T',
        '-i',
        key_file,
        "-o",
        "StrictHostKeyChecking=no",
        ec2_address,
        f"echo {ec2_address} > log-{region}-{ip}"
    ]
    subprocess.run(command)

# download the file
for ip, instance_id, ec2_address in zip(public_ips, instance_ids, ec2_addresses):
    command = [
        "scp", 
        "-i", 
        key_file,
        f"{ec2_address}:/home/ec2-user/*",
        "."
    ]
    subprocess.run(command)


# terminate the instance
try:
    response = ec2.terminate_instances(InstanceIds=instance_ids)
    print(f"Instance {instance_id} is terminating...")
    print(response)
except Exception as e:
    print(f"Error terminating instance {instance_id}: {str(e)}")

