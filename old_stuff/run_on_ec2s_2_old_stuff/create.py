import boto3

# from utils import imageids, instancetypes


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
imageids = {'us-east-2': 'ami-0649bea3443ede307'}
instancetypes = {'us-east-2': 't2.micro'}


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

EbsOptimized = False #True,

response = ec2.run_instances(
    ImageId=imageids[region],
    InstanceType=instancetypes[region],
    KeyName = region,
    MaxCount=1,
    MinCount=1,
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

print(response)

# {'Groups': [], 'Instances': [{'AmiLaunchIndex': 0, 'ImageId': 'ami-0649bea3443ede307', 'InstanceId': 'i-00fcb47ff97ec9a7f', 'InstanceType': 't2.micro', 'KeyName': 'us-east-2', 'LaunchTime': datetime.datetime(2024, 7, 22, 6, 22, 49, tzinfo=tzutc()), 'Monitoring': {'State': 'disabled'}, 'Placement': {'AvailabilityZone': 'us-east-2b', 'GroupName': '', 'Tenancy': 'default'}, 'PrivateDnsName': 'ip-172-31-29-215.us-east-2.compute.internal', 'PrivateIpAddress': '172.31.29.215', 'ProductCodes': [], 'PublicDnsName': '', 'State': {'Code': 0, 'Name': 'pending'}, 'StateTransitionReason': '', 'SubnetId': 'subnet-051334dff44d015fc', 'VpcId': 'vpc-0a1132926ae6f91e3', 'Architecture': 'x86_64', 'BlockDeviceMappings': [], 'ClientToken': 'faeb84d9-3ac0-4f7d-8cac-633c0ff14e92', 'EbsOptimized': False, 'EnaSupport': True, 'Hypervisor': 'xen', 'IamInstanceProfile': {'Arn': 'arn:aws:iam::456882910219:instance-profile/EnablesEC2ToAccessSystemsManagerRole', 'Id': 'AIPAWUYCX6QF636Z47YDZ'}, 'NetworkInterfaces': [{'Attachment': {'AttachTime': datetime.datetime(2024, 7, 22, 6, 22, 49, tzinfo=tzutc()), 'AttachmentId': 'eni-attach-06ffd35774bc506f0', 'DeleteOnTermination': True, 'DeviceIndex': 0, 'Status': 'attaching', 'NetworkCardIndex': 0}, 'Description': '', 'Groups': [{'GroupName': 'launch-wizard-us-east-2', 'GroupId': 'sg-0b442f2ba768ad4af'}], 'Ipv6Addresses': [], 'MacAddress': '06:37:20:a6:a1:9b', 'NetworkInterfaceId': 'eni-04e43af38ef51d683', 'OwnerId': '456882910219', 'PrivateDnsName': 'ip-172-31-29-215.us-east-2.compute.internal', 'PrivateIpAddress': '172.31.29.215', 'PrivateIpAddresses': [{'Primary': True, 'PrivateDnsName': 'ip-172-31-29-215.us-east-2.compute.internal', 'PrivateIpAddress': '172.31.29.215'}], 'SourceDestCheck': True, 'Status': 'in-use', 'SubnetId': 'subnet-051334dff44d015fc', 'VpcId': 'vpc-0a1132926ae6f91e3', 'InterfaceType': 'interface'}], 'RootDeviceName': '/dev/xvda', 'RootDeviceType': 'ebs', 'SecurityGroups': [{'GroupName': 'launch-wizard-us-east-2', 'GroupId': 'sg-0b442f2ba768ad4af'}], 'SourceDestCheck': True, 'StateReason': {'Code': 'pending', 'Message': 'pending'}, 'Tags': [{'Key': 'Name', 'Value': 'graphene'}], 'VirtualizationType': 'hvm', 'CpuOptions': {'CoreCount': 1, 'ThreadsPerCore': 1}, 'CapacityReservationSpecification': {'CapacityReservationPreference': 'open'}, 'MetadataOptions': {'State': 'pending', 'HttpTokens': 'required', 'HttpPutResponseHopLimit': 2, 'HttpEndpoint': 'enabled', 'HttpProtocolIpv6': 'disabled', 'InstanceMetadataTags': 'disabled'}, 'EnclaveOptions': {'Enabled': False}, 'BootMode': 'uefi-preferred', 'PrivateDnsNameOptions': {'HostnameType': 'ip-name', 'EnableResourceNameDnsARecord': True, 'EnableResourceNameDnsAAAARecord': False}, 'MaintenanceOptions': {'AutoRecovery': 'default'}, 'CurrentInstanceBootMode': 'legacy-bios'}], 'OwnerId': '456882910219', 'ReservationId': 'r-0bea119ebeba3a0e0', 'ResponseMetadata': {'RequestId': '87468287-0461-431d-b649-c4bc162c4b20', 'HTTPStatusCode': 200, 'HTTPHeaders': {'x-amzn-requestid': '87468287-0461-431d-b649-c4bc162c4b20', 'cache-control': 'no-cache, no-store', 'strict-transport-security': 'max-age=31536000; includeSubDomains', 'vary': 'accept-encoding', 'content-type': 'text/xml;charset=UTF-8', 'content-length': '6097', 'date': 'Mon, 22 Jul 2024 06:22:49 GMT', 'server': 'AmazonEC2'}, 'RetryAttempts': 0}}