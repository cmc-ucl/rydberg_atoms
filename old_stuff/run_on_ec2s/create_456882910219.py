# import os
# os.environ['AWS_PROFILE'] = 'maolinml2'

# import boto3

# regions = [
#     # 'us-east-1'
#     # 'us-east-2',
#     # 'us-west-1',
#     # 'us-west-2',
#     'ap-south-1',
#     # 'ap-northeast-3',
#     # 'ap-northeast-2',
#     # 'ap-northeast-1',
#     # 'ap-southeast-2',
#     # 'ap-southeast-1',
#     # 'ca-central-1',
#     # 'eu-central-1',
#     # 'eu-west-1',
#     # 'eu-west-2',
#     # 'eu-west-3',
#     # 'eu-north-1',
#     # 'sa-east-1',
# ]

# imageids = {
#     # 'us-east-1': 'c7i.48xlarge',
#     # 'us-east-2': 'c7i.48xlarge',
#     # 'us-west-1': 'c7i.48xlarge',
#     # 'us-west-2': 'c7i.48xlarge',
#     'ap-south-1': 'ami-0ec0e125bb6c6e8ec',
#     'ap-northeast-3': 'ami-0106c791cdf6b24fb',
#     'ap-northeast-2': 'ami-04ea5b2d3c8ceccf8',
#     'ap-northeast-1': 'ami-013a28d7c2ea10269',
#     'ap-southeast-2': 'ami-030a5acd7c996ef60',
#     'ap-southeast-1': 'ami-0e97ea97a2f374e3d',
#     'ca-central-1': 'ami-0a69ba12b33eaa951',
#     'eu-central-1': 'ami-0346fd83e3383dcb4',
#     'eu-west-1': 'ami-0b995c42184e99f98',
#     'eu-west-2': 'ami-026b2ae0ba2773e0a',
#     'eu-west-3': 'ami-080fa3659564ffbb1',
#     'eu-north-1': 'ami-0249211c9916306f8',
#     'sa-east-1': 'ami-000aa26b054f3a383',
# }
# instancetypes = {
#     'us-east-1': 'c7i.48xlarge',
#     'us-east-2': 'c7i.48xlarge',
#     'us-west-1': 'c7i.48xlarge',
#     'us-west-2': 'c7i.48xlarge',
#     'ap-south-1': 'c7i.48xlarge',
#     'ap-northeast-3': 'c6i.32xlarge', # 128 cores
#     'ap-northeast-2': 'c7i.48xlarge',
#     'ap-northeast-1': 'c7i.48xlarge',
#     'ap-southeast-2': 'c7i.48xlarge',
#     'ap-southeast-1': 'c7i.48xlarge',
#     'ca-central-1': 'c7i.48xlarge',
#     'eu-central-1': 'c7i.48xlarge',
#     'eu-west-1': 'c7i.48xlarge',
#     'eu-west-2': 'c7i.48xlarge',
#     'eu-west-3': 'c7i.48xlarge',
#     'eu-north-1': 'c7i.48xlarge',
#     'sa-east-1': 'c6a.48xlarge',
# }

# for region in regions:
#     print(region)
#     ec2 = boto3.resource('ec2', region_name=region)

#     try:
#         instances = ec2.create_instances(
#             BlockDeviceMappings=[
#                 {
#                     'DeviceName': '/dev/xvda',
#                     # 'VirtualName': 'string',
#                     'Ebs': {
#                         'DeleteOnTermination': True,
#                         # 'Iops': 123,
#                         # 'SnapshotId': 'string',
#                         'VolumeSize': 8,
#                         'VolumeType': 'gp3',
#                         # 'KmsKeyId': 'string',
#                         # 'Throughput': 123,
#                         # 'OutpostArn': 'string',
#                         # 'Encrypted': True|False
#                     },
#                     # 'NoDevice': 'string'
#                 },
#             ],        
#             ImageId=imageids[region],
#             MinCount=3,
#             MaxCount=3,
#             InstanceType=instancetypes[region],
#             IamInstanceProfile={
#                 # 'Arn': 'arn:aws:iam::456882910219:role/EnablesEC2ToAccessSystemsManagerRole',
#                 'Name': 'EnablesEC2ToAccessSystemsManagerRole'
#             },
#             KeyName=f"{region}",
#             SecurityGroups = ['launch-wizard-4']
#         )
#         print(instances[0].id)
#     except Exception as e:
#         print(e)