import boto3

regions = [
    'ap-northeast-3',
    # 'ap-northeast-2',
    # 'ap-northeast-1',
    # 'ap-southeast-2',
    # 'ap-southeast-1',
    # 'eu-central-1',
    # 'eu-west-1',
    # 'eu-west-2',
    # 'eu-west-3',
    # 'eu-north-1',
    # 'sa-east-1',
]

for region in regions:
    ec2 = boto3.resource('ec2', region_name=region)
    print(ec2.meta.client.describe_instances())
    # print(ec2.meta.__dict__)


    # ec2.meta.client.start_instances(
    #     InstanceIds = ['i-123'],
    #     DryRun=True
    # )
