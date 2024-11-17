import boto3
from datetime import datetime, timedelta
from operator import itemgetter
import numpy as np


def get_ec2_cpu_utilization(region, instance_id, start_time, end_time, period=600):
    """
    Fetches CPU utilization metrics for a given EC2 instance.
    
    :param instance_id: The ID of the EC2 instance.
    :param start_time: The start time of the period to fetch metrics (datetime object).
    :param end_time: The end time of the period to fetch metrics (datetime object).
    :return: List of CPU utilization datapoints.
    """
    cloudwatch_client = boto3.client('cloudwatch', region_name=region)

    # Define the metric to retrieve (Average CPU Utilization)
    metric_name = 'CPUUtilization'
    namespace = 'AWS/EC2'
    statistics = ['Average']
    # statistics = ['Maximum']

    # Convert start_time and end_time to UTC
    start_time = start_time.astimezone(datetime.now().astimezone().tzinfo)
    end_time = end_time.astimezone(datetime.now().astimezone().tzinfo)
    
    # Fetch metric data
    response = cloudwatch_client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=period,
        Statistics=statistics
    )

    return response['Datapoints']

def print_cpu_utilization(datapoints):
    """
    Prints CPU utilization datapoints.

    :param datapoints: List of CPU utilization datapoints.
    """
    # print("CPU Utilization:")

    datalist = []
    for datapoint in datapoints:
        # print(datapoint['Timestamp'], datapoint['Average'])
        timestamp = datapoint['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        average = datapoint['Average']
        datalist.append([timestamp, average])
        # print(f"  Timestamp: {timestamp}, Average CPU Utilization: {average}%")

    datalist.sort(key=itemgetter(0))

    usage = [item[1] for item in datalist]

    # print(np.average(usage))
    return np.average(usage)


import os
os.environ['AWS_PROFILE'] = 'maolinml'


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
    
    period = 600

    for instance_id in ids:
        
        end_time = datetime.now()
        # start_time = end_time - timedelta(hours=3)
        start_time = end_time - timedelta(minutes=10)

        # Get CPU utilization data
        cpu_utilization_data = get_ec2_cpu_utilization(region, instance_id, start_time, end_time, period=period)

        # Print CPU utilization data
        usage = print_cpu_utilization(cpu_utilization_data)
        print(instance_id, usage)

    print()

# if __name__ == '__main__':
#     # Replace 'your-instance-id' with your actual instance ID
#     instance_id = 'i-0a5adf680f5bb4eaa'

#     # Set the time range for which you want to fetch CPU utilization (last 1 hour)
#     end_time = datetime.now()
#     start_time = end_time - timedelta(hours=3)

#     # Get CPU utilization data
#     cpu_utilization_data = get_ec2_cpu_utilization(instance_id, start_time, end_time)

#     # Print CPU utilization data
#     print_cpu_utilization(cpu_utilization_data)
