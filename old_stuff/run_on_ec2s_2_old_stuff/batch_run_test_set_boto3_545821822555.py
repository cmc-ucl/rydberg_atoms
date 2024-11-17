import subprocess
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
k = 0

task_ids = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 90, 91, 92, 93, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 132, 133, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 191, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 228, 241]

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

    for ip, instance_id in zip(public_ips, ids):
        
        end_time = datetime.now()
        # start_time = end_time - timedelta(seconds=2 * period) # two datapoints
        start_time = end_time - timedelta(hours=3) # two datapoints

        # Get CPU utilization data
        cpu_utilization_data = get_ec2_cpu_utilization(region, instance_id, start_time, end_time, period=period)

        # Print CPU utilization data
        usage = print_cpu_utilization(cpu_utilization_data)
        
        if usage < 50:
            try:
                subprocess.run(['sh', 'run_test_set_545821822555.sh', f"{region}", f"{ip}", f"{task_ids[k]}"], stdout=subprocess.DEVNULL)
                print(instance_id, usage, k)
                k += 1
            except:
                print("submission failed")
        else:
            print(instance_id, usage)

    print()