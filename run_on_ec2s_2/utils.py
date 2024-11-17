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

# task_ids = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 90, 91, 92, 93, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 132, 133, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 191, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 228, 241]

# task_ids = [104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 132, 133, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 191, 196] # 104, 106, 107 maybe forced quit

# task_ids = [197, 198, 199, 200, 201]

# all_task_ids = [202]

# still need to run
# all_task_ids = [203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215]
                
# all_task_ids = [216, 217, 218, 219, ]
# all_task_ids = [220, 228, ]
# all_task_ids = [241, 104, 106, 2, 5, 6, 11, 12, 13, 16, 17, 19, 65, 97, 98, 99, 107, 117, 119, 157, 171, 202]

# all_task_ids = [91, 165, 206, 210, 213, 215]

# all_task_ids = [69, 218, 228]


# all_train_set_task_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]


# all_train_set_task_ids = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245]

# all_train_set_task_ids = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]

# all_train_set_task_ids = [66]

# all_train_set_task_ids = [92, 93, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

# [67, 68]

# all_train_set_task_ids = [69, 70, 71, 72, 73, 74, 75:q, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]

# all_train_set_task_ids = [89, 90, 91, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243]


# Still needs to run the following test sets
# all_task_ids = [105, 160]
all_train_set_task_ids = [49, 50, 51, 55, 62, 63, 64, 65, 66, 67, 68, 70, 97, 98, 99, 103, 106, 107, 110, 112, 113, 114, 115, 116, 117, 119, 120, 129, 134, 136, 142, 147, 148, 149, 150, 151, 153, 155, 156, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200, 202, 204, 207, 208, 209, 210, 211, 212, 213, 214, 215, 218, 219, 220, 221, 222, 223, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 245]

all_regions = [
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

imageids = {
    # 'us-east-1': 'c7i.48xlarge',
    'us-east-2': 'ami-0649bea3443ede307',
    # 'us-west-1': 'c7i.48xlarge',
    # 'us-west-2': 'c7i.48xlarge',
    'ap-south-1': 'ami-0ec0e125bb6c6e8ec',
    'ap-northeast-3': 'ami-0106c791cdf6b24fb',
    'ap-northeast-2': 'ami-04ea5b2d3c8ceccf8',
    'ap-northeast-1': 'ami-013a28d7c2ea10269',
    'ap-southeast-2': 'ami-030a5acd7c996ef60',
    'ap-southeast-1': 'ami-0e97ea97a2f374e3d',
    'ca-central-1': 'ami-0a69ba12b33eaa951',
    'eu-central-1': 'ami-0346fd83e3383dcb4',
    'eu-west-1': 'ami-0b995c42184e99f98',
    'eu-west-2': 'ami-026b2ae0ba2773e0a',
    'eu-west-3': 'ami-080fa3659564ffbb1',
    'eu-north-1': 'ami-0249211c9916306f8',
    'sa-east-1': 'ami-000aa26b054f3a383',
}

instancetypes = {
    'us-east-1': 'c7i.48xlarge',
    'us-east-2': 'c6a.48xlarge', #'c7i.48xlarge',
    'us-west-1': 'c7i.48xlarge',
    'us-west-2': 'c7i.48xlarge',
    'ap-south-1': 'c7i.48xlarge',
    'ap-northeast-3': 'c6i.32xlarge', # 128 cores
    'ap-northeast-2': 'c7i.48xlarge',
    'ap-northeast-1': 'c7i.48xlarge',
    'ap-southeast-2': 'c7i.48xlarge',
    'ap-southeast-1': 'c7i.48xlarge',
    'ca-central-1': 'c7i.48xlarge',
    'eu-central-1': 'c7i.48xlarge',
    'eu-west-1': 'c7i.48xlarge',
    'eu-west-2': 'c7i.48xlarge',
    'eu-west-3': 'c7i.48xlarge',
    'eu-north-1': 'c7i.48xlarge',
    'sa-east-1': 'c6a.48xlarge',
}