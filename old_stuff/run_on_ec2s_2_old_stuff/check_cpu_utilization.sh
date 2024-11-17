#!/bin/bash

# Set your AWS region and instance ID
region="us-east-1"
instance_id="i-05c9d6a147272d7e5"

# Calculate the current timestamp and timestamp 1 hour ago
current_time=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
# one_hour_ago=$(date -u -d '1 hour ago' +'%Y-%m-%dT%H:%M:%SZ')

# Specify the metric name and namespace
metric_name="CPUUtilization"
namespace="AWS/EC2"

# Use the AWS CLI to get the CPU utilization metric statistics
aws cloudwatch get-metric-statistics \
    --region $region \
    --metric-name $metric_name \
    --namespace $namespace \
    --dimensions Name=InstanceId,Value=$instance_id \
    --start-time $current_time \
    --end-time $current_time \
    --period 300 \
    --statistics Average \
    --output json
