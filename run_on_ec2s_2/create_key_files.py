import boto3
from botocore.exceptions import ClientError

def create_ec2_key_pair(region):
    key_name = region
    # Create EC2 client
    ec2 = boto3.client('ec2', region_name=region)

    try:
        # Create a new EC2 key pair
        response = ec2.create_key_pair(KeyName=key_name)
        
        # Extract the key material (private key)
        key_material = response['KeyMaterial']
        
        # Save the key material to a .pem file
        with open(f'{key_name}.pem', 'w') as key_file:
            key_file.write(key_material)
        
        # Set permissions to the file (optional but recommended)
        import os
        os.chmod(f'{key_name}.pem', 0o400)  # Read-only for the owner

        print(f"Key pair created and saved to {key_name}.pem")

    except ClientError as e:
        print("Error creating key pair:", e)

# Example usage
# create_ec2_key_pair('my-key-pair')

# create_ec2_key_pair("us-east-1")

all_regions = [
    # "us-east-1",
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

for region in all_regions:
    create_ec2_key_pair(region)
