import sys
import boto3

# Get the region
region = sys.argv[1]
print(sys.argv[0])


# Determine the files need to be run on each instance
folder = "test_set"
num_per_instance = 2

if region == "us-east-1":
    start_ind = 242
elif region == "us-east-2":
    # start_ind = 236
    start_ind = 140
elif region == "us-west-1":
    start_ind = 230
elif region == "us-west-2":
    # start_ind = 224
    start_ind = 134
elif region == "ap-south-1":
    start_ind = 218
elif region == "ca-central-1":
    start_ind = 212
elif region == "eu-north-1":
    start_ind = 206
elif region == "eu-west-2":
    start_ind = 200
elif region == "ap-northeast-3":
    start_ind = 194
elif region == "ap-northeast-2":
    start_ind = 188
elif region == "ap-northeast-1":
    start_ind = 182
elif region == "ap-southeast-2":
    start_ind = 176
elif region == "ap-southeast-1":
    start_ind = 170
elif region == "eu-central-1":
    start_ind = 164
elif region == "eu-west-1":
    start_ind = 158
elif region == "eu-west-3":
    start_ind = 152
elif region == "sa-east-1":
    start_ind = 146


# Get the instances in that region


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


if region == "us-east-1":
    public_ips = public_ips[1:]
    ids = ids[1:]
elif region == "us-east-2":
    public_ips = public_ips[:-1]
    ids = ids[:-1]

print(public_ips)
print(ids)
if region == "us-east-1":
    ec2_addresses = [f"ec2-user@ec2-{public_ip}.compute-1.amazonaws.com:./" for public_ip in public_ips]
else:
    ec2_addresses = [f"ec2-user@ec2-{public_ip}.{region}.compute.amazonaws.com:./" for public_ip in public_ips]

# elif region == "us-east-2":
#     ec2_addresses = [f"ec2-user@ec2-{public_ip}.us-east-2.compute.amazonaws.com:./" for public_ip in public_ips]
# elif region == "us-west-1":
#     ec2_addresses = [f"ec2-user@ec2-{public_ip}.us-west-1.compute.amazonaws.com:./" for public_ip in public_ips]


# Upload the files to the instances
import subprocess
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


print(key_file)
subprocess.run(['chmod', '400', key_file])

for (ind, ec2_address) in enumerate(ec2_addresses):
    for file_ind in range(start_ind + num_per_instance * ind, start_ind + num_per_instance * (ind+1)):
        file_name = f"../pyscf/{folder}/graphene_mol_r_6_6_h_{file_ind}.py"
        command = [
            "scp", 
            "-i", 
            key_file,
            file_name,
            ec2_address
        ]
        subprocess.run(command)

    # Upload the run-file that can run the simulations in parallel
    file_name = f"batch_run.py"
    command = [
        "scp", 
        "-i", 
        key_file,
        file_name,
        ec2_address
    ]
    subprocess.run(command)

# Extend the max time for ssm command
# The default max is 48 hours, we change it to 480 hours
# See here https://github.com/boto/boto3/issues/2558

import json
doc = {
  "schemaVersion": "1.2",
  "description": "Run a shell script or specify the commands to run.",
  "parameters": {
    "commands": {
      "type": "StringList",
      "description": "(Required) Specify a shell script or a command to run.",
      "minItems": 1,
      "displayType": "textarea"
    },
    "workingDirectory": {
      "type": "String",
      "default": "",
      "description": "(Optional) The path to the working directory on your instance.",
      "maxChars": 4096
    },
    "executionTimeout": {
      "type": "String",
      "default": "1728000",
      "description": "(Optional) The time in seconds for a command to complete before it is considered to have failed. Default is 3600 (1 hour). Maximum is 172800 (48 hours).",
      "allowedPattern": "([1-9][0-9]{0,4})|(1[0-6][0-9]{4})|(17[0-1][0-9]{3})|(172[0-7][0-9]{2})|(1728000)"
    }
  },
  "runtimeConfig": {
    "aws:runShellScript": {
      "properties": [
        {
          "id": "0.aws:runShellScript",
          "runCommand": "{{ commands }}",
          "workingDirectory": "{{ workingDirectory }}",
          "timeoutSeconds": "{{ executionTimeout }}"
        }
      ]
    }
  }
}

doc = json.dumps(doc)
doc_name = "AWS-RunShellScript" # "my_aws_run_shell_script_v2" # "AWS-RunShellScript"


# Create virtual env and install necessary dependencies and run
ssm_client = boto3.client('ssm', region_name=region)

try:
    ssm_client.describe_document(Name=doc_name)
except:
    ssm_client.create_document(Name=doc_name, Content=doc)

commands = [
    'cd ../../home/ec2-user',
    'python3 -m venv Graphene_venv',
    'source Graphene_venv/bin/activate',
    'pip install pyscf',
    'pip install geometric',
    'python3 batch_run.py',
]

response = ssm_client.send_command(
    InstanceIds=ids,
    DocumentName=doc_name,
    Parameters={
        'commands': commands, 
        'executionTimeout':["172800"]
        # 'executionTimeout':["100"]
        },
    TimeoutSeconds=300
)
print(response)
