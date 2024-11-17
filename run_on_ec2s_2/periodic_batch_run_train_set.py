import boto3
from datetime import datetime, timedelta
import time

from utils import get_ec2_cpu_utilization, print_cpu_utilization

from utils import all_regions as regions
from utils import all_train_set_task_ids as task_ids
import subprocess

import os
log_file = "periodic_run.log"


k = 0
usage_threashold = 5 # 5
minutes = 5

while k < len(task_ids):
    for which_account in [1,2,3]:
        if which_account == 1:
            os.environ['AWS_PROFILE'] = 'maolinml'
            boto3.setup_default_session(profile_name='maolinml')
        elif which_account == 2:
            os.environ['AWS_PROFILE'] = 'maolinml2'
            boto3.setup_default_session(profile_name='maolinml2')
        elif which_account == 3:
            os.environ['AWS_PROFILE'] = 'maolinml3'
            boto3.setup_default_session(profile_name='maolinml3')

        log = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.environ['AWS_PROFILE']
        print(log)
        with open(log_file, "a") as f:
            print(log, file=f)

        for region in regions:
            print(region)
            with open(log_file, "a") as f:
                print(region, file=f)

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

            subprocess.run(['chmod', '400', key_file])

            if region == "us-east-1":
                ec2_addresses = [f"ec2-user@ec2-{public_ip}.compute-1.amazonaws.com" for public_ip in public_ips]
            else:
                ec2_addresses = [f"ec2-user@ec2-{public_ip}.{region}.compute.amazonaws.com" for public_ip in public_ips]        

            for ip, instance_id, ec2_address in zip(public_ips, ids, ec2_addresses):
                
                end_time = datetime.now()
                # start_time = end_time - timedelta(hours=3)
                start_time = end_time - timedelta(minutes=minutes)

                # Get CPU utilization data
                cpu_utilization_data = get_ec2_cpu_utilization(region, instance_id, start_time, end_time, period=period)

                # Print CPU utilization data
                usage = print_cpu_utilization(cpu_utilization_data)

                if usage < usage_threashold:
                    # download
                    command = [
                        "scp", 
                        "-i", 
                        key_file,
                        f"{ec2_address}:/home/ec2-user/*graphene*",
                        "."
                    ]
                    subprocess.run(command)
                    
                    # remove the old files
                    command = [
                        'ssh',
                        '-T',
                        '-i',
                        key_file,
                        ec2_address,
                        "sudo rm -f -r *"
                    ]
                    # "sudo rm -f -r Graphene_venv; sudo rm *.py"
                    subprocess.run(command)

                    # upload new file
                    subprocess.run(
                        [
                            "scp", 
                            "-i", 
                            key_file,
                            f"../train_set_2/train_set_graphene_mol_r_6_6_h_{task_ids[k]}.py",
                            f"{ec2_address}:./"
                        ]
                    )
                    subprocess.run(
                        [
                            "scp", 
                            "-i", 
                            key_file,
                            "batch_run.py",
                            f"{ec2_address}:./"
                        ]
                    )

                    # run
                    ssh_command = f"""
                        python3 -m venv Graphene_venv;
                        sudo yum -y install tmux;
                        source Graphene_venv/bin/activate;
                        pip install pyscf;
                        pip install geometric;
                        nohup tmux new -d 'python3 batch_run.py' \; pipe-pane 'cat > train_set_graphene_mol_r_6_6_h_{task_ids[k]}.log'
                    """
                        # tmux new -d 'python3 batch_run.py';
                        # nohup tmux new -d 'mkdir abcd; echo q2421432' \; pipe-pane 'cat > train_set_graphene_mol_r_6_6_h_{task_ids[k]}.log'
                    command = [
                        'ssh',
                        '-T',
                        '-i',
                        key_file,
                        ec2_address,
                        f"{ssh_command}"
                    ]
                    subprocess.run(command)
                    
                    # print
                    print(instance_id, usage, k, task_ids[k])
                    with open(log_file, "a") as f:
                        print(instance_id, usage, k, task_ids[k], file=f)
                    k += 1
                else:
                    print(instance_id, usage)
                    with open(log_file, "a") as f:
                        print(instance_id, usage, file=f)

            print()
            with open(log_file, "a") as f:
                print(file=f)

    time.sleep(3600) # sleep for an hour

