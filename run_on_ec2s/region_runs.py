import subprocess

regions = [
    'eu-west-2', 'ap-northeast-3', 'ap-southeast-1', 'eu-central-1', 'eu-west-1', 'eu-west-3'
]

for region in regions:
    subprocess.run(['python', 'run_545821822555.py', region])