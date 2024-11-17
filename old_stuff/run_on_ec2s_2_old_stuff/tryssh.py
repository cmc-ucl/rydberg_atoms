import subprocess
subprocess.run(['ssh', '-i', '../../../../Braket/keys/graphene.pem', 'ec2-user@ec2-54-160-224-126.compute-1.amazonaws.com'])
subprocess.run(['mkdir', 'trytry3'])