region=$1
ip=$2
task_id=$3

echo $region
echo $ip
echo $task_id

keyfile="../../../../Braket/keys/maolinml2/$region.pem"

if [[ "$region" == "us-east-1" ]]; then
   ec2_address="ec2-user@ec2-$ip.compute-1.amazonaws.com"
else
   ec2_address="ec2-user@ec2-$ip.$region.compute.amazonaws.com"
fi

echo $ec2_address

ssh -i $keyfile $ec2_address << EOF
    rm *.py
    rm *.xyz
    rm *.json
    rm *.out
    exit
EOF

filename="../pyscf/test_set/graphene_mol_r_6_6_h_$task_id.py"
scp -i $keyfile $filename $ec2_address:./

scp -i $keyfile batch_run.py $ec2_address:./

ssh -i $keyfile $ec2_address << EOF
    sudo yum -y install tmux
    source Graphene_venv/bin/activate
    pip install pyscf
    pip install geometric
    tmux new -d 'python3 batch_run.py'
    exit
EOF