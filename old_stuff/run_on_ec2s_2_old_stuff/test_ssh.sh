
account="456882910219"
region="sa-east-1"
ip="15-229-85-86"

echo $account
echo $region
echo $ip

if [[ "$account" == "456882910219" ]]; then
   keyfile="../../../../Braket/keys/maolinml2/$region.pem"
else
    if [[ "$region" == "us-east-1" ]]; then
        keyfile="../../../../Braket/keys/graphene.pem"
    elif [[ "$region" == "us-east-2" ]]; then
        keyfile="../../../../Braket/keys/graphene_us_east_2.pem"
    elif [[ "$region" == "us-west-1" ]]; then
        keyfile="../../../../Braket/keys/docker-west-1.pem"
    elif [[ "$region" == "us-west-2" ]]; then
        keyfile="../../../../Braket/keys/trydocker.pem"
    elif [[ "$region" == "sa-east-1" ]]; then
        keyfile="../../../../Braket/keys/graphene-${region}.pem"
    else
        keyfile="../../../../Braket/keys/graphene_${region}.pem"
    fi
fi

echo $keyfile

if [[ "$region" == "us-east-1" ]]; then
   ec2_address="ec2-user@ec2-$ip.compute-1.amazonaws.com"
else
   ec2_address="ec2-user@ec2-$ip.$region.compute.amazonaws.com"
fi

echo $ec2_address

# ssh -T -i $keyfile $ec2_address mkdir abcde

command="
sudo yum -y install tmux;
source Graphene_venv/bin/activate;
pip install pyscf;
pip install geometric;
tmux new -d 'python3 batch_run.py';
"

ssh -T -i $keyfile $ec2_address $command