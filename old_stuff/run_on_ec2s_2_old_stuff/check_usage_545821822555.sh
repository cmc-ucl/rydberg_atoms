region=$1
ip=$2


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

# keyfile="../../../../Braket/keys/$region.pem"

if [[ "$region" == "us-east-1" ]]; then
   ec2_address="ec2-user@ec2-$ip.compute-1.amazonaws.com"
else
   ec2_address="ec2-user@ec2-$ip.$region.compute.amazonaws.com"
fi

ssh -i $keyfile $ec2_address << EOF
 mpstat 1 1 | grep "all" | awk '{ print 100 - \$NF; exit; }' > logfile
 exit
EOF

# ssh -i $keyfile $ec2_address << EOF
#  ls -lt > logfile
#  exit
# EOF

logfilename="usage_545821822555/${region}_$ip"
scp -i $keyfile $ec2_address:/home/ec2-user/logfile $logfilename