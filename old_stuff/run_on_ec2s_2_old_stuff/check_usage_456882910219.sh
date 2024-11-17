region=$1
ip=$2

keyfile="../../../../Braket/keys/maolinml2/$region.pem"

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

logfilename="usage_456882910219/${region}_$ip"
scp -i $keyfile $ec2_address:/home/ec2-user/logfile $logfilename