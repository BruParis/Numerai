#!/bin/bash

ADDR=$1
KEY_AWS=$2
NUM_ARG=$#

LOCAL_KEY_AWS=~/.ssh/$2

SCRIPT_AWS=deploy_aws.sh
AWS_FOLDER=Numerai

print_usage() {
    echo "deploy_local.sh ADDR KEY_AWS"
    exit 1;
}

script_not_found() {
    echo "script deploy_remote.sh not found"
    exit 1;
}

if [ $NUM_ARG -ne 2 ]
then
    print_usage
elif [ ! -f $LOCAL_FILE ]
then
    script_not_found
else
    eval "$(ssh-agent -s)"
    echo "LOCAL_KEY_AWS: " $LOCAL_KEY_AWS

    ssh -i $LOCAL_KEY_AWS ec2-user@$ADDR 'mkdir' $AWS_FOLDER

    cat aws_upload_files.txt | xargs -i{} scp -i $LOCAL_KEY_AWS {} ec2-user@$ADDR:~/$AWS_FOLDER
    scp -i $LOCAL_KEY_AWS 'deploy_aws.sh' ec2-user@$ADDR:~/

    ssh -i $LOCAL_KEY_AWS ec2-user@$ADDR "
        sudo chmod +x '$SCRIPT_AWS';
        sudo ./deploy_aws.sh
    "
    ssh -i $LOCAL_KEY_AWS ec2-user@$ADDR

    exit 0
fi
