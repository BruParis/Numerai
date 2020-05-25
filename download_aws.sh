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

    mkdir aws_files

    cat aws_download_files.txt | xargs -i{} scp -r -i $LOCAL_KEY_AWS ec2-user@$ADDR:~/$AWS_FOLDER/{} aws_files/

    exit 0
fi
