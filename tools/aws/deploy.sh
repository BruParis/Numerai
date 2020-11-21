#!/bin/bash

ADDR=$1
KEY_AWS=$2
NUM_ARG=$#

LOCAL_KEY_AWS=~/.ssh/$2

SCRIPT_AWS_PYTHON=aws_install_python.sh
SCRIPT_AWS_ENV=aws_env.sh
SCRIPT_AWS_EXECUTE=aws_execute.sh
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
    
    scp -i $LOCAL_KEY_AWS $SCRIPT_AWS_PYTHON ec2-user@$ADDR:~/
    scp -i $LOCAL_KEY_AWS $SCRIPT_AWS_ENV ec2-user@$ADDR:~/
    scp -i $LOCAL_KEY_AWS $SCRIPT_AWS_EXECUTE ec2-user@$ADDR:~/

    ssh -i $LOCAL_KEY_AWS ec2-user@$ADDR "
        sudo chmod +x '$SCRIPT_AWS_PYTHON';
        sudo ./'$SCRIPT_AWS_PYTHON'"

    ssh -i $LOCAL_KEY_AWS ec2-user@$ADDR "
        mv '$SCRIPT_AWS_ENV' Numerai/
        cd Numerai/
        sudo chmod +x '$SCRIPT_AWS_ENV';
        sudo ./'$SCRIPT_AWS_ENV'"
    
    ssh -i $LOCAL_KEY_AWS ec2-user@$ADDR "
        mv '$SCRIPT_AWS_EXECUTE' Numerai/
        cd Numerai/
        sudo chmod +x '$SCRIPT_AWS_EXECUTE';
        ./'$SCRIPT_AWS_EXECUTE'"

    ssh -i $LOCAL_KEY_AWS ec2-user@$ADDR

    exit 0
fi
