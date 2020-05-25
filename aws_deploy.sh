#!/bin/bash
cd /opt/
yum install gcc gcc-c++ openssl-devel bzip2-devel libffi-devel -y
wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz
tar -xvf Python-3.8.2.tgz
cd Python-3.8.2
./configure --enable-optimizations
make altinstall
rm -f /opt/Python-3.8.2.tgz

cd /home/ec2-user/Numerai/

yum install python-pip -y

sudo -u ec2-user pip install --user pipenv
pipenv --python 3.8
pipenv install

