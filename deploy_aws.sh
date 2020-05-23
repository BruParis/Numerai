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
pip install --user pipenv

pipenv --python 3.8
pipenv install
pipenv shell
pip install --no-cache-dir tensorflow

python feature_era_corr_split.py
python era_ft_graph.py
python data_subsets_format.py
python generate_models.py
python model_predict_tournament.py
