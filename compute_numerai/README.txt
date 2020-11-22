# clean data_cluster from useless files

# start pipenv
pipenv shell

# start docker daemon
sudo systemctl docker start

# setup compute node in AWS
sudo numerai setup

# build the docker container and deploys it to AWS
sudo numerai docker deploy

# trigger compute node in AWS
sudo numerai compute test-webhook

# logs
sudo numerai compute logs -f