#!/bin/bash
set -o xtrace
set -o nounset

ls
pwd
echo "docker is not installed - installing..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common wget
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io python3.8 python3.8-distutils
docker --version