#!/bin/bash
set -o xtrace
set -o nounset

ls
pwd

INSTALL_MARKER_FILE=$AZ_BATCH_NODE_SHARED_DIR/rlbakeoff_setup_done12
if [ ! -f $INSTALL_MARKER_FILE ]; then
  set -e

  # One time workaround...
  curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

  sudo apt-get update

  echo "docker is not installed - installing..."
  sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common wget
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
  sudo add-apt-repository -y ppa:deadsnakes/ppa

  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io python3.8 python3.8-distutils

  touch $INSTALL_MARKER_FILE
  set +e
fi

docker --version
sudo docker login perrlbousweppeacr.azurecr.io --username perrlbousweppeacr --password $ppid
sudo docker pull perrlbousweppeacr.azurecr.io/personalization-rlbakeoff-experiment-vw:8.9.0