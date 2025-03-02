## This repo will use both LLM and RL agents to play games.

## Diambra Engine
For starting we will focus of Diambra Engine https://docs.diambra.ai
Below instructions are for ubuntu but in official docs we have instruction for all major OS
### Installation
1. Install desktop docker

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update



sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin


2. Download docker desktop from here https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-linux-amd64

3. Install docker desktop 
sudo apt-get update
sudo apt-get install ./docker-desktop-amd64.deb

4. Check installation

systemctl --user start docker-desktop


follow instructions here https://docs.diambra.ai/#installation
