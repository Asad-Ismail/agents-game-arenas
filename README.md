# LLM and RL Game Playing Agents

This repository combines LLM and RL agents to play various games. 
This repo is currently under development and will be updated as we add more features and games.

## Diambra Engine Setup

We will initially focus on using the [Diambra Engine](https://docs.diambra.ai). While these instructions are for Ubuntu, you can find setup guides for all major operating systems in the [official documentation](https://docs.diambra.ai).

### Installation Steps

First we need to install Docker Desktop so we can run the Diambra Engine and games insdie the container.
#### Install Docker Desktop
1. Install Docker Desktop
   ```bash
   # Add Docker's official GPG key
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

   # Add the repository to Apt sources
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update

   # Install Docker packages
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

2. Download [Docker Desktop](https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-linux-amd64)

3. Install Docker Desktop
   ```bash
   sudo apt-get update
   sudo apt-get install ./docker-desktop-amd64.deb
   ```

4. Start Docker Desktop
   ```bash
   systemctl --user start docker-desktop
   ```

Optional: (fix the docker permission if permission error is thrown)

   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

### Install Diambra CLI and Arena
For further setup instructions, please refer to the [Diambra documentation](https://docs.diambra.ai/#installation).

Install diambra cli and arena

```bash
python3 -m pip install diambra
python3 -m pip install diambra-arena
```

### Download games

Put games roms in the `roms` directory, tekken tag for example should be in `roms/tektagt.zip` (**Important rename the file to `tektagt.zip`**) you can download it from [here](https://wowroms.com/en/roms/mame/tekken-tag-tournament-asia-clone/108661.html)

### Check game

```bash
diambra arena check-roms /absolute/path/to/roms/folder/romFileName.zip
```

### Add Rom directory to bashrc

```bash
echo "export DIAMBRAROMSPATH=/absolute/path/to/roms/folder" >> ~/.bashrc
```



### 

Tekken Tag Tournament with SpaceTypes.MULTI_DISCRETE, it returns an array with two random values, Actions of Tekken tag are 

[random_move, random_attack]

Where:

random_move is a random number from 0-8 (representing directional movements)
random_attack is a random number from 0-12 (representing attack buttons/combinations)