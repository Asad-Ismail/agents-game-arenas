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

   ## To run diambra docker and run just script for debugging
   docker run -d --rm --name engine   -v $HOME/.diambra/credentials:/tmp/.diambra/credentials   -v /home/asad/dev/agents-game-arenas/roms/:/opt/diambraArena/roms   -p 127.0.0.1:50051:50051 docker.io/diambra/engine:latest
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


## Running the Script

You can run the script in one of two ways:

### Option 1

You can runt the game easily using 

```bash
cd scripts
diambra run -r python custom_tekken_redering.py
```

### Option 2 for debugging/more fine grain control

```bash
#Run diambra arean docker
docker run -d --rm --name engine   -v $HOME/.diambra/credentials:/tmp/.diambra/credentials   -v /home/asad/dev/agents-game-arenas/roms:/opt/diambraArena/roms   -p 127.0.0.1:50051:50051 docker.io/diambra/engine:latest

#Run the script

DIAMBRA_ENVS=localhost:50051 python ./custom_tekken_redering.py
```


## Game Controls: Tekken Tag Tournament

Tekken Tag Tournament uses a `MULTI_DISCRETE` action space which takes an array of two values:

```
[movement_action, attack_action]
```

### Movement Actions (0-8)

| Code | Action    | Description             |
|------|-----------|-------------------------|
| 0    | NoMove    | Neutral stance/position |
| 1    | Left      | Move left               |
| 2    | UpLeft    | Move diagonally up-left |
| 3    | Up        | Move up/jump            |
| 4    | UpRight   | Move diagonally up-right|
| 5    | Right     | Move right              |
| 6    | DownRight | Move diagonally down-right |
| 7    | Down      | Move down/crouch        |
| 8    | DownLeft  | Move diagonally down-left |

### Attack Actions (0-12)

| Code | Action Type        | Description                   |
|------|-------------------|-------------------------------|
| 0    | No Attack         | No button pressed             |
| 1    | Left Punch (LP)   | Single button - light punch   |
| 2    | Right Punch (RP)  | Single button - heavy punch   |
| 3    | Left Kick (LK)    | Single button - light kick    |
| 4    | Right Kick (RK)   | Single button - heavy kick    |
| 5-12 | Button Combinations | Various combined button presses |

### Examples

```python
# Stand still (no movement) and do nothing
action = [0, 0]

# Move right while performing a left punch
action = [5, 1]

# Jump and execute a button combination
action = [3, 6]
```


### Install OLLAMA to run LLM locally

```python
curl -fsSL https://ollama.com/install.sh | sh
## Pull required model
ollama pull llava
```