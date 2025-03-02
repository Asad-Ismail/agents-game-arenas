# LLM and RL Game Playing Agents

This repository combines LLM and RL agents to play various games.

## Diambra Engine Setup

We will initially focus on using the [Diambra Engine](https://docs.diambra.ai). While these instructions are for Ubuntu, you can find setup guides for all major operating systems in the [official documentation](https://docs.diambra.ai).

### Installation Steps

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

For further setup instructions, please refer to the [Diambra documentation](https://docs.diambra.ai/#installation).
