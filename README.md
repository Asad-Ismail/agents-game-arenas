# 🎮 LLM and RL Game Playing Agents
This repository shows playing video games using both Reinforcement Learning (RL) and Large Language Models (LLMs) agents. 


### PPO (Reinforcement Learning) vs CPU

Better quality videos can be viewed in videos 

Lets see some fights of these agents

<div align="center">
  <h3>PPO vs CPU</h3>
  <img src="videos/ppo_224.gif" alt="PPO agent fighting" >
</div>

### LLMs vs CPU

We can run different language models either locally using Ollama or via cloud services.
<div align="center">
  <h3>Llama 3.2 1B vs CPU</h3>
  <img src="videos/llama3_224.gif" alt="Llama3 agent fighting">
  <h3>Qwen 0.5B vs CPU</h3>
  <img src="videos/qwen_224.gif" alt="Qwen agent fighting" >
</div>

### 🚀 Features

Multiple AI Approaches: Compare RL and LLM-based gaming agents

Variety of Models: Easily switch between different open-weight LLMs

Diambra Integration: Built on the Diambra game engine for realistic fighting games

Customizable: Adjust parameters to experiment with agent behavior

### 🔧 Quick Start

```bash
# Clone the repository
git clone https://github.com/Asad-Ismail/AGENTS_GAME-ARENAS.git

cd AGENTS_GAME-ARENAS
```



## Setup 


For detailed setup instructions including environment configuration, ROM installation, and troubleshooting, see [setup](setup/SETUP.md).


## Run the agents
 

```bash
cd scripts
diambra run -r python custom_tekken_rendering.py
```

For detailed setup instructions including environment configuration, ROM installation, and troubleshooting, see setup/SETUP.md.
