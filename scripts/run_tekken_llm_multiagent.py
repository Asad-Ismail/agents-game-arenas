#!/usr/bin/env python3
import cv2
import argparse
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent
from diambra.arena.stable_baselines3.make_sb3_env import WrappersSettings
import numpy as np
from llm_utils import  get_ollama_model, MOVES, ATTACKS
from utils import render_with_annotations, WINDOW_NAME, GAME_ID
from agent import ThreadedLLMAgent

def get_multiagent_settings(game_id):
    settings = EnvironmentSettingsMultiAgent()
    settings.game_id = game_id
    settings.step_ratio = 1
    settings.disable_keyboard = True
    settings.disable_joystick = True
    settings.render_mode = "rgb_array"
    settings.splash_screen = True
    settings.action_space = (SpaceTypes.MULTI_DISCRETE , SpaceTypes.MULTI_DISCRETE )
    settings.difficulty = 4
    settings.characters = (("Jin", "Devil"), ("Jin", "Devil"))
    settings.outfits = (1, 1)
    settings.frame_shape = (128, 128, 0)
    return settings


def get_agents_observations(observation):
    # Create observations for each agent
    observation_p1 = {
        'frame': observation['frame'],
        'stage': observation['stage'],
        'timer': observation.get('timer', 0)
    }
    
    observation_p2 = {
        'frame': observation['frame'],
        'stage': observation['stage'],
        'timer': observation.get('timer', 0)
    }
    
    # Extract agent-specific observations
    for key, value in observation.items():
        # For agent 0 (player 1)
        if key.startswith('agent_0_own_'):
            observation_p1['own_' + key[12:]] = value
        elif key.startswith('agent_0_opp_'):
            observation_p1['opp_' + key[12:]] = value
        
        # For agent 1 (player 2)
        if key.startswith('agent_1_own_'):
            observation_p2['own_' + key[12:]] = value
        elif key.startswith('agent_1_opp_'):
            observation_p2['opp_' + key[12:]] = value
    
    return observation_p1, observation_p2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model1", type=str, default="qwen:0.5b", help="Ollama model for player 1")
    parser.add_argument("--llm_model2", type=str, default="llama3.2:1b", help="Ollama model for player 2")
    parser.add_argument("--custom_wrapper", type=bool, default=False, help="True if model was trained on grayscale")
    args = parser.parse_args()
    
    # Get environment settings
    settings = get_multiagent_settings(GAME_ID)
    
    # Wrappers Settings - match training settings but adapt for multiagent
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 8
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True 
    
    # Update filter keys to include agent prefixes
    wrappers_settings.filter_keys = ['agent_0_action', 'agent_0_opp_active_character', 'agent_0_opp_bar_status',
                                     'agent_0_opp_character', 'agent_0_opp_character_1', 'agent_0_opp_character_2',
                                    'agent_0_opp_health_1', 'agent_0_opp_health_2', 'agent_0_opp_side', 'agent_0_opp_wins',
                                    'agent_0_own_active_character', 'agent_0_own_bar_status', 'agent_0_own_character', 
                                    'agent_0_own_character_1', 'agent_0_own_character_2', 'agent_0_own_health_1', 
                                    'agent_0_own_health_2', 'agent_0_own_side', 'agent_0_own_wins', 'agent_1_action', 
                                    'agent_1_opp_active_character', 'agent_1_opp_bar_status', 'agent_1_opp_character', 
                                    'agent_1_opp_character_1', 'agent_1_opp_character_2', 'agent_1_opp_health_1',
                                    'agent_1_opp_health_2', 'agent_1_opp_side', 'agent_1_opp_wins', 'agent_1_own_active_character',
                                    'agent_1_own_bar_status', 'agent_1_own_character', 'agent_1_own_character_1', 'agent_1_own_character_2', 
                                    'agent_1_own_health_1', 'agent_1_own_health_2', 'agent_1_own_side', 'agent_1_own_wins', 'frame', 'stage', 'timer']

    
    # Create environment with adapted wrappers
    env = diambra.arena.make(GAME_ID, settings,wrappers_settings=wrappers_settings)
    
    # Initialize LLM models
    model1 = get_ollama_model(model=args.llm_model1)
    model2 = get_ollama_model(model=args.llm_model2)  
    
    # Create LLM agents for both players
    llm_agent1 = ThreadedLLMAgent(model=model1)
    llm_agent2 = ThreadedLLMAgent(model=model2)
    
    llm_agent1.start()
    llm_agent2.start()
    
    observation, info = env.reset(seed=42)
    cumulative_reward = {args.llm_model1: 0, args.llm_model2: 0}
    players = [f"{args.llm_model1}", f"{args.llm_model2}"]

    observation_p1,observation_p2 = get_agents_observations(observation)
    llm_agent1.submit_observation(observation_p1)
    llm_agent2.submit_observation(observation_p2)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)

    try:
        while True:
            # Get actions from both agents
            move1, attack1, reasoning1 = llm_agent1.get_action()
            move2, attack2, reasoning2 = llm_agent2.get_action()
            
            # Combine actions for both players
            actions = {
            "agent_0": [MOVES.index(move1), ATTACKS.index(attack1)],
            "agent_1": [MOVES.index(move2), ATTACKS.index(attack2)]}

            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(actions)

            observation_p1,observation_p2 = get_agents_observations(observation)

            # Submit observations to LLM agents
            llm_agent1.submit_observation(observation_p1)
            llm_agent2.submit_observation(observation_p2)

            # Update cumulative rewards
            if reward>0:
                cumulative_reward[args.llm_model1] += reward
            elif reward<0:
                cumulative_reward[args.llm_model1] += -1 * reward
            
            # Render the game This is not working for multiagent somehow mode does not work and render does not return anything
            # for multiagent wrapper but works good for single agent :(
            if observation['frame'].shape[-1] == 3:  # RGB frames
                rgb_frame = observation['frame']
            else: 
                # Extract the most recent RGB frame (last 3 channels)
                rgb_frame = observation['frame'][:, :, -3:]
            #rgb_frame = env.render(mode="rgb_array")
            vis_data = observation_p1.copy()
            vis_data['rgb_frame'] = rgb_frame
            render_with_annotations(vis_data, players)
            
            if terminated or truncated:
                print(f"Episode finished! Cumulative rewards: {args.llm_model1}={cumulative_reward[args.llm_model1]}, {args.llm_model2}={cumulative_reward[args.llm_model2]}")
                observation, info = env.reset()
                cumulative_reward = {args.llm_model1: 0, args.llm_model2: 0}
    
    finally:
        llm_agent1.stop()
        llm_agent2.stop()
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 