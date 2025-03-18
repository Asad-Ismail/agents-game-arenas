#!/usr/bin/env python3
import os
import cv2
import argparse
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, WrappersSettings
from stable_baselines3 import PPO
from utils import render_with_annotations, GAME_ID, get_settings, WINDOW_NAME
from diambra.arena import make as diambra_make
from utils import RGBToGrayscaleWrapper, env_wrapping, DummyVecEnv, Monitor
import numpy as np
from llm_utils import get_ollama_model, MOVES, ATTACKS
from agent import ThreadedLLMAgent


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_wrapper", type=bool, default=False, help="True if model was trained on grayscale")
    parser.add_argument("--llm_model", type=str, default="qwen:0.5b", help="Ollama model to use")
    args = parser.parse_args()

    results_dir = os.path.join(os.getcwd(), "results", GAME_ID)
    log_dir = os.path.join(results_dir, "logs")

    # Get environment settings
    settings = get_settings(GAME_ID)
    settings.frame_shape = (128, 128, 0)  # Set frame shape for RL input

    # Wrappers Settings - match training settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 8
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ['action', 'frame', 'opp_active_character', 'opp_bar_status', 'opp_character', 
                                   'opp_character_1', 'opp_character_2', 'opp_health_1', 'opp_health_2', 'opp_side',
                                   'opp_wins', 'own_active_character', 'own_bar_status', 'own_character', 'own_character_1',
                                   'own_character_2', 'own_health_1', 'own_health_2', 'own_side', 'own_wins', 'stage', 'timer']

    # Create environments
    if args.custom_wrapper:
        print(f"Creating custom env!")
        env_base = diambra_make(GAME_ID, settings, render_mode="rgb_array")
        env_base = RGBToGrayscaleWrapper(env_base)
        env_wrapped = env_wrapping(env_base, wrappers_settings)
        env_monitor = Monitor(env_wrapped, log_dir)
        env = DummyVecEnv([lambda: env_monitor])
        num_envs = 1
    else:
        print(f"Creating default env from sb3!")
        env, num_envs = make_sb3_env(GAME_ID, settings, wrappers_settings)

    print(f"Activated {num_envs} environment(s)")

    SEED = 42
    env.seed(SEED)
    observation = env.reset()

    cumulative_reward = 0
    model = get_ollama_model(model=args.llm_model)
    players = [f"{model}", "cpu"]
    done = False

    llm_agent = ThreadedLLMAgent(model=model)
    llm_agent.start()
    
    # Submit the initial observation
    llm_agent.submit_observation(observation)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)  
    try:
        while not done:
            move, attack, reasoning = llm_agent.get_action()
            #result= get_llm_action(observation=observation, model=model)
            #if reasoning!="Waiting for LLM response":
            #    print(move,attack,reasoning) 
            move_index = MOVES.index(move)
            attack_index = ATTACKS.index(attack)
            action = np.array([move_index, attack_index]).reshape(1, -1)
            observation, reward, done, info = env.step(action)
            llm_agent.submit_observation(observation)
            cumulative_reward += reward
            rgb_frame = env.render(mode="rgb_array")
            vis_data = observation.copy()
            vis_data['rgb_frame'] = rgb_frame
            render_with_annotations(vis_data, players)
    finally:
        llm_agent.stop()
        env.close()
        print(f"Done with cumulative reward {cumulative_reward}!")
    
    return 0

if __name__ == "__main__":
    main()