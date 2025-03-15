#!/usr/bin/env python3
import os
import time
import cv2
import argparse
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, WrappersSettings
from stable_baselines3 import PPO
from custom_tekken_redering import render_with_annotations, GAME_ID, get_settings, WINDOW_NAME
from diambra.arena import make as diambra_make
from utils import RGBToGrayscaleWrapper, env_wrapping, DummyVecEnv, Monitor
import numpy as np


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/asad/dev/agents-game-arenas/scripts/results/tektagt/model/tekken_ppo_50000_steps", help="Path to the trained model")
    parser.add_argument("--custom_wrapper", type=bool, default=False, help="True if model was trained on grayscale")
    args = parser.parse_args()

    # Create results directories
    results_dir = os.path.join(os.getcwd(), "results", GAME_ID)
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

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

    # Load the trained model
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path, env=env)
    print("Model loaded successfully")

    # Display model architecture
    print("Model architecture:")
    print(model.policy)

    # Run the trained agent with custom rendering
    print("\nRunning trained agent with custom rendering...")
    SEED = 42
    env.seed(SEED)
    observation = env.reset()

    cumulative_reward = 0
    rl_controlled = {"P1": True, "P2": False}
    done = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)  
    # Run one episode
    while not done:
        action, _state = model.predict(observation, deterministic=True)
        #print(action.squeeze(0))
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        #print(observation)
        # Get RGB frame and render
        rgb_frame = env.render(mode="rgb_array")
        vis_data = observation.copy()
        vis_data['rgb_frame'] = rgb_frame
        render_with_annotations(vis_data, rl_controlled)
    
    env.close()
    print(f"Done with cumulative reward {cumulative_reward}!")
    
    return 0

if __name__ == "__main__":
    main()