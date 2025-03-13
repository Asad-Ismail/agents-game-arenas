#!/usr/bin/env python3
import os
import argparse
import cv2
from diambra.arena import EnvironmentSettings
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, WrappersSettings
from stable_baselines3 import PPO

# Import your custom rendering function
from scripts.custom_tekken_redering import render_with_annotations, GAME_ID, get_settings, WINDOW_NAME

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path + ".zip"):
        print(f"Model not found at {args.model_path}.zip")
        return 1

    # Get environment settings
    settings = get_settings(GAME_ID)
    settings.frame_shape = (128, 128, 1)  # Match training frame shape
    settings.render_mode = "rgb_array"

    # Wrappers Settings - should match training settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = False  # No need to normalize for inference
    wrappers_settings.stack_frames = 4
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 8
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ["frame", "action", "P1", "P2"]

    # Create environment - single, non-vectorized for rendering
    env, _ = make_sb3_env(GAME_ID, settings, wrappers_settings, no_vec=True)
    
    # Load trained model
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path, env=env)
    print("Model loaded successfully")

    # Create OpenCV window for custom rendering
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
    
    # Run agent for specified number of episodes
    rl_controlled = {"P1": True, "P2": False}
    episodes_completed = 0
    
    observation, info = env.reset()
    episode_reward = 0
    
    print(f"Running agent for {args.episodes} episodes...")
    
    while episodes_completed < args.episodes:
        # Get action from model
        action, _ = model.predict(observation, deterministic=True)
        
        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Custom rendering
        render_with_annotations(observation, rl_controlled)
        
        # Check if episode is done
        if terminated or truncated:
            print(f"Episode {episodes_completed + 1} completed with reward {episode_reward}")
            observation, info = env.reset()
            episode_reward = 0
            episodes_completed += 1
    
    # Close OpenCV windows and environment
    cv2.destroyAllWindows()
    env.close()
    print("Done!")
    
    return 0

if __name__ == "__main__":
    main()