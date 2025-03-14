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
from llm_utils import CHARACTERS



def decode_character(one_hot_vector):
    if isinstance(one_hot_vector, np.ndarray) and len(one_hot_vector) <= len(CHARACTERS):
        try:
            index = np.argmax(one_hot_vector)
            return CHARACTERS[index]
        except:
            return "Unknown"
    return "Unknown"

# Helper function to safely get values from Box spaces
def get_box_value(box_value):
    if isinstance(box_value, np.ndarray) and box_value.size > 0:
        return float(box_value[0])
    return 0.0

# Helper function to get value from Discrete space
def get_discrete_value(discrete_value):
    if isinstance(discrete_value, (int, np.ndarray, np.integer)):
        return int(discrete_value)
    return 0

# User prompt with the current game state information
# Get bar status explanation
def get_bar_status_explanation(bar_status_array):
    bar_status_array = bar_status_array.squeeze(0)
    bar_idx = np.argmax(bar_status_array)
    return bar_idx

def decoder_observations(observation):
    own_char_1 = decode_character(observation.get('own_character_1', np.zeros(39)))
    own_char_2 = decode_character(observation.get('own_character_2', np.zeros(39)))
    
    opp_char_1 = decode_character(observation.get('opp_character_1', np.zeros(39)))
    opp_char_2 = decode_character(observation.get('opp_character_2', np.zeros(39)))
    
    own_health_1 = get_box_value(observation.get('own_health_1', np.array([0.0])))
    own_health_2 = get_box_value(observation.get('own_health_2', np.array([0.0])))
    opp_health_1 = get_box_value(observation.get('opp_health_1', np.array([0.0])))
    opp_health_2 = get_box_value(observation.get('opp_health_2', np.array([0.0])))
    
    own_side = get_discrete_value(observation.get('own_side', 0))
    opp_side = get_discrete_value(observation.get('opp_side', 0))
    opp_side 
    
    timer = get_box_value(observation.get('timer', np.array([0.0])))
    stage = get_box_value(observation.get('stage', np.array([0.0])))
    
    own_wins = get_box_value(observation.get('own_wins', np.array([0.0])))
    opp_wins = get_box_value(observation.get('opp_wins', np.array([0.0])))
    
    # Get active character (0 or 1)
    own_active = get_discrete_value(observation.get('own_active_character', 0))
    opp_active = get_discrete_value(observation.get('opp_active_character', 0))
    
    # Format positions as string
    own_position = "Left" if own_side == 0 else "Right"
    opp_position = "Left" if opp_side == 0 else "Right"
    
    # Format active character information more clearly
    own_active_char = own_char_1 if own_active == 0 else own_char_2
    opp_active_char = opp_char_1 if opp_active == 0 else opp_char_2
    
    # Keep normalized values for consistency with the system prompt description
    own_health_1_pct = f"{own_health_1:.2f}"
    own_health_2_pct = f"{own_health_2:.2f}"
    opp_health_1_pct = f"{opp_health_1:.2f}"
    opp_health_2_pct = f"{opp_health_2:.2f}"
    
    
    own_bar_status_exp = get_bar_status_explanation(observation.get('own_bar_status', np.zeros(5)))
    opp_bar_status_exp = get_bar_status_explanation(observation.get('opp_bar_status', np.zeros(5)))

    user_prompt = f"""## CURRENT GAME STATE:
    - Stage: {stage:.2f} (normalized)
    - Time Left: {timer:.2f} (normalized)
    - Your Wins: {own_wins:.2f} (normalized)
    - Opponent Wins: {opp_wins:.2f} (normalized)
    - Your Characters: {own_char_1} and {own_char_2}
    - Your Active Character: {own_active_char}
    - Your Health (Active): {own_health_1_pct} (normalized)
    - Your Health (Reserve): {own_health_2_pct} (normalized)
    - Your Bar Status: {own_bar_status_exp}
    - Opponent Characters: {opp_char_1} and {opp_char_2}
    - Opponent Active Character: {opp_active_char}
    - Opponent Health (Active): {opp_health_1_pct} (normalized)
    - Opponent Health (Reserve): {opp_health_2_pct} (normalized)
    - Opponent Bar Status: {opp_bar_status_exp}
    - Your Position: {own_position}
    - Opponent Position: {opp_position}

    Based on this game state and the image, what is your next move and attack?"""

    print(user_prompt)


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
        decoder_observations(observation)
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