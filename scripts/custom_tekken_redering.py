#!/usr/bin/env python3
import diambra.arena
from diambra.arena import EnvironmentSettings
from diambra.arena import SpaceTypes, Roles
import cv2
import numpy as np
from fight_stratigies import *


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 2
WINDOW_NAME = "Tekken Tag"
GAME_ID = "tektagt"


def get_settings(game_id):
    settings = EnvironmentSettings()
    # General settings
    settings.game_id = GAME_ID  # Game ID
    settings.step_ratio = 1  # Game speed (1-6, where 1 is slowest, 6 is fastest)
    settings.disable_keyboard = True  # Disable keyboard input
    settings.disable_joystick = True  # Disable joystick input
    settings.render_mode = "rgb_array"  # Set to rgb_array to get frame data
    settings.splash_screen = True  # Show splash screen
    settings.rank = 0  # Environment rank (for distributed environments)
    settings.env_address = None  # Custom environment address
    settings.grpc_timeout = 600  # gRPC timeout in seconds

    # Action space
    settings.action_space = SpaceTypes.MULTI_DISCRETE  # MULTI_DISCRETE or DISCRETE

    # Episode settings
    settings.seed = None  # Random seed (None for automatic)
    settings.difficulty = 3  # Game difficulty (1-9, None for random)
    settings.continue_game = 0.0  # Continue game logic
    settings.show_final = False  # Show game finale when completed

    # Tekken-specific settings
    settings.role = Roles.P1  # Player role: Roles.P1, Roles.P2, or None (random)
    settings.characters = ("Jin", "Devil")  # (character1, character2) or None for random
    settings.outfits = 1  # Character outfits (1-5 for Tekken)

    return settings


def render_with_annotations(observation, rl_controlled):
    """
    Render the game frame with RL indicators overlaid on characters
    
    Args:
        observation: The current observation from the environment
        rl_controlled: Dictionary indicating which players are RL-controlled
        window_name: Name of the OpenCV window
    
    Returns:
        bool: True if rendering was successful
    """
    try:
        # Get frame from observation
        frame = observation["frame"].copy()[...,::-1].astype(np.uint8)  
        # Get player sides from observation
        p1_side = observation['P1']['side']
        p2_side = observation['P2']['side']
        
        # Frame shape
        height, width = frame.shape[:2]     
        print(f"Frame shape is {height}, {width}")
        
        # Calculate positions for "RL" text based on player sides, 0 left, 1 Right
        p1_pos = (int(width * 0.25), int(height * 0.22)) if p1_side == 0 else (int(width * 0.75), int(height * 0.22))
        p2_pos = (int(width * 0.25), int(height * 0.22)) if p2_side == 0 else (int(width * 0.75), int(height * 0.22))
        
        if rl_controlled["P1"]:
            cv2.putText(frame, "RL", p1_pos, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        
        if rl_controlled["P2"]:
            cv2.putText(frame, "RL", p2_pos, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        
        # Display frame with overlays
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f"Error in custom rendering: {e}")
        return False


def main():
    # Config options
    use_custom_rendering = True  # Set to False to use default env.render()
    rl_controlled = {"P1": True, "P2": False}  # Change as needed
    
    # Initialize settings and environment
    settings = get_settings(GAME_ID)
    env = diambra.arena.make(GAME_ID,settings)

    # Environment reset
    observation, info = env.reset(seed=42)
    
    # Create OpenCV window if using custom rendering
    if use_custom_rendering:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)  

    # Agent-Environment interaction loop
    while True:
        
        # Action random sampling
        #actions = env.action_space.sample()
        actions = basic_fighting_strategy(observation,"P1")

        #print(f"Actions are {actions}")

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        print(observation.keys())
        #print(f"P1 health: {observation['P1']['health_1']},{observation['P1']['health_2']}, P2 health: {observation['P2']['health_1']},{observation['P2']['health_2']}")
        

        # Rendering
        if use_custom_rendering:
            render_success = render_with_annotations(observation, rl_controlled)
        else:
            # Use default rendering
            env.render()

        # Episode end (Done condition) check
        if terminated or truncated:
            observation, info = env.reset()
            break

    # Environment shutdown
    if use_custom_rendering:
        cv2.destroyAllWindows()
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()