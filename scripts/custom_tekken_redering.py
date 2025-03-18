#!/usr/bin/env python3
import diambra.arena
import cv2
from fight_stratigies import *
from utils import get_settings, render_with_annotations, create_display_window, close_display_window, WINDOW_NAME, GAME_ID

def main():
    # Config options
    use_custom_rendering = True  # Set to False to use default env.render()
    players = ["RL", "CPU"]  # Change as needed
    
    # Initialize settings and environment
    settings = get_settings(GAME_ID)
    env = diambra.arena.make(GAME_ID, settings)

    # Environment reset
    observation, info = env.reset(seed=42)
    
    # Create OpenCV window if using custom rendering
    if use_custom_rendering:
        create_display_window()

    # Agent-Environment interaction loop
    while True:
        # Action from strategy
        actions = basic_fighting_strategy(observation, "P1")

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        
        # Rendering
        if use_custom_rendering:
            render_success = render_with_annotations(observation, players)
        else:
            # Use default rendering
            env.render()

        # Episode end (Done condition) check
        if terminated or truncated:
            observation, info = env.reset()
            break

    # Environment shutdown
    if use_custom_rendering:
        close_display_window()
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()