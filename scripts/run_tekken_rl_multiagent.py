#!/usr/bin/env python3
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettingsMultiAgent
from stable_baselines3 import PPO
import numpy as np



GAME_ID = "tektagt"


def get_settings(game_id):

    settings = EnvironmentSettingsMultiAgent()
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
    settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)

    # Episode settings
    settings.seed = None  # Random seed (None for automatic)
    settings.difficulty = 4  # Game difficulty (1-9, None for random)
    settings.continue_game = 0.0  # Continue game logic
    settings.show_final = False  # Show game finale when completed

    # Tekken-specific settings
    settings.characters = (("Jin", "Devil"),("Jin", "Devil")) 
    settings.outfits = (1, 1)  # Character outfits (1-5 for Tekken)

    return settings


def main():

    settings = get_settings(GAME_ID)
    settings.frame_shape = (128, 128, 0)

    env = diambra.arena.make(GAME_ID, settings, render_mode="human")

    # Load model
    model_path = "/home/asad/dev/agents-game-arenas/scripts/results/tektagt/model/tekken_ppo_700000_steps"

    # Load agent without passing the environment
    agent = PPO.load(model_path)
    
    # Begin evaluation
    observation, info = env.reset(seed=42)
    env.show_obs(observation)

    while True:
        env.render()

        # Extract observations for player 1 (P1), including shared environment information
        observation_p1 = {
            key: value for key, value in observation.items()
            if key.startswith('P1_') or key in ['frame', 'stage']
        }

        # Initialize player 2 (P2) observation with shared environment information
        observation_p2 = {'frame': observation['frame'], 'stage': observation['stage']}
        
        # Swap P1 and P2 keys for P2 observation
        # Modify P2 keys to match P1 format for the model, as it was trained with P1 observations
        observation_p2.update({
            key.replace('P2_', 'P1_'): value for key, value in observation.items()
            if key.startswith('P2_')
        })

        # Model prediction for P1 actions based on P1 observation
        action_p1, _ = agent.predict(observation_p1, deterministic=True)
        # Model prediction for P2 actions, using modified P2 observation
        action_p2, _ = agent.predict(observation_p2, deterministic=True)

        # Combine actions for both players
        actions = np.append(action_p1, action_p2)
        print("Actions: {}".format(actions))
        
        observation, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        env.show_obs(observation)
        print("Reward: {}".format(reward))
        print("Done: {}".format(done))
        print("Info: {}".format(info))

        if done:
            # Optionally, change episode settings here
            options = {}
            #options["characters"] = (None, None)
            options["char_outfits"] = (5, 5)
            observation, info = env.reset(options=options)
            env.show_obs(observation)
            break

    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()