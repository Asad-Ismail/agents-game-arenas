#!/usr/bin/env python3
import diambra.arena
from diambra.arena import EnvironmentSettings
from diambra.arena import EnvironmentSettingsMultiAgent, SpaceTypes, Roles


def get_settings():
    settings = EnvironmentSettings()
    # General settings
    settings.game_id = "tektagt"  # Game ID
    settings.step_ratio = 1  # Game speed (1-6, where 1 is slowest, 6 is fastest)
    #settings.frame_shape = (512, 512, 0)  # Frame size (width, height, channels)
    # 0 for RGB, 1 for grayscale
    settings.disable_keyboard = True  # Disable keyboard input
    settings.disable_joystick = True  # Disable joystick input
    settings.render_mode = "human"  # Render mode: "human" or "rgb_array"
    settings.splash_screen = True  # Show splash screen
    settings.rank = 0  # Environment rank (for distributed environments)
    settings.env_address = None  # Custom environment address
    settings.grpc_timeout = 600  # gRPC timeout in seconds

    # Action space
    settings.action_space = SpaceTypes.MULTI_DISCRETE  # MULTI_DISCRETE or DISCRETE
    # MULTI_DISCRETE: [move_action, attack_action]
    # DISCRETE: single action incorporating both move and attack

    # Episode settings
    settings.seed = None  # Random seed (None for automatic)
    settings.difficulty = None  # Game difficulty (1-9, None for random)
    # For Tekken: 1-5 (Easy), 6-7 (Medium), 8-9 (Hard)
    settings.continue_game = 0.0  # Continue game logic:
    # - [0.0, 1.0]: probability of continuing after game over
    # - int((-inf, -1.0]): number of continues before episode ends

    settings.show_final = False  # Show game finale when completed

    # Tekken-specific settings
    settings.role = None  # Player role: Roles.P1, Roles.P2, or None (random)
    # Tekken requires selecting 2 characters per player
    settings.characters = ("Jin", "Devil")  # (character1, character2) or None for random
    # Tekken character options include: "Xiaoyu", "Yoshimitsu", "Nina", "Law", 
    # "Hwoarang", "Eddy", "Paul", "King", "Lei", "Jin", "Baek", "Michelle", 
    # "Armorking", "Gunjack", "Anna", "Brian", "Heihachi", "Ganryu", "Julia", 
    # "Jun", "Kunimitsu", "Kazuya", "Bruce", "Kuma", "Jack-Z", "Lee", "Wang", 
    # "P.Jack", "Devil", "True Ogre", "Ogre", "Roger", "Tetsujin", "Panda", 
    # "Tiger", "Angel", "Alex", "Mokujin"

    settings.outfits = 1  # Character outfits (1-5 for Tekken)

    return settings

def main():

    settings = get_settings()
    env = diambra.arena.make("tektagt",settings,render_mode="human")

    # Environment reset
    observation, info = env.reset(seed=42)

    # Agent-Environment interaction loop
    while True:
        # (Optional) Environment rendering
        env.render()

        # Action random sampling
        actions = env.action_space.sample()

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        print(f"P1 health: {observation['P1']['health_1']},{observation['P1']['health_2']}, P2 health: {observation['P2']['health_1']},{observation['P2']['health_2']}")
        print(f"Info: {info}")

        # Episode end (Done condition) check
        if terminated:
            observation, info = env.reset()
            break

    # Environment shutdown
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()
