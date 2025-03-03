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
    settings.difficulty = 3  # Game difficulty (1-9, None for random)
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



def basic_fighting_strategy(observation, player="P1"):
    """
    A simple fighting strategy that decides between blocking and attacking
    based on opponent position.
    
    Args:
        observation: Current game observation
        player: Which player this strategy controls ("P1" or "P2")
    
    Returns:
        list: [move_action, attack_action] to perform
    """
    # Determine opponent label
    opponent = "P2" if player == "P1" else "P1"
    
    # Get player and opponent sides
    player_side = observation[player]['side']  # 0 = left, 1 = right
    opponent_side = observation[opponent]['side']  # 0 = left, 1 = right
    
    # Get health values to determine if we should be aggressive or defensive
    player_health = observation[player]['health_1']  # Using active character's health
    opponent_health = observation[opponent]['health_1']
    
    # Move actions reference:
    # 0 = NoMove, 1 = Left, 2 = UpLeft, 3 = Up, 4 = UpRight
    # 5 = Right, 6 = DownRight, 7 = Down, 8 = DownLeft
    
    # Attack actions reference:
    # 0 = No attack, 1-4 = single buttons, 5+ = combinations
    
    # Distance-based strategy (based on sides):
    if player_side == opponent_side:
        # Same side - they're close, either block or attack
        
        # Defensive logic when health is low
        if player_health < opponent_health * 0.7:  
            # Block by moving away from opponent
            if player_side == 0:  # Player on left
                move_action = 5  # Move right (away)
            else:  # Player on right
                move_action = 1  # Move left (away)
            attack_action = 0  # No attack while blocking
            
        # Offensive logic when health is higher or similar
        else:
            # Basic attack pattern
            if player_side == 0:  # Player on left
                move_action = 5  # Move right (toward opponent)
            else:  # Player on right
                move_action = 1  # Move left (toward opponent)
    else:
        # Different sides - need to approach opponent
        if player_side == 0:  # Player on left, opponent on right
            move_action = 5  # Move right (toward opponent)
        else:  # Player on right, opponent on left
            move_action = 1  # Move left (toward opponent)
        
        # No attack while approaching from distance
        attack_action = 0
    
    return [move_action, attack_action]


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
        print(f"Length of actions are")
        print(len(actions))

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
