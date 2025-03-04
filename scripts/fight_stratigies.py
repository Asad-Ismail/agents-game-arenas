import random

def basic_fighting_strategy(observation, player="P1"):
    """
    A simple random fighting strategy that mixes movement and attacks.
    
    Args:
        observation: Current game observation
        player: Which player this strategy controls ("P1" or "P2")
    
    Returns:
        list: [move_action, attack_action] to perform
    """
    # Move actions reference:
    # 0 = NoMove, 1 = Left, 2 = UpLeft, 3 = Up, 4 = UpRight
    # 5 = Right, 6 = DownRight, 7 = Down, 8 = DownLeft
    
    # Attack actions reference:
    # 0 = No attack, 1-4 = single buttons, 5+ = combinations
    
    # Choose a random movement action (0-8)
    move_action = random.randint(0, 8)
    
    # Choose a random attack action (0-4)
    # Using 0-4 for simplicity, where 0 is no attack and 1-4 are basic attacks
    attack_action = random.randint(0, 4)
    
    # 30% chance of no movement
    if random.random() < 0.3:
        move_action = 0
    
    # 30% chance of no attack
    if random.random() < 0.3:
        attack_action = 0
    
    print(f"Move: {move_action}, Attack: {attack_action}")
    
    return [move_action, attack_action]