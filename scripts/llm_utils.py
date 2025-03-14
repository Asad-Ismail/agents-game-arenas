import ollama
import random
import base64
import io
from PIL import Image
import numpy as np
import json
import re

# Define valid moves & attacks there can be special combos attacks which might not be listed below
MOVES = ["No-Move", "Left", "Left+Up", "Up", "Up+Right", "Right", "Right+Down", "Down", "Down+Left"]

ATTACKS = ["No-Attack", "Left Punch", "Right Punch", "Left Kick", "Right Kick", "Tag",
           "Left Punch+Right Punch", "Left Punch+Left Kick", "Left Punch+Right Kick",
           "Right Punch+Left Kick", "Right Punch+Right Kick", "Right Punch+Tag", "Left Kick+Right Kick"]

CHARACTERS = ["Xiaoyu", "Yoshimitsu", "Nina", "Law", "Hwoarang", "Eddy", "Paul", "King", "Lei", 
              "Jin", "Baek", "Michelle", "Armorking", "Gunjack", "Anna", "Brian", "Heihachi", 
              "Ganryu", "Julia", "Jun", "Kunimitsu", "Kazuya", "Bruce", "Kuma", "Jack-Z", "Lee", 
              "Wang", "P.Jack", "Devil", "True Ogre", "Ogre", "Roger", "Tetsujin", "Panda", 
              "Tiger", "Angel", "Alex", "Mokujin", "Unknown"]



system_prompt = f"""# Tekken Tag Tournament - AI Player
You are an expert Tekken Tag Tournament player. Your job is to analyze the game state and choose the optimal move and attack to win
each round.

## AVAILABLE MOVES:
{', '.join(MOVES)}

## AVAILABLE ATTACKS:
{', '.join(ATTACKS)}

## OBSERVATION SPACE DETAILS:
- timer: Time remaining in match (range: 0.0-1.0), where 0.0 means time's up and 1.0 is maximum time
- stage: Current stage ID (range: 0.0-1.0), where different values represent different arenas

## PLAYER INFORMATION:
- own_wins/opp_wins: Number of rounds won (range: 0.0-1.0), where 0.0 means no wins and 1.0 is maximum (typically 2 wins)
- own_side/opp_side: Position on stage (0: left side, 1: right side)
- own_active_character/opp_active_character: Which character is currently fighting (0: first character, 1: second character)

## CHARACTER DETAILS:
- own_character_1/opp_character_1: First character in the tag team
- own_character_2/opp_character_2: Second character in the tag team
- own_character/opp_character: Currently active character

## HEALTH SYSTEM:
- own_health_1/opp_health_1: Health of first character (range: 0.0-1.0), where 0.0 means defeated and 1.0 is full health
- own_health_2/opp_health_2: Health of second character (range: 0.0-1.0), where 0.0 means defeated and 1.0 is full health

## BAR STATUS MEANING:
Bar status describes the condition of the reserve (tag) character:
- 0: Reserve health bar almost filled, character in good condition
- 1: Small amount of health lost, recharging in progress
- 2: Large amount of health lost, recharging in progress
- 3: Rage mode on, combo attack ready (special attacks available)
- 4: No background character (final boss battle or character defeated)

## STRATEGY GUIDELINES:
- Use Tag when your active character is low on health
- Consider position advantage when choosing moves
- Use combo attacks when opponent is vulnerable
- Defensive moves are better when your health is low
- Tag strategically to maximize health recovery of reserve character
- Different characters have different strengths (speed, power, range)

## INSTRUCTIONS:
1. Analyze the game state information
2. Consider character positions, health, and match situation
3. Choose the optimal move and attack combination based on the current state

## RESPONSE FORMAT:
Respond ONLY with a valid JSON object with this exact structure:

```json
{{
  "move": "MOVE_NAME",
  "attack": "ATTACK_NAME",
  "reasoning": "Brief explanation of your decision"
}}
```

Where MOVE_NAME is one of the available moves and ATTACK_NAME is one of the available attacks.
The reasoning should be short (max 20 words).
"""


def encode_image(image_array):
    """Convert numpy array to base64 encoded string"""
    if image_array is None:
        return None
    
    img = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def format_game_state(observation, include_image=False):
    """
    Convert DIAMBRA game state into a structured prompt for VLMs.
    Separates system prompt (constant) from user prompt (game state).
    """
    # Extract frame if available and include_image is True
    image_data = None
    if include_image and 'rgb_frame' in observation:
        image_data = encode_image(observation['rgb_frame'])
    

    # User prompt - just the current game state information
    user_prompt = f"""## CURRENT GAME STATE:
    - Stage: {observation.get('stage', 'Unknown')}
    - Time Left: {observation.get('timer', 'Unknown')}
    - Your Wins: {observation.get('own_wins', observation.get('wins', 'Unknown'))}
    - Opponent Wins: {observation.get('opp_wins', 'Unknown')}
    - Your Character: {observation.get('own_character', observation.get('character', 'Unknown'))}
    - Your Health (Active): {observation.get('own_health_1', observation.get('health_1', 'Unknown'))}
    - Your Health (Reserve): {observation.get('own_health_2', observation.get('health_2', 'Unknown'))}
    - Opponent Character: {observation.get('opp_character', 'Unknown')}
    - Opponent Health: {observation.get('opp_health_1', 'Unknown')}
    - Your Position: {observation.get('own_side', 'Unknown')}
    - Opponent Position: {observation.get('opp_side', 'Unknown')}

    Based on this game state and the image, what is your next move and attack?"""

    return  user_prompt.strip(), image_data


def find_closest_match(text, options):
    """Find the closest matching option to the given text"""
    if not text:
        return random.choice(options)
        
    # Simple case-insensitive partial matching
    text_lower = text.lower()
    for option in options:
        if text_lower in option.lower() or option.lower() in text_lower:
            return option
    
    # If no match, return random choice
    return random.choice(options)

def parse_llm_response(response_text):
    """
    Parse the VLM response to extract move, attack and reasoning.
    Handles different response formats with robust fallbacks.
    """
    # First try: Look for a JSON block with regex
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if not json_match:
        # Second try: Look for any JSON-like structure
        json_match = re.search(r'{.*}', response_text, re.DOTALL)
    
    if json_match:
        try:
            # Parse the JSON block
            json_str = json_match.group(1) if '```json' in response_text else json_match.group(0)
            action_data = json.loads(json_str)
            
            # Extract move and attack, with validation
            move = action_data.get('move', '')
            attack = action_data.get('attack', '')
            reasoning = action_data.get('reasoning', 'No reasoning provided')
            
            # Validate move
            if move not in MOVES:
                closest_move = find_closest_match(move, MOVES)
                print(f"Invalid move '{move}', using closest match: '{closest_move}'")
                move = closest_move
                
            # Validate attack
            if attack not in ATTACKS:
                closest_attack = find_closest_match(attack, ATTACKS)
                print(f"Invalid attack '{attack}', using closest match: '{closest_attack}'")
                attack = closest_attack
                
            return move, attack, reasoning
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {json_match.group(0)}")
    
    # Fallback parsing for non-JSON responses
    print("Falling back to text parsing for response")
    lines = response_text.split("\n")
    
    # Look for move and attack in response lines
    move = next((m for m in MOVES if any(m.lower() in line.lower() for line in lines)), None)
    attack = next((a for a in ATTACKS if any(a.lower() in line.lower() for line in lines)), None)
    
    # If still not found, use random selections
    if not move:
        move = random.choice(MOVES)
    if not attack:
        attack = random.choice(ATTACKS)
        
    return move, attack, "Extracted from text response"


def get_llm_action(observation, model="gemma3:12b", temperature=0.2, timeout=3.0):
    """
    Query a VLM via Ollama to get the next move and attack.
    Returns a tuple of (move, attack, reasoning).
    Uses separate system and user prompts for efficiency.
    """
    global system_prompt
    system_prompt = system_prompt.strip()
    user_prompt, image_data = format_game_state(observation)
    
    try:
        options = {
            "temperature": temperature,  # Lower temperature for more consistent output
            "num_predict": 150,  # Enough tokens for JSON output
            #"stop": ["```"],  # Stop at the end of the JSON block
        }
        
        # Prepare messages with system and user roles
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # If we have an image, add it to the user message
        if image_data:
            messages[1]["images"] = [image_data]
        
        # Send the request to Ollama
        response = ollama.chat(
            model=model,
            messages=messages,
            options=options,
        )
        
        # Extract the content from the response
        if 'message' in response and 'content' in response['message']:
            action_text = response['message']['content']
            print(f"LLM response is {action_text}")
        else:
            print("Warning: Unexpected response format from Ollama")
            return random.choice(MOVES), random.choice(ATTACKS), "Random fallback (API error)"
        
        return parse_llm_response(action_text)
        
    except Exception as e:
        print(f"Error querying VLM: {str(e)}")
        return random.choice(MOVES), random.choice(ATTACKS), f"Random fallback (Exception: {type(e).__name__})"


def get_ollama_model(model="gemma3:12b"):
    """Check available models and return the best option for Tekken"""
    # Get list of available models
    available_models = ollama.list()
    print(f"Available models in ollama are {available_models}")
    
    # Define preferred models in order (best first)
    preferred_models = [
        "gemma3:12b",
        "llava",  # Default LLaVA
        "llava:34b",  # Largest LLaVA model
        "bakllava:7b",  # BakLLaVA model
        "llava:13b",  # Medium LLaVA model
        "llava:7b",  # Smallest LLaVA model 
        "gemma:2b-vision",  # Small but fast vision model
        "phi:vision"  # Another option
    ]
    
    # Return first available preferred model
    if model in preferred_models:
        return model
    else:
        raise("Specified model is not available!")



if __name__ =="__main__":

    # Example game state
    example_observation = {
        "stage": 3,
        "timer": 45,
        "wins": 1,
        "health_1": 180,
        "health_2": 100,
        "character": "Jin",
        "bar_status": 2
    }

    model = get_ollama_model()
    print(f"Using model: {model}")

    # Get action from LLM
    move, attack, reasoning = get_llm_action(example_observation, model=model)

    print(f"Predicted Move: {move}")
    print(f"Predicted Attack: {attack}")
    print(f"Reasoning: {reasoning}")

