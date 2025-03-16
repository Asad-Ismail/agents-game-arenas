import ollama
import random
import base64
import io
from PIL import Image
import numpy as np
import json
import re
from tqdm import tqdm
import time
import statistics


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



system_prompt = f"""# Tekken Tag Tournament - You are an expert Tekken Tag Tournament player. Your job is to analyze the current game state and choose the optimal move and attack to win
each round. Pay close attention to yours and opponent position which are either Left or Right. If you are Left going Left will block and moving right
will go to opponenet and vice versa. 

## AVAILABLE MOVES:
{', '.join(MOVES)}

## AVAILABLE ATTACKS:
{', '.join(ATTACKS)}

## OBSERVATION SPACE DETAILS:
- timer: Time remaining in match (range: 0.0-1.0), where 0.0 means time's up and 1.0 is maximum time
- stage: Current stage ID (range: 0.0-1.0), 0.0 means first and 1.0 means last stage

## PLAYER INFORMATION:
- own_wins/opp_wins: Number of rounds won (range: 0.0-1.0), where 0.0 means no wins and 1.0 is maximum 
- own_side/opp_side: Position on stage (0: left side, 1: right side)
- own_active_character/opp_active_character: Which character is currently fighting (0: first character, 1: second character)

## CHARACTER DETAILS:
- own_character_1/opp_character_1: First character in the tag team
- own_character_2/opp_character_2: Second character in the tag team
- own_character/opp_character: Currently active character Name

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
- Defensive moves are better when your health is very low
- Tag strategically to maximize health recovery of reserve character
- Different characters have different strengths (speed, power, range)
- In general play very aggresively!

## POSITION-BASED STRATEGY:
If you are on the Left side:
- Moving Left will block/defend
- Moving Right will approach opponent
- Down+Left can dodge certain attacks

If you are on the Right side:
- Moving Right will block/defend
- Moving Left will approach opponent
- Down+Right can dodge certain attacks

## Example Responses

Example if opponent is close and you are on Left:
- Move: "Right"
- Attack: "Left Punch+Right Punch"
- Reasoning: "Close range combo when approaching"

Example if opponent is close and you are on Right:
- Move: "Left"
- Attack: "Right Kick"
- Reasoning: "Fast strike to create space"

Example if opponent is far and you are on Left:
- Move: "Right"
- Attack: "No-Attack"
- Reasoning: "Closing distance first"

Example if opponent is far and you are on Right:
- Move: "Left+Up"
- Attack: "No-Attack"
- Reasoning: "Jump approach to avoid projectiles"

## DEFENSIVE EXAMPLES:
Example when low on health and you are on Left:
- Move: "Left"
- Attack: "No-Attack"
- Reasoning: "Blocking to avoid damage"

Example when low on health and you are on Right:
- Move: "Right"
- Attack: "Tag"
- Reasoning: "Switch to character with more health"

## OFFENSIVE EXAMPLES:
Example when opponent is stunned and you are on Left:
- Move: "Right"
- Attack: "Left Punch+Left Kick"
- Reasoning: "Combo attack on vulnerable opponent"

Example when opponent is stunned and you are on Right:
- Move: "Left"
- Attack: "Right Punch+Right Kick"
- Reasoning: "Maximum damage opportunity"


## INSTRUCTIONS:
1. Analyze the game state information
2. Consider character positions, health, and match situation
3. Choose exactly ONE move and ONE attack based on the current state
4. Return your decision in the required JSON format

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

def decoder_observations(observation, include_image=False):

    # Extract frame if available and include_image is True
    image_data = None
    if include_image and 'rgb_frame' in observation:
        image_data = encode_image(observation['rgb_frame'])

    
    # Cache characters
    decoded_characters = {}

    def get_decoded_character(key):
        """Helper function to get decoded character info with safer caching."""
        if key not in observation:
            return "Unknown"
        char_array = observation.get(key, np.zeros(39))
        if isinstance(char_array, np.ndarray) and char_array.size > 0:
            char_index = np.argmax(char_array)
            cache_key = f"{key}_{char_index}"
            
            if cache_key not in decoded_characters:
                decoded_characters[cache_key] = decode_character(char_array)
            return decoded_characters[cache_key]
        return "Unknown"

    own_char_1 = get_decoded_character('own_character_1')
    own_char_2 = get_decoded_character('own_character_2')
    
    opp_char_1 = get_decoded_character('opp_character_1')
    opp_char_2 = get_decoded_character('opp_character_2')
    
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

    # Get character name
    own_char = get_decoded_character('own_character')
    opp_char = get_decoded_character('opp_character')
    
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
    - Your Active Character Name: {own_char}
    - Your Health (Active): {own_health_1_pct} (normalized)
    - Your Health (Reserve): {own_health_2_pct} (normalized)
    - Your Bar Status: {own_bar_status_exp}
    - Opponent Characters: {opp_char_1} and {opp_char_2}
    - Opponent Active Character: {opp_active_char}
    - Opponent Active Character Name: {opp_char}
    - Opponent Health (Active): {opp_health_1_pct} (normalized)
    - Opponent Health (Reserve): {opp_health_2_pct} (normalized)
    - Opponent Bar Status: {opp_bar_status_exp}
    - Your Position: {own_position}
    - Opponent Position: {opp_position}

    Based on this game state and the image, what is your next move and attack?"""

    return user_prompt, image_data


def encode_image(image_array):
    """Convert numpy array to base64 encoded string"""
    if image_array is None:
        return None
    
    img = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


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
    user_prompt, image_data = decoder_observations(observation)
    #print(f"Usser prompt is {user_prompt}")
    
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
            #print(f"LLM response is {action_text}")
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
        "phi3:mini",
        "qwen:0.5b",
        "llama3.2:1b",
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
        raise ValueError("Specified model is not available!") 



if __name__ =="__main__":

    model = get_ollama_model(model="llama3.2:1b")
    print(f"Using model: {model}")

    iterations = 200
    response_times = []
    
    print(f"Running benchmark for {iterations} iterations...")
    
    for i in tqdm(range(iterations)):
        # Get action from LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Choose a random move and action!"}
        ]
        
        # Time the request
        start_time = time.time()
        
        # Send the request to Ollama
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        
        # Calculate and store response time
        elapsed_time = time.time() - start_time
        response_times.append(elapsed_time)

    # Calculate and print statistics
    avg_time = statistics.mean(response_times)
    median_time = statistics.median(response_times)
    min_time = min(response_times)
    max_time = max(response_times)
    stdev_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
    
    print("\nBenchmark Results:")
    print(f"Total iterations: {iterations}")
    print(f"Average response time: {avg_time:.4f} seconds")
    print(f"Median response time: {median_time:.4f} seconds")
    print(f"Minimum response time: {min_time:.4f} seconds")
    print(f"Maximum response time: {max_time:.4f} seconds")
    print(f"Standard deviation: {stdev_time:.4f} seconds")
    print(f"Total time elapsed: {sum(response_times):.2f} seconds")
    print(f"Requests per second: {iterations/sum(response_times):.2f}")

