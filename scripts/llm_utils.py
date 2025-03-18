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


num_moves= 3

system_prompt = f"""# Tekken Tag Tournament - AI Agent

You are an expert Tekken Tag Tournament player. Your job is to analyze the current game state and choose the optimal moves and attacks to win each round.

## CORE MECHANICS:
- Position matters: If you are on the Left side, moving Left blocks and Right approaches (and vice versa)
- Health management: Tag out when low on health to allow recovery
- Aggression: Generally play aggressively to maintain pressure

## AVAILABLE MOVES:
{', '.join(MOVES)}

## AVAILABLE ATTACKS:
{', '.join(ATTACKS)}

## GAME STATE INFORMATION:
- timer: Time remaining (0.0-1.0)
- own_side/opp_side: Position (0: left side, 1: right side)
- own_wins/opp_wins: Rounds won (0.0-1.0)
- own_active_character/opp_active_character: Currently fighting (0: first character, 1: second character)
- own_health_1/opp_health_1: Health of first character (0.0-1.0)
- own_health_2/opp_health_2: Health of second character (0.0-1.0)
- own_character/opp_character: Name of currently active character

## BAR STATUS:
- 0: Reserve health bar almost filled
- 1: Small health loss, recharging
- 2: Large health loss, recharging
- 3: Rage mode on, combo attack ready
- 4: No reserve character

## POSITION-BASED STRATEGY:
Left side:
- Left → Block/defend
- Right → Approach opponent
- Down+Left → Dodge certain attacks

Right side:
- Right → Block/defend
- Left → Approach opponent
- Down+Right → Dodge certain attacks

## TACTICAL GUIDELINES:
- Tag when active character health is low
- Use combos when opponent is vulnerable
- Block when very low on health
- Use powerful attacks in rage mode
- Switch characters strategically for health recovery

## RESPONSE INSTRUCTIONS:
Plan the next {num_moves} moves as a sequence. Return EXACTLY {num_moves} JSON objects in a single code block.
IMPORTANT: You MUST choose moves ONLY from the AVAILABLE MOVES list and attacks ONLY from the AVAILABLE ATTACKS list. Do not combine or create new moves or attacks.

Use this format:

```json
{{
  "move": "MOVE_FROM_LIST",
  "attack": "ATTACK_FROM_LIST",
  "reasoning": "Brief tactical explanation (20 words max)"
}}
{{
  "move": "MOVE_FROM_LIST",
  "attack": "ATTACK_FROM_LIST",
  "reasoning": "Brief tactical explanation (20 words max)"
}}
...
```

IMPORTANT: Place all {num_moves} JSON objects within a single ```json code block, one after another (not as an array). Each object should be a complete, valid JSON object.

Example for a 3-move plan:

```json
{{
  "move": "Right",
  "attack": "Left Punch",
  "reasoning": "Approach and strike"
}}
{{
  "move": "Left",
  "attack": "Right Kick",
  "reasoning": "Create space with counter"
}}
{{
  "move": "Right+Down",
  "attack": "Left Punch+Right Punch",
  "reasoning": "Approach with combo attack"
}}
```

RESPOND ONLY WITH THE JSON CODE BLOCK AND NOTHING ELSE.

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
    if len(bar_status_array.shape)==2:
        bar_status_array = bar_status_array.squeeze(0)
    bar_idx = np.argmax(bar_status_array)
    return bar_idx


def get_bar_status_meaning(status_index):
    meanings = [
        "(Reserve almost full health)",
        "(Small health loss, recharging)",
        "(Large health loss, recharging)",
        "(Rage mode, special attacks available)",
        "(No reserve character)"
    ]
    return meanings[status_index] if 0 <= status_index < len(meanings) else ""

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
    - Time Left: {timer:.2f} (0.0=time's up, 1.0=max time)
    - Your Position: {own_position} side
    - Opponent Position: {opp_position} side
    - Match Score: You {own_wins:.2f} vs Opponent {opp_wins:.2f}

    ## YOUR TEAM:
    - Active: {own_char} ({own_health_1_pct} health)
    - Reserve: {own_char_2 if own_active == 0 else own_char_1} ({own_health_2_pct} health)
    - Bar Status: {own_bar_status_exp}

    ## OPPONENT TEAM:
    - Active: {opp_char} ({opp_health_1_pct} health)
    - Reserve: {opp_char_2 if opp_active == 0 else opp_char_1} ({opp_health_2_pct} health)
    - Bar Status: {opp_bar_status_exp} 

    Analyze this game state and provide your next {num_moves} moves in the required JSON format.
    """
    #print(f"User prompt is {user_prompt}")
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

def parse_llm_response(response_text, num_moves=3):
    """
    Parse the VLM response to extract move, attack and reasoning.
    Simplified approach that handles the JSON code block as a whole.
    """
    move_sequence = []
    
    # Extract the content between ```json and ``` markers
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    
    if json_match:
        json_content = json_match.group(1)
        
        # Wrap the content in array brackets to make it valid JSON
        json_array = f"[{json_content}]"
        
        # Remove any commas between JSON objects if they exist
        json_array = re.sub(r'}\s*,?\s*{', "},{", json_array)
        
        try:
            # Parse as a JSON array
            actions = json.loads(json_array)
            
            for action_data in actions:
                # Extract move and attack, with validation
                move = action_data.get('move', '')
                attack = action_data.get('attack', '')
                reasoning = action_data.get('reasoning', f'Move {len(move_sequence)+1} in sequence')
                
                # Validate move
                if move not in MOVES:
                    closest_move = find_closest_match(move, MOVES)
                    print(f"Invalid move '{move}', using closest match: '{closest_move}'")
                    move = closest_move
                    reasoning = "Using fallback move"
                    
                # Validate attack
                if attack not in ATTACKS:
                    closest_attack = find_closest_match(attack, ATTACKS)
                    print(f"Invalid attack '{attack}', using closest match: '{closest_attack}'")
                    attack = closest_attack
                    reasoning = "Using fallback attack"
                    
                move_sequence.append((move, attack, reasoning))
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON array: {str(e)}")
            print(f"Content attempted to parse: {json_array}...")
    
    # Fill with fallback moves if needed
    if len(move_sequence) < num_moves:
        fallback_moves = [
            (random.choice(MOVES), random.choice(ATTACKS), "Fallback move") 
            for _ in range(num_moves - len(move_sequence))
        ]
        move_sequence.extend(fallback_moves)
    
    return move_sequence


def get_llm_action(observation, model="gemma3:12b", temperature=0.2, timeout=3.0, num_moves=1):
    """
    Query a VLM via Ollama to get the next moves and attacks.
    Returns a list of tuples (move, attack, reasoning) for each action.
    Uses separate system and user prompts for efficiency.
    """
    global system_prompt
    system_prompt = system_prompt.strip()
    user_prompt, image_data = decoder_observations(observation)
    
    try:
        options = {
            "temperature": temperature,  # Lower temperature for more consistent output
            "num_predict": 400,  # Enough tokens for JSON output
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
        else:
            print("Warning: Unexpected response format from Ollama")
            return [(random.choice(MOVES), random.choice(ATTACKS), "Random fallback (API error)") for _ in range(num_moves)]
        
        # Parse the response to get multiple actions
        actions = parse_llm_response(action_text)
        return actions
        
    except Exception as e:
        print(f"Error querying LLM: {str(e)}")
        return [(random.choice(MOVES), random.choice(ATTACKS), f"Random fallback (Exception: {type(e).__name__})") 
                for _ in range(num_moves)]


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

