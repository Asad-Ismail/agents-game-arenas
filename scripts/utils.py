#!/usr/bin/env python3
import cv2
import numpy as np
from diambra.arena import EnvironmentSettings
from diambra.arena import SpaceTypes, Roles
import gymnasium as gym

# Constants for rendering
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 1
WINDOW_NAME = "Tekken Tag"
GAME_ID = "tektagt"

def get_settings(game_id=GAME_ID):
    """
    Create and return environment settings for DIAMBRA Arena.
    
    Args:
        game_id: The ID of the game to configure
        
    Returns:
        EnvironmentSettings object with configured parameters
    """
    settings = EnvironmentSettings()
    # General settings
    settings.game_id = game_id  # Game ID
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
    settings.difficulty = 4  # Game difficulty (1-9, None for random)
    settings.continue_game = 0.0  # Continue game logic
    settings.show_final = False  # Show game finale when completed

    # Tekken-specific settings
    settings.role = Roles.P1  # Player role: Roles.P1, Roles.P2, or None (random)
    settings.characters = ("Jin", "Devil")  # (character1, character2) or None for random
    settings.outfits = 1  # Character outfits (1-5 for Tekken)

    return settings

def render_with_annotations(observation, players, window_name=WINDOW_NAME):
    """
    Render the game frame with indicators overlaid on characters
    
    Args:
        observation: The current observation from the environment
        players: Dictionary or list indicating player labels
        window_name: Name of the OpenCV window
    
    Returns:
        bool: True if rendering was successful
    """
    try:
        # Get frame from observation
        if 'rgb_frame' in observation:
            frame = observation['rgb_frame'].copy()[...,::-1].astype(np.uint8)  
        else:
            frame = observation["frame"].copy()[...,::-1].astype(np.uint8)  
            
        # Get player sides from observation
        if 'own_side' in observation:
            p1_side = int(observation['own_side'])
        else:
            p1_side = observation['P1']['side']
            
        if 'opp_side' in observation:
            p2_side = int(observation['opp_side'])
        else:
            p2_side = observation['P2']['side']
        
        frame = cv2.resize(frame, (224, 200))
        # Frame shape
        height, width = frame.shape[:2]     
        
        # Calculate positions for text based on player sides, 0 left, 1 Right
        p1_pos = (int(width * 0.22), int(height * 0.22)) if p1_side == 0 else (int(width * 0.75), int(height * 0.22))
        p2_pos = (int(width * 0.22), int(height * 0.22)) if p2_side == 0 else (int(width * 0.75), int(height * 0.22))
        
        cv2.putText(frame, players[0], p1_pos, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(frame, players[1], p2_pos, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        
        # Display frame with overlays
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f"Error in custom rendering: {e}")
        return False

def create_display_window(window_name=WINDOW_NAME):
    """Create an OpenCV window for displaying the game"""
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    
def close_display_window():
    """Close all OpenCV windows"""
    cv2.destroyAllWindows()

class RGBToGrayscaleWrapper(gym.Wrapper):
    """
    A wrapper that converts RGB frames to grayscale for training
    but preserves RGB frames for visualization.
    """
    def __init__(self, env):
        super().__init__(env)
        # Store original observation space
        self._original_obs_space = env.observation_space
        
        # Modify observation space for grayscale
        obs_space = env.observation_space.spaces.copy()
        frame_shape = obs_space["frame"].shape
        if frame_shape[2] == 3:  # Only if it's RGB
            gray_shape = (frame_shape[0], frame_shape[1], 1)
            obs_space["frame"] = gym.spaces.Box(
                low=0, high=255, shape=gray_shape, dtype=np.uint8
            )
            self.observation_space = gym.spaces.Dict(obs_space)
            self.needs_conversion = True
        else:
            self.needs_conversion = False
            
        # Store for rgb rendering
        self._last_rgb_frame = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Store RGB frame
        if self.needs_conversion and "frame" in obs:
            self._last_rgb_frame = obs["frame"].copy()
            # Convert frame to grayscale
            obs["frame"] = cv2.cvtColor(obs["frame"], cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store RGB frame
        if self.needs_conversion and "frame" in obs:
            self._last_rgb_frame = obs["frame"].copy()
            # Convert frame to grayscale
            obs["frame"] = cv2.cvtColor(obs["frame"], cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
        return obs, reward, terminated, truncated, info
    
    def get_rgb_frame(self):
        """Return the last RGB frame"""
        return self._last_rgb_frame