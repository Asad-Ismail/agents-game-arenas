import cv2
import numpy as np
from diambra.arena.wrappers.arena_wrappers import env_wrapping
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

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