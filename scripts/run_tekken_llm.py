#!/usr/bin/env python3
import os
import time
import cv2
import argparse
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, WrappersSettings
from stable_baselines3 import PPO
from custom_tekken_redering import render_with_annotations, GAME_ID, get_settings, WINDOW_NAME
from diambra.arena import make as diambra_make
from utils import RGBToGrayscaleWrapper, env_wrapping, DummyVecEnv, Monitor
import numpy as np
import queue
import threading
from llm_utils import decoder_observations,get_llm_action,get_ollama_model, MOVES, ATTACKS



class ThreadedLLMAgent:
    def __init__(self, model):
        self.model = model
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.current_result = ("No-Move", "No-Attack", "Initial state")
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the worker thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_requests)
        self.thread.daemon = True  # Thread will exit when main program exits
        self.thread.start()
        print("LLM worker thread started")
        
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("LLM worker thread stopped")
    
    def _process_requests(self):
        """Worker thread that processes LLM requests"""
        while self.running:
            try:
                # Get the newest observation from the queue (non-blocking)
                try:
                    # Get the latest observation (clear the queue first)
                    latest_observation = None
                    while not self.request_queue.empty():
                        latest_observation = self.request_queue.get_nowait()
                        self.request_queue.task_done()
                    
                    if latest_observation is not None:
                        # Process the observation with LLM
                        move, attack, reasoning = get_llm_action(latest_observation, model=self.model)
                        
                        # Put the result in the result queue
                        self.result_queue.put((move, attack, reasoning))
                        
                except queue.Empty:
                    # No observations to process, sleep a bit
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error in LLM worker thread: {e}")
                time.sleep(0.1)
    
    def submit_observation(self, observation):
        """Submit a new observation to be processed asynchronously"""
        try:
            # Make a copy of the observation to avoid issues if it changes
            obs_copy = observation.copy()
            self.request_queue.put_nowait(obs_copy)
        except queue.Full:
            # Queue is full, just ignore this observation
            pass
    
    def get_action(self):
        """Get the most recent LLM action result"""
        # Check if we have any new results
        try:
            while not self.result_queue.empty():
                # Update with the newest result
                self.current_result = self.result_queue.get_nowait()
                self.result_queue.task_done()
        except queue.Empty:
            pass
        
        # Return the most recent result
        return self.current_result



def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/asad/dev/agents-game-arenas/scripts/results/tektagt/model/tekken_ppo_50000_steps", help="Path to the trained model")
    parser.add_argument("--custom_wrapper", type=bool, default=False, help="True if model was trained on grayscale")
    args = parser.parse_args()

    results_dir = os.path.join(os.getcwd(), "results", GAME_ID)
    log_dir = os.path.join(results_dir, "logs")

    # Get environment settings
    settings = get_settings(GAME_ID)
    settings.frame_shape = (128, 128, 0)  # Set frame shape for RL input

    # Wrappers Settings - match training settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 8
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ['action', 'frame', 'opp_active_character', 'opp_bar_status', 'opp_character', 
                                   'opp_character_1', 'opp_character_2', 'opp_health_1', 'opp_health_2', 'opp_side',
                                   'opp_wins', 'own_active_character', 'own_bar_status', 'own_character', 'own_character_1',
                                   'own_character_2', 'own_health_1', 'own_health_2', 'own_side', 'own_wins', 'stage', 'timer']

    # Create environments
    if args.custom_wrapper:
        print(f"Creating custom env!")
        env_base = diambra_make(GAME_ID, settings, render_mode="rgb_array")
        env_base = RGBToGrayscaleWrapper(env_base)
        env_wrapped = env_wrapping(env_base, wrappers_settings)
        env_monitor = Monitor(env_wrapped, log_dir)
        env = DummyVecEnv([lambda: env_monitor])
        num_envs = 1
    else:
        print(f"Creating default env from sb3!")
        env, num_envs = make_sb3_env(GAME_ID, settings, wrappers_settings)

    print(f"Activated {num_envs} environment(s)")

    SEED = 42
    env.seed(SEED)
    observation = env.reset()

    cumulative_reward = 0
    model = get_ollama_model(model="llama3.2:1b")
    players = [f"{model}", "cpu"]
    done = False

    llm_agent = ThreadedLLMAgent(model=model)
    llm_agent.start()
    
    # Submit the initial observation
    llm_agent.submit_observation(observation)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)  
    try:
        while not done:
            move, attack, reasoning = llm_agent.get_action()
            move_index = MOVES.index(move)
            attack_index = ATTACKS.index(attack)
            action = np.array([move_index, attack_index]).reshape(1, -1)
            #move, attack, reasoning = get_llm_action(observation,model=model)
            # Print the current prediction (optional)
            print(f"Current Prediction - Move: {move}, Attack: {attack}")
            #print(f"Reasoning: {reasoning}")
            #action = env.action_space.sample()
            #action=np.array(action).reshape(1,-1)
            observation, reward, done, info = env.step(action)
            llm_agent.submit_observation(observation)
            #print(f"Obs is {observation}")
            cumulative_reward += reward
            #print(observation)
            # Get RGB frame and render
            rgb_frame = env.render(mode="rgb_array")
            vis_data = observation.copy()
            vis_data['rgb_frame'] = rgb_frame
            render_with_annotations(vis_data, players)
    finally:
        llm_agent.stop()
        env.close()
        print(f"Done with cumulative reward {cumulative_reward}!")
    
    return 0

if __name__ == "__main__":
    main()