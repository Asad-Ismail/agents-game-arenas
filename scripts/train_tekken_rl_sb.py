#!/usr/bin/env python3
import os
import time
import argparse
from diambra.arena import EnvironmentSettings
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, WrappersSettings
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback


from custom_tekken_redering import render_with_annotations, GAME_ID, get_settings

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=500000, help="Total timesteps for training")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--checkpoint_freq", type=int, default=10000, help="Frequency of checkpoints")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load a pre-trained model")
    args = parser.parse_args()

    # Create results directories
    results_dir = os.path.join(os.getcwd(), "results", GAME_ID)
    model_dir = os.path.join(results_dir, "model")
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Get environment settings from your custom script
    settings = get_settings(GAME_ID)
    settings.frame_shape = (128, 128, 3)  # Set frame shape for RL input

    # Wrappers Settings - customize as needed for Tekken
    wrappers_settings = WrappersSettings()
    wrappers_settings.normalize_reward = True
    wrappers_settings.stack_frames = 4
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 8
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ["frame", "action", "P1", "P2"]

    # Create environment
    env, num_envs = make_sb3_env(GAME_ID, settings, wrappers_settings)
    print(f"Activated {num_envs} environment(s)")

    # Define policy kwargs - network architecture
    policy_kwargs = {
        "net_arch": [64, 64]  # Simple network to start with
    }

    # Initialize the agent or load a pre-trained one
    if args.load_model is None:
        # Create a new PPO agent
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            gamma=0.95,
            policy_kwargs=policy_kwargs
        )
    else:
        # Load pre-trained model
        print(f"Loading pre-trained model from {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            tensorboard_log=log_dir
        )

    # Display model architecture
    print("Model architecture:")
    print(model.policy)

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // num_envs,
        save_path=model_dir,
        name_prefix="tekken_ppo"
    )

    # Train the agent
    print(f"\nStarting training for {args.total_timesteps} timesteps...\n")
    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Save the final model
    final_model_path = os.path.join(model_dir, "tekken_ppo_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Evaluate the trained agent
    print("\nEvaluating the trained agent...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=args.eval_episodes,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Run the trained agent with custom rendering
    print("\nRunning trained agent with custom rendering...")
    # Create a separate environment for visualization
    eval_settings = get_settings(GAME_ID)
    eval_settings.render_mode = "rgb_array"
    eval_env = make_sb3_env(GAME_ID, eval_settings, wrappers_settings, no_vec=True)[0]
    
    observation, info = eval_env.reset()
    rl_controlled = {"P1": True, "P2": False}
    done = False
    
    # Run one episode
    while not done:
        # Get action from model
        action, _ = model.predict(observation, deterministic=True)
        
        # Step environment
        observation, reward, terminated, truncated, info = eval_env.step(action)
        
        # Custom rendering
        frame_data = eval_env.render()
        render_with_annotations(observation, rl_controlled)
        
        # Check if done
        done = terminated or truncated
    
    # Close environment
    eval_env.close()
    env.close()
    print("Done!")
    
    return 0

if __name__ == "__main__":
    main()