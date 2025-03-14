#!/usr/bin/env python3
import os
import time
import argparse
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, WrappersSettings
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_tekken_redering import render_with_annotations, GAME_ID, get_settings
from diambra.arena import make as diambra_make
from utils import RGBToGrayscaleWrapper, env_wrapping,DummyVecEnv, Monitor


def linear_schedule(initial_value, final_value=3e-6):
    """
    Linear learning rate schedule with specified initial and final values.
    
    Args:
        initial_value: Starting learning rate.
        final_value: Ending learning rate.
        
    Returns:
        Schedule function that adjusts learning rate based on training progress.
    """
    def func(progress):
        # If progress goes from 1 to 0 during training
        return initial_value + (final_value - initial_value) * (1 - progress)
    
    return func


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=5000000, help="Total timesteps for training")
    parser.add_argument("--eval_episodes", type=int, default=3, help="Number of episodes for evaluation")
    parser.add_argument("--checkpoint_freq", type=int, default=10000, help="Frequency of checkpoints")
    parser.add_argument("--custom_wrapper", type=bool, default=False, help="True if you want train on gray scale and want rgb visualizations false if the model was trained on rgb")
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
    settings.frame_shape = (128, 128, 0)  # Set frame shape for RL input

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
    wrappers_settings.filter_keys = ['action', 'frame', 'opp_active_character', 'opp_bar_status', 'opp_character', 
                                     'opp_character_1', 'opp_character_2', 'opp_health_1', 'opp_health_2', 'opp_side',
                                    'opp_wins', 'own_active_character', 'own_bar_status', 'own_character', 'own_character_1',
                                    'own_character_2', 'own_health_1', 'own_health_2', 'own_side', 'own_wins', 'stage', 'timer']

    # Create environments
    if args.custom_wrapper:
        print(f"Creating custom env!")
        env_base = diambra_make(GAME_ID, settings, render_mode="rgb_array")
        # Apply our custom wrapper to convert RGB to grayscale
        env_base = RGBToGrayscaleWrapper(env_base)
        env_wrapped = env_wrapping(env_base, wrappers_settings)
        # Add monitoring
        env_monitor = Monitor(env_wrapped, log_dir)
        # Create vectorized environment
        env = DummyVecEnv([lambda: env_monitor])
        num_envs = 1
    else:
        print(f"Creating default env from sb3!")
        env, num_envs = make_sb3_env(GAME_ID, settings, wrappers_settings)

    #env, num_envs = make_sb3_env(GAME_ID, settings, wrappers_settings)
    print(f"*"*100)
    print(f"Activated {num_envs} environment(s)")
    print(f"*"*100)

    # Define policy kwargs - network architecture
    policy_kwargs = {
        "net_arch": [64, 64]  
    }


    # Initialize the agent or load a pre-trained one
    if args.load_model is None:
        # Create a new PPO agent
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=linear_schedule(2.5e-4),
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
    print(f"\nTraining completed in {training_time/60:.2f} minutes")

    # Save the final model
    final_model_path = os.path.join(model_dir, "tekken_ppo_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Evaluate the trained agent
    print("\nEvaluating the trained agent it might take a while please wait...")
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
    SEED = 42
    env.seed(SEED)
    observation = env.reset()

    cumulative_reward = [0.0 for _ in range(num_envs)]
    #rl_controlled = {"P1": True, "P2": False}
    done = False

    # Run one episode
    while not done:
        # Get action from model using training observation
        action, _state = model.predict(observation, deterministic=True)
        
        # Step training environment and get next observation for model
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if any(x != 0 for x in reward):
            print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            observation = env.reset()
            break
        
        # Get RGB frame from visualization environment # TODO it will not work for multiple env change it to get one env
        #rgb_frame = env.render(mode="rgb_array")
        #print(f"RGB frame shape: {rgb_frame.shape}")  # Should be (128, 128, 3)
        #vis_data = observation.copy()
        #vis_data['rgb_frame'] = rgb_frame 
        # Pass RGB observation to your rendering function
        #render_with_annotations(vis_data, rl_controlled)
    
    env.close()
    print(f"Done with cummulative reward {cumulative_reward }!")
    
    return 0

if __name__ == "__main__":
    main()