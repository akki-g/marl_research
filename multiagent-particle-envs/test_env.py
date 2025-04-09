"""
Example script to test the simple_spread environment with random actions.
This validates that the environment is working correctly.
"""
import numpy as np
from make_env import make_env
from tqdm import tqdm

def main():
    print("Creating simple_spread environment...")
    env = make_env('simple_spread')
    
    # Environment information
    print(f"Number of agents: {env.n}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few episodes
    num_episodes = 3
    max_episode_length = 100
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs_n, _ = env.reset(seed=42 + episode)
        
        # Run episode
        episode_rewards = []
        
        for step in tqdm(range(max_episode_length)):
            # Random actions (as per your original implementation)
            actions_n = []
            for i in range(env.n):
                # Sample random action
                action = np.zeros(5)  # 5 actions: [no-op, left, right, up, down]
                action_idx = np.random.choice(5)
                action[action_idx] = 1
                
                # Add communication dimension (zeros since no communication)
                actions_n.append(np.concatenate([action, np.zeros(env.world.dim_c)]))
            
            # Step environment
            next_obs_n, reward_n, done_n, truncated_n, info_n = env.step(actions_n)
            
            # Store rewards
            episode_rewards.append(sum(reward_n))
            
            # Render environment
            
            # If all agents are done, end episode
            if all(done_n):
                break
            
            # Update observations for next step
            obs_n = next_obs_n
            
        # Print episode statistics
        print(f"Episode {episode+1} rewards: sum={sum(episode_rewards):.2f}, mean={np.mean(episode_rewards):.2f}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()