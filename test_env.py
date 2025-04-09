#!/usr/bin/env python
"""
Test script for the simple_spread environment to verify it works correctly.

This script:
1. Creates the simple_spread environment
2. Runs multiple episodes with random actions
3. Collects metrics to verify functionality
4. Provides a summary of performance

Usage:
    python test_simple_spread.py [--render] [--episodes N] [--steps M]
"""

import argparse
import sys
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple

# Add the multiagent-particle-envs directory to the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIAGENT_DIR = os.path.join(SCRIPT_DIR, "multiagent-particle-envs")
sys.path.append(MULTIAGENT_DIR)

# Import the environment creator
from make_env import make_env


def run_episode(
    env, 
    max_steps: int = 25, 
    render: bool = False,
    seed: int = None
) -> Tuple[List[float], int, int, bool]:
    """
    Run a single episode of the environment with random actions.
    
    Args:
        env: The environment to run
        max_steps: Maximum number of steps per episode
        render: Whether to render the environment
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (rewards, steps, collisions, success)
    """
    # Reset the environment
    observations = env.reset()
    
    # Initialize metrics
    episode_rewards = [0.0 for _ in range(env.n)]
    step_count = 0
    collision_count = 0
    success = False
    
    # Run the episode
    for step in range(max_steps):
        # Sample random actions for each agent
        actions = [space.sample() for space in env.action_space]
        
        # Step the environment
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Update metrics
        for i, reward in enumerate(rewards):
            episode_rewards[i] += reward
            
        # Count collisions (approximately)
        if min(rewards) < -10:  # Collision likely occurred if reward is very negative
            collision_count += 1
            
        # Render if requested
        if render:
            env.render()
            time.sleep(0.05)  # Slow down rendering for visibility
        
        # Update step count
        step_count += 1
        
        # Check if episode is done
        done = any(terminated) or any(truncated)
        if done:
            # Check if all landmarks are covered (success)
            success = all(np.min([np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) 
                                for agent in env.world.agents]) < 0.04 
                          for landmark in env.world.landmarks)
            break
            
        # Update observations
        observations = next_obs
        
    return episode_rewards, step_count, collision_count, success


def test_environment(
    num_episodes: int = 10, 
    max_steps: int = 25, 
    render: bool = False,
    seed: int = 42
) -> None:
    """
    Test the simple_spread environment across multiple episodes.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        seed: Random seed for reproducibility
    """
    print("Creating simple_spread environment...")
    env = make_env("simple_spread", benchmark=False)
    
    print(f"Environment created with {env.n} agents and {len(env.world.landmarks)} landmarks")
    print(f"Action spaces: {env.action_space}")
    print(f"Observation spaces: {env.observation_space}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Metrics to track
    all_rewards = []
    all_steps = []
    all_collisions = []
    success_count = 0
    
    print(f"\nRunning {num_episodes} episodes with random actions...")
    
    # Run episodes
    for episode in range(num_episodes):
        episode_seed = seed + episode  # Different seed for each episode
        rewards, steps, collisions, success = run_episode(
            env, max_steps=max_steps, render=render, seed=episode_seed
        )
        
        # Track metrics
        all_rewards.append(rewards)
        all_steps.append(steps)
        all_collisions.append(collisions)
        if success:
            success_count += 1
            
        # Print episode results
        print(f"Episode {episode+1}/{num_episodes}: " + 
              f"Steps={steps}, " + 
              f"Collisions={collisions}, " +
              f"Rewards={[round(r, 2) for r in rewards]}, " +
              f"Success={'Yes' if success else 'No'}")
    
    # Calculate statistics
    avg_reward_per_agent = np.mean(all_rewards, axis=0)
    avg_steps = np.mean(all_steps)
    avg_collisions = np.mean(all_collisions)
    success_rate = success_count / num_episodes
    
    # Print summary
    print("\n===== Test Results =====")
    print(f"Average steps per episode: {avg_steps:.2f}")
    print(f"Average collisions per episode: {avg_collisions:.2f}")
    print(f"Average reward per agent: {[round(r, 2) for r in avg_reward_per_agent]}")
    print(f"Success rate: {success_rate:.2%}")
    
    # Close the environment
    env.close()
    print("\nEnvironment test completed successfully!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test the simple_spread environment")
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Enable rendering"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=10, 
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=25, 
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Run the test
    test_environment(
        num_episodes=args.episodes,
        max_steps=args.steps,
        render=args.render,
        seed=args.seed
    )