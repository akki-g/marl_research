#!/usr/bin/env python
"""
Run this script to interact with a multi-agent environment
using the keyboard.

For example:
    python bin/interactive.py --scenario simple

Controls:
    Arrow keys move the agent
    WASD move the camera
    R resets the environment
    Q or ESC quits
"""

import argparse
import sys
import time
import os
from typing import List, Dict, Any

import numpy as np
from pyglet.window import key

# Add parent directory to path so we can import modules
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from make_env import make_env


class InteractivePolicy:
    """
    Interactive policy using keyboard input
    """
    
    def __init__(self, env, agent_index):
        self.env = env
        self.agent_index = agent_index
        
        # Movement keys (arrow keys)
        self.move = [False for _ in range(4)]
        
        # Register keyboard events with the environment's window
        if hasattr(env, '_viewers') and env.render_mode in env._viewers:
            env._viewers[env.render_mode].window.on_key_press = self.key_press
            env._viewers[env.render_mode].window.on_key_release = self.key_release
            
    def action(self, obs):
        """
        Return action based on keyboard input
        """
        # Ignore observation and act based on keyboard events
        if hasattr(self.env, 'discrete_action_input') and self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1  # Left
            if self.move[1]: u = 2  # Right
            if self.move[2]: u = 3  # Up
            if self.move[3]: u = 4  # Down
        else:
            # 5-d action space (no-move action plus 4 directions)
            u = np.zeros(5)
            if self.move[0]: u[1] += 1.0  # Left
            if self.move[1]: u[2] += 1.0  # Right
            if self.move[2]: u[3] += 1.0  # Up
            if self.move[3]: u[4] += 1.0  # Down
            if True not in self.move:
                u[0] += 1.0  # No move
                
        # Add zeros for communication actions
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
        
    def key_press(self, k, mod):
        """Handle key press events"""
        # Movement keys
        if k == key.LEFT:  self.move[0] = True
        if k == key.RIGHT: self.move[1] = True
        if k == key.UP:    self.move[2] = True
        if k == key.DOWN:  self.move[3] = True
        
        # Reset environment
        if k == key.R:
            self.env.reset()
            
        # Quit
        if k == key.Q or k == key.ESCAPE:
            self.env.close()
            sys.exit(0)
            
    def key_release(self, k, mod):
        """Handle key release events"""
        if k == key.LEFT:  self.move[0] = False
        if k == key.RIGHT: self.move[1] = False
        if k == key.UP:    self.move[2] = False
        if k == key.DOWN:  self.move[3] = False


def run_interactive(args):
    """Run interactive mode with keyboard control"""
    # Create environment
    env = make_env(
        args.scenario,
        benchmark=args.benchmark,
        render_mode='human',
        max_cycles=args.max_cycles
    )
    
    # Print environment info
    print(f"Environment: {args.scenario}")
    print(f"Number of agents: {env.n}")
    print(f"Action spaces: {env.action_space}")
    print(f"Observation spaces: {env.observation_space}")
    print("\nControls:")
    print("  Arrow keys - move agent")
    print("  R - reset environment")
    print("  Q/ESC - quit")
    
    # Create interactive policies
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    
    # Reset environment
    obs_n, _ = env.reset(seed=args.seed)
    
    # Main loop
    while True:
        # Get actions from policies
        actions = []
        for i, policy in enumerate(policies):
            actions.append(policy.action(obs_n[i]))
            
        # Step environment
        obs_n, reward_n, term_n, trunc_n, _ = env.step(actions)
        
        # Check for episode end
        done = any(term_n) or any(trunc_n)
        if done:
            print("Episode finished")
            obs_n, _ = env.reset(seed=args.seed)
            
        # Render
        env.render()
        
        # Display rewards
        rewards_text = " ".join([f"{env.agents[i].name}: {reward:.3f}" for i, reward in enumerate(reward_n)])
        print(f"\rRewards: {rewards_text}", end="")
        
        # Limit frame rate
        time.sleep(0.1)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Interactive Multi-Agent Environment")
    parser.add_argument(
        '--scenario', 
        default='simple',
        help='Name of the scenario script (default: %(default)s)'
    )
    parser.add_argument(
        '--benchmark', 
        action='store_true', 
        default=False,
        help='Whether to use benchmark data callbacks'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--max-cycles', 
        type=int, 
        default=50,
        help='Maximum number of steps in an episode'
    )
    
    args = parser.parse_args()
    
    # Run interactive mode
    run_interactive(args)