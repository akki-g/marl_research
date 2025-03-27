import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
import time
import sys
import gym

def test_comprehensive():
    """
    Comprehensive test of the multiagent-particle-envs environment
    Tests scenario loading, environment creation, stepping, and rendering
    """
    print("===== COMPREHENSIVE ENVIRONMENT TEST =====")
    print(f"Python version: {sys.version}")
    
    try:
        # Test scenario loading
        print("\n1. Testing scenario loading...")
        try:
            scenario = scenarios.load("simple_spread.py").Scenario()
            print("✓ Scenario loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load scenario: {e}")
            raise
        
        # Test world creation
        print("\n2. Testing world creation...")
        try:
            world = scenario.make_world()
            print(f"✓ World created with {len(world.agents)} agents and {len(world.landmarks)} landmarks")
        except Exception as e:
            print(f"✗ Failed to create world: {e}")
            raise
        
        # Test environment creation
        print("\n3. Testing environment creation...")
        try:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
            print(f"✓ Environment created successfully")
            print(f"  - Number of agents: {len(env.agents)}")
            print(f"  - Action space: {env.action_space}")
            print(f"  - Observation space: {env.observation_space}")
        except Exception as e:
            print(f"✗ Failed to create environment: {e}")
            raise
        
        # Test reset
        print("\n4. Testing environment reset...")
        try:
            obs_n = env.reset()
            print(f"✓ Environment reset successfully")
            print(f"  - Observation shapes: {[obs.shape for obs in obs_n]}")
        except Exception as e:
            print(f"✗ Failed to reset environment: {e}")
            raise
        
        # Test step - with correct action handling for different action spaces
        print("\n5. Testing environment step...")
        try:
            # Generate appropriate actions based on action space type
            actions = []
            for i in range(len(env.agents)):
                if isinstance(env.action_space[i], gym.spaces.Discrete):
                    # For Discrete action spaces, sample an integer
                    actions.append(env.action_space[i].sample())
                    print(f"  - Agent {i}: Using discrete action {actions[-1]}")
                elif isinstance(env.action_space[i], gym.spaces.Box):
                    # For continuous action spaces, sample from the space
                    actions.append(env.action_space[i].sample())
                    print(f"  - Agent {i}: Using continuous action with shape {actions[-1].shape}")
                else:
                    # For other types like MultiDiscrete
                    actions.append(env.action_space[i].sample())
                    print(f"  - Agent {i}: Using other action type: {actions[-1]}")
            
            next_obs_n, rew_n, done_n, info_n = env.step(actions)
            print(f"✓ Environment step successful")
            print(f"  - Rewards: {rew_n}")
            print(f"  - Dones: {done_n}")
        except Exception as e:
            print(f"✗ Failed to step environment: {e}")
            raise
        
        # Test multiple steps
        print("\n6. Testing multiple steps...")
        try:
            for i in range(5):
                actions = []
                for j in range(len(env.agents)):
                    actions.append(env.action_space[j].sample())
                next_obs_n, rew_n, done_n, info_n = env.step(actions)
                print(f"  - Step {i+1}: Rewards = {[round(r, 3) for r in rew_n]}")
            print(f"✓ Multiple steps successful")
        except Exception as e:
            print(f"✗ Failed during multiple steps: {e}")
            raise
        
        # Try rendering (optional)
        print("\n7. Testing rendering (optional)...")
        try:
            env.render()
            time.sleep(0.5)  # Brief pause to show the rendering
            env.close()
            print(f"✓ Rendering successful")
        except Exception as e:
            print(f"⚠ Rendering not available or failed: {e}")
            print("  (This is not critical if you don't need visualization)")
        
        print("\n===== ALL TESTS PASSED =====")
        return True
        
    except Exception as e:
        print(f"\n===== TEST FAILED =====")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_comprehensive()
    if success:
        print("Environment is ready for use!")
    else:
        print("Environment setup has issues that need to be fixed.")