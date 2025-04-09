"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.

Can be called by using, for example:
    env = make_env('simple_spread')
"""

from typing import Dict, Optional, Any
import os
import sys
import importlib.util

from multiagent.environment import MultiAgentEnv


def make_env(
    scenario_name: str,
    benchmark: bool = False,
    render_mode: Optional[str] = None,
    max_cycles: int = 25,
    **kwargs
) -> MultiAgentEnv:
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gymnasium
    environment by calling env.reset() and env.step().
    """
    # Try to load the scenario module
    scenario_path = os.path.join(os.path.dirname(__file__), "multiagent", "scenarios", f"{scenario_name}.py")
    
    if not os.path.exists(scenario_path):
        raise ValueError(f"Scenario '{scenario_name}' not found at {scenario_path}")
    
    # Use importlib.util to load the module from path
    spec = importlib.util.spec_from_file_location(f"scenarios.{scenario_name}", scenario_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load scenario module: {scenario_name}")
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenario = module.Scenario()

    # Create world
    world = scenario.make_world()
    
    # Create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            info_callback=scenario.benchmark_data,
            done_callback=getattr(scenario, "done", None),
            render_mode=render_mode,
            max_cycles=max_cycles,
            **kwargs
        )
    else:
        env = MultiAgentEnv(
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            done_callback=getattr(scenario, "done", None),
            render_mode=render_mode,
            max_cycles=max_cycles,
            **kwargs
        )
        
    return env