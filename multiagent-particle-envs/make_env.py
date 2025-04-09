"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.

Can be called by using, for example:
    env = make_env('simple_spread')

After producing the env object, can be used similarly to a Gymnasium
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

from typing import Dict, Optional, Any
import importlib
import os

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
    Use env.render() to view the environment on the screen.

    Args:
        scenario_name: name of the scenario from ./scenarios/ to load (without the .py extension)
        benchmark: whether you want to produce benchmarking data (usually only done during evaluation)
        render_mode: the render mode to use ('human', 'rgb_array', or None)
        max_cycles: maximum number of steps in an episode
        **kwargs: additional kwargs to pass to the environment

    Returns:
        env: A MultiAgentEnv object

    Some useful env properties:
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    # Try to load the scenario module
    try:
        # First try to load from multiagent.scenarios (the package)
        scenario = importlib.import_module(f"multiagent.scenarios.{scenario_name}").Scenario()
    except ImportError:
        # If that fails, try to load from the scenarios directory directly
        scenario_path = os.path.join(os.path.dirname(__file__), "scenarios", f"{scenario_name}.py")
        if not os.path.exists(scenario_path):
            raise ValueError(f"Scenario '{scenario_name}' not found")
            
        # Use importlib.util to load the module from path
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"scenarios.{scenario_name}", scenario_path)
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