"""
Scenarios for the multi-agent particle environment.

Each scenario is defined as a separate Python module implementing 
the BaseScenario interface.

Available scenarios:
- simple: Single agent, single landmark for debugging
- simple_adversary: Cooperative agents trying to deceive an adversary
- simple_crypto: Two agents securely communicating with an eavesdropper
- simple_push: One agent pushing another away from a landmark
- simple_reference: Agents communicating about landmarks
- simple_speaker_listener: One speaker directing a listener to a landmark
- simple_spread: Multiple agents covering landmarks while avoiding collisions
- simple_tag: Predator-prey environment
- simple_world_comm: Complex environment with food, forests, and communication
"""

import importlib.util
import os

def load(name: str):
    """
    Load a scenario module by name.
    
    Args:
        name: The name of the scenario module to load (with .py extension)
        
    Returns:
        The loaded module
    """
    # Try to find the file in the scenarios directory
    dirname = os.path.dirname(__file__)
    pathname = os.path.join(dirname, name)
    
    # Check if file exists
    if not os.path.exists(pathname):
        raise ValueError(f"Scenario file not found: {pathname}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module