"""
Base class for all scenarios.
Defines the interface for creating and interacting with scenarios.
"""

from typing import Any, Optional, Dict
import numpy as np

from multiagent.core import World, Agent


class BaseScenario:
    """
    Base class for all scenarios.
    Scenarios inherit from this class and implement the required methods.
    """
    
    def make_world(self) -> World:
        """
        Create a World object with agents, landmarks, etc., according to the scenario.
        Must be implemented by all scenarios.
        
        Returns:
            world: A World object
        """
        raise NotImplementedError()
    
    def reset_world(self, world: World) -> None:
        """
        Reset the world to initial conditions.
        Must be implemented by all scenarios.
        
        Args:
            world: The world to reset
        """
        raise NotImplementedError()
        
    def reward(self, agent: Agent, world: World) -> float:
        """
        Compute the reward for an agent.
        Must be implemented by all scenarios.
        
        Args:
            agent: The agent to compute the reward for
            world: The world containing all entities
            
        Returns:
            float: The reward value
        """
        raise NotImplementedError()
        
    def observation(self, agent: Agent, world: World) -> np.ndarray:
        """
        Get observations for an agent.
        Must be implemented by all scenarios.
        
        Args:
            agent: The agent to get observations for
            world: The world containing all entities
            
        Returns:
            np.ndarray: The observation vector
        """
        raise NotImplementedError()
        
    def done(self, agent: Agent, world: World) -> bool:
        """
        Check if the episode is done for an agent.
        May be implemented by scenarios, otherwise defaults to False.
        
        Args:
            agent: The agent to check
            world: The world containing all entities
            
        Returns:
            bool: True if the episode is done, False otherwise
        """
        return False
        
    def benchmark_data(self, agent: Agent, world: World) -> Dict[str, Any]:
        """
        Get benchmark data for an agent.
        May be implemented by scenarios for evaluation purposes.
        
        Args:
            agent: The agent to get benchmark data for
            world: The world containing all entities
            
        Returns:
            Dict[str, Any]: Benchmark data
        """
        return {}
        
    def info(self, agent: Agent, world: World) -> Dict[str, Any]:
        """
        Get additional info for an agent.
        May be implemented by scenarios to provide additional information.
        
        Args:
            agent: The agent to get info for
            world: The world containing all entities
            
        Returns:
            Dict[str, Any]: Additional info
        """
        return {}