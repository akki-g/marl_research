"""
Simple environment with a single agent and a single landmark.
Used mainly for debugging purposes.
"""

import numpy as np
from typing import Dict, List, Optional

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    """
    Simple scenario with a single agent and a single landmark.
    The agent is rewarded based on its distance to the landmark.
    """
    
    def make_world(self) -> World:
        """
        Create a world with a single agent and a single landmark.
        """
        world = World()
        
        # Add agents
        world.agents = [Agent() for _ in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = False
            agent.silent = True
            
        # Add landmarks
        world.landmarks = [Landmark() for _ in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark {i}'
            landmark.collide = False
            landmark.movable = False
            
        # Make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        """
        Reset the world with random positions for all entities.
        """
        # Random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            
        # Random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
            
        # Special color for the first landmark
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        
        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent: Agent, world: World) -> float:
        """
        Compute the reward for an agent based on its distance to the landmark.
        
        Args:
            agent: The agent to compute the reward for
            world: The world containing all entities
            
        Returns:
            float: The negative squared distance to the landmark
        """
        # Squared Euclidean distance to the landmark
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        """
        Get observations for an agent.
        
        Args:
            agent: The agent to get observations for
            world: The world containing all entities
            
        Returns:
            np.ndarray: The observation vector
        """
        # Get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            
        # Combine observations into a single vector
        return np.concatenate([agent.state.p_vel] + entity_pos)
        
    def done(self, agent: Agent, world: World) -> bool:
        """
        Check if the episode is done.
        
        Args:
            agent: The agent to check
            world: The world containing all entities
            
        Returns:
            bool: True if the episode is done, False otherwise
        """
        # Episode is done if the agent is very close to the landmark
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        return dist < 0.1