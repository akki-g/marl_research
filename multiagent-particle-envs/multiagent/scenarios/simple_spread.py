"""
Simple spread scenario for multi-agent particle environment.

N agents, N landmarks. Agents are rewarded based on how far any agent is from each landmark,
and are penalized if they collide with other agents. Agents need to spread out to reach all landmarks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    """
    Simple spread scenario for multi-agent particle environment.
    
    N agents are trying to cover N landmarks while avoiding collisions.
    Agents are rewarded based on how close they are to landmarks.
    """
    
    def make_world(self) -> World:
        """
        Create a world with N agents and N landmarks.
        Agents are rewarded based on distance to landmarks.
        """
        world = World()
        
        # Set world properties
        world.dim_c = 2  # communication channel dimensionality
        num_agents = 9
        num_landmarks = 9
        world.collaborative = True
        
        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            
        # Add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
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
            agent.color = np.array([0.35, 0.35, 0.85])
            
        # Random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            
        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        """
        Check if two agents are colliding.
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            bool: True if agents are colliding, False otherwise
        """
        # Get the delta position between agents
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        
        # Calculate the distance between agents
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        
        # Minimum distance to avoid collision
        dist_min = agent1.size + agent2.size
        
        # Return True if distance is less than minimum distance
        return dist < dist_min

    def reward(self, agent: Agent, world: World) -> float:
        """
        Compute the reward for an agent.
        
        The reward is based on:
        1. Negative minimum distance of any agent to each landmark
        2. Penalty for collisions with other agents
        
        Args:
            agent: The agent to compute the reward for
            world: The world containing all entities
            
        Returns:
            float: The reward value
        """
        # Initialize reward
        reward = 0.0
        
        # Calculate negative minimum distance to each landmark
        for landmark in world.landmarks:
            # Calculate distance from each agent to this landmark
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
            
            # Add negative minimum distance to reward
            reward -= min(dists)
            
        # Penalty for collisions with other agents
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent:
                    continue
                    
                # Add penalty for collision
                if self.is_collision(agent, other_agent):
                    reward -= 1.0
                    
        return reward

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        """
        Get observations for an agent.
        
        The observation includes:
        1. Agent's velocity
        2. Agent's position
        3. Relative positions of all landmarks
        4. Relative positions of all other agents
        5. Communication of all other agents (if any)
        
        Args:
            agent: The agent to get observations for
            world: The world containing all entities
            
        Returns:
            np.ndarray: The observation vector
        """
        # Get positions of all landmarks in this agent's reference frame
        landmark_pos = []
        for landmark in world.landmarks:
            landmark_pos.append(landmark.state.p_pos - agent.state.p_pos)
            
        # Get colors of all landmarks
        landmark_color = []
        for landmark in world.landmarks:
            landmark_color.append(landmark.color)
            
        # Get positions of all other agents in this agent's reference frame
        other_pos = []
        # Get communications of all other agents
        comm = []
        
        for other in world.agents:
            if other is agent:
                continue
                
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            
        # Combine all observations into a single vector
        return np.concatenate(
            [agent.state.p_vel] +        # Agent's velocity
            [agent.state.p_pos] +        # Agent's position
            landmark_pos +               # Landmarks' relative positions
            other_pos +                  # Other agents' relative positions
            comm                         # Communication from other agents
        )

    def done(self, agent: Agent, world: World) -> bool:
        """
        Check if the episode is done.
        
        In this scenario, episodes don't terminate based on agent actions.
        They terminate after max_cycles (defined in environment.py).
        
        Args:
            agent: The agent to check
            world: The world containing all entities
            
        Returns:
            bool: Always False in this scenario
        """
        return False

    def benchmark_data(self, agent: Agent, world: World) -> Tuple[float, int, float, int]:
        """
        Get benchmark data for evaluation.
        
        Returns data for:
        - Reward
        - Number of collisions
        - Minimum distance sum
        - Number of occupied landmarks
        
        Args:
            agent: The agent to get benchmark data for
            world: The world containing all entities
            
        Returns:
            Tuple: (reward, collisions, min_dists, occupied_landmarks)
        """
        reward = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        
        # Calculate minimum distance to each landmark
        for landmark in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            
            # Check if a landmark is occupied
            if min(dists) < 0.1:
                occupied_landmarks += 1
                
            # Add negative minimum distance to reward
            reward -= min(dists)
            
        # Count collisions
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent:
                    continue
                    
                # Add collision to count
                if self.is_collision(agent, other_agent):
                    reward -= 1
                    collisions += 1
                    
        return (reward, collisions, min_dists, occupied_landmarks)