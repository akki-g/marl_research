"""
Environment for all agents in the multiagent world.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from .multi_discrete import MultiDiscrete

# Environment for all agents in the multiagent world
# Currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(
        self,
        world,
        reset_callback: Optional[Callable] = None,
        reward_callback: Optional[Callable] = None,
        observation_callback: Optional[Callable] = None,
        info_callback: Optional[Callable] = None,
        done_callback: Optional[Callable] = None,
        shared_viewer: bool = True,
        render_mode: Optional[str] = None,
        max_cycles: int = 25
    ):
        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        
        # Rendering
        self.render_mode = render_mode
        self._viewers = {}
        
        # Cycle limit for the environment
        self.max_cycles = max_cycles
        self.current_step = 0
        
        # Random number generator
        self.np_random = None
        self._seed = None
        self.reset(seed=42)  # Initialize RNG
        
        # configure spaces
        self.action_space = []
        self.observation_space = []
        
        for agent in self.agents:
            total_action_space = []
            
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(
                    low=-agent.u_range, 
                    high=+agent.u_range, 
                    shape=(world.dim_p,), 
                    dtype=np.float32
                )
            
            if agent.movable:
                total_action_space.append(u_action_space)
                
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(
                    low=0.0, 
                    high=1.0, 
                    shape=(world.dim_c,), 
                    dtype=np.float32
                )
                
            if not agent.silent:
                total_action_space.append(c_action_space)
                
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n - 1] for act_space in total_action_space]
                    )
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
                
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf, 
                    high=+np.inf, 
                    shape=(obs_dim,), 
                    dtype=np.float32
                )
            )
            
            # Initialize agent's action
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self._viewers = {}
        else:
            self._viewers = {agent: None for agent in self.agents}

    def reset(
        self, 
        *,
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[NDArray[np.float32]], Dict[str, Any]]:
        """
        Reset the environment and return initial observations.
        
        Args:
            seed: Seed for the random number generator
            options: Additional options for reset (not used currently)
            
        Returns:
            observations: Initial observations for each agent
            info: Additional information
        """
        # Set random seed if provided
        if seed is not None:
            self._seed = seed
            self.np_random, _ = seeding.np_random(seed)
            # Set the numpy random seed for consistency
            np.random.seed(seed)
        
        # Reset the step counter
        self.current_step = 0
        
        # reset world
        self.reset_callback(self.world)
        
        # reset renderer
        self._reset_render()
        
        # record observations for each agent
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            
        info_n = {agent.name: self._get_info(agent) for agent in self.agents}
        return obs_n, info_n

    def step(self, action_n: List) -> Tuple[List, List, List, List, Dict]:
        """
        Step the environment with actions for all agents.
        
        Args:
            action_n: List of actions for each agent
            
        Returns:
            obs_n: List of observations for each agent
            reward_n: List of rewards for each agent
            terminated_n: List of termination flags for each agent
            truncated_n: List of truncation flags for each agent
            info_n: Dictionary of info for each agent
        """
        self.current_step += 1
        
        # Set actions for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
            
        # Advance world state
        self.world.step()
        
        # Record observations for each agent
        obs_n = []
        reward_n = []
        terminated_n = []
        truncated_n = []
        info_n = {'n': {}}
        
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            terminated = self._get_done(agent)
            # Check for cycle limit truncation
            truncated = self.current_step >= self.max_cycles
            
            terminated_n.append(terminated)
            truncated_n.append(truncated)
            info_n['n'][agent.name] = self._get_info(agent)
            
        # All agents get total reward in cooperative case
        if self.shared_reward:
            reward = np.sum(reward_n)
            reward_n = [reward] * self.n

        return obs_n, reward_n, terminated_n, truncated_n, info_n

    # Get info used for benchmarking
    def _get_info(self, agent) -> Dict:
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # Get observation for a particular agent
    def _get_obs(self, agent) -> NDArray[np.float32]:
        if self.observation_callback is None:
            return np.zeros(0, dtype=np.float32)
        return self.observation_callback(agent, self.world)

    # Get termination status for a particular agent
    def _get_done(self, agent) -> bool:
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # Get reward for a particular agent
    def _get_reward(self, agent) -> float:
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # Set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        
        # Process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # Physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # Process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
            
        if not agent.silent:
            # Communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
                
        # Make sure we used all elements of action
        assert len(action) == 0

    # Reset rendering assets
    def _reset_render(self):
        self._viewers = {}

    def render(self):
        """
        Render the environment using pyglet.
        
        Returns:
            A rendered image of the environment, or None if rendering is not enabled
        """
        if self.render_mode is None:
            return None
            
        # Import rendering only when needed (avoid dependency for headless machines)
        from . import rendering
        
        # Create viewer if it doesn't exist
        if self.render_mode not in self._viewers:
            # Create a viewer of correct type
            viewer = rendering.Viewer(700, 700)
            self._viewers[self.render_mode] = viewer
        
        viewer = self._viewers[self.render_mode]
        
        # Create geoms for rendering
        from . import rendering
        
        # Clear existing geoms
        viewer.geoms = []
        
        # Create new geoms for each entity
        for entity in self.world.entities:
            geom = rendering.make_circle(entity.size)
            xform = rendering.Transform()
            
            # Set entity color
            if 'agent' in entity.name:
                geom.set_color(*entity.color, alpha=0.5)
            else:
                geom.set_color(*entity.color)
                
            geom.add_attr(xform)
            viewer.add_geom(geom)
            
            # Store transform for later updates
            entity._xform = xform
        
        # Update positions for rendering
        self._update_render_positions()
        
        # Render and return
        if self.render_mode == 'human':
            viewer.render(return_rgb_array=False)
            return None
        elif self.render_mode == 'rgb_array':
            return viewer.render(return_rgb_array=True)
            
    def _update_render_positions(self):
        """Update the positions of entities for rendering"""
        for entity in self.world.entities:
            if hasattr(entity, '_xform'):
                entity._xform.set_translation(*entity.state.p_pos)
                
    def close(self):
        """Close the environment and any viewers"""
        for viewer in self._viewers.values():
            if viewer is not None:
                viewer.close()
        self._viewers = {}