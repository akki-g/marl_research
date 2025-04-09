"""
Core components for multi-agent particle environments.
"""

from typing import Optional, List, Tuple, Union, Any
import numpy as np
from numpy.typing import NDArray


class EntityState:
    """Physical/external base state of all entities"""
    
    def __init__(self) -> None:
        # physical position
        self.p_pos: Optional[NDArray[np.float_]] = None
        # physical velocity
        self.p_vel: Optional[NDArray[np.float_]] = None


class AgentState(EntityState):
    """State of agents (including communication and internal/mental state)"""
    
    def __init__(self) -> None:
        super().__init__()
        # communication utterance
        self.c: Optional[NDArray[np.float_]] = None


class Action:
    """Action of the agent"""
    
    def __init__(self) -> None:
        # physical action
        self.u: Optional[NDArray[np.float_]] = None
        # communication action
        self.c: Optional[NDArray[np.float_]] = None


class Entity:
    """Properties and state of physical world entity"""
    
    def __init__(self) -> None:
        # name 
        self.name: str = ''
        # properties:
        self.size: float = 0.050
        # entity can move / be pushed
        self.movable: bool = False
        # entity collides with others
        self.collide: bool = True
        # material density (affects mass)
        self.density: float = 25.0
        # color
        self.color: Optional[NDArray[np.float_]] = None
        # max speed and acceleration
        self.max_speed: Optional[float] = None
        self.accel: Optional[float] = None
        # state
        self.state: EntityState = EntityState()
        # mass
        self.initial_mass: float = 1.0

    @property
    def mass(self) -> float:
        return self.initial_mass


class Landmark(Entity):
    """Properties of landmark entities"""
    
    def __init__(self) -> None:
        super().__init__()


class Agent(Entity):
    """Properties of agent entities"""
    
    def __init__(self) -> None:
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent: bool = False
        # cannot observe the world
        self.blind: bool = False
        # physical motor noise amount
        self.u_noise: Optional[float] = None
        # communication noise amount
        self.c_noise: Optional[float] = None
        # control range
        self.u_range: float = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback: Optional[callable] = None
        # The agent's goal (used in some scenarios)
        self.goal_a: Optional[Any] = None
        self.goal_b: Optional[Any] = None


class World:
    """Multi-agent world"""
    
    def __init__(self) -> None:
        # list of agents and entities (can change at execution-time!)
        self.agents: List[Agent] = []
        self.landmarks: List[Landmark] = []
        # communication channel dimensionality
        self.dim_c: int = 0
        # position dimensionality
        self.dim_p: int = 2
        # color dimensionality
        self.dim_color: int = 3
        # simulation timestep
        self.dt: float = 0.1
        # physical damping
        self.damping: float = 0.25
        # contact response parameters
        self.contact_force: float = 1e+2
        self.contact_margin: float = 1e-3
        # Whether the environment is collaborative
        self.collaborative: bool = False
        # Whether to use discrete action space
        self.discrete_action: bool = False

    @property
    def entities(self) -> List[Entity]:
        """Return all entities in the world"""
        return self.agents + self.landmarks

    @property
    def policy_agents(self) -> List[Agent]:
        """Return all agents controllable by external policies"""
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self) -> List[Agent]:
        """Return all agents controlled by world scripts"""
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self) -> None:
        """Update state of the world"""
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
            
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        
        # integrate physical state
        self.integrate_state(p_force)
        
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    def apply_action_force(self, p_force: List[Optional[NDArray[np.float_]]]) -> List[Optional[NDArray[np.float_]]]:
        """Gather agent action forces"""
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    def apply_environment_force(self, p_force: List[Optional[NDArray[np.float_]]]) -> List[Optional[NDArray[np.float_]]]:
        """Gather physical forces acting on entities"""
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                    
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                    
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
                    
        return p_force

    def integrate_state(self, p_force: List[Optional[NDArray[np.float_]]]) -> None:
        """Integrate physical state"""
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
                
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                
            if entity.max_speed is not None:
                speed = np.sqrt(np.sum(np.square(entity.state.p_vel)))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.sum(np.square(entity.state.p_vel))) * entity.max_speed
                    
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent: Agent) -> None:
        """Set communication state"""
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def get_collision_force(self, entity_a: Entity, entity_b: Entity) -> Tuple[Optional[NDArray[np.float_]], Optional[NDArray[np.float_]]]:
        """Get collision forces for any contact between two entities"""
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
            
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
            
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        
        return [force_a, force_b]