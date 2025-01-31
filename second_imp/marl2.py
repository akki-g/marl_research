import numpy as np
from scipy.linalg import block_diag

class NetworkedMultiAgentMDP:
    def __init__(self, num_agents, num_states, num_actions):
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Random transition matrix and rewards
        self.P = np.random.dirichlet(np.ones(num_states), size=(num_states, num_actions))
        self.rewards = np.random.uniform(0, 1, size=(num_agents, num_states, num_actions))
        
    def reset(self):
        self.current_state = np.random.randint(self.num_states)
        return self.current_state
    
    def step(self, actions):
        next_state = np.random.choice(self.num_states, p=self.P[self.current_state, actions.mean().astype(int)])
        rewards = self.rewards[np.arange(self.num_agents), self.current_state, actions]
        self.current_state = next_state
        return next_state, rewards

class Agent:
    def __init__(self, agent_id, num_states, feat_dim, beta):
        self.id = agent_id
        self.w = np.random.randn(feat_dim) * 0.1  # Value function parameters
        self.mu = 0.0  # Average reward tracker
        self.beta = beta
        
    def get_features(self, state):
        """Simple one-hot feature encoding"""
        phi = np.zeros(feat_dim)
        phi[state % feat_dim] = 1
        return phi
    
    def update(self, phi, phi_next, reward):
        # Calculate TD error
        td_error = reward - self.mu + phi_next.dot(self.w) - phi.dot(self.w)
        
        # Update value function parameters
        self.w += self.beta * td_error * phi
        
        # Update average reward estimate
        self.mu = (1 - self.beta) * self.mu + self.beta * reward

def consensus_update(agents, A):
    """Perform consensus step using weight matrix A"""
    all_w = np.array([agent.w for agent in agents])
    new_w = A @ all_w
    for i, agent in enumerate(agents):
        agent.w = new_w[i]

def compute_msbe(agents, state, next_state):
    """Compute Mean Squared Bellman Error"""
    errors = []
    for agent in agents:
        phi = agent.get_features(state)
        phi_next = agent.get_features(next_state)
        error = (agent.reward - agent.mu + 
                 phi_next.dot(agent.w) - 
                 phi.dot(agent.w))
        errors.append(error**2)
    return np.mean(errors)

# Hyperparameters
num_agents = 5
num_states = 10
num_actions = 2
feat_dim = 5
beta = 0.005
K = 50       # Local TD update steps
L = 200      # Communication rounds

# Create environment and agents
env = NetworkedMultiAgentMDP(num_agents, num_states, num_actions)
agents = [Agent(i, num_states, feat_dim, beta) for i in range(num_agents)]

# Create ring topology consensus matrix
A = np.eye(num_agents) * 0.4
for i in range(num_agents):
    A[i, (i+1)%num_agents] = 0.3
    A[i, (i-1)%num_agents] = 0.3

# Training loop
msbe_history = []
state = env.reset()

for l in range(L):
    # K local TD updates
    for k in range(K):
        actions = np.array([np.random.choice(num_actions) for _ in range(num_agents)])
        next_state, rewards = env.step(actions)
        
        for i, agent in enumerate(agents):
            phi = agent.get_features(state)
            phi_next = agent.get_features(next_state)
            agent.update(phi, phi_next, rewards[i])
            agent.reward = rewards[i]  # Store for MSBE calculation
            
        # Compute and store MSBE
        msbe = compute_msbe(agents, state, next_state)
        msbe_history.append(msbe)
        
        state = next_state
    
    # Perform consensus after K steps
    consensus_update(agents, A)

# Plot MSBE convergence
import matplotlib.pyplot as plt
plt.plot(np.convolve(msbe_history, np.ones(100)/100, mode='valid'))
plt.xlabel('Training Steps')
plt.ylabel('Average MSBE')
plt.title('Mean Squared Bellman Error Convergence')
plt.grid(True)
plt.show()
