import numpy as np

class MPDEnvironment:

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_matrix = self.get_transition_matrix()
        self.reward_matrix = self.get_reward_matrix()

    def get_transition_matrix(self):
        # Random transition probabilities for state-action pairs
        matrix = np.random.rand(self.num_states, self.num_actions, self.num_states)
        return matrix / matrix.sum(axis=2, keepdims=True)  # Normalize to ensure stochasticity
    
    def get_reward_matrix(self):
        # Random rewards for state-action pairs
        return np.random.rand(self.num_states, self.num_actions)
    
    def step(self, state, action):
        next_state = np.random.choice(self.num_states, p=self.transition_matrix[state, action])
        reward = self.reward_matrix[state, action]
        return next_state, reward
    


class Agent:

    def __init__(self, agent_id, feature_dim, step_size=0.1, seed=42):
        np.random.seed(seed + agent_id)  # so each agent can differ slightly
        self.agent_id = agent_id
        self.weights = np.random.rand(feature_dim)
        self.average_reward = 0.0
        self.step_size = step_size
    
    def td_error(self, state, next_state, reward, features):
        """
        δ^i_{l,k} = r^i_{l,k+1} - μ^i_{l,k} + φ(s_{l,k+1})^T w^i_{l,k} - φ(s_{l,k})^T w^i_{l,k}
        """
        phi_s     = features[state]
        phi_s_next= features[next_state]
        v_s       = phi_s.dot(self.weights)      # φ(s)ᵀ w
        v_s_next  = phi_s_next.dot(self.weights) # φ(s')ᵀ w

        delta = reward - self.average_reward + v_s_next - v_s
        return delta, phi_s
    
    def update(self, delta, phi_s, reward):

        self.weights += self.step_size * delta * phi_s
        self.average_reward = (1 - self.step_size) * self.average_reward + self.step_size * reward

    def td_update(self, state, next_state, reward, features):

        delta, phi_s = self.td_error(state, next_state, reward, features)
        self.update(delta, phi_s, reward)

    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights

    
class DecentralizedMARL:
    
    def __init__(self, num_agents, num_states, num_actions, feature_dim, step_size, K):
        self.env = MPDEnvironment(num_states, num_actions)
        self.num_agents = num_agents
        self.agents = [Agent(agent_id, feature_dim, step_size) for agent_id in range(num_agents)]
        self.num_states = num_states
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.feature_matrix = np.random.rand(num_states, feature_dim)
        self.K = K # num local TD unpdates

    def communicate(self):
        """
        Perform the consensus step (Eq. 4 in the paper):
        w_i <- Σ_j A_ij * w_j
        """
        avg_weights = np.mean([agent.get_weights() for agent in self.agents], axis=0) # Σ_j A_ij * w_j
        for agent in self.agents:
            agent.set_weights(avg_weights) # w_i <- Σ_j A_ij * w_j
        
    def train(self, num_rounds):

        for i in range(num_rounds): #outer loop: communication rounds
            
            for k in range(self.K): #inner loop: local TD updates
                for agent in self.agents:
                    state = np.random.choice(self.num_states)
                    action = np.random.choice(self.env.num_actions)
                    next_state, reward = self.env.step(state, action)
                    agent.td_update(state, next_state, reward, self.feature_matrix)
            self.communicate()

    def eval_consensus_error(self):
        """
        Compute the consensus error (Eq. 9 in the paper):
        ||Q|| = ||w_i - average(w)||
        """
        avg_weights = np.mean([agent.get_weights() for agent in self.agents], axis=0)
        consensus_error = np.linalg.norm([agent.get_weights() - avg_weights for agent in self.agents])
        return consensus_error
    
    def eval_value_function(self):
        """
        Evaluate the approximate global value function for each state:
        V(s) = ϕ(s)ᵀw
        """
        w_bar = np.mean([agent.get_weights() for agent in self.agents], axis=0)
        val_func = self.feature_matrix.dot(w_bar)
        return val_func
    
                    


# Parameters
NUM_AGENTS = 20
NUM_STATES = 10
NUM_ACTIONS = 2
FEATURE_DIM = 5
STEP_SIZE = 0.005
K = 50
NUM_ROUNDS = 100
NUM_SAMPLES = NUM_AGENTS * NUM_ROUNDS * K

# Initialize and train the MARL system
marl_system = DecentralizedMARL(NUM_AGENTS, NUM_STATES, NUM_ACTIONS, FEATURE_DIM, STEP_SIZE, K)
marl_system.train(NUM_ROUNDS)

print(NUM_SAMPLES)

# Evaluate the system
consensus_error = marl_system.eval_consensus_error()
value_function = marl_system.eval_value_function()

print(f"Consensus Error: {consensus_error}")
print("Global Value Function (V(s)) for each state:")
print(value_function)