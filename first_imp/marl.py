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
        return matrix / matrix.sum(axis=1, keepdims=True)  # Normalize to ensure stochasticity
    
    def get_reward_matrix(self):
        # Random rewards for state-action pairs
        return np.random.rand(self.num_states, self.num_actions)
    
    def get_policy_matrix(self):

        return np.full((self.num_states, self.num_actions), 1.0/self.num_actions)
    
    def sample_actin(self, state):
        return np.random.choice(self.num_actions, p=self.policy_matrix[state])
    

    def step(self, state, action):
        next_state = np.random.choice(self.num_states, p=self.transition_matrix[state, action])
        reward = self.reward_matrix[state, action] + np.random.uniform(-0.5, 0.5)
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
        return delta

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
        self.phi = self.get_feature_matrix(feature_dim, num_states)
        self.K = K # num local TD unpdates
        self.adjacency_matrix = self.get_ring_topology_matrix(num_agents)
        self.sample_round = 0
        self.msbe = []

    def get_feature_matrix(self, feature_dim, num_states):
        """
        Create a random feature matrix (ϕ) for the states.
        """
        phi = np.random.rand(num_states, feature_dim)
        phi = phi / np.linalg.norm(phi, axis=1, keepdims=True)
        return phi 


    def get_ring_topology_matrix(self, num_agents):
        """
        Create a ring topology adjacency matrix (A) for the agents.
        A_ij = 0.5 if agent i is connected to agent j, else 0.
        """
        A = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            A[i, (i-1) % num_agents] = 0.5
            A[i, (i+1) % num_agents] = 0.5
        return A

    def communicate(self):
        """
        Perform the consensus step (Eq. 4 in the paper):
        w_i <- Σ_j A_ij * w_j
        """
        new_weights = np.zeros_like(self.agents[0].get_weights())
        for i, agent in enumerate(self.agents):
            neighbors = np.nonzero(self.adjacency_matrix[i])[0]
            neighbor_weights = np.sum([self.adjacency_matrix[i, j] * self.agents[j].get_weights() for j in neighbors], axis=0)
            new_weights += neighbor_weights
        for agent in self.agents:
            agent.set_weights(new_weights)


    def train(self, num_rounds):

        for i in range(num_rounds): #outer loop: communication rounds
            for k in range(self.K): #inner loop: local TD updates
                for agent in self.agents:
                    state = np.random.choice(self.num_states)
                    action = self.env.sample_actin(state)
                    next_state, reward = self.env.step(state, action)
                    delta = agent.td_update(state, next_state, reward, self.phi)
                    self.sample_round += 1


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
        val_func = self.phi.dot(w_bar)
        return val_func
    
                    


# Parameters
NUM_AGENTS = 20
NUM_STATES = 10
NUM_ACTIONS = 2
TOTAL_ACTION_SPACE = NUM_ACTIONS ** NUM_AGENTS
FEATURE_DIM = 5
STEP_SIZE = 0.005
K = 50
NUM_ROUNDS = 100
NUM_SAMPLES = NUM_AGENTS * NUM_ROUNDS * K
GAMMA = 0.99

# Initialize and train the MARL system
marl_system = DecentralizedMARL(NUM_AGENTS, NUM_STATES, NUM_ACTIONS, FEATURE_DIM, STEP_SIZE, K)
marl_system.train(NUM_ROUNDS)

print("Num Samples: {NUM_SAMPLES}")

# Evaluate the system
consensus_error = marl_system.eval_consensus_error()
value_function = marl_system.eval_value_function()

print(f"Consensus Error: {consensus_error}")
print("Global Value Function (V(s)) for each state:")
print(value_function)