import numpy as np
import matplotlib.pyplot as plt

def a_id(action_vec):
    """
    Convert a binary action vector (size N) to a single integer in [0, 2^N-1].
    """
    idx = 0
    for bit in action_vec:
        idx = idx * 2 + bit
    return idx

def id_to_vec(idx, N):
    """
    Convert an integer idx in [0, 2^N-1] to a binary vector of length N.
    """
    a = np.zeros(N, dtype=int)
    for i in reversed(range(N)):
        a[i] = idx % 2
        idx //= 2
    return a

class MPDEnvironment:

    def __init__(self, num_states, num_actions, num_agents, seed=42):
        np.random.seed(seed)
        self.num_states = num_states
        self.num_actions_local = num_actions
        self.num_actions_global = num_actions ** num_agents
        self.num_agents = num_agents

        self.reward_matrix = self.get_reward_matrix()
        self.transition_matrix = self.get_transition_matrix()
        self.policy = 0.5 * np.ones((num_agents, num_states, num_actions))

    def get_policy_matrix(self):
        pi = np.zeros((self.num_agents, self.num_states, self.num_actions_local ))
        pi[:, :, 0] = 0.5
        pi[:, :, 1] = 0.5
        return pi
    
    def get_transition_matrix(self):
        # Random transition probabilities for state-action pairs
        P = np.random.rand(self.num_states, self.num_states)
        # Normalize each row
        for i in range(self.num_states):
            row_sum = np.sum(P[i, :])
            P[i, :] /= row_sum # Normalize to ensure stochasticity
        return P
    
    def get_reward_matrix(self):
        # Random rewards for state-action pairs
        R_mean = np.zeros((self.num_agents, self.num_states, self.num_actions_global))
        for i in range(self.num_agents):
            for s in range(self.num_states):
                for a_idx in range(self.num_actions_global):
                    R_mean[i, s, a_idx] = 4.0 * np.random.rand()
        return R_mean

    def step(self, state, joint_action_idx):
        """
        Take a step in the environment.
        """
        p = self.transition_matrix[state, :]
        next_state = np.random.choice(self.num_states, p=p)
        rewards = self.reward_matrix[:, state, joint_action_idx]
        rewards = rewards + np.random.uniform(-0.5, 0.5, size=(self.num_agents,))
        return next_state, rewards

    def sample_joint_action(self, state):
        local_actions = []
        for i in range(self.num_agents):
            prob = self.policy[i, state]
            a_i = np.random.choice(self.num_actions_local, p=prob)
            local_actions.append(a_i)
        joint_action_idx = a_id(local_actions)
        return joint_action_idx 

    


class Agent:

    def __init__(self, agent_id, feature_dim, gamma, step_size=0.005, seed=42):

        np.random.seed(seed + agent_id)  # so each agent can differ slightly
        self.agent_id = agent_id
        self.weights = np.random.rand(feature_dim)

        self.average_reward = 0.0
        self.gamma = gamma
        self.step_size = step_size
    
    def td_error(self, state, next_state, reward, features):
        """
        δ^i_{l,k} = r^i_{l,k+1} - μ^i_{l,k} + φ(s_{l,k+1})^T w^i_{l,k} - φ(s_{l,k})^T w^i_{l,k}
        """
        """
        Discounted env
        δ^i = r^i + γφ(s')ᵀ w - φ(s)ᵀ w
        """
        phi_s     = features[state]
        phi_s_next= features[next_state]
        v_s       = phi_s.dot(self.weights)      # φ(s)ᵀ w
        v_s_next  = phi_s_next.dot(self.weights) # φ(s')ᵀ w

        delta = reward + self.gamma*v_s_next - v_s
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
    
    def __init__(self, num_agents, num_states, num_actions, feature_dim, step_size, K, gamma, seed=42):

        self.env = MPDEnvironment(num_states, num_actions, num_agents)
        self.num_agents = num_agents
        self.agents = [Agent(agent_id, feature_dim, gamma, step_size, seed=seed+1000) for agent_id in range(num_agents)]

        self.num_states = num_states
        self.num_actions_local = num_actions
        self.num_actions_global = self.env.num_actions_global

        self.phi = self.get_feature_matrix(feature_dim, num_states)

        self.K = K # num local TD unpdates
        self.adjacency_matrix = self.get_ring_topology_matrix(num_agents)
        self.gamma = gamma
        self.msbe = []
        self.mse = []
        self.w_star = self.compute_w_star()

    def get_feature_matrix(self, feature_dim, num_states):
        """
        Create a random feature matrix (ϕ) for the states.
        """
        phi = np.random.rand(num_states, feature_dim)
        for s in range(num_states):
            norm_s = np.linalg.norm(phi[s, :], 2)
            if norm_s > 0:
                phi[s, :] /= norm_s
        return phi


    def get_ring_topology_matrix(self, num_agents):
        # ================================
        # Setup the network adjacency A
        # ================================
        if num_agents > 1:
            diag_element = 0.4
            off_diag     = 0.3
            A = diag_element * np.eye(num_agents)
            A[0, num_agents - 1] = off_diag
            A[0, 1]     = off_diag
            for i in range(1, num_agents - 1):
                A[i, i - 1] = off_diag
                A[i, i + 1] = off_diag
            A[num_agents - 1, 0] = off_diag
            A[num_agents - 1, num_agents - 2] = off_diag
        else:
            A = np.array([[1.0]])
        return A

    def communicate(self):
        """
        Perform the consensus step (Eq. 4 in the paper):
        w_i <- Σ_j A_ij * w_j
        """
        current_weights = np.stack([agent.get_weights() for agent in self.agents], axis=1)
        new_weights = current_weights @ self.adjacency_matrix.T
        for i, agent in enumerate(self.agents):
            agent.set_weights(new_weights[:, i])

    def compute_w_star(self):
        """
        Compute the optimal weights w* for the global value function.
        """
        P = self.env.transition_matrix

        eigvals, eigvecs = np.linalg.eig(P.T)
        ix_1 = np.argmin(np.abs(eigvals - 1.0))
        dist_vec = np.real(eigvecs[:, ix_1])
        dist_vec /= np.sum(dist_vec)
        diag_dist = np.diag(dist_vec)

        num_states = self.num_states
        num_agents = self.num_agents
        r_s = np.zeros(self.num_states)
        for s in range(num_states):
            sum_for_s = 0.0

            for a_id in range(self.num_actions_global):
                a_vec = id_to_vec(a_id, num_agents)

                p_a = 1.0
                for i in range(num_agents):
                    p_a *= self.env.policy[i, s, a_vec[i]]
                
                r_bar = 0.0
                for i in range(num_agents):
                    r_bar += (1.0 / num_agents) * self.env.reward_matrix[i, s, a_id]
                
                sum_for_s += p_a * r_bar
            r_s[s] = sum_for_s 
        
        
        phi_T_dist = self.phi.T @ diag_dist
        P_phi = P @ self.phi
        A_ode = phi_T_dist @ (self.phi - self.gamma * P_phi)
        b_ode = phi_T_dist @ r_s
        w_star = np.linalg.solve(A_ode, b_ode)
        return w_star
    
    def eval_mse(self):
        """
        Compute the mean squared error (MSE) between the optimal weights w* and the agents' weights.
        """
        w_star = self.w_star
        w_matrix = np.stack([agent.get_weights() for agent in self.agents], axis=1)  # (d, N)
        diff = w_matrix - w_star[:, None]
        mse = np.mean(diff**2)
        return mse
    
    def eval_msbe(self):
        """
        Compute the mean squared Bellman error (MSBE
        """
        msbe = 0.0
        samples = 0
        for i, agent in enumerate(self.agents):
            for state in range(self.num_states):
                joint_action_idx = self.env.sample_joint_action(state)
                next_state, reward = self.env.step(state, joint_action_idx)
                v_s = self.phi[state].dot(agent.get_weights())
                v_s_next = self.phi[next_state].dot(agent.get_weights())
                r_i = reward[i]

                diff = v_s - (r_i + agent.gamma * v_s_next)
                msbe += diff**2
                samples += 1
        return msbe / samples if samples > 0 else 0.0
    
    def train(self, num_rounds):

        s = np.random.randint(self.num_states)
        for i in range(num_rounds): #outer loop: communication rounds
            for k in range(self.K): #inner loop: local TD updates
                msbe = self.eval_msbe()
                self.msbe.append(msbe)

                mse = self.eval_mse()
                self.mse.append(mse)
                joint_action_idx = self.env.sample_joint_action(s)
                next_state, rewards = self.env.step(s, joint_action_idx)
                for idx, agent in enumerate(self.agents):
                    agent.td_update(state=s,
                                next_state=next_state,
                                reward=rewards[idx],
                                features=self.phi)
                s = next_state
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
K = [40, 50, 100, 200, 250]
L = [250, 200, 100, 50, 40]

GAMMA = 0.99


msbes = []
mses = []

# Initialize and train the MARL system
for k, l in zip(K, L):
    marl_system = DecentralizedMARL(NUM_AGENTS, NUM_STATES, NUM_ACTIONS, FEATURE_DIM, STEP_SIZE, k, GAMMA)
    marl_system.train(l)
    msbes.append(marl_system.msbe)
    mses.append(marl_system.mse)
    # Evaluate the system
    consensus_error = marl_system.eval_consensus_error()
    value_function = marl_system.eval_value_function()

    print(f"K={k}, L={l}")
    print(f"Consensus Error: {consensus_error}")
    print("Global Value Function (V(s)) for each state:")
    print(value_function)
    print()


# Plot the MSBE
plt.figure()
for i, msbe in enumerate(msbes):
    x_vals = range(1, K[i]*L[i]+1)
    plt.plot(x_vals, msbe, label=f"K={K[i]}, L={L[i]}")
plt.xlabel("Communication Rounds")
plt.ylabel("MSBE")
plt.legend()
plt.show()


plt.figure()
for i, mse in enumerate(mses):
    x_vals = range(1, K[i]*L[i]+1)
    plt.plot(x_vals, mse, label=f"K={K[i]}, L={L[i]}")
plt.xlabel("Communication Rounds")
plt.ylabel("Objective Error")
plt.legend()
plt.show()