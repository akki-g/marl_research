import numpy as np
import matplotlib.pyplot as plt
import os
import time
import networkx as nx
from tqdm import tqdm
import sys

# Add path to import from multiagent-particle-envs
sys.path.append('.')  # Assuming we're running from the project root

# Import from multiagent-particle-envs
from make_env import make_env

# -------------------- Constants and Configuration --------------------
SEED = 42  # Random seed for reproducibility
L = 200    # Number of communication rounds
K = 50     # Number of local TD-update steps
BETA = 0.1 # Step size parameter
NUM_AGENTS = 9
NUM_LANDMARKS = 9
FEAT_NORMALIZATION = 'l2'
INIT_SCALE = 0.01
SAVE_DIR = "results_local_td"

# -------------------- Set Random Seeds --------------------
def set_seeds(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)

# -------------------- Network Topology Functions --------------------
def generate_ring_network(N):
    """
    Generate a ring network as described in the paper:
    - Self weight: 0.4
    - Neighbor weights: 0.3 each
    """
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 0.4  # Self weight
        A[i, (i-1) % N] = 0.3  # Left neighbor
        A[i, (i+1) % N] = 0.3  # Right neighbor
    return A

def generate_regular_network(N, degree, self_weight=0.4):
    """Generate a k-regular network with specified degree"""
    # Set seed for NetworkX to ensure reproducibility
    G = nx.random_regular_graph(degree, N, seed=SEED)
    A = np.zeros((N, N))
    
    # Calculate edge weight based on degree
    edge_weight = (1.0 - self_weight) / degree
    
    # Set weights
    for i in range(N):
        A[i, i] = self_weight
        for j in list(G.neighbors(i)):
            A[i, j] = edge_weight
            
    return A

def generate_erdos_renyi_network(N, p=0.5, self_weight=0.4):
    """Generate an Erdos-Renyi network with connection probability p"""
    # Set seed for NetworkX to ensure reproducibility
    G = nx.erdos_renyi_graph(N, p, seed=SEED)
    A = np.zeros((N, N))
    
    # Set weights
    for i in range(N):
        neighbors = list(G.neighbors(i))
        degree = len(neighbors)
        
        # If node has no neighbors, set self-loop to 1
        if degree == 0:
            A[i, i] = 1.0
            continue
            
        # Otherwise, distribute weights
        edge_weight = (1.0 - self_weight) / degree
        A[i, i] = self_weight
        for j in neighbors:
            A[i, j] = edge_weight
            
    return A

def generate_complete_network(N):
    """
    Generate a complete network where every node is connected to every other node
    with equal weight 1/N
    """
    A = np.ones((N, N)) / N
    return A

def verify_doubly_stochastic(A, tol=1e-6):
    """Verify that a matrix is doubly stochastic (rows and columns sum to 1)"""
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    
    row_check = np.allclose(row_sums, 1.0, atol=tol)
    col_check = np.allclose(col_sums, 1.0, atol=tol)
    
    return row_check and col_check

# -------------------- Feature Extraction --------------------
def get_feature_vector(obs, num_agents=9, num_landmarks=9, normalization='l2'):
     """
     Constructs the feature vector φ(s) for an agent based on its observation.
     For simple_spread_v3, the observation is assumed to be arranged as:
       [self_vel (2), self_pos (2), landmark_rel_positions (num_landmarks*2),
        other_agents_rel_positions ((num_agents-1)*2), communication (remaining dims)]
     We ignore communication and extract:
       - self_pos = obs[2:4]
       - landmark_rel_pos = obs[4:4+2*num_landmarks]
       - other_agents_rel_pos = obs[4+2*num_landmarks : 4+2*num_landmarks+2*(num_agents-1)]
     For num_agents=9 and num_landmarks=9, this gives 2+18+16 = 36 dims.
     The resulting vector is normalized.
     Extract features from the observation as described in the paper:
     - Agent's own position (2 dims)
     - Landmark relative positions (num_landmarks * 2 dims)
     - Other agents' relative positions ((num_agents-1) * 2 dims)
     """
     agent_pos = obs[2:4]
     agent_pos = obs[2:4]  # Self position (2D)
     
     # Landmark relative positions
     start_landmark = 4
     end_landmark = start_landmark + num_landmarks * 2
     landmark_rel_pos = obs[start_landmark:end_landmark]
     # Extract other agents relative positions (ignoring communication)
     
     # Other agents' relative positions
     other_agents_rel_pos = obs[end_landmark:end_landmark + (num_agents - 1) * 2]
     
     # Concatenate all features
     feature_vector = np.concatenate([agent_pos, landmark_rel_pos, other_agents_rel_pos])
     
     # Normalize the feature vector (using L1 norm as per the paper)
     if normalization == 'l1':
            norm = np.linalg.norm(feature_vector, ord=1)
     elif normalization == 'l2':
            norm = np.linalg.norm(feature_vector, ord=2)

     if norm > 0:
         feature_vector = feature_vector / norm
         
     return feature_vector

# -------------------- Error Metrics --------------------
def calculate_consensus_error(weights):
    """
    Calculate consensus error across agents:
    CE = (1/N) * sum_i ||w_i - w_bar||^2
    """
    agent_ids = list(weights.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
        
    W = np.stack([weights[agent] for agent in agent_ids])
    w_bar = np.mean(W, axis=0)
    errors = np.sum((W - w_bar)**2, axis=1)
    
    return np.mean(errors)

def calculate_SBE(sample_data):
    """
    Calculate Squared Bellman Error:
    SBE = (1/N) * sum_i (phi(s)^T*w_i + mu_bar - r_bar - phi(s')^T*w_i)^2
    """
    agent_ids = list(sample_data.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
        
    # Calculate average reward and average value estimation
    r_bar = sum(sample_data[agent]['reward'] for agent in agent_ids) / N
    mu_bar = sum(sample_data[agent]['mu'] for agent in agent_ids) / N
    
    total_error = 0.0
    for agent in agent_ids:
        w = sample_data[agent]['w']
        phi_s = sample_data[agent]['phi_s']
        phi_s_next = sample_data[agent]['phi_s_next']
        
        # Calculate TD target
        value_current = np.dot(phi_s, w)
        value_next = np.dot(phi_s_next, w)
        
        # Calculate Bellman error
        bellman_error = value_current + mu_bar - r_bar - value_next
        squared_error = bellman_error ** 2
        
        total_error += squared_error
    
    return total_error / N

# -------------------- Agent Model --------------------
class LocalTDAgent:
    """Agent implementing the Local TD update from Algorithm 1"""
    def __init__(self, agent_id, feat_dim, init_scale=INIT_SCALE):
        self.agent_id = agent_id
        
        # Initialize weights with small random values as in the paper
        self.w = np.random.randn(feat_dim) * init_scale
        
        # Initialize average reward estimate to 0
        self.mu = 0.0
        
        # Previous feature vector (for storing between steps)
        self.prev_phi = None
        
    def update(self, phi_s, phi_s_next, reward, beta):
        """
        Perform local TD update using current transition
        
        Args:
            phi_s: Feature vector for current state
            phi_s_next: Feature vector for next state
            reward: Reward received
            beta: Step size parameter
            
        Returns:
            Dictionary with update information
        """
        # Calculate current state value
        value_current = np.dot(phi_s, self.w)
        
        # Calculate next state value
        value_next = np.dot(phi_s_next, self.w)
        
        # Calculate TD error: δ = r - μ + φ(s')ᵀw - φ(s)ᵀw
        delta = reward - self.mu + value_next - value_current
        
        # Update average reward estimate: μ = (1-β)μ + βr
        self.mu = (1 - beta) * self.mu + beta * reward
        
        # Update weight vector: w = w + βδφ(s)
        self.w = self.w + beta * delta * phi_s
        
        # Return relevant data for metrics calculation
        return {
            'phi_s': phi_s,
            'phi_s_next': phi_s_next,
            'reward': reward,
            'mu': self.mu,
            'w': self.w,
            'delta': delta
        }

# -------------------- Main Algorithm --------------------
def run_local_td_experiment(network_type='er'):
    """
    Run Local TD experiment following Algorithm 1 from the paper
    
    Args:
        network_type: Type of network topology to use
        
    Returns:
        Dictionary with experiment results
    """
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seeds(SEED)
    
    print(f"Starting experiment with {network_type} network topology")
    print(f"Parameters: L={L}, K={K}, beta={BETA}, num_agents={NUM_AGENTS}")
    
    # Initialize environment
    env = make_env('simple_spread', benchmark=False)
    
    # Reset environment
    obs_n, _ = env.reset(seed=SEED)
    
    # Create consensus matrix based on network topology
    if network_type == 'ring':
        A_consensus = generate_ring_network(NUM_AGENTS)
    elif network_type == '4-regular':
        A_consensus = generate_regular_network(NUM_AGENTS, 4)
    elif network_type == '6-regular':
        A_consensus = generate_regular_network(NUM_AGENTS, 6)
    elif network_type == 'er':
        A_consensus = generate_erdos_renyi_network(NUM_AGENTS, p=0.5)
    elif network_type == 'complete':
        A_consensus = generate_complete_network(NUM_AGENTS)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Verify consensus matrix is doubly stochastic
    if not verify_doubly_stochastic(A_consensus):
        print(f"WARNING: Consensus matrix for {network_type} is not doubly stochastic!")
    
    # Initialize agents
    # Get a sample observation to determine feature dimension
    sample_obs = get_feature_vector(
        obs_n[0], 
        NUM_AGENTS, 
        NUM_LANDMARKS,
        normalization=FEAT_NORMALIZATION
    )
    feat_dim = sample_obs.shape[0]
    print(f"Feature dimension: {feat_dim}")
    
    # Create a dictionary mapping agent indices to LocalTDAgent objects
    agents = {
        i: LocalTDAgent(i, feat_dim) 
        for i in range(NUM_AGENTS)
    }
    
    # Initialize feature vectors for each agent
    for i, agent_obs in enumerate(obs_n):
        phi_init = get_feature_vector(
            agent_obs, 
            NUM_AGENTS, 
            NUM_LANDMARKS,
            normalization=FEAT_NORMALIZATION
        )
        agents[i].prev_phi = phi_init
    
    # Metrics to track
    sbe_history = []
    msbe_running = []
    consensus_error_history = []
    
    # Track total samples and communication rounds
    total_samples = 0
    
    # Main experiment loop - following Algorithm 1
    pbar = tqdm(total=L*K, desc=f"Local TD with {network_type} network")
    start_time = time.time()
    
    for l in range(L):
        # For each communication round
        for k in range(K):
            # For each local TD update step
            
            # Select random actions - uniform random policy as in paper
            actions_n = []
            for i in range(NUM_AGENTS):
                # Sample random action from the action space
                # For simple_spread, actions are one-hot encoded for discrete movement
                action = np.zeros(5)  # 5 actions: [no-op, left, right, up, down]
                action_idx = np.random.choice(5)
                action[action_idx] = 1
                
                # Add communication dimension (zeros since no communication)
                actions_n.append(np.concatenate([action, np.zeros(env.world.dim_c)]))
            
            # Step environment
            next_obs_n, reward_n, done_n, truncated_n, info_n = env.step(actions_n)

            # Reset environment if episode terminates
            if any(done_n):
                obs_n, _ = env.reset()
                for i in range(NUM_AGENTS):
                    phi_reset = get_feature_vector(
                        obs_n[i], 
                        NUM_AGENTS, 
                        NUM_LANDMARKS,
                        normalization=FEAT_NORMALIZATION
                    )
                    agents[i].prev_phi = phi_reset
                continue
            
            # Perform local TD updates for all agents
            sample_data = {}
            
            for i in range(NUM_AGENTS):
                # Get feature vectors
                phi_s = agents[i].prev_phi
                phi_s_next = get_feature_vector(
                    next_obs_n[i], 
                    NUM_AGENTS, 
                    NUM_LANDMARKS,
                    normalization=FEAT_NORMALIZATION
                )
                
                # Get reward
                reward = reward_n[i]
                
                # Update agent and store data for metrics
                update_data = agents[i].update(phi_s, phi_s_next, reward, BETA)
                sample_data[i] = update_data
                
                # Store next state feature
                agents[i].prev_phi = phi_s_next
            
            # Calculate metrics
            sbe = calculate_SBE(sample_data)
            sbe_history.append(sbe)
            msbe_running.append(np.mean(sbe_history))
            
            # Calculate consensus error
            weights_dict = {i: agents[i].w for i in range(NUM_AGENTS)}
            consensus_error = calculate_consensus_error(weights_dict)
            consensus_error_history.append(consensus_error)
            
            # Update observations for next step
            obs_n = next_obs_n.copy()
            
            # Increment counter and update progress bar
            total_samples += 1
            pbar.update(1)
        
        # After K local updates, perform consensus - Line 12 in Algorithm 1
        W = np.stack([agents[i].w for i in range(NUM_AGENTS)])
        W_new = np.matmul(A_consensus, W)
        
        for i in range(NUM_AGENTS):
            agents[i].w = W_new[i]
    
    # Close environment and progress bar
    env.close()
    pbar.close()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Experiment completed in {elapsed_time:.2f} seconds")
    
    # Save metrics
    results_path = os.path.join(SAVE_DIR, f"local_td_{network_type}_L{L}_K{K}_beta{BETA}")
    np.save(f"{results_path}_msbe.npy", np.array(msbe_running))
    np.save(f"{results_path}_consensus.npy", np.array(consensus_error_history))
    
    # Return results
    return {
        'msbe': msbe_running,
        'consensus_error': consensus_error_history,
        'params': {
            'network': network_type,
            'num_agents': NUM_AGENTS,
            'num_landmarks': NUM_LANDMARKS,
            'beta': BETA,
            'L': L,
            'K': K,
            'feat_normalization': FEAT_NORMALIZATION,
            'init_scale': INIT_SCALE
        }
    }

# -------------------- Run Experiments on Different Topologies --------------------
def run_topology_comparison():
    """Run experiments with different network topologies and compare results"""
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Network topologies to test
    network_types = ['ring']
    results = {}
    
    # Run experiment for each network type
    for network in network_types:
        print(f"\nRunning Local TD with {network} network (L={L}, K={K}, beta={BETA})")
        
        results[network] = run_local_td_experiment(network_type=network)
    
    # Plot comparison (like Figure 11 in the paper)
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        msbe = results[network]['msbe']
        plt.plot(range(len(msbe)), msbe, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title(f"Effect of Network Topology on MSBE (L={L}, K={K}, β={BETA})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{SAVE_DIR}/topology_comparison_msbe.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot consensus error
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        consensus = results[network]['consensus_error']
        plt.plot(range(len(consensus)), consensus, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title(f"Effect of Network Topology on Consensus Error (L={L}, K={K}, β={BETA})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{SAVE_DIR}/topology_comparison_consensus.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seeds(SEED)
    
    # Run experiments with different network topologies
    print("\n============ Running Topology Comparison ============")
    results = run_topology_comparison()
    
    print("Experiment completed!")