import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import imageio
import os
import networkx as nx
from tqdm import tqdm

# -------------------- Device Setup --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Network Topology Functions --------------------
def generate_ring_network(N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 0.4  # Exactly as specified in paper
        A[i, (i-1) % N] = 0.3  # Left neighbor
        A[i, (i+1) % N] = 0.3  # Right neighbor
    return torch.tensor(A, dtype=torch.float32, device=device)

def generate_regular_network(N, degree, diag_weight=0.4):
    """
    Generate k-regular network (each node has exactly k neighbors)
    """
    if degree >= N:
        return generate_complete_network(N)
    
    G = nx.random_regular_graph(degree, N)
    A = np.zeros((N, N))
    
    # Calculate appropriate weights for neighbors
    neighbor_weight = (1.0 - diag_weight) / degree
    
    # Set the adjacency matrix
    for i in range(N):
        A[i, i] = diag_weight
        for j in list(G.neighbors(i)):
            A[i, j] = neighbor_weight
            
    return torch.tensor(A, dtype=torch.float32, device=device)

def generate_erdos_renyi_network(N, p=0.5, diag_weight=0.4):
    """
    Generate an Erdos-Renyi network with connection probability p
    """
    G = nx.erdos_renyi_graph(N, p)
    A = np.zeros((N, N))
    
    # Calculate weights for each node based on its degree
    for i in range(N):
        neighbors = list(G.neighbors(i))
        degree = len(neighbors)
        
        # If the node is isolated, make it self-connected with weight 1
        if degree == 0:
            A[i, i] = 1.0
            continue
            
        # Otherwise, assign appropriate weights
        neighbor_weight = (1.0 - diag_weight) / degree
        
        A[i, i] = diag_weight
        for j in neighbors:
            A[i, j] = neighbor_weight
            
    return torch.tensor(A, dtype=torch.float32, device=device)

def generate_complete_network(N):
    """
    Generate a complete network where all agents are connected
    """
    weight = 1.0 / N
    A = np.ones((N, N)) * weight
    return torch.tensor(A, dtype=torch.float32, device=device)

# -------------------- Feature Extraction --------------------
def get_feature_vector(obs, num_agents=9, num_landmarks=9):
    """
    Extract features from the observation as described in the paper:
    - Agent's own position (2 dims)
    - Landmark relative positions (num_landmarks * 2 dims)
    - Other agents' relative positions ((num_agents-1) * 2 dims)
    """
    agent_pos = obs[2:4]  # Self position (2D)
    
    # Landmark relative positions
    start_landmark = 4
    end_landmark = start_landmark + num_landmarks * 2
    landmark_rel_pos = obs[start_landmark:end_landmark]
    
    # Other agents' relative positions
    other_agents_rel_pos = obs[end_landmark:end_landmark + (num_agents - 1) * 2]
    
    # Concatenate all features
    feature_vector = np.concatenate([agent_pos, landmark_rel_pos, other_agents_rel_pos])
    
    # Normalize the feature vector (using L1 norm as per the paper)
    norm = np.linalg.norm(feature_vector, ord=1)
    if norm > 0:
        feature_vector = feature_vector / norm
        
    return feature_vector

# -------------------- Error Metrics --------------------
def calculate_consensus_error(weights):
    """
    Calculate the consensus error across all agents:
    CE = (1/N) * sum_i ||w_i - w_bar||^2
    where w_bar is the average weight vector
    """
    agent_ids = list(weights.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
        
    W = torch.stack([weights[agent] for agent in agent_ids], dim=0)
    w_bar = torch.mean(W, dim=0)
    errors = torch.norm(W - w_bar, dim=1) ** 2
    
    # Move tensor to CPU before converting to numpy/scalar
    return torch.mean(errors).cpu().item()

def calculate_SBE(sample_data):
    """
    Calculate the Squared Bellman Error for a given sample
    """
    agent_ids = list(sample_data.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
        
    # Calculate averages across agents
    r_bar = sum(sample_data[agent]['reward'] for agent in agent_ids) / N
    mu_bar = sum(sample_data[agent]['mu'].cpu().item() for agent in agent_ids) / N
    
    errors = []
    for agent in agent_ids:
        w = sample_data[agent]['w']
        phi_s = sample_data[agent]['phi_s']
        phi_s_next = sample_data[agent]['phi_s_next']
        
        value_current = torch.dot(phi_s, w)
        value_next = torch.dot(phi_s_next, w)
        
        error = value_current + mu_bar - r_bar - value_next
        errors.append(error ** 2)
    
    # Sum errors and convert to CPU scalar
    total_error = sum(error.cpu().item() if isinstance(error, torch.Tensor) else error for error in errors)
    return total_error / N

# -------------------- Agent Model --------------------
class LocalTDAgent:
    """
    Implementation of an agent for Local TD updates following Algorithm 1
    """
    def __init__(self, agent_id, feat_dim, init_mu=0.0):
        self.agent_id = agent_id
        self.w = torch.randn(feat_dim, device=device, dtype=torch.float32) * 0.01
        self.mu = torch.tensor(init_mu, device=device, dtype=torch.float32)
        self.prev_phi = None
        
    def update(self, phi_s, phi_s_next, reward, beta):
        """
        Perform a TD update based on a transition (s, s', r) with step size beta
        """
        # Calculate current and next state values
        value_current = torch.dot(phi_s, self.w)
        value_next = torch.dot(phi_s_next, self.w)
        
        # Calculate TD error (δ = r - μ + φ(s')ᵀw - φ(s)ᵀw)
        delta = reward - self.mu + value_next - value_current
        
        # Update running average reward (μ = (1-β)μ + βr)
        self.mu = (1 - beta) * self.mu + beta * reward
        
        # Update weight vector (w = w + βδφ(s))
        self.w = self.w + beta * delta * phi_s
        
        return {
            'phi_s': phi_s,
            'phi_s_next': phi_s_next,
            'reward': reward.cpu().item(),  # Move to CPU for safe storage
            'mu': self.mu,
            'w': self.w,
            'delta': delta.cpu().item()  # Move to CPU for safe storage
        }

# -------------------- Local TD(0) Algorithm (Algorithm 1) --------------------
def run_local_td_algorithm(network_type='er', num_agents=9, num_landmarks=9, 
                           L=200, K=50, beta=0.05, seed=42, save_dir="results"):
    """
    Run the Local TD(0) Algorithm (Algorithm 1 from the paper)
    
    Parameters:
    - network_type: 'ring', '4-regular', '6-regular', 'er', or 'complete'
    - num_agents: number of agents
    - num_landmarks: number of landmarks
    - L: number of communication rounds
    - K: number of local TD-update steps per round
    - beta: step size
    - seed: random seed
    - save_dir: directory to save results
    
    Returns:
    - Dictionary with MSBE and consensus error history
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment
    env = simple_spread_v3.parallel_env(
        N=num_agents, 
        local_ratio=0.5,
        max_cycles=L*K + 100,  # Add buffer for resets
        continuous_actions=False
    )
    
    # Generate consensus matrix based on network type
    if network_type == 'ring':
        A_consensus = generate_ring_network(num_agents)
    elif network_type == '4-regular':
        A_consensus = generate_regular_network(num_agents, 4)
    elif network_type == '6-regular':
        A_consensus = generate_regular_network(num_agents, 6)
    elif network_type == 'er':
        A_consensus = generate_erdos_renyi_network(num_agents, p=0.5)
    elif network_type == 'complete':
        A_consensus = generate_complete_network(num_agents)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Reset environment
    obs, infos = env.reset(seed=seed)
    
    # Initialize agent models
    sample_phi = get_feature_vector(obs[env.agents[0]], num_agents, num_landmarks)
    feat_dim = sample_phi.shape[0]
    print(f"Feature dimension: {feat_dim}")
    
    agents = {agent: LocalTDAgent(agent, feat_dim) for agent in env.agents}
    
    # Initialize feature vectors
    for agent in env.agents:
        phi_init = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                               device=device, dtype=torch.float32)
        agents[agent].prev_phi = phi_init
    
    # Metrics to track
    sbe_history = []
    msbe_running = []
    consensus_error_history = []
    
    # Main loop - following Algorithm 1 structure
    pbar = tqdm(total=L, desc=f"Local TD(0) with {network_type} network")
    
    for l in range(L):
        # For each communication round
        for k in range(K):
            # For each local step
            
            # Get random actions (following uniform random policy)
            actions = {agent: int(env.action_space(agent).sample()) for agent in env.agents}
            
            # Step the environment
            next_obs, rewards, dones, truncs, infos = env.step(actions)
            
            # Check if we need to reset
            if not env.agents:
                obs, infos = env.reset()
                for agent in env.agents:
                    phi_reset = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                                           device=device, dtype=torch.float32)
                    agents[agent].prev_phi = phi_reset
                continue
            
            # Perform local TD updates for each agent
            sample_data = {}
            for agent in env.agents:
                phi_s = agents[agent].prev_phi
                phi_s_next = torch.tensor(get_feature_vector(next_obs[agent], num_agents, num_landmarks),
                                       device=device, dtype=torch.float32)
                r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
                
                # Update agent using TD update rule
                update_data = agents[agent].update(phi_s, phi_s_next, r, beta)
                sample_data[agent] = update_data
                
                # Store next state feature for next update
                agents[agent].prev_phi = phi_s_next
            
            # Calculate and store metrics for this step
            instantaneous_sbe = calculate_SBE(sample_data)
            sbe_history.append(instantaneous_sbe)
            msbe_running.append(np.mean(sbe_history))
            
            weights_dict = {agent: agents[agent].w for agent in env.agents}
            consensus_error = calculate_consensus_error(weights_dict)
            consensus_error_history.append(consensus_error)
            
            # Update observation for next step
            obs = next_obs
        
        # Perform consensus update after K local steps (Line 12 in Algorithm 1)
        if env.agents:
            W = torch.stack([agents[agent].w for agent in env.agents])
            W_new = torch.matmul(A_consensus, W)
            
            for i, agent in enumerate(env.agents):
                agents[agent].w = W_new[i]
        
        pbar.update(1)
    
    # Clean up
    env.close()
    pbar.close()
    
    # Save metrics
    np.save(f"{save_dir}/local_td_{network_type}_L{L}_K{K}_msbe.npy", np.array(msbe_running))
    np.save(f"{save_dir}/local_td_{network_type}_L{L}_K{K}_consensus.npy", np.array(consensus_error_history))
    
    # Return results
    return {
        'msbe': msbe_running,
        'consensus_error': consensus_error_history,
        'params': {
            'network': network_type,
            'num_agents': num_agents,
            'num_landmarks': num_landmarks,
            'beta': beta,
            'L': L,
            'K': K
        }
    }

# -------------------- Run the Algorithm on Different Topologies --------------------
def run_comparison_across_topologies(L=200, K=50, beta=0.05, save_dir="topology_comparison"):
    """
    Run the Local TD algorithm on different network topologies and plot the results
    to recreate Figure 11 from the paper
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Network topologies to test
    network_types = ['ring', '4-regular', '6-regular', 'er', 'complete']
    results = {}
    
    # Run algorithm for each network type
    for network in network_types:
        print(f"\nRunning Local TD with {network} network (L={L}, K={K})")
        results[network] = run_local_td_algorithm(
            network_type=network,
            L=L,
            K=K,
            beta=beta,
            save_dir=save_dir
        )
    
    # Plot comparison of MSBE for all network types (like Figure 11)
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        msbe = results[network]['msbe']
        plt.plot(range(len(msbe)), msbe, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title(f"Effect of Network Topology on MSBE (L={L}, K={K})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/figure11_msbe_comparison.png", dpi=300)
    plt.close()
    
    # Plot comparison of consensus error for all network types
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        consensus = results[network]['consensus_error']
        plt.plot(range(len(consensus)), consensus, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title(f"Effect of Network Topology on Consensus Error (L={L}, K={K})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/figure11_consensus_comparison.png", dpi=300)
    plt.close()
    
    return results

# -------------------- Visualize Different Topology Types --------------------
def visualize_topologies():
    """Create visualizations of the different network topologies"""
    
    network_types = ['ring', '4-regular', '6-regular', 'er', 'complete']
    N = 9  # Number of agents
    
    plt.figure(figsize=(15, 3))
    
    for i, network_type in enumerate(network_types):
        plt.subplot(1, 5, i+1)
        
        if network_type == 'ring':
            G = nx.cycle_graph(N)
        elif network_type == '4-regular':
            G = nx.random_regular_graph(4, N)
        elif network_type == '6-regular':
            G = nx.random_regular_graph(6, N)
        elif network_type == 'er':
            G = nx.erdos_renyi_graph(N, p=0.5)
        elif network_type == 'complete':
            G = nx.complete_graph(N)
        
        # Draw the graph
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
        plt.title(f"{network_type.capitalize()} Network")
    
    plt.tight_layout()
    plt.savefig("network_topologies.png", dpi=300)
    plt.close()

# -------------------- Main Entry Point --------------------
if __name__ == "__main__":
    # Visualize network topologies
    visualize_topologies()
    
    # Run the Local TD algorithm with parameters L=200, K=50 as requested
    results = run_comparison_across_topologies(L=200, K=50, beta=0.05)
    
    print("Experiment completed!")