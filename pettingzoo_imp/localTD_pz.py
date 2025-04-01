import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import os
import networkx as nx
from tqdm import tqdm
import time
import random

# -------------------- Constants and Configuration --------------------
SEED = 42  # Global seed for reproducibility

# -------------------- Device Setup --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Set Random Seeds --------------------
def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -------------------- Network Topology Functions --------------------
def generate_ring_network(N):
    """
    Generate a ring network exactly as described in the paper:
    - Self weight: 0.4
    - Neighbor weights: 0.3 each
    """
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 0.4
        A[i, (i-1) % N] = 0.3
        A[i, (i+1) % N] = 0.3
    return torch.tensor(A, dtype=torch.float32, device=device)

def generate_regular_network(N, degree, self_weight=0.4):
    """
    Generate a k-regular network with specified degree
    """
    if degree >= N:
        return generate_complete_network(N)
    
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
            
    return torch.tensor(A, dtype=torch.float32, device=device)

def generate_erdos_renyi_network(N, p=0.5, self_weight=0.4):
    """
    Generate an Erdos-Renyi network with connection probability p
    """
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
            
    return torch.tensor(A, dtype=torch.float32, device=device)

def generate_complete_network(N):
    """
    Generate a complete network where every node is connected to every other node
    with equal weight 1/N
    """
    A = np.ones((N, N)) / N
    return torch.tensor(A, dtype=torch.float32, device=device)

def verify_doubly_stochastic(A, tol=1e-6):
    """Verify that a matrix is doubly stochastic (rows and columns sum to 1)"""
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    
    row_check = np.allclose(row_sums, 1.0, atol=tol)
    col_check = np.allclose(col_sums, 1.0, atol=tol)
    
    return row_check and col_check

# -------------------- Feature Extraction --------------------
def get_feature_vector(obs, num_agents=9, num_landmarks=9, normalization='l1'):
    """
    Extract features from observation as described in the paper.
    
    Args:
        obs: Observation from the environment
        num_agents: Number of agents in the environment
        num_landmarks: Number of landmarks in the environment
        normalization: Normalization method ('l1', 'l2', or None)
        
    Returns:
        Normalized feature vector
    """
    # Extract agent's position (2 dimensions)
    agent_pos = obs[2:4]
    
    # Extract landmark positions (2 dimensions per landmark)
    start_landmark = 4
    end_landmark = start_landmark + num_landmarks * 2
    landmark_rel_pos = obs[start_landmark:end_landmark]
    
    # Extract other agents' positions (2 dimensions per agent)
    other_agents_rel_pos = obs[end_landmark:end_landmark + (num_agents - 1) * 2]
    
    # Concatenate all features
    feature_vector = np.concatenate([agent_pos, landmark_rel_pos, other_agents_rel_pos])
    
    # Normalize feature vector
    if normalization == 'l1':
        norm = np.linalg.norm(feature_vector, ord=1)
        if norm > 0:
            feature_vector = feature_vector / norm
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
        
    W = torch.stack([weights[agent] for agent in agent_ids])
    w_bar = torch.mean(W, dim=0)
    errors = torch.sum((W - w_bar)**2, dim=1)
    
    return torch.mean(errors).cpu().item()

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
    mu_bar = sum(sample_data[agent]['mu'].cpu().item() for agent in agent_ids) / N
    
    total_error = 0.0
    for agent in agent_ids:
        w = sample_data[agent]['w']
        phi_s = sample_data[agent]['phi_s']
        phi_s_next = sample_data[agent]['phi_s_next']
        
        # Calculate TD target
        value_current = torch.dot(phi_s, w)
        value_next = torch.dot(phi_s_next, w)
        
        # Calculate Bellman error
        bellman_error = value_current + mu_bar - r_bar - value_next
        squared_error = bellman_error ** 2
        
        total_error += squared_error.cpu().item()
    
    return total_error / N

# -------------------- Agent Model --------------------
class LocalTDAgent:
    """Agent implementing the Local TD update from Algorithm 1"""
    def __init__(self, agent_id, feat_dim, init_scale=0.01):
        self.agent_id = agent_id
        
        # Initialize weights with small random values as in the paper
        self.w = torch.randn(feat_dim, device=device, dtype=torch.float32) * init_scale
        
        # Initialize average reward estimate to 0
        self.mu = torch.tensor(0.0, device=device, dtype=torch.float32)
        
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
        value_current = torch.dot(phi_s, self.w)
        
        # Calculate next state value
        value_next = torch.dot(phi_s_next, self.w)
        
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
            'reward': reward.cpu().item(),
            'mu': self.mu,
            'w': self.w,
            'delta': delta.cpu().item()
        }

# -------------------- Experiment Runner --------------------
def run_local_td_experiment(
    network_type='er',
    num_agents=9,
    num_landmarks=9,
    L=200,
    K=50,
    beta=0.01,
    feat_normalization='l2',
    init_scale=0.01,
    seed=SEED,
    save_dir="results_local_td"
):
    """
    Run Local TD experiment following Algorithm 1 from the paper
    
    Args:
        network_type: Type of network topology to use
        num_agents: Number of agents
        num_landmarks: Number of landmarks
        L: Number of communication rounds
        K: Number of local TD updates between communication
        beta: Step size parameter
        feat_normalization: Feature normalization method
        init_scale: Scale for weight initialization
        seed: Random seed
        save_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seeds(seed)
    
    print(f"Starting experiment with {network_type} network topology")
    print(f"Parameters: L={L}, K={K}, beta={beta}, num_agents={num_agents}")
    
    # Initialize environment with global rewards (local_ratio=0.0)
    env = simple_spread_v3.parallel_env(
        N=num_agents,
        local_ratio=0.0,  # Use global rewards only
        max_cycles=L*K + 100,  # Add buffer for potential resets
        continuous_actions=False
    )
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Create consensus matrix based on network topology
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
    
    # Verify consensus matrix is doubly stochastic
    if not verify_doubly_stochastic(A_consensus.cpu().numpy()):
        print(f"WARNING: Consensus matrix for {network_type} is not doubly stochastic!")
    
    # Initialize agents
    sample_obs = get_feature_vector(
        obs[env.agents[0]], 
        num_agents, 
        num_landmarks,
        normalization=feat_normalization
    )
    feat_dim = sample_obs.shape[0]
    print(f"Feature dimension: {feat_dim}")
    
    agents = {
        agent: LocalTDAgent(agent, feat_dim, init_scale) 
        for agent in env.agents
    }
    
    # Initialize feature vectors for each agent
    for agent in env.agents:
        phi_init = torch.tensor(
            get_feature_vector(obs[agent], num_agents, num_landmarks, normalization=feat_normalization),
            device=device, 
            dtype=torch.float32
        )
        agents[agent].prev_phi = phi_init
    
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
            actions = {agent: int(env.action_space(agent).sample()) for agent in env.agents}
            
            # Step environment
            next_obs, rewards, dones, truncs, infos = env.step(actions)
            
            # Reset environment if episode terminates
            if not env.agents or any(dones.values()):
                obs, info = env.reset(seed=seed + total_samples)  # Use different seed for each reset
                for agent in env.agents:
                    phi_reset = torch.tensor(
                        get_feature_vector(obs[agent], num_agents, num_landmarks, normalization=feat_normalization),
                        device=device, 
                        dtype=torch.float32
                    )
                    agents[agent].prev_phi = phi_reset
                continue
            
            # Perform local TD updates for all agents
            sample_data = {}
            
            for agent in env.agents:
                # Get feature vectors
                phi_s = agents[agent].prev_phi
                phi_s_next = torch.tensor(
                    get_feature_vector(next_obs[agent], num_agents, num_landmarks, normalization=feat_normalization),
                    device=device, 
                    dtype=torch.float32
                )
                
                # Get reward
                r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
                
                # Update agent and store data for metrics
                update_data = agents[agent].update(phi_s, phi_s_next, r, beta)
                sample_data[agent] = update_data
                
                # Store next state feature
                agents[agent].prev_phi = phi_s_next
            
            # Calculate metrics
            sbe = calculate_SBE(sample_data)
            sbe_history.append(sbe)
            msbe_running.append(np.mean(sbe_history))
            
            # Calculate consensus error
            weights_dict = {agent: agents[agent].w for agent in env.agents}
            consensus_error = calculate_consensus_error(weights_dict)
            consensus_error_history.append(consensus_error)
            
            # Update observations for next step
            obs = next_obs.copy()
            
            # Increment counter and update progress bar
            total_samples += 1
            pbar.update(1)
        
        # After K local updates, perform consensus - Line 12 in Algorithm 1
        if env.agents:
            W = torch.stack([agents[agent].w for agent in env.agents])
            W_new = torch.matmul(A_consensus, W)
            
            for i, agent in enumerate(env.agents):
                agents[agent].w = W_new[i]
    
    # Close environment and progress bar
    env.close()
    pbar.close()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Experiment completed in {elapsed_time:.2f} seconds")
    
    # Save metrics
    results_path = os.path.join(save_dir, f"local_td_{network_type}_L{L}_K{K}_beta{beta}")
    np.save(f"{results_path}_msbe.npy", np.array(msbe_running))
    np.save(f"{results_path}_consensus.npy", np.array(consensus_error_history))
    
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
            'K': K,
            'feat_normalization': feat_normalization,
            'init_scale': init_scale
        }
    }

# -------------------- Comparison Across Topologies --------------------
def run_topology_comparison(
    L=200,
    K=50,
    beta=0.01,
    feat_normalization='l2',
    init_scale=0.01,
    seed=SEED,
    save_dir="topology_comparison"
):
    """
    Run Local TD algorithm on different network topologies and compare results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Network topologies to test
    network_types = ['ring']
    results = {}
    
    # Run experiment for each network type
    for network in network_types:
        print(f"\nRunning Local TD with {network} network (L={L}, K={K}, beta={beta})")
        
        results[network] = run_local_td_experiment(
            network_type=network,
            L=L,
            K=K,
            beta=beta,
            feat_normalization=feat_normalization,
            init_scale=init_scale,
            seed=seed,
            save_dir=save_dir
        )
    
    # Create smoothed versions of results for plotting
    window_size = 100  # Smoothing window size
    smoothed_results = {}
    
    for network in network_types:
        msbe = results[network]['msbe']
        consensus = results[network]['consensus_error']
        
        # Apply smoothing
        smoothed_msbe = np.convolve(msbe, np.ones(window_size)/window_size, mode='valid')
        smoothed_consensus = np.convolve(consensus, np.ones(window_size)/window_size, mode='valid')
        
        smoothed_results[network] = {
            'msbe': smoothed_msbe,
            'consensus_error': smoothed_consensus
        }
    
    # Plot MSBE comparison (like Figure 11 in the paper)
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        msbe = results[network]['msbe']
        plt.plot(range(len(msbe)), msbe, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title(f"Effect of Network Topology on MSBE (L={L}, K={K}, β={beta})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/figure11_msbe_raw.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot smoothed MSBE
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        msbe = smoothed_results[network]['msbe']
        x_vals = range(window_size-1, window_size-1+len(msbe))
        plt.plot(x_vals, msbe, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error (Smoothed)")
    plt.title(f"Effect of Network Topology on MSBE (L={L}, K={K}, β={beta})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/figure11_msbe_smoothed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot consensus error
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        consensus = results[network]['consensus_error']
        plt.plot(range(len(consensus)), consensus, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title(f"Effect of Network Topology on Consensus Error (L={L}, K={K}, β={beta})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/figure11_consensus_raw.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot smoothed consensus error
    plt.figure(figsize=(10, 6))
    
    for network in network_types:
        consensus = smoothed_results[network]['consensus_error']
        x_vals = range(window_size-1, window_size-1+len(consensus))
        plt.plot(x_vals, consensus, label=f"{network.capitalize()} Network")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error (Smoothed)")
    plt.title(f"Effect of Network Topology on Consensus Error (L={L}, K={K}, β={beta})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/figure11_consensus_smoothed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

# -------------------- Parameter Sensitivity Experiments --------------------
def run_step_size_comparison(
    network_type='er',
    L=200,
    K=50,
    betas=0.1,
    save_dir="step_size_comparison"
):
    """Run experiments with different step sizes to find optimal value"""
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    for beta in betas:
        print(f"\nRunning with step size β={beta}")
        results[beta] = run_local_td_experiment(
            network_type=network_type,
            L=L,
            K=K,
            beta=beta,
            save_dir=save_dir
        )
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    for beta in betas:
        msbe = results[beta]['msbe']
        plt.plot(range(len(msbe)), msbe, label=f"β={beta}")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title(f"Effect of Step Size on MSBE ({network_type.capitalize()} Network, L={L}, K={K})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/step_size_comparison_msbe.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def run_K_value_comparison(
    network_type='er',
    L=200,
    K_values=[10, 20, 50, 100, 200],
    beta=0.01,
    save_dir="K_value_comparison"
):
    """Run experiments with different K values to study local update effect"""
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    for K in K_values:
        print(f"\nRunning with K={K} local updates")
        results[K] = run_local_td_experiment(
            network_type=network_type,
            L=L,
            K=K,
            beta=beta,
            save_dir=save_dir
        )
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    for K in K_values:
        msbe = results[K]['msbe']
        plt.plot(range(len(msbe)), msbe, label=f"K={K}")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title(f"Effect of Local Update Steps on MSBE ({network_type.capitalize()} Network, L={L}, β={beta})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/K_value_comparison_msbe.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

# -------------------- Main Entry Point --------------------
if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seeds(SEED)
    
    # Run topology comparison with recommended parameters
    print("\n============ Running Topology Comparison ============")
    results = run_topology_comparison(
        L=200,
        K=50,
        beta=0.1,  # Try a smaller beta for more stability
        feat_normalization='l1',  # Try L2 normalization
        init_scale=0.01,
        save_dir="topology_comparison_revised"
    )
    
    print("Experiment completed!")