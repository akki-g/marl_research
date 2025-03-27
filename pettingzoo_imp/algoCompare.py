import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import imageio
import os
import time
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -------------------- Device Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Feature Extraction --------------------
def get_feature_vector(obs, num_agents=9, num_landmarks=9):
    """
    Constructs the feature vector φ(s) for an agent based on its observation.
    For simple_spread_v3, the observation includes:
      - Agent's position (2D)
      - Landmark relative positions (2D for each landmark)
      - Other agents' relative positions (2D for each other agent)
    
    Returns a normalized feature vector.
    """
    agent_pos = obs[2:4]
    start_landmark = 4
    end_landmark = start_landmark + num_landmarks * 2
    landmark_rel_pos = obs[start_landmark:end_landmark]
    other_agents_rel_pos = obs[end_landmark:end_landmark + (num_agents - 1) * 2]
    feature_vector = np.concatenate([agent_pos, landmark_rel_pos, other_agents_rel_pos])
    
    # Normalize the feature vector as per the paper's assumptions
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    return feature_vector

# -------------------- Error Metrics --------------------
def calculate_consensus_error(weights):
    """
    Computes the consensus error as defined in the paper:
    CE(w_i^{k}) = (1/N) * sum_{i=1}^N ||w_i^k - \bar{w}^k||^2
    where \bar{w}^k is the average weight vector across all agents.
    """
    agent_ids = list(weights.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
    
    W = torch.stack([weights[agent] for agent in agent_ids])
    w_bar = torch.mean(W, dim=0)
    consensus_error = torch.mean(torch.sum((W - w_bar)**2, dim=1))
    return consensus_error.item()

def calculate_SBE(sample_data):
    """
    Computes the instantaneous squared Bellman error (SBE) for a given sample.
    
    SBE = (1/N) * sum_{i=1}^N (φ(s_k)^T w_i^k + \bar{μ}_k - \bar{r}_k - φ(s_{k+1})^T w_i^k)^2
    
    where \bar{μ}_k and \bar{r}_k are the averages across all agents.
    """
    agent_ids = list(sample_data.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
    
    r_bar = sum(sample_data[agent]['reward'] for agent in agent_ids) / N
    mu_bar = sum(sample_data[agent]['mu'].item() for agent in agent_ids) / N
    
    errors = []
    for agent in agent_ids:
        w = sample_data[agent]['w']
        phi_s = sample_data[agent]['phi_s']
        phi_s_next = sample_data[agent]['phi_s_next']
        
        value_current = torch.dot(phi_s, w)
        value_next = torch.dot(phi_s_next, w)
        
        error = value_current + mu_bar - r_bar - value_next
        errors.append(error ** 2)
    
    SBE = sum(errors) / N
    return SBE.item()

# -------------------- Network Topologies --------------------
def generate_ring_network(N):
    """
    Generates a ring topology consensus matrix.
    Each agent is connected to its left and right neighbors.
    """
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 0.4  # Self-weight
        A[i, (i-1) % N] = 0.3  # Left neighbor
        A[i, (i+1) % N] = 0.3  # Right neighbor
    return torch.tensor(A, device=device, dtype=torch.float32)

def generate_k_regular_network(N, k=4):
    """
    Generates a k-regular network where each agent is connected to k other agents.
    """
    if k >= N:
        # If k is too large, return fully connected network
        return generate_complete_network(N)
    
    # Initialize the adjacency matrix
    A = np.zeros((N, N))
    
    # Ensure k is even for symmetric connections
    if k % 2 != 0:
        k = k - 1
    
    # Connect each node to k/2 nodes on each side
    half_k = k // 2
    for i in range(N):
        A[i, i] = 1.0  # Self-connection
        for j in range(1, half_k + 1):
            A[i, (i + j) % N] = 1.0
            A[i, (i - j) % N] = 1.0
    
    # Convert to doubly stochastic
    row_sums = A.sum(axis=1)
    A = A / row_sums[:, np.newaxis]
    
    return torch.tensor(A, device=device, dtype=torch.float32)

def generate_erdos_renyi_network(N, p=0.5):
    """
    Generates an Erdos-Renyi random graph with connection probability p.
    """
    # Generate random adjacency matrix
    A = np.random.rand(N, N) < p
    # Make it symmetric
    A = np.logical_or(A, A.T).astype(float)
    # Set diagonal to 1
    np.fill_diagonal(A, 1.0)
    
    # Convert to doubly stochastic
    row_sums = A.sum(axis=1)
    A = A / row_sums[:, np.newaxis]
    
    return torch.tensor(A, device=device, dtype=torch.float32)

def generate_complete_network(N):
    """
    Generates a complete network where all agents are connected to each other.
    """
    A = np.ones((N, N)) / N
    return torch.tensor(A, device=device, dtype=torch.float32)

# -------------------- Agent Model --------------------
class AgentModel:
    """
    Agent model for TD learning with function approximation.
    Stores and updates the weight vector w and average reward estimate μ.
    """
    def __init__(self, agent_id, feat_dim, init_mu=0.0):
        self.agent_id = agent_id
        self.w = torch.zeros(feat_dim, device=device, dtype=torch.float32)
        self.mu = torch.tensor(init_mu, device=device, dtype=torch.float32)
        self.prev_phi = None

# -------------------- Algorithm Implementations --------------------
def vanilla_td_learning(env, num_agents, num_landmarks, num_samples, beta=0.1):
    """
    Implements vanilla TD learning with consensus after each sample.
    """
    # Initialize environment
    obs, info = env.reset(seed=42)
    
    # Initialize agent models
    sample_phi = get_feature_vector(obs[env.agents[0]], num_agents, num_landmarks)
    feat_dim = sample_phi.shape[0]
    agents_model = {agent: AgentModel(agent, feat_dim) for agent in env.agents}
    
    # Set initial feature vectors
    for agent in env.agents:
        phi_init = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                                device=device, dtype=torch.float32)
        agents_model[agent].prev_phi = phi_init
    
    # Consensus matrix (complete graph for vanilla TD)
    A_consensus = generate_complete_network(num_agents)
    
    # Metrics tracking
    msbe_history = []
    consensus_history = []
    
    # Main loop
    sample_count = 0
    progress_bar = tqdm(total=num_samples, desc="Vanilla TD")
    
    while sample_count < num_samples:
        # Take random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        # If episode terminated, reset environment
        if all(terms.values()) or all(truncs.values()):
            obs, info = env.reset()
            for agent in env.agents:
                phi_init = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                                       device=device, dtype=torch.float32)
                agents_model[agent].prev_phi = phi_init
            continue
        
        # Sample data for metrics
        sample_data = {}
        
        # Update weights for each agent
        for agent in env.agents:
            model = agents_model[agent]
            phi_s = model.prev_phi
            phi_s_next = torch.tensor(get_feature_vector(next_obs[agent], num_agents, num_landmarks),
                                      device=device, dtype=torch.float32)
            
            # TD update
            value_current = torch.dot(phi_s, model.w)
            value_next = torch.dot(phi_s_next, model.w)
            r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
            
            # TD error (average reward TD update)
            delta = r - model.mu + value_next - value_current
            
            # Update running average reward
            model.mu = (1 - beta) * model.mu + beta * r
            
            # Update weights
            model.w = model.w + beta * delta * phi_s
            
            # Store data for metrics
            sample_data[agent] = {
                'phi_s': phi_s,
                'phi_s_next': phi_s_next,
                'reward': r.item(),
                'mu': model.mu,
                'w': model.w
            }
            
            # Update feature vector for next step
            model.prev_phi = phi_s_next
        
        # Calculate metrics
        instantaneous_sbe = calculate_SBE(sample_data)
        msbe_history.append(instantaneous_sbe)
        
        # Consensus update after each sample
        # Get all agent weights
        all_agents = list(env.agents)
        W = torch.stack([agents_model[agent].w for agent in all_agents])
        
        # Perform consensus
        new_W = torch.matmul(A_consensus, W)
        
        # Update agent weights
        for i, agent in enumerate(all_agents):
            agents_model[agent].w = new_W[i]
        
        # Calculate consensus error after update
        weights_dict = {agent: agents_model[agent].w for agent in env.agents}
        consensus_error = calculate_consensus_error(weights_dict)
        consensus_history.append(consensus_error)
        
        # Update observations
        obs = next_obs.copy()
        
        # Increment counter and update progress bar
        sample_count += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate running MSBE
    running_msbe = []
    for i in range(len(msbe_history)):
        running_msbe.append(np.mean(msbe_history[:i+1]))
    
    return running_msbe, consensus_history

def batch_td_learning(env, num_agents, num_landmarks, num_samples, batch_size=20, beta=0.1):
    """
    Implements batch TD learning with consensus after each batch.
    """
    # Initialize environment
    obs, info = env.reset(seed=42)
    
    # Initialize agent models
    sample_phi = get_feature_vector(obs[env.agents[0]], num_agents, num_landmarks)
    feat_dim = sample_phi.shape[0]
    agents_model = {agent: AgentModel(agent, feat_dim) for agent in env.agents}
    
    # Set initial feature vectors
    for agent in env.agents:
        phi_init = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                               device=device, dtype=torch.float32)
        agents_model[agent].prev_phi = phi_init
    
    # Consensus matrix (complete for simplicity)
    A_consensus = generate_complete_network(num_agents)
    
    # Metrics tracking
    msbe_history = []
    consensus_history = []
    
    # Main loop
    sample_count = 0
    batch_count = 0
    
    # For batch approach, we need to store the batch updates
    batch_updates = {agent: torch.zeros_like(agents_model[agent].w) for agent in env.agents}
    batch_samples = {agent: [] for agent in env.agents}
    
    progress_bar = tqdm(total=num_samples, desc="Batch TD")
    
    while sample_count < num_samples:
        # Take random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        # If episode terminated, reset environment
        if all(terms.values()) or all(truncs.values()):
            obs, info = env.reset()
            for agent in env.agents:
                phi_init = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                                       device=device, dtype=torch.float32)
                agents_model[agent].prev_phi = phi_init
            continue
        
        # Sample data for metrics
        sample_data = {}
        
        # Collect batch samples for each agent
        for agent in env.agents:
            model = agents_model[agent]
            phi_s = model.prev_phi
            phi_s_next = torch.tensor(get_feature_vector(next_obs[agent], num_agents, num_landmarks),
                                     device=device, dtype=torch.float32)
            r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
            
            # Update running average reward (this is still updated per sample)
            model.mu = (1 - beta) * model.mu + beta * r
            
            # Store sample for batch update
            batch_samples[agent].append({
                'phi_s': phi_s,
                'phi_s_next': phi_s_next,
                'reward': r.item(),
                'mu': model.mu.clone()
            })
            
            # Store data for metrics
            sample_data[agent] = {
                'phi_s': phi_s,
                'phi_s_next': phi_s_next,
                'reward': r.item(),
                'mu': model.mu,
                'w': model.w
            }
            
            # Update feature vector for next step
            model.prev_phi = phi_s_next
        
        # Calculate metrics (based on current weights, not yet updated)
        instantaneous_sbe = calculate_SBE(sample_data)
        msbe_history.append(instantaneous_sbe)
        
        # Calculate consensus error before potential batch update
        weights_dict = {agent: agents_model[agent].w for agent in env.agents}
        consensus_error = calculate_consensus_error(weights_dict)
        consensus_history.append(consensus_error)
        
        # Check if we need to perform batch update
        batch_count += 1
        if batch_count == batch_size:
            # Process batch updates for each agent
            for agent in env.agents:
                model = agents_model[agent]
                batch_delta = torch.zeros(feat_dim, device=device, dtype=torch.float32)
                
                # Calculate average TD update over batch
                for sample in batch_samples[agent]:
                    phi_s = sample['phi_s']
                    phi_s_next = sample['phi_s_next']
                    r = sample['reward']
                    mu = sample['mu']
                    
                    value_current = torch.dot(phi_s, model.w)
                    value_next = torch.dot(phi_s_next, model.w)
                    
                    # TD error
                    delta = r - mu.item() + value_next - value_current
                    
                    # Accumulate batch update
                    batch_delta += delta * phi_s
                
                # Apply average batch update
                model.w = model.w + beta * batch_delta / batch_size
            
            # Perform consensus after batch update
            all_agents = list(env.agents)
            W = torch.stack([agents_model[agent].w for agent in all_agents])
            new_W = torch.matmul(A_consensus, W)
            
            for i, agent in enumerate(all_agents):
                agents_model[agent].w = new_W[i]
            
            # Reset batch counters and storage
            batch_count = 0
            batch_samples = {agent: [] for agent in env.agents}
        
        # Update observations
        obs = next_obs.copy()
        
        # Increment counter and update progress bar
        sample_count += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate running MSBE
    running_msbe = []
    for i in range(len(msbe_history)):
        running_msbe.append(np.mean(msbe_history[:i+1]))
    
    return running_msbe, consensus_history

def local_td_learning(env, num_agents, num_landmarks, num_samples, K=20, beta=0.05, topology='er'):
    """
    Implements local TD learning with consensus after K local updates.
    
    Args:
        env: PettingZoo environment
        num_agents: Number of agents
        num_landmarks: Number of landmarks
        num_samples: Total number of environment samples to collect
        K: Number of local TD updates between consensus steps
        beta: Step size for TD update
        topology: Network topology ('ring', '4-regular', '6-regular', 'er', 'complete')
    """
    # Initialize environment
    obs, info = env.reset(seed=42)
    
    # Initialize agent models
    sample_phi = get_feature_vector(obs[env.agents[0]], num_agents, num_landmarks)
    feat_dim = sample_phi.shape[0]
    agents_model = {agent: AgentModel(agent, feat_dim) for agent in env.agents}
    
    # Set initial feature vectors
    for agent in env.agents:
        phi_init = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                               device=device, dtype=torch.float32)
        agents_model[agent].prev_phi = phi_init
    
    # Set up consensus matrix based on topology
    if topology == 'ring':
        A_consensus = generate_ring_network(num_agents)
    elif topology == '4-regular':
        A_consensus = generate_k_regular_network(num_agents, k=4)
    elif topology == '6-regular':
        A_consensus = generate_k_regular_network(num_agents, k=6)
    elif topology == 'er':
        A_consensus = generate_erdos_renyi_network(num_agents, p=0.5)
    elif topology == 'complete':
        A_consensus = generate_complete_network(num_agents)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Metrics tracking
    msbe_history = []
    consensus_history = []
    
    # Main loop
    sample_count = 0
    local_step_count = 0
    
    progress_bar = tqdm(total=num_samples, desc=f"Local TD ({topology})")
    
    while sample_count < num_samples:
        # Take random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        # If episode terminated, reset environment
        if all(terms.values()) or all(truncs.values()):
            obs, info = env.reset()
            for agent in env.agents:
                phi_init = torch.tensor(get_feature_vector(obs[agent], num_agents, num_landmarks),
                                       device=device, dtype=torch.float32)
                agents_model[agent].prev_phi = phi_init
            continue
        
        # Sample data for metrics
        sample_data = {}
        
        # Update weights for each agent (local TD update)
        for agent in env.agents:
            model = agents_model[agent]
            phi_s = model.prev_phi
            phi_s_next = torch.tensor(get_feature_vector(next_obs[agent], num_agents, num_landmarks),
                                     device=device, dtype=torch.float32)
            
            value_current = torch.dot(phi_s, model.w)
            value_next = torch.dot(phi_s_next, model.w)
            r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
            
            # TD error (average reward TD update)
            delta = r - model.mu + value_next - value_current
            
            # Update running average reward
            model.mu = (1 - beta) * model.mu + beta * r
            
            # Update weights (local TD update)
            model.w = model.w + beta * delta * phi_s
            
            # Store data for metrics
            sample_data[agent] = {
                'phi_s': phi_s,
                'phi_s_next': phi_s_next,
                'reward': r.item(),
                'mu': model.mu,
                'w': model.w
            }
            
            # Update feature vector for next step
            model.prev_phi = phi_s_next
        
        # Calculate metrics
        instantaneous_sbe = calculate_SBE(sample_data)
        msbe_history.append(instantaneous_sbe)
        
        # Calculate consensus error
        weights_dict = {agent: agents_model[agent].w for agent in env.agents}
        consensus_error = calculate_consensus_error(weights_dict)
        consensus_history.append(consensus_error)
        
        # Increment local step counter
        local_step_count += 1
        
        # Check if we need to perform consensus
        if local_step_count == K:
            # Perform consensus
            all_agents = list(env.agents)
            W = torch.stack([agents_model[agent].w for agent in all_agents])
            new_W = torch.matmul(A_consensus, W)
            
            for i, agent in enumerate(all_agents):
                agents_model[agent].w = new_W[i]
            
            # Reset local step counter
            local_step_count = 0
            
            # Calculate consensus error after consensus (append the same MSBE as before)
            weights_dict = {agent: agents_model[agent].w for agent in env.agents}
            consensus_error = calculate_consensus_error(weights_dict)
            if sample_count + 1 < num_samples:  # Avoid appending beyond num_samples
                consensus_history.append(consensus_error)
                msbe_history.append(instantaneous_sbe)  # Repeat MSBE for visualization
                sample_count += 1  # Count this consensus step as a sample for tracking
                progress_bar.update(1)
        
        # Update observations
        obs = next_obs.copy()
        
        # Increment counter and update progress bar
        sample_count += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate running MSBE
    running_msbe = []
    for i in range(len(msbe_history)):
        running_msbe.append(np.mean(msbe_history[:i+1]))
    
    return running_msbe, consensus_history

# -------------------- Main Experiment --------------------
def run_experiment(num_agents=9, num_landmarks=9, num_samples=10000):
    """
    Runs the complete experiment comparing vanilla TD, batch TD, and local TD with different topologies.
    """
    results = {}
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/msbe", exist_ok=True)
    os.makedirs("results/consensus", exist_ok=True)
    
    # Create environment
    env = simple_spread_v3.parallel_env(
        N=num_agents, local_ratio=0.5, max_cycles=1000,
        continuous_actions=False
    )
    
    # Run vanilla TD
    print("Running Vanilla TD...")
    vanilla_msbe, vanilla_consensus = vanilla_td_learning(
        env, num_agents, num_landmarks, num_samples, beta=0.1
    )
    results["vanilla"] = {"msbe": vanilla_msbe, "consensus": vanilla_consensus}
    
    # Save results for vanilla TD
    np.save("results/msbe/vanilla.npy", np.array(vanilla_msbe))
    np.save("results/consensus/vanilla.npy", np.array(vanilla_consensus))
    
    # Run batch TD
    print("Running Batch TD...")
    batch_msbe, batch_consensus = batch_td_learning(
        env, num_agents, num_landmarks, num_samples, batch_size=20, beta=0.1
    )
    results["batch"] = {"msbe": batch_msbe, "consensus": batch_consensus}
    
    # Save results for batch TD
    np.save("results/msbe/batch.npy", np.array(batch_msbe))
    np.save("results/consensus/batch.npy", np.array(batch_consensus))
    
    # Run local TD with different topologies
    topologies = ["ring", "4-regular", "6-regular", "er", "complete"]
    
    for topology in topologies:
        print(f"Running Local TD with {topology} topology...")
        local_msbe, local_consensus = local_td_learning(
            env, num_agents, num_landmarks, num_samples, K=20, beta=0.05, topology=topology
        )
        results[f"local_{topology}"] = {"msbe": local_msbe, "consensus": local_consensus}
        
        # Save results for this topology
        np.save(f"results/msbe/local_{topology}.npy", np.array(local_msbe))
        np.save(f"results/consensus/local_{topology}.npy", np.array(local_consensus))
    
    # Close environment
    env.close()
    
    # Plot results
    plot_results(results, num_samples)
    
    return results

def plot_results(results, num_samples):
    """
    Plots MSBE and consensus error for all algorithms.
    """
    # Set up plots
    plt.figure(figsize=(16, 6))
    
    # Plot MSBE
    plt.subplot(1, 2, 1)
    plt.plot(results["vanilla"]["msbe"], label="Vanilla TD")
    plt.plot(results["batch"]["msbe"], label="Batch TD")
    
    # Add local TD with different topologies
    topologies = ["ring", "4-regular", "6-regular", "er", "complete"]
    colors = ['r', 'g', 'b', 'c', 'm']
    
    for i, topology in enumerate(topologies):
        plt.plot(results[f"local_{topology}"]["msbe"], 
                label=f"Local TD ({topology})",
                color=colors[i])
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title("MSBE Comparison")
    plt.legend()
    plt.grid(True)
    
    # Plot consensus error
    plt.subplot(1, 2, 2)
    plt.plot(results["vanilla"]["consensus"], label="Vanilla TD")
    plt.plot(results["batch"]["consensus"], label="Batch TD")
    
    for i, topology in enumerate(topologies):
        plt.plot(results[f"local_{topology}"]["consensus"], 
                label=f"Local TD ({topology})",
                color=colors[i])
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title("Consensus Error Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/comparison_plot.png", dpi=300, bbox_inches='tight')
    
    # Plot individual algorithm comparisons to match the paper's figures
    plt.figure(figsize=(16, 6))
    
    # Plot MSBE for vanilla, batch, and local (ER)
    plt.subplot(1, 2, 1)
    plt.plot(results["vanilla"]["msbe"], label="Vanilla TD")
    plt.plot(results["batch"]["msbe"], label="Batch TD")
    plt.plot(results["local_er"]["msbe"], label="Local TD")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title("MSBE for Main Algorithms (ER Network)")
    plt.legend()
    plt.grid(True)
    
    # Plot consensus error for the same algorithms
    plt.subplot(1, 2, 2)
    plt.plot(results["vanilla"]["consensus"], label="Vanilla TD")
    plt.plot(results["batch"]["consensus"], label="Batch TD")
    plt.plot(results["local_er"]["consensus"], label="Local TD")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title("Consensus Error for Main Algorithms (ER Network)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/main_comparison.png", dpi=300, bbox_inches='tight')
    
    # Also create plots showing communication rounds vs. MSBE
    # This requires some additional calculations for local TD and batch TD
    plt.figure(figsize=(12, 5))
    
    # Calculate MSBE per communication round
    comm_rounds_vanilla = np.arange(len(results["vanilla"]["msbe"]))
    comm_rounds_batch = np.arange(0, len(results["batch"]["msbe"]), 20)
    comm_rounds_local = np.arange(0, len(results["local_er"]["msbe"]), 20)
    
    # Get corresponding MSBE values (taking every 20th point for batch and local)
    msbe_per_comm_vanilla = results["vanilla"]["msbe"]
    msbe_per_comm_batch = [results["batch"]["msbe"][min(i, len(results["batch"]["msbe"])-1)] for i in range(0, len(results["batch"]["msbe"]), 20)]
    msbe_per_comm_local = [results["local_er"]["msbe"][min(i, len(results["local_er"]["msbe"])-1)] for i in range(0, len(results["local_er"]["msbe"]), 20)]
    
    # Trim to the same length if needed
    min_length = min(len(comm_rounds_vanilla), len(comm_rounds_batch), len(comm_rounds_local))
    comm_rounds_vanilla = comm_rounds_vanilla[:min_length]
    comm_rounds_batch = comm_rounds_batch[:min_length]
    comm_rounds_local = comm_rounds_local[:min_length]
    msbe_per_comm_vanilla = msbe_per_comm_vanilla[:min_length]
    msbe_per_comm_batch = msbe_per_comm_batch[:len(comm_rounds_batch)]
    msbe_per_comm_local = msbe_per_comm_local[:len(comm_rounds_local)]
    
    # Plot MSBE vs communication rounds
    plt.subplot(1, 2, 1)
    plt.plot(comm_rounds_vanilla, msbe_per_comm_vanilla, label="Vanilla TD")
    plt.plot(comm_rounds_batch, msbe_per_comm_batch, label="Batch TD")
    plt.plot(comm_rounds_local, msbe_per_comm_local, label="Local TD")
    
    plt.xlabel("Communication Round")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title("MSBE vs Communication Rounds")
    plt.legend()
    plt.grid(True)
    
    # Plot MSBE vs Samples
    plt.subplot(1, 2, 2)
    plt.plot(results["vanilla"]["msbe"][:num_samples], label="Vanilla TD")
    plt.plot(results["batch"]["msbe"][:num_samples], label="Batch TD")
    plt.plot(results["local_er"]["msbe"][:num_samples], label="Local TD")
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title("MSBE vs Samples")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/communication_vs_sample.png", dpi=300, bbox_inches='tight')
    
def run_experiment_varying_k(num_agents=9, num_landmarks=9, num_samples=10000):
    """
    Runs experiments with varying K values for the local TD update.
    This replicates Figure 14 from the paper.
    """
    results = {}
    
    # Create environment
    env = simple_spread_v3.parallel_env(
        N=num_agents, local_ratio=0.5, max_cycles=1000,
        continuous_actions=False
    )
    
    # Run local TD with different K values
    K_values = [10, 20, 50, 100, 200]
    
    for K in K_values:
        print(f"Running Local TD with K={K}...")
        local_msbe, local_consensus = local_td_learning(
            env, num_agents, num_landmarks, num_samples, K=K, beta=0.05, topology="4-regular"
        )
        results[f"K_{K}"] = {"msbe": local_msbe, "consensus": local_consensus}
        
        # Save results
        np.save(f"results/msbe/local_K_{K}.npy", np.array(local_msbe))
        np.save(f"results/consensus/local_K_{K}.npy", np.array(local_consensus))
    
    # Close environment
    env.close()
    
    # Plot results
    plt.figure(figsize=(16, 6))
    
    # Plot MSBE
    plt.subplot(1, 2, 1)
    colors = ['r', 'g', 'b', 'c', 'm']
    
    for i, K in enumerate(K_values):
        plt.plot(results[f"K_{K}"]["msbe"], 
                label=f"K={K}",
                color=colors[i])
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title("MSBE for Different K Values (4-Regular Network)")
    plt.legend()
    plt.grid(True)
    
    # Plot consensus error
    plt.subplot(1, 2, 2)
    
    for i, K in enumerate(K_values):
        plt.plot(results[f"K_{K}"]["consensus"], 
                label=f"K={K}",
                color=colors[i])
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title("Consensus Error for Different K Values (4-Regular Network)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/varying_K.png", dpi=300, bbox_inches='tight')
    
    # Also create a zoomed-in view of the consensus error for the first 2000 samples
    plt.figure(figsize=(8, 6))
    for i, K in enumerate(K_values):
        consensus_data = results[f"K_{K}"]["consensus"]
        zoomed_data = consensus_data[:min(2000, len(consensus_data))]
        plt.plot(range(len(zoomed_data)), zoomed_data, 
                label=f"K={K}",
                color=colors[i])
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title("Consensus Error for Different K Values (First 2000 Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/varying_K_zoomed.png", dpi=300, bbox_inches='tight')
    
    return results

def run_experiment_varying_beta(num_agents=9, num_landmarks=9, num_samples=10000):
    """
    Runs experiments with varying step sizes (beta) for local TD.
    This replicates Figures 18-20 from the paper.
    """
    results = {}
    
    # Create environment
    env = simple_spread_v3.parallel_env(
        N=num_agents, local_ratio=0.5, max_cycles=1000,
        continuous_actions=False
    )
    
    # Run local TD with different beta values
    beta_values = [0.005, 0.01, 0.05, 0.1]
    K = 20  # Fixed K value as in the paper
    
    for beta in beta_values:
        print(f"Running Local TD with beta={beta}...")
        local_msbe, local_consensus = local_td_learning(
            env, num_agents, num_landmarks, num_samples, K=K, beta=beta, topology="4-regular"
        )
        results[f"beta_{beta}"] = {"msbe": local_msbe, "consensus": local_consensus}
        
        # Save results
        np.save(f"results/msbe/local_beta_{beta}.npy", np.array(local_msbe))
        np.save(f"results/consensus/local_beta_{beta}.npy", np.array(local_consensus))
    
    # Close environment
    env.close()
    
    # Plot results
    plt.figure(figsize=(16, 6))
    
    # Plot MSBE
    plt.subplot(1, 2, 1)
    colors = ['r', 'g', 'b', 'c']
    
    for i, beta in enumerate(beta_values):
        plt.plot(results[f"beta_{beta}"]["msbe"], 
                label=f"β={beta}",
                color=colors[i])
    
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Squared Bellman Error")
    plt.title(f"MSBE for Different Step Sizes (K={K}, 4-Regular Network)")
    plt.legend()
    plt.grid(True)
    
    # Plot consensus error
    plt.subplot(1, 2, 2)
    
    for i, beta in enumerate(beta_values):
        plt.plot(results[f"beta_{beta}"]["consensus"], 
                label=f"β={beta}",
                color=colors[i])
    
    plt.xlabel("Sample Number")
    plt.ylabel("Consensus Error")
    plt.title(f"Consensus Error for Different Step Sizes (K={K}, 4-Regular Network)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/varying_beta.png", dpi=300, bbox_inches='tight')
    
    return results

if __name__ == "__main__":
    # Run the main experiment
    print("Running main experiment...")
    run_experiment(num_agents=9, num_landmarks=9, num_samples=10000)
    
    # Run experiment with varying K values
    print("\nRunning experiment with varying K values...")
    run_experiment_varying_k(num_agents=9, num_landmarks=9, num_samples=10000)
    
    # Run experiment with varying step sizes
    print("\nRunning experiment with varying step sizes...")
    run_experiment_varying_beta(num_agents=9, num_landmarks=9, num_samples=10000)