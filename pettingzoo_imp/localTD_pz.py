import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3

# Use MPS if available; otherwise, use CPU (or "cuda" if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

def get_feature_vector(obs, num_landmarks, num_agents):
    """
    Constructs the feature vector φ(s) for an agent.
    Assumes the observation from simple_spread_v3 is structured as:
      [self_vel (2), self_pos (2), landmark_rel_positions (num_landmarks*2),
       other_agent_rel_positions ((num_agents-1)*2), communication (remaining dims)]
       
    For our cooperative navigation experiment we ignore the communication part.
    For N agents and N landmarks (with N=9 in the paper), we extract:
      - self_pos: 2 dims (indices 2:4)
      - landmark_rel_pos: 2*num_landmarks dims (indices 4:4+num_landmarks*2)
      - other_agents_rel_pos: 2*(num_agents-1) dims (following immediately)
      
    This yields a 2 + 2*num_landmarks + 2*(num_agents-1) dimensional vector.
    For N=9 and num_landmarks=9, that gives 2 + 18 + 16 = 36 dimensions.
    The vector is then normalized.
    """
    # Extract self position from indices 2:4 (since obs[0:2] are self_vel)
    agent_pos = obs[2:4]
    start_landmark = 4
    end_landmark = start_landmark + num_landmarks * 2
    landmark_rel_pos = obs[start_landmark:end_landmark]
    # Extract other agents relative positions (ignoring communication)
    other_agents_rel_pos = obs[end_landmark:end_landmark + (num_agents - 1) * 2]
    feature_vector = np.concatenate([agent_pos, landmark_rel_pos, other_agents_rel_pos])
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    return feature_vector

def calculate_consensus_error(weights):
    """
    Computes the consensus error:
      ConsensusError = (1/N) ∑_i ||w_i - w̄||²,
    where w̄ is the average weight vector over all agents.
    """
    agent_ids = list(weights.keys())
    N = len(agent_ids)
    W = torch.stack([weights[agent] for agent in agent_ids], dim=0)  # shape (N, feat_dim)
    w_bar = torch.mean(W, dim=0)
    consensus_error = torch.mean(torch.sum((W - w_bar) ** 2, dim=1))
    return consensus_error.item()

def calculate_msbe(sample_data):
    """
    Computes the squared Bellman error (SBE) for one sample and returns the MSBE over agents.
    
    For each agent i, define:
      error_i = φ(s)ᵀw_i + μ̄ - r̄ - φ(s')ᵀw_i,
    where r̄ and μ̄ are the averages of rewards and running averages over all agents.
    Then, MSBE = (1/N) ∑_i (error_i)².
    
    Args:
        sample_data: dict mapping agent IDs to a dict with keys:
           'phi_s'      : φ(s) (torch.Tensor, shape [feat_dim])
           'phi_s_next' : φ(s') (torch.Tensor, shape [feat_dim])
           'reward'     : observed reward (float)
           'mu'         : current running average (torch scalar)
           'w'          : weight vector (torch.Tensor, shape [feat_dim])
    Returns:
        msbe (float)
    """
    agent_ids = list(sample_data.keys())
    N = len(agent_ids)
    r_bar = sum(sample_data[agent]['reward'] for agent in agent_ids) / N
    mu_bar = sum(sample_data[agent]['mu'] for agent in agent_ids) / N
    errors = []
    for agent in agent_ids:
        w = sample_data[agent]['w']
        phi_s = sample_data[agent]['phi_s']
        phi_s_next = sample_data[agent]['phi_s_next']
        value_current = torch.dot(phi_s, w)
        value_next = torch.dot(phi_s_next, w)
        error = value_current + mu_bar - r_bar - value_next
        errors.append(error ** 2)
    msbe = sum(errors) / N
    return msbe.item()

def generate_consensus_matrix(N, p=0.5):
    """
    Generates a consensus matrix A for N agents based on an Erdos–Rényi (ER) graph
    with connection probability p. We use a Metropolis–Hastings weighting rule:
      For i ≠ j, if an edge exists then:
          A[i,j] = 1 / (1 + max{deg(i), deg(j)}),
      and A[i,i] = 1 - ∑_{j ≠ i} A[i,j].
    Returns:
        A: torch.Tensor of shape (N, N)
    """
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() < p:
                A[i, j] = 1
                A[j, i] = 1
    degrees = A.sum(axis=1)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        W[i, i] = 1.0 - W[i].sum()
    return torch.tensor(W, dtype=torch.float32)

class AgentModel:
    """
    Stores the learnable weight vector w and running average μ for an agent.
    The value function is approximated as V(s) = φ(s)ᵀw.
    """
    def __init__(self, agent_id, feat_dim, init_mu=0.0):
        self.agent_id = agent_id
        self.w = torch.zeros(feat_dim, device=device, dtype=torch.float32, requires_grad=False)
        self.mu = torch.tensor(init_mu, device=device, dtype=torch.float32)
        self.prev_phi = None  # Stores φ(s) from the previous step
        self.delta_history = []  # Records TD errors for analysis

# Hyperparameters and settings
L = 200            # Number of communication rounds
K = 50            # Number of local TD-update (sample) steps per round
beta = 0.1        # Learning rate (step size)
num_agents = 9    # Number of agents (and landmarks, per cooperative navigation task)
num_landmarks = num_agents  # In simple spread, typically number of landmarks equals number of agents
consensus_p = 0.5 # ER connection probability

# Create the environment using the parallel API.
env = simple_spread_v3.env(N=num_agents, local_ratio=0.5, max_cycles=None, continuous_actions=False, render_mode='rgb_array')
obs = env.reset(seed=42)  # obs is a dict mapping agent IDs to observations

# Initialize AgentModel for each agent.
# Compute feature dimension using one agent's observation.
sample_phi = get_feature_vector(obs[env.agents[0]], num_landmarks, num_agents)
feat_dim = len(sample_phi)  # should be 36 for N=9 and num_landmarks=9
agents_model = {agent: AgentModel(agent, feat_dim) for agent in env.agents}

# Initialize each agent's prev_phi.
for agent in env.agents:
    phi = torch.tensor(get_feature_vector(obs[agent], num_landmarks, num_agents), device=device, dtype=torch.float32)
    agents_model[agent].prev_phi = phi

# Prepare lists to record MSBE and consensus error at every sample update.
msbe_per_sample = []
consensus_per_sample = []
sample_count = 0

# Generate consensus matrix based on ER network.
A_consensus = generate_consensus_matrix(num_agents, p=consensus_p)
print("Consensus matrix A:\n", A_consensus.numpy())

# Main loop: for each communication round (L rounds), perform K local TD-update steps.
for l in range(L):
    print(f"\n=== Communication Round {l} ===")
    for k in range(K):
        # Sample one step: select random actions for all agents.
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        new_obs, rewards, dones, truncs, infos = env.step(actions)
        
        # Prepare a dictionary to collect sample data for MSBE calculation at this sample.
        sample_data = {}
        
        for agent in env.agents:
            model = agents_model[agent]
            # φ(s) is stored in model.prev_phi.
            phi_s = model.prev_phi
            # Compute φ(s') for the new observation.
            phi_s_next = torch.tensor(get_feature_vector(new_obs[agent], num_landmarks, num_agents),
                                        device=device, dtype=torch.float32)
            value_current = torch.dot(phi_s, model.w)
            value_next = torch.dot(phi_s_next, model.w)
            r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
            # TD error: δ = r - μ + φ(s')ᵀw - φ(s)ᵀw.
            delta = r - model.mu + value_next - value_current
            model.delta_history.append(delta.item())
            # Update weight: w ← w + beta * δ * φ(s)
            model.w = model.w + beta * delta * phi_s
            # Update running average: μ ← beta * r + (1 - beta) * μ.
            model.mu = beta * r + (1 - beta) * model.mu
            # Record data for MSBE calculation.
            sample_data[agent] = {
                'phi_s': phi_s,
                'phi_s_next': phi_s_next,
                'reward': r.item(),
                'mu': model.mu,
                'w': model.w
            }
            # Update prev_phi for the next sample.
            model.prev_phi = phi_s_next
        
        # Record MSBE and consensus error for this sample update.
        weights_dict = {agent: agents_model[agent].w for agent in env.agents}
        msbe_val = calculate_msbe(sample_data)
        consensus_val = calculate_consensus_error(weights_dict)
        msbe_per_sample.append(msbe_val)
        consensus_per_sample.append(consensus_val)
        sample_count += 1
        
        # Update obs for next sample.
        obs = new_obs

    # End of K local steps in the round: perform consensus update.
    # Update each agent's weight using the consensus matrix: w_i = sum_j A[i,j] * w_j.
    all_agents = env.agents
    W = torch.stack([agents_model[agent].w for agent in all_agents], dim=0)  # shape (N, feat_dim)
    new_W = torch.matmul(A_consensus, W)
    for i, agent in enumerate(all_agents):
        agents_model[agent].w = new_W[i].clone()
    
    # Record consensus error immediately after the consensus update.
    weights_dict = {agent: agents_model[agent].w for agent in all_agents}
    consensus_after = calculate_consensus_error(weights_dict)
    # Optionally, append the consensus error as an extra sample.
    msbe_per_sample.append(msbe_per_sample[-1])
    consensus_per_sample.append(consensus_after)
    sample_count += 1
    print(f"After Round {l}, post-consensus consensus error = {consensus_after:.4f}")

env.close()

# Plot the MSBE and consensus error versus sample update step.
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(msbe_per_sample, marker='.')
plt.xlabel("Sample Update Step")
plt.ylabel("MSBE")
plt.title("MSBE per Sample Update")

plt.subplot(1,2,2)
plt.plot(consensus_per_sample, marker='.', color='red')
plt.xlabel("Sample Update Step")
plt.ylabel("Consensus Error")
plt.title("Consensus Error per Sample Update")

plt.tight_layout()
plt.show()
