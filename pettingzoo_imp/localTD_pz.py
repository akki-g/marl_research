import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import imageio  


# -------------------- Device Setup --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
print("Using device:", device)

# -------------------- Feature Extraction --------------------
def get_feature_vector(obs, num_agents=9, num_landmarks=9, d = 2):
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
    """
    agent_pos = obs[2:4]
    start_landmark = 4
    end_landmark = start_landmark + num_landmarks * 2
    landmark_rel_pos = obs[start_landmark:end_landmark]
    # Extract other agents relative positions (ignoring communication)
    other_agents_rel_pos = obs[end_landmark:end_landmark + (num_agents - 1) * 2]
    feature_vector = np.concatenate([agent_pos, landmark_rel_pos, other_agents_rel_pos])
    norm = np.linalg.norm(feature_vector, ord=1)
    if norm > 0:
        feature_vector = feature_vector / norm
    return feature_vector

# -------------------- Error Metrics --------------------
def calculate_consensus_error(weights):
    """
    Computes the consensus error:
      ConsensusError = (1/N) ∑_{i=1}^N ||w_i - w̄||²,
    where w̄ is the average weight vector.
    """
    agent_ids = list(weights.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
    W = torch.stack([weights[agent] for agent in agent_ids], dim=0)
    w_bar = torch.mean(W, dim=0)
    eucild = torch.norm(W - w_bar, dim=1)
    consensus_error = torch.mean(eucild ** 2)
    return consensus_error.item()

def calculate_SBE(sample_data):
    """
    Computes the instantaneous squared Bellman error (SBE) for a sample update.
    For each agent i, define:
      error_i = φ(s)ᵀw_i + μ̄ - r̄ - φ(s')ᵀw_i,
    where r̄ and μ̄ are the averages over agents.
    SBE = (1/N) ∑_{i=1}^N error_i².
    """
    agent_ids = list(sample_data.keys())
    N = len(agent_ids)
    if N == 0:
        return 0.0
    r_bar = sum(sample_data[agent]['reward'] for agent in agent_ids) / N
    mu_bar = sum(sample_data[agent]['mu'] for agent in agent_ids) / N
    errors = []
    for agent in agent_ids:
        w = sample_data[agent]['w']
        phi_s = sample_data[agent]['phi_s']
        phi_s_next = sample_data[agent]['phi_s_next']
        value_current = torch.dot(phi_s.t(), w)
        value_next = torch.dot(phi_s_next.t(), w)
        error = value_current + mu_bar - r_bar - value_next
        errors.append(error ** 2)
    SBE = sum(errors) / N
    return SBE.item()

# -------------------- Consensus Matrix --------------------
def generate_ring_matrix(N, diag=0.4, off_diag=0.3):
    """
    Generates a ring consensus matrix for N agents.
    For agent 0, neighbors: agent 1 and agent N-1.
    For agent i (1 <= i <= N-2): neighbors: i-1 and i+1.
    For agent N-1, neighbors: agent N-2 and agent 0.
    """
    A = np.zeros((N, N))
    if N == 1:
        A[0,0] = 1.0
        return torch.tensor(A, dtype=torch.float32)
    A[0,0] = diag
    A[0,1] = off_diag
    A[0,N-1] = off_diag
    for i in range(1, N-1):
        A[i,i] = diag
        A[i,i-1] = off_diag
        A[i,i+1] = off_diag
    A[N-1,N-1] = diag
    A[N-1,N-2] = off_diag
    A[N-1,0] = off_diag
    return torch.tensor(A, dtype=torch.float32)

def generate_consensus_matrix(N, p=0.5):
    """
    Generates a consensus matrix A for N agents based on an Erdos–Rényi (ER) graph with connection probability p.
    Using a simple Metropolis–Hastings rule:
      For i ≠ j, if an edge exists: A[i,j] = 1 / (1 + max{deg(i), deg(j)}),
      Then A[i,i] = 1 - ∑₍j ≠ i₎ A[i,j].
    Returns:
        A: torch.Tensor of shape (N, N)
    """
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
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
# -------------------- Agent Model --------------------
class AgentModel:
    """
    Stores the learnable weight vector w, the running average reward μ,
    and the previous feature vector φ(s).
    The value function is approximated as V(s) = φ(s)ᵀw.
    """
    def __init__(self, agent_id, feat_dim, init_mu=0.0):
        self.agent_id = agent_id
        self.w = torch.ones(feat_dim, device=device, dtype=torch.float32)
        self.mu = torch.tensor(init_mu, device=device, dtype=torch.float32)
        self.prev_phi = None
        self.delta_history = []

# -------------------- Hyperparameters --------------------
L = 500           # Number of communication rounds
K = 20            # Number of local TD-update (sample) steps per round
beta = 0.1        # Step size for the update
num_agents = 9    # Number of agents (and landmarks)
num_landmarks = num_agents  # For simple_spread, typically equal to number of agents
A_consensus = generate_ring_matrix(num_agents).to(device)

# -------------------- Environment Setup --------------------
env = simple_spread_v3.parallel_env(
    N=num_agents, local_ratio=0.5, max_cycles=10_000, 
    continuous_actions=False, render_mode='rgb_array'
)
obs, infos = env.reset(seed=42)  # Parallel API returns (obs, infos)

# Initialize each agent's model.
sample_phi = get_feature_vector(obs[env.agents[0]])
feat_dim = sample_phi.shape[0]
print(f"Feature dimension: {feat_dim}")
agents_model = {agent: AgentModel(agent, feat_dim) for agent in env.agents}

# Set initial φ(s) for each agent.
for agent in env.agents:
    phi_init = torch.tensor(get_feature_vector(obs[agent]),
                              device=device, dtype=torch.float32)
    agents_model[agent].prev_phi = phi_init

# -------------------- Main Loop --------------------
# We also keep a list of instantaneous SBE values, then compute the running average (MSBE) over history.
sbe_history = []
msbe_running = []  # running average over sbe_history
consensus_per_sample = []
sample_count = 0
video_frames = []     
comm_snapshots = [] 
for l in range(L):
    print(f"\n=== Communication Round {l} ===")
    obs, infos = env.reset(seed=42)
    for agent in env.agents:
        phi_reset = torch.tensor(get_feature_vector(obs[agent]),
                                  device=device, dtype=torch.float32)
        agents_model[agent].prev_phi = phi_reset
    for k in range(K):
        actions = {agent: int(env.action_space(agent).sample()) for agent in env.agents}
        obs, rewards, dones, truncs, infos = env.step(actions)
        
        if not env.agents:
            print("No active agents; resetting environment.")
            obs, infos = env.reset()
            for agent in env.agents:
                phi_reset = torch.tensor(get_feature_vector(obs[agent]),
                                           device=device, dtype=torch.float32)
                agents_model[agent].prev_phi = phi_reset
            break
        
        sample_data = {}
        for agent in env.agents:
            model = agents_model[agent]
            phi_s = model.prev_phi
            phi_s_next = torch.tensor(get_feature_vector(obs[agent]),
                                        device=device, dtype=torch.float32)
            value_current = torch.dot(phi_s.t(), model.w)
            value_next = torch.dot(phi_s_next.t(), model.w)
            r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
            
            # TD error (average reward TD update):
            delta = r - model.mu + value_next - value_current
            model.delta_history.append(delta.item())
            # Update running average reward: μ ← (1 - β) * μ + β * r.
            model.mu = (1 - beta) * model.mu + beta * r
            model.w = model.w + beta * delta * phi_s
            sample_data[agent] = {
                'phi_s': phi_s,
                'phi_s_next': phi_s_next,
                'reward': r.item(),
                'mu': model.mu,
                'w': model.w
            }
            model.prev_phi = phi_s_next
        instantaneous_sbe = calculate_SBE(sample_data)
        sbe_history.append(instantaneous_sbe)
        running_avg = np.mean(sbe_history)
        msbe_running.append(running_avg)
        frame = env.render()
        video_frames.append(frame)
        weights_dict = {agent: agents_model[agent].w for agent in env.agents}
        cons_val = calculate_consensus_error(weights_dict)
        consensus_per_sample.append(cons_val)
        sample_count += 1
        
    # Consensus update at end of communication round.
    comm_snapshots.append(env.render())
    all_agents = env.agents
    print(f"Round {l}: Number of active agents = {len(all_agents)}")
    if len(all_agents) > 0:
        W = torch.stack([agents_model[agent].w for agent in all_agents], dim=0)
        new_W = torch.matmul(A_consensus, W)
        for i, agent in enumerate(all_agents):
            agents_model[agent].w = new_W[i].clone()
        weights_dict = {agent: agents_model[agent].w for agent in all_agents}
        cons_after = calculate_consensus_error(weights_dict)
        # Append consensus error after consensus update (keeping MSBE same as last sample)
        print(f"After Round {l}: Post-consensus consensus error = {cons_after:.4f}")

env.close()
def save_video(frames, filename="local_ratio.mp4", fps=10):
    imageio.mimwrite(filename, frames, fps=fps)
    print(f"Video saved to {filename}")

def save_images(frames, prefix="comm_snapshot"):
    for i, frame in enumerate(frames):
        imageio.imwrite(f"{prefix}_{i:03d}.png", frame)
    print(f"{len(frames)} images saved with prefix '{prefix}_'.")
# -------------------- Plotting --------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(msbe_running, marker='.')
plt.xlabel("Sample Update Step")
plt.ylabel("MSBE")
plt.title("Running MSBE per Sample Update (Average Reward)")

plt.subplot(1,2,2)
plt.plot(consensus_per_sample, marker='.', color='red')
plt.xlabel("Sample Update Step")
plt.ylabel("Consensus Error")
plt.title("Consensus Error per Sample Update")
plt.tight_layout()
plt.show()

save_video(video_frames)

