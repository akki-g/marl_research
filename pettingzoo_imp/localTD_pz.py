import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio

from pettingzoo.sisl import pursuit_v4
import supersuit as ss

# --- Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --- Feature Extractor Definition ---
class FeatureExtractor(nn.Module):
    def __init__(self, input_shape, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            cnn_out_dim = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out_dim, feature_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# --- Recursive Extraction Helper ---
def recursive_extract(obs, default_shape=(84,84,3)):
    """
    Recursively extract the actual image data from nested dictionaries.
    If an empty dict is encountered, return a zero array with default_shape.
    """
    while isinstance(obs, dict):
        if len(obs) == 0:
            return np.zeros(default_shape, dtype=np.float32)
        if "observation" in obs:
            obs = obs["observation"]
        else:
            obs = list(obs.values())[0]
    return obs

# --- Helper Functions ---
def preprocess_obs(obs, device, default_shape=(84,84,3)):
    """
    Convert an observation to a normalized torch tensor in (C, H, W) format.
    Uses recursive_extract to ensure the image is extracted.
    """
    if obs is None:
        return None
    obs = recursive_extract(obs, default_shape=default_shape)
    obs = np.array(obs, dtype=np.float32) / 255.0  # Normalize pixels
    if obs.ndim == 2:
        obs = np.expand_dims(obs, axis=-1)
    obs = np.transpose(obs, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    return torch.tensor(obs, device=device).unsqueeze(0)

def ensure_dict(observations, agents, default_shape=(84,84,3)):
    """
    Ensure observations is a dict with an entry for every agent in agents.
    If observations is a tuple or list and its length is less than len(agents),
    missing agents are filled with a default zero array.
    """
    if isinstance(observations, dict):
        return observations
    elif isinstance(observations, (tuple, list)):
        obs_dict = {}
        for i, agent in enumerate(agents):
            if i < len(observations):
                obs_dict[agent] = observations[i]
            else:
                obs_dict[agent] = np.zeros(default_shape, dtype=np.float32)
        return obs_dict
    else:
        raise TypeError("Observations must be a dict, tuple, or list.")

def consensus_update_custom(agent_params, A, agents):
    """
    For each agent i, update: w_i ← ∑_{j in N_i} A[i,j] * w_j.
    """
    new_params = {}
    n = len(agents)
    for i, agent in enumerate(agents):
        weighted_sum = 0
        for j in range(n):
            weighted_sum += A[i, j] * agent_params[agents[j]]
        new_params[agent] = weighted_sum.clone()
    return new_params

def ring_adjacency(n):
    """
    Create a ring adjacency matrix for n agents.
    Each agent averages with itself and its two neighbors.
    """
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1/3
        A[i, (i-1) % n] = 1/3
        A[i, (i+1) % n] = 1/3
    return A

def get_obs(observations, agent_idx):
    """
    Return the observation for the agent at agent_idx from a dict.
    """
    return observations[agents[agent_idx]]

def save_video(frames, filename="simulation_video.mp4", fps=10):
    """
    Save a video file from a list of RGB frames.
    """
    imageio.mimwrite(filename, frames, fps=fps)
    print(f"Video saved to {filename}")

def save_images(frames, prefix="comm_snapshot"):
    """
    Save each frame in frames as a separate image file.
    """
    for i, frame in enumerate(frames):
        imageio.imwrite(f"{prefix}_{i:03d}.png", frame)
    print(f"{len(frames)} images saved with prefix '{prefix}_'.")

# --- Environment Setup ---
# Use pursuit_v4.parallel_env with 10 pursuers and 10 evaders (20 agents).
env = pursuit_v4.parallel_env(
    max_cycles=10000,
    x_size=16,
    y_size=16,
    shared_reward=False,
    n_evaders=50,
    n_pursuers=20,
    obs_range=7,
    n_catch=2,
    freeze_evaders=False,
    tag_reward=0.01,
    catch_reward=5.0,
    urgency_reward=-0.1,
    surround=True,
    constraint_window=1.0,
    render_mode="rgb_array"
)
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.dtype_v0(env, dtype=np.float32)
observations = ensure_dict(env.reset(), env.agents, default_shape=(84,84,3))

agents = env.agents  # Should now return 20 agent names.
print("Agents:", agents)
print("Action space for", agents[0], ":", env.action_space(agents[0]))

# --- Debug: Check a Sample Observation ---
sample_obs = get_obs(observations, 0)
print("Raw sample_obs type:", type(sample_obs))
print("Raw sample_obs (raw):", sample_obs)
sample_obs = recursive_extract(sample_obs, default_shape=(84,84,3))
print("Processed sample_obs shape (before transpose):", np.array(sample_obs).shape)
if np.array(sample_obs).ndim == 2:
    sample_obs = np.expand_dims(sample_obs, axis=-1)
input_shape = np.transpose(sample_obs, (2, 0, 1)).shape
print("Using input shape:", input_shape)
feature_dim = 128

# --- Instantiate Feature Extractor ---
feature_extractor = FeatureExtractor(input_shape, feature_dim=feature_dim).to(device)
feature_extractor.eval()

# --- Initialize Agents' Value Parameters and Average Reward Estimates ---
agent_params = {}
agent_mu = {}
for agent in agents:
    w = 0.01 * torch.randn(feature_dim, 1, device=device)
    agent_params[agent] = w
    agent_mu[agent] = torch.tensor(0.0, device=device, dtype=torch.float32)

# --- Create Custom Adjacency Matrix (Ring Topology) ---
A = ring_adjacency(len(agents))
print("Adjacency matrix A:\n", A)

# --- Storage for MSBE, and lists to store frames ---
msbe_list = []
video_frames = []      # to store RGB frames for video
comm_snapshots = []    # to store one snapshot per communication round

# --- Fixed Policy (Uniform Random) ---
def fixed_policy(obs, action_space):
    action = action_space.sample()
    if isinstance(action, np.ndarray) and action.size == 1:
        return int(action.item())
    return action

# --- Main Simulation Loop ---
num_comm_rounds = 200
local_updates = 50
beta = 0.001
total_sample_updates = 0

for comm_round in range(num_comm_rounds):
    for update in range(local_updates):
        actions = {}
        for i, agent in enumerate(agents):
            obs = get_obs(observations, i)
            actions[agent] = fixed_policy(obs, env.action_space(agent))
        
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        next_obs = ensure_dict(next_obs, agents, default_shape=(84,84,3))
        dones = {agent: terminations[agent] or truncations[agent] for agent in agents}
        
        # Capture an RGB frame at each sample update.
        frame = env.render()  # returns an RGB array
        video_frames.append(frame)
        
        sample_msbe = []
        for i, agent in enumerate(agents):
            obs = get_obs(observations, i)
            next_obs_agent = get_obs(next_obs, i)
            if obs is None or next_obs_agent is None:
                continue

            s_tensor = preprocess_obs(obs, device, default_shape=(84,84,3))
            s_next_tensor = preprocess_obs(next_obs_agent, device, default_shape=(84,84,3))
            if s_tensor is None or s_next_tensor is None:
                continue

            with torch.no_grad():
                phi_s = feature_extractor(s_tensor).view(-1, 1)
                phi_next = feature_extractor(s_next_tensor).view(-1, 1)
            
            w = agent_params[agent]
            v = torch.matmul(phi_s.t(), w)
            v_next = torch.matmul(phi_next.t(), w)
            
            r = torch.tensor(rewards[agent], device=device, dtype=torch.float32)
            mu = agent_mu[agent]
            # Average reward TD error: delta = r - mu + v_next - v.
            delta = r - mu + v_next - v
            td_error_sq = (delta ** 2).item()
            sample_msbe.append(td_error_sq)
            
            agent_params[agent] = w + beta * delta * phi_s
            agent_mu[agent] = (1 - beta) * mu + beta * r

        if sample_msbe:
            avg_msbe = np.mean(sample_msbe)
            msbe_list.append(avg_msbe)
            total_sample_updates += 1

        observations = ensure_dict(next_obs, agents, default_shape=(84,84,3))
        for agent in agents:
            if dones[agent]:
                observations = ensure_dict(env.reset(), agents, default_shape=(84,84,3))
                print(f"Episode ended at update {total_sample_updates}.")
                break

    # --- Communication (Consensus) Step ---
    agent_params = consensus_update_custom(agent_params, A, agents)

    # Save a snapshot of the current rendered frame at the end of each communication round.
    comm_frame = env.render()
    comm_snapshots.append(comm_frame)

    if (comm_round + 1) % 10 == 0:
        with torch.no_grad():
            sample_tensor = preprocess_obs(sample_obs, device, default_shape=(84,84,3))
            phi_sample = feature_extractor(sample_tensor).view(-1, 1)
            values = [torch.matmul(phi_sample.t(), agent_params[agent]).item() for agent in agents]
            avg_value = np.mean(values)
        print(f"Comm round {comm_round+1}/{num_comm_rounds}, average V(s) estimate: {avg_value:.4f}")

env.close()

# --- Save Video and Snapshot Images ---
save_images(comm_snapshots, prefix="comm_snapshot")
save_video(video_frames, filename="simulation_video.mp4", fps=10)

# --- Plot MSBE Over Time ---
plt.figure(figsize=(10, 5))
plt.plot(msbe_list, label="Average MSBE")
plt.xlabel("Sample Update Step")
plt.ylabel("MSBE")
plt.title("Evolution of Mean Squared Bellman Error (Average Reward Setting)")
plt.legend()
plt.grid(True)
plt.show()
