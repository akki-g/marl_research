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
def recursive_extract(obs, default_shape=(7,7,3)):
    """
    Recursively extract the image data from nested dictionaries.
    For pursuit_v4, each observation is a 7x7x3 array.
    """
    while isinstance(obs, dict):
        if len(obs) == 0:
            return np.zeros(default_shape, dtype=np.float32)
        if "observation" in obs:
            obs = obs["observation"]
        else:
            obs = list(obs.values())[0]
    return obs

# --- Preprocessing Helper ---
def preprocess_obs(obs, device, default_shape=(7,7,3)):
    """
    Convert an observation to a normalized torch tensor in (C,H,W) format.
    For pursuit_v4, observations are 7x7x3; we normalize by dividing by 30.
    """
    if obs is None:
        return None
    obs = recursive_extract(obs, default_shape=default_shape)
    obs = np.array(obs, dtype=np.float32) / 30.0
    if obs.ndim == 2:
        obs = np.expand_dims(obs, axis=-1)
    obs = np.transpose(obs, (2, 0, 1))
    return torch.tensor(obs, device=device).unsqueeze(0)

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

def get_obs(observations, agent):
    """
    Return the observation for the given agent from a dictionary.
    """
    return observations[agent]

def fixed_policy(obs, action_space):
    action = action_space.sample()
    if isinstance(action, np.ndarray) and action.size == 1:
        return int(action.item())
    return action

# --- AEC Environment Setup ---
env = pursuit_v4.env(render_mode="rgb_array",
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
                     constraint_window=1.0)
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.dtype_v0(env, dtype=np.float32)
env.reset(seed=42)

agents = env.agents
print("Agents:", agents)
print("Action space for", agents[0], ":", env.action_space(agents[0]))

# Get a sample observation to build the feature extractor.
agent_iter = env.agent_iter()
sample = None
for agent in agent_iter:
    obs, reward, termination, truncation, info = env.last()
    sample = obs
    env.step(0)  # use dummy valid action instead of None
    break
sample = recursive_extract(sample, default_shape=(7,7,3))
if np.array(sample).ndim == 2:
    sample = np.expand_dims(sample, axis=-1)
input_shape = np.transpose(sample, (2, 0, 1)).shape
print("Using input shape:", input_shape)

feature_dim = 128
feature_extractor = FeatureExtractor(input_shape, feature_dim=feature_dim).to(device)
feature_extractor.eval()

agent_params = {agent: 0.01 * torch.randn(feature_dim, 1, device=device) for agent in agents}
agent_mu = {agent: torch.tensor(0.0, device=device, dtype=torch.float32) for agent in agents}

A = ring_adjacency(len(agents))
print("Adjacency matrix A:\n", A.shape)

msbe_list = []
video_frames = []      # For saving every rendered frame (for video)
comm_snapshots = []    # One snapshot per communication round

num_comm_rounds = 200
local_updates = 50   # Each agent performs K = local_updates local TD updates per communication round.
beta = 0.001
comm_round_steps = local_updates * len(agents)
step_count = 0
comm_rounds_done = 0

last_obs = {agent: None for agent in agents}

# Main simulation loop (double-loop structure).
for l in range(num_comm_rounds):
    # Outer loop: Communication rounds.
    # For each communication round, reinitialize the agent iterator.
    agent_iter = iter(env.agent_iter())
    for k in range(local_updates):
        # Inner loop: Local TD-update steps.
        for _ in range(len(agents)):
            try:
                agent = next(agent_iter)
            except StopIteration:
                # If the episode ends, reset and reinitialize the iterator.
                env.reset()
                agent_iter = iter(env.agent_iter())
                agent = next(agent_iter)
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if last_obs[agent] is not None:
                s = last_obs[agent]
                s_next = obs
                s_tensor = preprocess_obs(s, device, default_shape=(7,7,3))
                s_next_tensor = preprocess_obs(s_next, device, default_shape=(7,7,3))
                if s_tensor is not None and s_next_tensor is not None:
                    with torch.no_grad():
                        phi_s = feature_extractor(s_tensor).view(-1, 1)
                        phi_next = feature_extractor(s_next_tensor).view(-1, 1)
                    w = agent_params[agent]
                    v = torch.matmul(phi_s.t(), w)
                    v_next = torch.matmul(phi_next.t(), w)
                    r = torch.tensor(reward, device=device, dtype=torch.float32)
                    mu = agent_mu[agent]
                    delta = r - mu + v_next - v
                    agent_params[agent] = w + beta * delta * phi_s
                    agent_mu[agent] = (1 - beta) * mu + beta * r
                    msbe_list.append((delta ** 2).item())
            last_obs[agent] = obs

            if done:
                action = None
            else:
                action = fixed_policy(obs, env.action_space(agent))
                if not env.action_space(agent).contains(action):
                    print(f"Warning: action {action} not in action space for {agent}. Defaulting to 0.")
                    action = 0
            env.step(action)
            frame = env.render()
            video_frames.append(frame)
        step_count += 1
    # End inner loop: perform consensus update.
    agent_params = consensus_update_custom(agent_params, A, agents)
    # Save communication round snapshot.
    comm_snapshots.append(env.render())
    comm_rounds_done += 1
    print(f"Comm round {comm_rounds_done} completed at step {step_count}.")
    # Optionally, reset the environment if the episode ended.
    # (This design assumes the algorithm continues across episodes.)
    # If needed, call env.reset() here.

env.close()

def save_video(frames, filename="simulation_video.mp4", fps=10):
    imageio.mimwrite(filename, frames, fps=fps)
    print(f"Video saved to {filename}")

def save_images(frames, prefix="comm_snapshot"):
    for i, frame in enumerate(frames):
        imageio.imwrite(f"{prefix}_{i:03d}.png", frame)
    print(f"{len(frames)} images saved with prefix '{prefix}_'.")

save_video(video_frames, filename="simulation_video.mp4", fps=10)
save_images(comm_snapshots, prefix="comm_snapshot")

plt.figure(figsize=(10, 5))
plt.plot(msbe_list, label="Average MSBE")
plt.xlabel("Sample Update Step")
plt.ylabel("MSBE")
plt.title("Evolution of MSBE (Average Reward Setting)")
plt.legend()
plt.grid(True)
plt.show()
