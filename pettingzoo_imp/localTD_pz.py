import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pettingzoo.butterfly import cooperative_pong_v5
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
    Recursively extract the observation from nested dictionaries.
    If the dict is empty, return a default zero array with default_shape.
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
    Uses recursive_extract to obtain the image from nested dict(s).
    """
    if obs is None:
        return None
    obs = recursive_extract(obs, default_shape=default_shape)
    obs = np.array(obs, dtype=np.float32) / 255.0  # Normalize
    if obs.ndim == 2:
        obs = np.expand_dims(obs, axis=-1)
    obs = np.transpose(obs, (2, 0, 1))
    return torch.tensor(obs, device=device).unsqueeze(0)

def consensus_update(agent_params):
    """
    Average the parameters across agents.
    """
    param_list = list(agent_params.values())
    avg_param = sum(param_list) / len(param_list)
    return {agent: avg_param.clone() for agent in agent_params.keys()}

def get_obs(observations, agent_idx):
    """
    Return the observation corresponding to agent_idx.
    If observations is a dict, return observations[agent_name].
    If it is a list or tuple, index it by agent_idx.
    """
    if isinstance(observations, dict):
        obs = observations[agents[agent_idx]]
        return obs
    elif isinstance(observations, (tuple, list)):
        return observations[agent_idx]
    else:
        raise TypeError("Observations must be a dict, tuple, or list.")

# --- Simulation Parameters ---
num_comm_rounds = 1000      # Communication rounds (outer loop)
local_updates = 10         # Number of local TD updates per communication round
beta = 0.001                # TD learning rate for average reward update

# --- Create and Wrap Environment ---
env = cooperative_pong_v5.parallel_env(render_mode="rgb_array")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.dtype_v0(env, dtype=np.float32)
observations = env.reset()

agents = env.agents  # e.g. ['paddle_0', 'paddle_1']
print("Agents:", agents)

# --- Debug: Check Sample Observation ---
sample_obs = get_obs(observations, 0)
print("Raw sample_obs type:", type(sample_obs))
print("Raw sample_obs (raw):", sample_obs)
sample_obs = recursive_extract(sample_obs)
print("Processed sample_obs shape (before transpose):", np.array(sample_obs).shape)
if np.array(sample_obs).ndim == 2:
    sample_obs = np.expand_dims(sample_obs, axis=-1)
input_shape = np.transpose(sample_obs, (2, 0, 1)).shape
print("Using input shape:", input_shape)
feature_dim = 128

# --- Instantiate Feature Extractor ---
feature_extractor = FeatureExtractor(input_shape, feature_dim=feature_dim).to(device)
feature_extractor.eval()

# --- Initialize Agents' Parameters ---
agent_params = {}
agent_mu = {}  # Average reward estimates
for agent in agents:
    w = 0.01 * torch.randn(feature_dim, 1, device=device)
    agent_params[agent] = w
    agent_mu[agent] = torch.tensor(0.0, device=device)

# --- Storage for MSBE ---
msbe_list = []

# --- Fixed Policy (Uniform Random) ---
def fixed_policy(obs, action_space):
    return action_space.sample()

# --- Main Simulation Loop ---
total_sample_updates = 0

for comm_round in range(num_comm_rounds):
    for update in range(local_updates):
        actions = {}
        for i, agent in enumerate(agents):
            obs = get_obs(observations, i)
            actions[agent] = fixed_policy(obs, env.action_space(agent))
        
        # Unpack 5 outputs from step()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = {agent: terminations[agent] or truncations[agent] for agent in agents}
        
        sample_msbe = []
        for i, agent in enumerate(agents):
            obs = get_obs(observations, i)
            next_obs_agent = get_obs(next_obs, i)
            if obs is None or next_obs_agent is None:
                continue

            s_tensor = preprocess_obs(obs, device)
            s_next_tensor = preprocess_obs(next_obs_agent, device)
            if s_tensor is None or s_next_tensor is None:
                continue

            with torch.no_grad():
                phi_s = feature_extractor(s_tensor).view(-1, 1)
                phi_next = feature_extractor(s_next_tensor).view(-1, 1)
            
            w = agent_params[agent]
            v = torch.matmul(phi_s.t(), w)
            v_next = torch.matmul(phi_next.t(), w)
            
            r = torch.tensor(rewards[agent], device=device)
            mu = agent_mu[agent]
            # Average reward TD error
            delta = r - mu + v_next - v
            td_error_sq = (delta ** 2).item()
            sample_msbe.append(td_error_sq)
            
            agent_params[agent] = w + beta * delta * phi_s
            agent_mu[agent] = (1 - beta) * mu + beta * r

        if sample_msbe:
            avg_msbe = np.mean(sample_msbe)
            msbe_list.append(avg_msbe)
            total_sample_updates += 1

        observations = next_obs
        for agent in agents:
            if dones[agent]:
                env.reset()
                print(f"agent {agent} reset at update {total_sample_updates}")

    # --- Communication Step ---
    agent_params = consensus_update(agent_params)

    if (comm_round + 1) % 10 == 0:
        with torch.no_grad():
            sample_tensor = preprocess_obs(sample_obs, device)
            phi_sample = feature_extractor(sample_tensor).view(-1, 1)
            values = [torch.matmul(phi_sample.t(), agent_params[agent]).item() for agent in agents]
            avg_value = np.mean(values)
        print(f"Comm round {comm_round+1}/{num_comm_rounds}, average V(s) estimate: {avg_value:.4f}")

env.close()

# --- Plot MSBE Over Time ---
plt.figure(figsize=(10, 5))
plt.plot(msbe_list, label="Average MSBE")
plt.xlabel("Sample Update Step")
plt.ylabel("MSBE")
plt.title("Evolution of MSBE (Average Reward Setting)")
plt.legend()
plt.grid(True)
plt.show()
