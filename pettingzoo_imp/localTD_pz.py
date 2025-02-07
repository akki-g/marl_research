from pettingzoo.butterfly import cooperative_pong_v5
import numpy as np
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print("Using device:", device)

env = cooperative_pong_v5.parallel_env(render_mode="human")
observations, infos = env.reset(seed=42)


class Agent:
    def __init__(self, agent_id, feat_dim, beta):
        self.id = agent_id
        self.w = torch.randn(feat_dim, device=device) * 0.1
        self.beta = beta
        self.mu = torch.tensor(0.0, device=device)
        self.feature_dim = feat_dim

    def get_features(self, observation):
        obs_tensor = torch.tensor(observation, device=device, dtype=torch.float32)
        reshaped_obs = obs_tensor.view(-1, self.feature_dim)
        return torch.mean(reshaped_obs, dim=0)
    
    def update(self, phi, phi_next, reward):
        reward_tensor = torch.tensor(reward, device=device)
        td_error = reward_tensor - self.mu + torch.dot(phi_next, self.w) - torch.dot(phi, self.w)
        self.w += self.beta * td_error * phi
        self.mu = (1 - self.beta) * self.mu + self.beta * reward_tensor 
        return td_error.cpu().numpy()
    
    def policy(self, observation):
        phi = self.get_features(observation)
        action_values = torch.dot(self.w, phi)
        return torch.argmax(action_values).item()


def consensus_update(agents, A):
    """Perform consensus step using weight matrix A"""
    with torch.no_grad():
        all_w = torch.stack([agent.w for agent in agents])
        new_w = torch.matmul(A, all_w)
        for i, agent in enumerate(agents):
            agent.w = new_w[i]


def compute_msbe(agents, phi, phi_next, rewards):
    """Compute Mean Squared Bellman Error"""
    errors = []
    reward_tensor = torch.tensor(rewards, device=device)
    for agent in agents:
        error = reward_tensor - agent.mu + torch.dot(phi_next, agent.w) - torch.dot(phi, agent.w)
        errors.append(error ** 2)
    return torch.mean(torch.stack(errors)).cpu().item()


agent_ids = env.possible_agents
num_agents = len(agent_ids)


feat_dim = 256
beta = 0.01

agents = [Agent(agent_id, feat_dim, beta) for agent_id in agent_ids]

if num_agents > 1:
    diag_element = 0.4
    off_diag = 0.3
    # Create an identity matrix scaled by the diagonal weight
    A = diag_element * torch.eye(num_agents, device=device)
    
    # Set neighbor connections for a circular (ring) topology
    A[0, num_agents - 1] = off_diag  # last neighbor of first node
    A[0, 1] = off_diag               # second neighbor of first node
    
    for i in range(1, num_agents - 1):
        A[i, i - 1] = off_diag       # left neighbor
        A[i, i + 1] = off_diag       # right neighbor

    A[num_agents - 1, 0] = off_diag      # first neighbor of last node
    A[num_agents - 1, num_agents - 2] = off_diag  # second neighbor of last node
else:
    A = torch.tensor([[1.0]], device=device)
    
K = 50
L = 200

consenesus_errors = []


msbes = []

for l in range(L):
    observations, infos = env.reset()
    for k in range(K):
        actions = {agent.id: agent.policy(observations[agent.id]) for agent in agents if agent.id in observations}
        next_observations, rewards, term, trunc, infos = env.step(actions)
        for agent_id in observations.key():
            agent = next(a for a in agents if a.id == agent_id)
            phi = agent.get_features(observations[agent_id])
            phi_next = agent.get_features(next_observations[agent_id])
            td_error = agent.update(phi, phi_next, rewards[agent_id])

        observations = next_observations
        
        if all(term.values()) or all(trunc.values()):
            break

    consensus_update(agents, A)
    current_msbe = compute_msbe(agents, 
                                {agent.id: agent.get_features(observations[agent.id]) for agent in agents if agent.id in observations},
                                {agent.id: agent.get_features(next_observations[agent.id]) for agent in agents if agent.id in next_observations},
                                rewards)
    msbes.append(current_msbe)

        
env.close()
# Plot MSBE
import matplotlib.pyplot as plt
plt.plot(msbes)
plt.xlabel("Local TD Update Step")
plt.ylabel("Mean Squared Bellman Error (MSBE)")
plt.title("Tracking MSBE at Each Local Sample Round")
plt.grid(True)
plt.show()

        
