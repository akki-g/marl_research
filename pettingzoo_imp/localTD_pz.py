from pettingzoo.butterfly import cooperative_pong_v5
import numpy as np
import torch
from collections import deque
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print("Using device:", device)

env = cooperative_pong_v5.parallel_env(render_mode="human")
observations, infos = env.reset(seed=42)


class Agent:
    def __init__(self, agent_id, feat_dim, beta, obs_dim, act_dim):
        self.id = agent_id
        self.w = torch.randn(feat_dim, device=device) * 0.1
        self.beta = beta
        self.mu = torch.tensor(0.0, device=device)
        self.feature_dim = feat_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.phi = torch.randn(feat_dim, obs_dim, device=device) * (1.0/np.sqrt(feat_dim))

        self.normailze_phi()

        self.theta = torch.randn(act_dim, feat_dim, device=device) * 0.1

    def normailze_phi(self):
        for s in range(self.phi.shape[1]):
            norm_s = torch.linalg.norm(self.phi[:, s])
            if norm_s > 0:
                self.phi[:, s] /= norm_s

    def get_features(self, observation):
        obs_tensor = torch.tensor(observation, device=device, dtype=torch.float32).flatten()
        obs_tensor = obs_tensor / 255.0
        return torch.matmul(self.phi, obs_tensor)
    
    def update(self, phi, phi_next, reward):
        td_error = reward - self.mu + torch.dot(phi_next, self.w) - torch.dot(phi, self.w)
        self.w += self.beta * td_error * phi
        self.mu = (1 - self.beta) * self.mu + self.beta * reward
        return td_error
    
    def policy(self, observation):
        features = self.get_features(observation)
        logits = torch.matmul(self.theta, features)
        action_probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(action_probs, 1).item()

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
    reward_tensor = torch.tensor(
        [rewards.get(agent.id, 0.0) for agent in agents],
        device=device
    )
    for i, agent in enumerate(agents):
        phi_i = phi[agent.id]
        phi_next_i = phi_next[agent.id]
        td_error = reward_tensor[i] - agent.mu + torch.dot(phi_next_i, agent.w) - torch.dot(phi_i, agent.w)
        errors.append(td_error ** 2)
    return torch.mean(torch.stack(errors)).cpu().item()


agent_ids = env.possible_agents
num_agents = len(agent_ids)


feat_dim = 32
beta = 0.001
obs_dim = np.prod(env.observation_space(agent_ids[0]).shape)
act_dim = env.action_space(agent_ids[0]).n
print("Observation dimension:", obs_dim)
agents = [Agent(agent_id, feat_dim, beta, obs_dim, act_dim) for agent_id in agent_ids]

A = torch.zeros((num_agents, num_agents))
for i in range(num_agents):
    neighbors = [(i-1)%num_agents, (i+1)%num_agents]
    A[i, neighbors] = 0.3
    A[i,i] = 0.4
A = A / A.sum(dim=1, keepdim=True)
print("A matrix row sums:", A.sum(dim=1))    
K = 50
L = 200

consenesus_errors = []

msbes = []
i = 0
observations, _ = env.reset()
for l in range(L):  
    for k in range(K):
        
        actions = {a.id: a.policy(observations[a.id]) for a in agents}
        next_observations, rewards, term, trunc, infos = env.step(actions)

        td_errors = []
        current_phi = {}
        next_phi = {}

        for agent in agents:
            current_phi[agent.id] = agent.get_features(observations[agent.id])
            next_phi[agent.id] = agent.get_features(next_observations[agent.id])

        for agent in agents:
            reward = rewards[agent.id]
            td_error = agent.update(current_phi[agent.id], next_phi[agent.id], reward)
            td_errors.append(td_error**2)

        
        msbes.append(torch.mean(torch.stack(td_errors)).cpu().item())
        consensus_error = torch.std(torch.stack([a.w for a in agents]), dim=0).mean().item()
        print(f"Consensus Error: {consensus_error}")
        i += 1
        print(f"Step {i}, MSBE: {msbes[-1]}")
        observations = next_observations
        if all(term.values()) or all(trunc.values()):
            observations = env.reset()
            break
    consensus_update(agents, A)



        
env.close()
# Plot MSBE
import matplotlib.pyplot as plt
plt.plot(msbes)
plt.xlabel("Local TD Update Step")
plt.ylabel("Mean Squared Bellman Error (MSBE)")
plt.title("Tracking MSBE at Each Local Sample Round")
plt.grid(True)
plt.show()

        
