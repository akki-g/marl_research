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
    def __init__(self, agent_id, feat_dim, beta):
        self.id = agent_id
        self.w = torch.randn(feat_dim, device=device) * 0.1
        self.beta = beta
        self.mu = torch.tensor(0.0, device=device)
        self.feature_dim = feat_dim

    def get_features(self, observation):
        obs_tensor = torch.tensor(observation, device=device, dtype=torch.float32)
        projection_matrix = torch.randn(self.feature_dim, obs_tensor.numel(), device=device)
        return torch.matmul(projection_matrix, obs_tensor.float())
    
    def update(self, phi, phi_next, reward):
        reward_tensor = torch.tensor(reward, device=device)
        td_error = reward_tensor - self.mu + torch.dot(phi_next, self.w) - torch.dot(phi, self.w)
        self.w += self.beta * td_error * phi
        self.mu = (1 - self.beta) * self.mu + self.beta * reward_tensor 
        return td_error.cpu().numpy()
    
    def policy(self):
        return np.random.randint(0, env.action_space(agent.id).n)
    
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, s, a, r, s_next):
        self.buffer.append((
            torch.tensor(s, device=device).flatten(),
            a,
            r,
            torch.tensor(s_next, device=device).flatten()
        ))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, device=device),
            torch.tensor(rewards, device=device),
            torch.stack(next_states)
        )

def consensus_update(agents, A):
    """Perform consensus step using weight matrix A"""
    with torch.no_grad():
        all_w = torch.stack([agent.w for agent in agents])
        all_mu = torch.stack([agent.mu for agent in agents])

        new_w = torch.matmul(A, all_w)
        new_mu = torch.matmul(A, all_mu)

        for i, agent in enumerate(agents):
            agent.w = new_w[i]
            agent.mu = new_mu[i]


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


feat_dim = 5
beta = 0.005

agents = [Agent(agent_id, feat_dim, beta) for agent_id in agent_ids]

A = torch.zeros((num_agents, num_agents), device=device)
for i in range(num_agents):
    neighbors = [(i-1)%num_agents, (i+1)%num_agents]
    A[i, neighbors] = 0.3
    A[i, i] = 0.4

A = A / A.sum(dim=1, keepdim=True)
A = A / A.sum(dim=0, keepdim=True)
    
K = 50
L = 200

consenesus_errors = []
replay_buff = ReplayBuffer()

msbes = []
i = 0
for l in range(L):
    observations, infos = env.reset()
  
    for k in range(K):
        actions = {a.id: a.policy(obs) for a, obs in    zip(agents, observations.items())}
        next_observations, rewards, dones, _ = env.step(actions)
        for agent in agents:
            if agent.id in observations:
                state = observations[agent.id].flatten()
                next_state = next_observations[agent.id].flatten()
                replay_buff.add(state, actions[agent.id], rewards.get(agent.id, 0.0), next_state)

        observations = next_observations

        for _ in range(K):
            states, actions, rewards, next_states = replay_buff.sample(batch_size=32)

            states = torch.matmul(torch.randn(feat_dim, states.shape[1], device=device), states.T).T
            next_states = torch.matmul(torch.randn(feat_dim, next_states.shape[1], device=device), next_states.T).T

            batch_msbes = []
            for i, agent in enumerate(agents):
                phi = states[:, i+feat_dim].mean(dim=0)
                phi_next = next_states[:, i+feat_dim].mean(dim=0)
                rewards = rewards[:, i].mean()
                batch_msbes.append(agent.update(phi, phi_next, rewards))
            
            msbes.append(torch.mean(batch_msbes))
            i += 1
            print(f"Step {i}, MSBE: {msbes[-1]}")
    
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

        
