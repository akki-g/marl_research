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

A = torch.ones(num_agents, num_agents, device=device) / num_agents    # Weight matrix for consensus step

K = 50
L = 200

consenesus_errors = []


msbes = []

for l in range(L):
    observations, infos = env.reset()
    for k in range(K):
        actions = {}
        active_agents = [agent.id for agent in agents if agent.id in observations]
        for agent_id in active_agents:
            agent = next(a for a in agents if a.id == agent_id)
            phi = agent.get_features(observations[agent.id])
            actions[agent.id] = env.action_space(agent.id).sample()
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent in agents:
            if agent.id in next_observations:
                phi = agent.get_features(observations[agent.id])
                phi_next = agent.get_features(next_observations[agent.id])
                agent.update(phi, phi_next, rewards[agent.id])
        observations = next_observations

        if all(terminations.values()) or all(truncations.values()):
            break

        active_ids = [agent.id for agent in agents if agent.id in observations]
        if active_ids:
            sample_agent = next(a for a in agents if a.id == active_ids[0])
            phi_sample = sample_agent.get_features(observations[active_ids[0]])
            phi_next_sample = sample_agent.get_features(observations[active_ids[0]])
            current_msbe = compute_msbe(agents, phi_sample, phi_next_sample, rewards[active_ids[0]])
            msbes.append(current_msbe)
            
        consensus_update(agents, A)
        print(f"Local TD Update Step: {k}, MSBE: {current_msbe}")
        
env.close()
# Plot MSBE
import matplotlib.pyplot as plt
plt.plot(msbes)
plt.xlabel("Local TD Update Step")
plt.ylabel("Mean Squared Bellman Error (MSBE)")
plt.title("Tracking MSBE at Each Local Sample Round")
plt.grid(True)
plt.show()

        
