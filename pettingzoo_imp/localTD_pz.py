from pettingzoo.butterfly import cooperative_pong_v5
import numpy as np
import torch


env = cooperative_pong_v5.parallel_env()
env.reset(seed=42)



class Agent:
    def __init__(self, agent_id, feat_dim, beta):
        self.id = agent_id
        self.w = np.random.randn(feat_dim) * 0.1
        self.beta = beta
        self.mu = 0.0

    def get_features(self, observation):
        feat = np.mean(observation, axis=(0, 1))
        return feat
    
    def update(self, phi, phi_next, reward):
        td_error = reward - self.mu + phi_next.dot(self.w) - phi.dot(self.w)
        self.w += self.beta * td_error * phi
        self.mu = (1 - self.beta) * self.mu + self.beta * reward
        return td_error


def consensus_update(agents, A):
    """Perform consensus step using weight matrix A"""
    all_w = np.array([agent.w for agent in agents])
    new_w = A @ all_w
    for i, agent in enumerate(agents):
        agent.w = new_w[i]

def compute_msbe(agents, phi, phi_next, rewards):
    """Compute Mean Squared Bellman Error"""
    squared_errors = []
    for agent in agents:
        error = rewards - agent.mu + phi_next.dot(agent.w) - phi.dot(agent.w)
        squared_errors.append(error ** 2)

    return np.mean(squared_errors)


agent_ids = env.agents
num_agents = len(agent_ids)


feat_dim = 256
beta = 0.01

agents = [Agent(agent_id, feat_dim, beta) for agent_id in agent_ids]

A = np.ones(num_agents, num_agents) / num_agents    # Weight matrix for consensus step

K = 50
L = 200

consenesus_errors = []


msbes = []

for l in range(L):

    for k in range(K):

        for i, agent in enumerate(agents):
            obs, reward, term, trun, info = env.last()
            agent_idx = agent_ids.index(agent.id)

            if term or trun:
                continue
            else:
                phi = agent.get_features(obs)
                action = np.random(range(env.action_spaces(agent).n))
            env.step(action)

            next_obs, _, _, _, _ = env.last()
            phi_next = agent.get_features(next_obs)

            td_error = agent.update(phi, phi_next, reward)

        msbe_samples = []
        for i in range(num_agents):
            phi_sample = agents[i].get_features(obs)
            phi_next_sample = agents[i].get_features(next_obs)
            msbe_samples.append(reward)
        current_msbe = compute_msbe(agents, phi_sample, phi_next_sample, msbe_samples)
        msbes.append(current_msbe)

        consensus_update(agents, A)
        print(f"After consensus round {l+1}/{L}, last MSBE = {msbes[-1]:.5f}")
        
env.close()
# Plot MSBE
import matplotlib.pyplot as plt
plt.plot(msbes)
plt.xlabel("Local TD Update Step")
plt.ylabel("Mean Squared Bellman Error (MSBE)")
plt.title("Tracking MSBE at Each Local Sample Round")
plt.grid(True)
plt.show()

        
