from pettingzoo.butterfly import cooperative_pong_v5
import numpy as np
import torch


env = cooperative_pong_v5.parallel_env(render_mode="human")
observations, infos = env.reset(seed=42)



class Agent:
    def __init__(self, agent_id, feat_dim, beta):
        self.id = agent_id
        self.w = np.random.randn(feat_dim) * 0.1
        self.beta = beta
        self.mu = 0.0
        self.feature_dim = feat_dim

    def get_features(self, observation):
        reshaped_obs = observation.reshape((-1, self.feature_dim))
        return np.mean(reshaped_obs, axis=0)
    
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


agent_ids = env.possible_agents
num_agents = len(agent_ids)


feat_dim = 256
beta = 0.01

agents = [Agent(agent_id, feat_dim, beta) for agent_id in agent_ids]

A = np.ones((num_agents, num_agents)) / num_agents    # Weight matrix for consensus step

K = 50
L = 200

consenesus_errors = []


msbes = []

for l in range(L):
    observations, infos = env.reset()
    env.render()
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
        
env.close()
# Plot MSBE
import matplotlib.pyplot as plt
plt.plot(msbes)
plt.xlabel("Local TD Update Step")
plt.ylabel("Mean Squared Bellman Error (MSBE)")
plt.title("Tracking MSBE at Each Local Sample Round")
plt.grid(True)
plt.show()

        
