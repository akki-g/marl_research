from pettingzoo.butterfly import cooperative_pong_v5
import numpy as np
import torch


env = cooperative_pong_v5.parallel_env()
env.reset(seed=42)


def consensus_update(agents, A):
    """Perform consensus step using weight matrix A"""
    all_w = np.array([agent.w for agent in agents])
    new_w = A @ all_w
    for i, agent in enumerate(agents):
        agent.w = new_w[i]

def compute_msbe(agents, state, next_state):
    """Compute Mean Squared Bellman Error"""
    errors = []
    for agent in agents:
        phi = agent.get_features(state)
        phi_next = agent.get_features(next_state)
        error = (agent.reward - agent.mu + 
                 phi_next.dot(agent.w) - 
                 phi.dot(agent.w))
        errors.append(error**2)
    return np.mean(errors)

for agent in env.agent_iter():
    obs, reward, term, trun, info = env.last()
        
    if term or trun:
        action = None
        continue
    else:
        action = np.random.choice(env.action_spaces[agent])

def local_TD(env):

        
