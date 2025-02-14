import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio

from pettingzoo.mpe import simple_spread_v3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device: ", device)



L = 200 
K = 50
beta = 0.1

env = simple_spread__v3.env(N=9, local_ratio=0.5, max_cycles=None, continuous_actions=False, render_mode='rgb_array')

class Agent:
    def __init__(self, agent, feat_dim):
        self.agent = agent
        self.feat_dim = feat_dim
        self.delta = []
        self.phi = torch.zeros(feat_dim, agent.observation_space.shape[0], agent.action_space.n).to(device)
        self.phi.requires_grad = True   
        self.mu = 0







for l in range(0, L-1):
    for k in range(0, K-1):
        for agent in env.agent_iter():
            obs, reward, term, trun, info = env.last()
            print(obs)

            if term or trun:
                action = None

            else:
                action = env.action_spaces[agent].sample()

                delta_i = reward - agent.mu 

            