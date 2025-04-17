import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3


RUN_TYPE = input("What is the run env?\n")
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

torch.device(DEVICE)


class Agents:
    def __init__(self, agent_id, feat_dim, init_scale=0.01):
        self.agent_id = agent_id
        
        # Initialize weights with small random values as in the paper
        self.w = np.random.randn(feat_dim) * init_scale
        
        # Initialize average reward estimate to 0
        self.mu = 0.0
        
        # Previous feature vector (for storing between steps)
        self.prev_phi = None
        
    def update(self, phi_s_next, reward, beta):
        """
        Perform local TD update using current transition
        
        Args:
            phi_s: Feature vector for current state
            phi_s_next: Feature vector for next state
            reward: Reward received
            beta: Step size parameter
            
        Returns:
            Dictionary with update information
        """
        # Calculate current state value
        value_current = np.dot(self.prev_phi.t(), self.w)
        
        # Calculate next state value
        value_next = np.dot(phi_s_next.t(), self.w)
        
        # Calculate TD error: δ = r - μ + φ(s')ᵀw - φ(s)ᵀw
        delta = reward - self.mu + value_next - value_current
        
        # Update average reward estimate: μ = (1-β)μ + βr
        self.mu = (1 - beta) * self.mu + beta * reward
        
        # Update weight vector: w = w + βδφ(s)
        self.w = self.w + beta * delta * self.prev_phi
        
        # Return relevant data for metrics calculation
        return {
            'phi_s': self.prev_phi,
            'phi_s_next': phi_s_next,
            'reward': reward,
            'mu': self.mu,
            'w': self.w,
            'delta': delta
        }

class LocalTD_Algorithm():
    def __init__(self, env, L, K, beta=0.01, norm='l2'):
        self.env = env
        self.numAgents = len(env.agents)
        self.numLm = self.numAgents
        self.L = L
        self.K = K
        self.norm = 'l2'
        self.agents = self.init_agents()
        self.A

        self.beta = beta
        self.SBE_history = []
        self.MSBE_history = []
        self.consensus_history = []
    
    def init_agents(self):
        obs, infos = self.env.reset(seed=42)
        sample_phi = self.get_feature_matrix(obs)
        feat_dim = sample_phi.shape[0]

        if RUN_TYPE == 'test':
            print(f"Feature Matrix Dim: {feat_dim} ")

        self.agents = {agent: Agents(agent, feat_dim) 
                       for agent in self.env.agents}
        
        for agent in self.env.agents:
            phi_init = torch.tensor(self.get_feature_matrix(obs[agent]),
                                    device=DEVICE, dtype=torch.float32)
            self.agents[agent].prev_phi = phi_init

    def get_feature_matrix(self, obs):
        agent_pos = obs[2:4]
        
        start_lm = 4
        end_lm = start_lm + self.numLm * 2
        lm_rel_pos = obs[start_lm:end_lm]

        other_agents_rel_pos = obs[end_lm:end_lm + (self.numAgents - 1) *2]
        feat_vec = np.concatenate([agent_pos, lm_rel_pos, other_agents_rel_pos])
        ord = int(self.norm[1])
        norm = np.linalg.norm(feat_vec, ord=ord)
        if norm != 0:
            feat_vec /= norm
        return feat_vec


    def conseneus_error_update(self):
        if len(self.agents == 0):
            return 0.0

        W = torch.stack(self.agents[agent].w for agent in self.agents)
        w_bar = torch.mean(W, dim=0)
        consensus_error = torch.mean(torch.sum((W - w_bar) ** 2, dim=1))
        self.consensus_history.append(consensus_error)
        
    
    def consenus_update(self):
        W = torch.stack([self.agents[agent].w for agent in self.agents], dim=0)
        new_w = torch.matmul(self.A, W)
        for i, agent in enumerate(self.agents):
            self.agents[agent].w = new_w[i].clone()
    
    def msbe_update(self, sample_data):
        agent_ids = list(sample_data.keys())
        N = len(agent_ids)
        if N == 0:
            return 0.0
        r_bar = sum(sample_data[agent]['reward'] for agent in agent_ids) / N
        mu_bar = sum(sample_data[agent]['mu'] for agent in agent_ids) / N
        errors = []
        for agent in agent_ids:
            w = sample_data[agent]['w']
            phi_s = sample_data[agent]['phi_s']
            phi_s_next = sample_data[agent]['phi_s_next']
            value_current = torch.dot(phi_s.t(), w)
            value_next = torch.dot(phi_s_next.t(), w)
            error = value_current + mu_bar - r_bar - value_next
            errors.append(error ** 2)
        SBE = sum(errors) / N

        self.SBE_history.append(SBE)
        current_msbe = np.mean(self.SBE_history)
        if RUN_TYPE == 'test':
            print(current_msbe)
        self.MSBE_history.append(current_msbe)

    def run_local_td_update(self):
        for l in range(self.L):

            for k in range(self.K):
                actions = {agent: int(self.env.action_space(agent).sample())
                                      for agent in self.agents}
                obs, rewards, done, truncs, terms = self.env.step(actions)
                
                if not self.agents:
                    print("No more agents resetting env")
                    self.agents = self.init_agents()

                sample_data = {}
                for agent in self.env.agents:
                    td_agent = self.agents[agent]
                    phi_s1 = torch.tensor(self.get_feature_matrix(obs[agent], 
                                                                  device=DEVICE, dtype=torch.float32))
                    
                    sample_data[agent] = td_agent.update(phi_s_next=phi_s1, reward=rewards[agent], beta=self.beta)
                    td_agent.prev_phi = phi_s1
                self.msbe_update(sample_data=sample_data)
                self.conseneus_error_update()
            self.consenus_update()
        return self.consensus_history, self.MSBE_history



def run_maxCycles_comp(L, K, beta, numAgents):
    numCycles = [20, 50, 100, 200, 10000]

    for cycle in numCycles:
        env = simple_spread_v3.parallel_env(
            N=numAgents, local_ratio=0.5, max_cycles=cycle
        )

        localTD = LocalTD_Algorithm(env=env, L=L, K=K, beta=beta)
        test_consensus, test_msbe = localTD.run_local_td_update()
        env.close()

def run_test():
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25)
    localTD = LocalTD_Algorithm(env=env, L=500, K=20, beta=0.01)
    consensus, msbe = localTD.run_local_td_update()
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(msbe, marker='.')
    plt.xlabel("Sample Update Step")
    plt.ylabel("MSBE")
    plt.title("Running MSBE per Sample Update (Average Reward)")

    plt.subplot(1,2,2)
    plt.plot(consensus, marker='.', color='red')
    plt.xlabel("Sample Update Step")
    plt.ylabel("Consensus Error")
    plt.title("Consensus Error per Sample Update")
    plt.tight_layout()
    plt.show()


run_test()


    