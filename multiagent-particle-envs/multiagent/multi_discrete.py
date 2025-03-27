import numpy as np
import gym
from gym.spaces import Discrete

# Fix for newer gym versions that don't have prng
# Instead of importing prng, we'll use numpy's random

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different number of actions in each
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of positive integers specifying number of actions for each discrete action space
    
    Note: Some environment wrappers assume a value of 0 always represents the NOOP action.
    
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: [5]
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: [2]
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: [2]
    
    - Can be initialized as
        MultiDiscrete([ 5, 2, 2 ])
    """
    def __init__(self, nvec):
        """
        nvec: vector of counts of discrete actions for each discrete action space
        """
        assert (np.array(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = np.asarray(nvec, dtype=np.int64)
        gym.Space.__init__(self, self.nvec.shape, np.int64)
        
    def sample(self):
        return (np.random.random_sample(self.nvec.shape) * self.nvec).astype(self.dtype)
    
    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
            
        # if nvec is one dimensional, x can be a number
        if self.nvec.ndim == 1 and np.isscalar(x):
            x = np.array([x])
            
        return x.shape == self.nvec.shape and (0 <= x).all() and (x < self.nvec).all()
        
    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]
    
    def from_jsonable(self, sample_n):
        return np.array(sample_n)
    
    def __repr__(self):
        return "MultiDiscrete({})".format(self.nvec)