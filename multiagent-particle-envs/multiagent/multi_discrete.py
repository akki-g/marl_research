"""
Multi-discrete space.
Adapted from the original implementation by OpenAI.
"""

from typing import List, Tuple, Union, Dict, Any
import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces import Space


class MultiDiscrete(Space):
    """
    Multi-discrete action space.
    
    The multi-discrete action space consists of a series of discrete action spaces with different parameters.
    It can be adapted to both a Discrete action space or a continuous (Box) action space.
    It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space.
    
    It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
    where the discrete action space can take any integers from `min` to `max` (both inclusive).
    
    Note: A value of 0 always need to represent the NOOP action.
    
    Example:
    Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5 - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4] - params: min: 0, max: 4
        2) Button A: Discrete 2 - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B: Discrete 2 - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as MultiDiscrete([[0,4], [0,1], [0,1]])
    """

    def __init__(self, array_of_param_array: List[List[int]]):
        """
        Initialize the MultiDiscrete space.
        
        Args:
            array_of_param_array: List of [min, max] pairs for each discrete action space
        """
        self.low = np.array([x[0] for x in array_of_param_array], dtype=np.int64)
        self.high = np.array([x[1] for x in array_of_param_array], dtype=np.int64)
        
        # Number of discrete spaces
        self.num_discrete_space = self.low.shape[0]
        
        # Set the shape and dtype
        self.shape = (self.num_discrete_space,)
        self.dtype = np.int64
        
        super().__init__(self.shape, self.dtype)

    def sample(self, mask=None) -> np.ndarray:
        """
        Sample a random point from the space.
        
        Args:
            mask: Optional mask for sampling (not implemented for this space)
            
        Returns:
            A random sample from the space
        """
        if mask is not None:
            raise NotImplementedError("Masked sampling is not implemented for MultiDiscrete")
            
        # For each row: round(random * (max - min) + min, 0)
        random_array = self.np_random.random(self.num_discrete_space)
        
        # Generate samples as integers between low and high
        samples = np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)
        
        return samples.astype(self.dtype)

    def contains(self, x) -> bool:
        """
        Check if x is contained in the space.
        
        Args:
            x: The point to check
            
        Returns:
            True if x is contained in the space, False otherwise
        """
        if isinstance(x, list):
            x = np.array(x)  # Ensure x is an array for easier checks
            
        # Check if the array shape matches and all values are within bounds
        return (
            isinstance(x, np.ndarray) and
            x.shape == self.shape and
            np.all(x >= self.low) and
            np.all(x <= self.high) and
            np.all(x.astype(int) == x)  # All values must be integers
        )

    def __repr__(self) -> str:
        """String representation of the space."""
        return f"MultiDiscrete({self.low}, {self.high})"

    def __eq__(self, other) -> bool:
        """Check if two spaces are equal."""
        return (
            isinstance(other, MultiDiscrete) and
            np.array_equal(self.low, other.low) and
            np.array_equal(self.high, other.high)
        )