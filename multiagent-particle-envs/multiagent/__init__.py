"""
Multi-Agent Particle Environment

A modernized version of the OpenAI Multi-Agent Particle Environment.
"""

__version__ = '0.1.0'

from typing import Dict, List

# Dictionary mapping scenario names to their file names
SCENARIOS: Dict[str, str] = {
    'simple': 'simple.py',
    'simple_adversary': 'simple_adversary.py',
    'simple_crypto': 'simple_crypto.py',
    'simple_push': 'simple_push.py',
    'simple_reference': 'simple_reference.py',
    'simple_speaker_listener': 'simple_speaker_listener.py',
    'simple_spread': 'simple_spread.py',
    'simple_tag': 'simple_tag.py',
    'simple_world_comm': 'simple_world_comm.py',
}

# List of all available scenarios
AVAILABLE_SCENARIOS: List[str] = list(SCENARIOS.keys())