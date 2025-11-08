"""
Replay Buffer for Soft Actor-Critic (SAC)

Stores and samples experience tuples (state, action, reward, next_state, done)
for off-policy RL. Supports sequential sampling for LSTM-based networks.
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional
import torch


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.
    
    Each transition (s, a, r, s', done) is stored in a fixed-size circular buffer (FIFO).
    For LSTM-based agents, sequences of consecutive transitions can be sampled.
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        state_dim: int = 23,
        action_dim: int = 7,
    ):
        """
        Initialize the buffer.
        
        Args:
            capacity: Maximum number of stored transitions.
            state_dim: State observation dimension.
            action_dim: Action dimension.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.ptr = 0   # Write pointer
        self.size = 0  # Current buffer size
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store one transition (s, a, r, s', done).
        Overwrites the oldest transition when full (FIFO behavior).
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        seq_len: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions or sequences for LSTM networks.
        
        Args:
            batch_size: Number of samples.
            seq_len: Sequence length (default 5).
            
        Returns:
            states: [batch, seq, state_dim]
            actions: [batch, action_dim]
            rewards: [batch]
            next_states: [batch, seq, state_dim]
            dones: [batch]
        """
        assert self.size >= batch_size, f"Not enough samples: {self.size} < {batch_size}"
        
        max_idx = min(self.size, self.capacity) - 1
        min_idx = seq_len - 1
        
        if max_idx < min_idx:
            # Not enough data for sequences → sample single transitions
            indices = np.random.randint(0, self.size, size=batch_size)
            
            states = torch.FloatTensor(self.states[indices])
            actions = torch.FloatTensor(self.actions[indices])
            rewards = torch.FloatTensor(self.rewards[indices])
            next_states = torch.FloatTensor(self.next_states[indices])
            dones = torch.FloatTensor(self.dones[indices])
            
            # Add sequence dim for LSTM compatibility
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)
            
        else:
            # Sample indices allowing backward look-up of seq_len steps
            indices = []
            max_attempts = batch_size * 10  # Avoid infinite loops
            
            for _ in range(max_attempts):
                if len(indices) >= batch_size:
                    break
                
                idx = np.random.randint(min_idx, max_idx + 1)
                
                # Ensure sequence doesn't cross episode boundary
                seq_indices = np.arange(idx - seq_len + 1, idx + 1)
                if idx >= seq_len:
                    episode_break = np.any(self.dones[seq_indices[:-1]])
                    if episode_break:
                        continue
                indices.append(idx)
            
            # Fill remaining slots if needed
            while len(indices) < batch_size:
                idx = np.random.randint(min_idx, max_idx + 1)
                indices.append(idx)
            
            # Build sequences by looking back seq_len steps
            states_list, next_states_list = [], []
            actions_list, rewards_list, dones_list = [], [], []
            
            for idx in indices:
                seq_indices = np.arange(idx - seq_len + 1, idx + 1)
                states_list.append(self.states[seq_indices])
                next_states_list.append(self.next_states[seq_indices])
                actions_list.append(self.actions[idx])
                rewards_list.append(self.rewards[idx])
                dones_list.append(self.dones[idx])
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states_list))
            next_states = torch.FloatTensor(np.array(next_states_list))
            actions = torch.FloatTensor(np.array(actions_list))
            rewards = torch.FloatTensor(np.array(rewards_list))
            dones = torch.FloatTensor(np.array(dones_list))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def clear(self) -> None:
        """Clear all stored transitions."""
        self.ptr = 0
        self.size = 0
