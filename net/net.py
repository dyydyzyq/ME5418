import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PandaActorCriticNetwork(nn.Module):
    """
    Neural network for Panda robot with moving obstacles environment.
    
    Architecture:
    - Input: 23-dimensional state vector
    - FC1: 23 -> 64 (feature extraction)
    - FC2: 64 -> 64 (pre-LSTM processing)
    - Residual connection: FC1 + FC2
    - LSTM: 64 -> 64 (sequence processing)
    - Policy head: 64 -> 7 (action output)
    - Value head: 64 -> 1 (state value)
    """
    
    def __init__(
        self,
        state_dim: int = 23,
        action_dim: int = 7,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        seq_len: int = 5,
        device: str = "cpu"
    ):
        super(PandaActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.seq_len = seq_len
        self.device = device
        
        # Feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )
        
        # Policy head (actor)
        self.policy_head = nn.Linear(lstm_hidden_dim, action_dim)
        
        # Value head (critic)
        self.value_head = nn.Linear(lstm_hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def forward(
        self, 
        states: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state tuple (h, c)
            
        Returns:
            policy: Action probabilities/logits [batch_size, action_dim]
            value: State values [batch_size, 1]
            new_hidden_state: Updated LSTM hidden state tuple
        """
        batch_size = states.size(0)
        
        # Feature extraction
        # states: [batch_size, seq_len, state_dim]
        fc1_out = F.relu(self.fc1(states))  # [batch_size, seq_len, hidden_dim]
        fc2_out = F.relu(self.fc2(fc1_out))  # [batch_size, seq_len, hidden_dim]
        
        # Residual connection: FC1 + FC2
        residual_out = fc1_out + fc2_out  # [batch_size, seq_len, hidden_dim]
        
        # LSTM processing
        lstm_out, new_hidden_state = self.lstm(residual_out, hidden_state)
        # lstm_out: [batch_size, seq_len, lstm_hidden_dim]
        
        # Use the last timestep output for policy and value
        last_output = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # Policy and value outputs
        policy = self.policy_head(last_output)  # [batch_size, action_dim]
        value = self.value_head(last_output)    # [batch_size, 1]
        
        return policy, value, new_hidden_state
    
    def get_action(
        self, 
        states: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action from the policy network.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state tuple
            deterministic: If True, return mean action; if False, sample from distribution
            
        Returns:
            action: Actions [batch_size, action_dim]
            log_prob: Log probabilities of actions [batch_size, action_dim]
            new_hidden_state: Updated LSTM hidden state tuple
        """
        policy, value, new_hidden_state = self.forward(states, hidden_state)
        
        if deterministic:
            action = torch.tanh(policy)  # Assuming bounded actions [-1, 1]
            log_prob = None
        else:
            # For SAC, we typically use a Gaussian policy
            # Here we'll use tanh squashing for bounded actions
            action = torch.tanh(policy)
            # Note: For proper SAC implementation, you'd need to compute log probabilities
            # considering the tanh transformation and Jacobian determinant
            log_prob = None  # Placeholder - implement proper log prob calculation for SAC
        
        return action, log_prob, new_hidden_state
    
    def get_action_and_log_prob(
        self, 
        states: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action and log probability for SAC training.
        This is a placeholder for proper SAC implementation.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state tuple
            
        Returns:
            action: Actions [batch_size, action_dim]
            log_prob: Log probabilities of actions [batch_size, action_dim]
            new_hidden_state: Updated LSTM hidden state tuple
        """
        policy, value, new_hidden_state = self.forward(states, hidden_state)
        
        # For SAC, you would typically:
        # 1. Use policy as mean of Gaussian distribution
        # 2. Add learnable log_std parameter
        # 3. Sample from Gaussian: action = mean + std * noise
        # 4. Apply tanh squashing: action = tanh(raw_action)
        # 5. Compute log probability considering tanh transformation
        
        # Placeholder implementation - just return tanh(policy) and None
        action = torch.tanh(policy)
        log_prob = None  # Would need proper implementation for SAC
        
        return action, log_prob, new_hidden_state
    
    def get_value(
        self, 
        states: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get state value from the critic network.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state tuple
            
        Returns:
            value: State values [batch_size, 1]
        """
        _, value, _ = self.forward(states, hidden_state)
        return value
    
    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            hidden_state: Initial LSTM hidden state tuple (h, c)
        """
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device)
        return (h, c)


def create_network(
    state_dim: int = 23,
    action_dim: int = 7,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    seq_len: int = 5,
    device: str = "cpu"
) -> PandaActorCriticNetwork:
    """
    Factory function to create the Panda Actor-Critic network.
    
    Args:
        state_dim: Dimension of state space (default: 23)
        action_dim: Dimension of action space (default: 7)
        hidden_dim: Hidden layer dimension (default: 64)
        lstm_hidden_dim: LSTM hidden dimension (default: 64)
        seq_len: Sequence length for LSTM (default: 5)
        device: Device to place network on (default: "cpu")
        
    Returns:
        PandaActorCriticNetwork: Initialized network
    """
    return PandaActorCriticNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        seq_len=seq_len,
        device=device
    )


def create_actor_network(
    state_dim: int = 23,
    action_dim: int = 7,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    seq_len: int = 5,
    device: str = "cpu"
) -> PandaActorCriticNetwork:
    """
    Factory function to create the Actor network for SAC.
    This is the same as create_network but with clearer naming for SAC usage.
    
    Args:
        state_dim: Dimension of state space (default: 23)
        action_dim: Dimension of action space (default: 7)
        hidden_dim: Hidden layer dimension (default: 64)
        lstm_hidden_dim: LSTM hidden dimension (default: 64)
        seq_len: Sequence length for LSTM (default: 5)
        device: Device to place network on (default: "cpu")
        
    Returns:
        PandaActorCriticNetwork: Initialized actor network
    """
    return create_network(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        seq_len=seq_len,
        device=device
    )


def create_critic_network(
    state_dim: int = 23,
    action_dim: int = 7,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    seq_len: int = 5,
    device: str = "cpu"
) -> PandaActorCriticNetwork:
    """
    Factory function to create the Critic network for SAC.
    This is the same as create_network but with clearer naming for SAC usage.
    
    Args:
        state_dim: Dimension of state space (default: 23)
        action_dim: Dimension of action space (default: 7)
        hidden_dim: Hidden layer dimension (default: 64)
        lstm_hidden_dim: LSTM hidden dimension (default: 64)
        seq_len: Sequence length for LSTM (default: 5)
        device: Device to place network on (default: "cpu")
        
    Returns:
        PandaActorCriticNetwork: Initialized critic network
    """
    return create_network(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        seq_len=seq_len,
        device=device
    )


