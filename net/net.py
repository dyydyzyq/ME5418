import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import copy


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor for SAC networks.
    Extracts features(obstacle moving pattern, task awareness) from state sequences using FC1 + FC2 + LSTM.
    """
    
    def __init__(
        self,
        state_dim: int = 23,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        device: str = "cpu"
    ):
        super(FeatureExtractor, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = device
        
        # Feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # LSTM layer for capturing temporal dependencies of dynamic obstacles
        # and helping the robotic arm remember obstacle movements over time
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract features from state sequences.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state tuple (h, c)
            
        Returns:
            features: Extracted features [batch_size, lstm_hidden_dim] of the last timestep
            new_hidden_state: Updated LSTM hidden state tuple
        """
        # Feature extraction
        fc1_out = F.relu(self.fc1(states))  # [batch_size, seq_len, hidden_dim]
        fc2_out = F.relu(self.fc2(fc1_out))  # [batch_size, seq_len, hidden_dim]
        
        # Residual connection: FC1 + FC2
        residual_out = fc1_out + fc2_out  # [batch_size, seq_len, hidden_dim]
        
        # LSTM processing
        lstm_out, new_hidden_state = self.lstm(residual_out, hidden_state)
        # lstm_out: [batch_size, seq_len, lstm_hidden_dim]
        
        # Use the last timestep output
        features = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        return features, new_hidden_state
    
    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state
  
           Returns:
            hidden_state (tuple): (h, c), where
                h: hidden state tensor of shape [num_layers, batch_size, lstm_hidden_dim]
                c: cell state tensor of shape [num_layers, batch_size, lstm_hidden_dim]
        """
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device)
        return (h, c)


class Actor(nn.Module):
    """
    SAC Actor network.
    Outputs Gaussian policy parameters (mean and log_std).
    """
    
    def __init__(
        self,
        state_dim: int = 23,
        action_dim: int = 7,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        device: str = "cpu"
    ):
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=device
        )
        
        # Policy heads
        self.policy_mean_head = nn.Linear(lstm_hidden_dim, action_dim)
        self.policy_log_std_head = nn.Linear(lstm_hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        states: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the actor network.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state tuple
            
        Returns:
            policy_mean: Mean of action distribution [batch_size, action_dim]
            policy_log_std: Log standard deviation of action distribution [batch_size, action_dim]
            new_hidden_state: Updated LSTM hidden state tuple
        """
        # Extract features
        features, new_hidden_state = self.feature_extractor(states, hidden_state)
        
        # Policy outputs
        policy_mean = self.policy_mean_head(features)
        policy_log_std = self.policy_log_std_head(features)
        
        return policy_mean, policy_log_std, new_hidden_state
    
    def get_action_and_logprob(
        self, 
        states: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action from the policy network using Gaussian distribution.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state tuple
            deterministic: If True, return mean action; if False, sample from distribution
            
        Returns:
            action: Actions [batch_size, action_dim] (bounded in [-1, 1])
            log_prob: Log probabilities of actions [batch_size, 1]
            new_hidden_state: Updated LSTM hidden state tuple
        """
        policy_mean, policy_log_std, new_hidden_state = self.forward(states, hidden_state)
        
        # Clamp log_std to prevent numerical instability
        policy_log_std = torch.clamp(policy_log_std, min=-20, max=2)
        policy_std = torch.exp(policy_log_std)
        
        if deterministic:
            # Return mean action (no sampling)
            raw_action = policy_mean
            action = torch.tanh(raw_action)
            log_prob = None
        else:
            # Sample from Gaussian distribution
            normal = torch.distributions.Normal(policy_mean, policy_std)
            raw_action = normal.rsample()  # Reparameterization trick
            
            # Apply tanh squashing for bounded actions
            action = torch.tanh(raw_action)
            
            # Compute log probability considering tanh transformation
            log_prob = normal.log_prob(raw_action)
            # Apply tanh correction: log_prob -= log(1 - tanh^2(raw_action))
            log_prob -= torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)  # Sum over action dimensions
        
        return action, log_prob, new_hidden_state

class Critic(nn.Module):
    """
    SAC Critic network (Q-network).
    Takes (state,action) pair as input, outputs Q-value.
    """
    
    def __init__(
        self,
        state_dim: int = 23,
        action_dim: int = 7,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        device: str = "cpu"
    ):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=device
        )
        
        # Q-value head (takes features + action as input)
        self.q_head = nn.Linear(lstm_hidden_dim + action_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the critic network.
        
        Args:
            states: Batch of state sequences [batch_size, seq_len, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            hidden_state: Optional LSTM hidden state tuple
            
        Returns:
            q_value: Q-values [batch_size, 1]
            new_hidden_state: Updated LSTM hidden state tuple
        """
        # Extract features from states
        features, new_hidden_state = self.feature_extractor(states, hidden_state)
        
        # Concatenate features with actions
        q_input = torch.cat([features, actions], dim=-1)  # [batch_size, lstm_hidden_dim + action_dim]
        
        # Compute Q-value
        q_value = self.q_head(q_input)  # [batch_size, 1]
        
        return q_value, new_hidden_state


class SACNetworks:
    """
    Complete SAC network collection.
    Includes Actor, twin Critic networks (Q1, Q2) to mitigate Q-value overestimation, and their target networks for stable training.
    """
    
    def __init__(
        self,
        state_dim: int = 23,
        action_dim: int = 7,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Create networks
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=device
        )
        
        self.critic1 = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=device
        )
        
        self.critic2 = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=device
        )
        
        # Create target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Move all networks to device
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.target_critic1.to(device)
        self.target_critic2.to(device)
        
        # Freeze target networks
        self._freeze_target_networks()
    
    def _freeze_target_networks(self):
        """Freeze target networks (no gradient updates)"""
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
    
    def get_actor(self) -> Actor:
        """Get the actor network"""
        return self.actor
    
    def get_critics(self) -> Tuple[Critic, Critic]:
        """Get both critic networks"""
        return self.critic1, self.critic2
    
    def get_target_critics(self) -> Tuple[Critic, Critic]:
        """Get both target critic networks"""
        return self.target_critic1, self.target_critic2
    
    def update_target_networks(self, tau: float = 0.005):
        """
        Soft update target networks.
        
        Args:
            tau: Soft update coefficient (0 < tau <= 1)
        """
        # Update target_critic1
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Update target_critic2
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def get_parameters(self):
        """Get all trainable parameters"""
        return {
            'actor': list(self.actor.parameters()),
            'critic1': list(self.critic1.parameters()),
            'critic2': list(self.critic2.parameters())
        }
    
    def get_parameter_count(self):
        """Get parameter count for each network"""
        return {
            'actor': sum(p.numel() for p in self.actor.parameters()),
            'critic1': sum(p.numel() for p in self.critic1.parameters()),
            'critic2': sum(p.numel() for p in self.critic2.parameters()),
            'total': sum(p.numel() for p in self.actor.parameters()) + 
                     sum(p.numel() for p in self.critic1.parameters()) + 
                     sum(p.numel() for p in self.critic2.parameters())
        }


# Factory functions for backward compatibility and ease of use
def create_feature_extractor(
    state_dim: int = 23,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    device: str = "cpu"
) -> FeatureExtractor:
    """Create a feature extractor network"""
    return FeatureExtractor(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        device=device
    )

def create_actor(
    state_dim: int = 23,
    action_dim: int = 7,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    device: str = "cpu"
) -> Actor:
    """Create an actor network"""
    return Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        device=device
    )

def create_critic(
    state_dim: int = 23,
    action_dim: int = 7,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    device: str = "cpu"
) -> Critic:
    """Create a critic network"""
    return Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        device=device
    )

def create_sac_networks(
    state_dim: int = 23,
    action_dim: int = 7,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    device: str = "cpu"
) -> SACNetworks:
    """Create complete SAC networks (Actor + Twin Critics + Target networks)"""
    return SACNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        device=device
    )

# Legacy compatibility functions
def create_network(
    state_dim: int = 23,
    action_dim: int = 7,
    hidden_dim: int = 64,
    lstm_hidden_dim: int = 64,
    seq_len: int = 5,
    device: str = "cpu"
) -> Actor:
    """
    Legacy function for backward compatibility.
    Returns an Actor network (for simple use cases).
    """
    return create_actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        device=device
    )