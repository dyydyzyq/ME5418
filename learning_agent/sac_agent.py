"""
Soft Actor-Critic (SAC) Agent

Implements the SAC algorithm, an off-policy actor-critic method with maximum entropy
regularization. Supports LSTM-based feature extraction for temporal dependencies.

Key features:
- Twin Q networks to reduce overestimation bias
- Automatic temperature (entropy coefficient) tuning
- Target networks for training stability
- LSTM encoder for sequence inputs
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to import network definitions
sys.path.append(str(Path(__file__).resolve().parents[1]))
from net.net_sac import SACNetworks


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent.

    Learns both an actor (policy) and two critic (Q-value) networks to
    maximize expected return and entropy for better exploration.
    """

    def __init__(
        self,
        state_dim: int = 23,
        action_dim: int = 7,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        seq_len: int = 5,
        device: str = "cpu",
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            hidden_dim: Hidden layer width.
            lstm_hidden_dim: LSTM hidden size.
            lr: Learning rate for all optimizers.
            gamma: Discount factor.
            tau: Soft update rate for target networks.
            alpha: Initial entropy coefficient.
            auto_alpha: Enable automatic temperature tuning.
            seq_len: LSTM sequence length.
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.seq_len = seq_len

        # Initialize networks
        self.networks = SACNetworks(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=device,
        )

        self.actor = self.networks.get_actor()
        self.critic1, self.critic2 = self.networks.get_critics()
        self.target_critic1, self.target_critic2 = self.networks.get_target_critics()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Temperature parameter (entropy coefficient)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
            self.log_alpha = None
            self.alpha_optimizer = None

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Select an action using the current policy.

        Args:
            state: Current observation [state_dim] or [seq_len, state_dim].
            deterministic: If True, use the mean action (no sampling).
            hidden_state: Optional LSTM hidden state.

        Returns:
            action: Selected action [action_dim].
            new_hidden_state: Updated LSTM hidden state.
        """
        self.actor.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            state_tensor = state_tensor.to(self.device)

            action, _, new_hidden_state = self.actor.get_action_and_logprob(
                state_tensor,
                hidden_state=hidden_state,
                deterministic=deterministic,
            )
            action = action.cpu().numpy().squeeze(0)
        self.actor.train()
        return action, new_hidden_state

    def update(self, replay_buffer, batch_size: int = 256) -> dict:
        """
        Perform one SAC update step.

        Samples a batch, updates:
        - Critics (TD learning)
        - Actor (policy gradient)
        - Target networks (soft update)
        - Temperature alpha (if enabled)

        Returns:
            A dictionary of loss values for logging.
        """
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size=batch_size, seq_len=self.seq_len
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ---------- Update Critics ----------
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action_and_logprob(
                next_states, hidden_state=None, deterministic=False
            )
            target_q1, _ = self.target_critic1(next_states, next_actions, hidden_state=None)
            target_q2, _ = self.target_critic2(next_states, next_actions, hidden_state=None)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + self.gamma * (
                target_q - self.alpha * next_log_probs
            ) * (1 - dones.unsqueeze(1))

        current_q1, _ = self.critic1(states, actions, hidden_state=None)
        current_q2, _ = self.critic2(states, actions, hidden_state=None)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 10.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.critic2_optimizer.step()

        # ---------- Update Actor ----------
        new_actions, log_probs, _ = self.actor.get_action_and_logprob(
            states, hidden_state=None, deterministic=False
        )
        q1_new, _ = self.critic1(states, new_actions, hidden_state=None)
        q2_new, _ = self.critic2(states, new_actions, hidden_state=None)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # ---------- Update Temperature ----------
        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # ---------- Soft Update Target Networks ----------
        self.networks.update_target_networks(tau=self.tau)

        grad_norms = self.compute_grad_norm()

        losses = {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item()
            if isinstance(self.alpha, torch.Tensor)
            else self.alpha,
        }
        losses.update(grad_norms)
        if alpha_loss is not None:
            losses["alpha_loss"] = alpha_loss.item()
        return losses

    def save(self, filepath: str) -> None:
        """Save all networks and optimizers to a checkpoint file."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "alpha": self.alpha.item()
            if isinstance(self.alpha, torch.Tensor)
            else self.alpha,
            "log_alpha": self.log_alpha.item() if self.log_alpha is not None else None,
        }

        if self.alpha_optimizer is not None:
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """Load agent state (networks + optimizers) from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])
        if self.auto_alpha and "alpha_optimizer_state_dict" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            if "log_alpha" in checkpoint and checkpoint["log_alpha"] is not None:
                self.log_alpha.data = torch.tensor(checkpoint["log_alpha"], device=self.device)
                self.alpha = self.log_alpha.exp()

    def compute_grad_norm(self):
        """Compute gradient norms of actor and critics for monitoring."""
        grad_norms = {}

        def total_grad_norm(model):
            total = 0
            for p in model.parameters():
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
            return total ** 0.5

        grad_norms["actor_grad_norm"] = total_grad_norm(self.actor)
        grad_norms["critic1_grad_norm"] = total_grad_norm(self.critic1)
        grad_norms["critic2_grad_norm"] = total_grad_norm(self.critic2)
        return grad_norms
