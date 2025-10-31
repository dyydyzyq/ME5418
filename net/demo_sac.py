"""
Minimal SAC Network Forward/Backward Test
"""

import torch
import sys, os

# Import your network creator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net_sac import create_sac_networks

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create full SAC network bundle
    sac = create_sac_networks(device=device)
    actor = sac.get_actor()
    critic1, critic2 = sac.get_critics()

    # Example input
    batch_size, seq_len, state_dim, action_dim = 4, 5, 37, 7
    states = torch.randn(batch_size, seq_len, state_dim, device=device, requires_grad=True)

    # ---- Forward pass ----
    policy_mean, policy_log_std, hidden = actor(states)
    action, log_prob, _ = actor.get_action_and_logprob(states, deterministic=False)
    q1_value, _ = critic1(states, action)
    q2_value, _ = critic2(states, action)

    print(f"policy mean shape: {policy_mean.shape}")
    print(f"policy log_std shape: {policy_log_std.shape}")
    print(f"Action shape: {action.shape}")
    print(f"lop prob shape: {log_prob.shape}")
    print(f"Q1 value shape: {q1_value.shape}")
    print(f"Q2 value shape: {q2_value.shape}")


    # ---- Compute dummy loss ----
    policy_loss = -policy_mean.mean()
    q_loss = (q1_value.mean() + q2_value.mean()) / 2
    total_loss = policy_loss + q_loss

    print(f"Total loss: {total_loss.item():.6f}")

    # ---- Backward pass ----
    total_loss.backward()

    # Print gradient norms for sanity check
    actor_grad = sum(p.grad.norm().item() for p in actor.parameters() if p.grad is not None)
    critic_grad = sum(p.grad.norm().item() for p in critic1.parameters() if p.grad is not None)
    print(f"Actor grad norm sum: {actor_grad:.6f}")
    print(f"Critic1 grad norm sum: {critic_grad:.6f}")

    print("✅ Forward & backward test completed successfully!")
    print("-------------------\n ")

if __name__ == "__main__":
    main()
