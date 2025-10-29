"""
Panda SAC Networks Demo
Test the complete SAC network architecture with Actor, Twin Critics, and Target networks
"""

import torch
import numpy as np
import sys
import os
import time

# Add parent directory to path to import net module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net import create_sac_networks, create_actor, create_critic

# Delay (seconds) used throughout the demo to make the creation/execution feel paced.
# Can be overridden with environment variable NET_DEMO_DELAY (e.g. NET_DEMO_DELAY=0.2)
DEMO_DELAY = float(os.getenv("NET_DEMO_DELAY", "0.8"))


def test_sac_networks_creation():
    """Test SAC networks creation"""
    print("=" * 60)
    print("1. SAC Networks Creation Test")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    time.sleep(DEMO_DELAY)
    
    # Create complete SAC networks
    print("\n📝 Creating complete SAC networks...")
    time.sleep(DEMO_DELAY)
    sac_networks = create_sac_networks(device=device)
    time.sleep(DEMO_DELAY)
    print("✅ SAC networks created successfully!")
    time.sleep(DEMO_DELAY)
    
    # Get individual networks
    actor = sac_networks.get_actor()
    critic1, critic2 = sac_networks.get_critics()
    target_critic1, target_critic2 = sac_networks.get_target_critics()
    
    # Parameter count
    param_counts = sac_networks.get_parameter_count()
    print(f"\n📊 Parameter counts:")
    for name, count in param_counts.items():
        print(f"   - {name}: {count:,}")
        time.sleep(0.03)
    
    return sac_networks, device


def test_actor_forward_pass(sac_networks, device):
    """Test actor forward pass"""
    print("\n" + "=" * 60)
    print("2. Actor Forward Pass Test")
    print("=" * 60)
    
    actor = sac_networks.get_actor()
    
    # Test parameters
    batch_size = 4
    seq_len = 5
    state_dim = 23
    action_dim = 7
    
    print(f"📊 Test parameters:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - State dimension: {state_dim}")
    print(f"   - Action dimension: {action_dim}")
    time.sleep(DEMO_DELAY)
    
    # Create test data
    states = torch.randn(batch_size, seq_len, state_dim, device=device)
    print(f"   - Input shape: {states.shape}")
    
    # Forward pass
    print(f"\n🔄 Executing actor forward pass...")
    time.sleep(DEMO_DELAY)
    with torch.no_grad():
        policy_mean, policy_log_std, hidden_state = actor(states)
    time.sleep(DEMO_DELAY)
    print(f"✅ Actor forward pass completed!")
    print(f"\n📤 Output results:")
    print(f"   - Policy mean shape: {policy_mean.shape}")
    print(f"   -{policy_mean}")
    print(f"   - Policy mean range: [{policy_mean.min().item():.3f}, {policy_mean.max().item():.3f}]")
    print(f"   - Policy log_std shape: {policy_log_std.shape}")
    print(f"   -{policy_log_std}")
    print(f"   - Policy log_std range: [{policy_log_std.min().item():.3f}, {policy_log_std.max().item():.3f}]")
    print(f"   - Hidden state shape: h={hidden_state[0].shape}, c={hidden_state[1].shape}")
    
    # Test action generation
    print(f"\n🎯 Testing action generation...")
    time.sleep(DEMO_DELAY)
    with torch.no_grad():
        action, log_prob, new_hidden = actor.get_action(states, deterministic=False)
    time.sleep(DEMO_DELAY)
    print(f"   - Action shape: {action.shape}")
    print(f"   -{action}")
    print(f"   - Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
    print(f"   - Log prob shape: {log_prob.shape}")
    print(f"   -{log_prob}")
    print(f"   - Log prob range: [{log_prob.min().item():.3f}, {log_prob.max().item():.3f}]")
    
    return states, action, log_prob


def test_critics_forward_pass(sac_networks, states, actions, device):
    """Test critics forward pass"""
    print("\n" + "=" * 60)
    print("3. Critics Forward Pass Test")
    print("=" * 60)
    
    critic1, critic2 = sac_networks.get_critics()
    target_critic1, target_critic2 = sac_networks.get_target_critics()
    
    print(f"📊 Testing Twin Q-networks...")
    print(f"   - States shape: {states.shape}")
    print(f"   - Actions shape: {actions.shape}")
    time.sleep(DEMO_DELAY)
    
    # Test Q1 forward pass
    print(f"\n🔄 Testing Q1 forward pass...")
    time.sleep(DEMO_DELAY)
    with torch.no_grad():
        q1_value, q1_hidden = critic1(states, actions)
    time.sleep(DEMO_DELAY)
    print(f"✅ Q1 forward pass completed!")
    print(f"   - Q1 value shape: {q1_value.shape}")
    print(f"   -{q1_value}")
    print(f"   - Q1 value range: [{q1_value.min().item():.3f}, {q1_value.max().item():.3f}]")
    
    # Test Q2 forward pass
    print(f"\n🔄 Testing Q2 forward pass...")
    time.sleep(DEMO_DELAY)
    with torch.no_grad():
        q2_value, q2_hidden = critic2(states, actions)
    time.sleep(DEMO_DELAY)
    print(f"✅ Q2 forward pass completed!")
    print(f"   - Q2 value shape: {q2_value.shape}")
    print(f"   -{q2_value}")
    print(f"   - Q2 value range: [{q2_value.min().item():.3f}, {q2_value.max().item():.3f}]")
    
    # Test target networks
    print(f"\n🔄 Testing target networks...")
    time.sleep(DEMO_DELAY)
    with torch.no_grad():
        target_q1_value, _ = target_critic1(states, actions)
        target_q2_value, _ = target_critic2(states, actions)
    time.sleep(DEMO_DELAY)
    print(f"✅ Target networks forward pass completed!")
    print(f"   - Target Q1 value shape: {target_q1_value.shape}")
    print(f"   -{target_q1_value}")
    print(f"   - Target Q2 value shape: {target_q2_value.shape}")
    print(f"   -{target_q2_value}")
    
    # Verify networks are different (not identical)
    q1_q2_diff = torch.abs(q1_value - q2_value).mean().item()
    print(f"   - Q1-Q2 difference (mean): {q1_q2_diff:.6f}")
    print(f"   - Networks are independent: {q1_q2_diff > 1e-6}")
    
    return q1_value, q2_value, target_q1_value, target_q2_value


def test_training_mode(sac_networks, states, actions, device):
    """Test training mode with gradients"""
    print("\n" + "=" * 60)
    print("4. Training Mode Test")
    print("=" * 60)
    
    actor = sac_networks.get_actor()
    critic1, critic2 = sac_networks.get_critics()
    
    # Set training mode
    actor.train()
    critic1.train()
    critic2.train()
    print(f"🔄 Set all networks to training mode")
    time.sleep(DEMO_DELAY)
    
    # Prepare data with gradients
    states_grad = states.clone().detach().requires_grad_(True)
    actions_grad = actions.clone().detach().requires_grad_(True)
    
    print(f"📊 Prepared data with gradients:")
    print(f"   - States shape: {states_grad.shape}")
    print(f"   - Actions shape: {actions_grad.shape}")
    time.sleep(DEMO_DELAY)
    
    # Actor forward pass
    print(f"\n🔄 Actor forward pass (training mode)...")
    time.sleep(DEMO_DELAY)
    policy_mean, policy_log_std, hidden_state = actor(states_grad)
    time.sleep(DEMO_DELAY)
    # Critic forward pass
    print(f"🔄 Critics forward pass (training mode)...")
    q1_value, _ = critic1(states_grad, actions_grad)
    q2_value, _ = critic2(states_grad, actions_grad)
    time.sleep(DEMO_DELAY)
    print(f"✅ Forward passes completed!")
    print(f"   - Policy mean shape: {policy_mean.shape}")
    print(f"   - Q1 value shape: {q1_value.shape}")
    print(f"   - Q2 value shape: {q2_value.shape}")
    
    # Compute losses
    print(f"\n📉 Computing losses...")
    time.sleep(DEMO_DELAY)
    policy_loss = policy_mean.mean()
    q1_loss = q1_value.mean()
    q2_loss = q2_value.mean()
    total_loss = policy_loss + q1_loss + q2_loss
    
    print(f"   - Policy loss: {policy_loss.item():.6f}")
    print(f"   - Q1 loss: {q1_loss.item():.6f}")
    print(f"   - Q2 loss: {q2_loss.item():.6f}")
    print(f"   - Total loss: {total_loss.item():.6f}")
    
    # Backward pass
    print(f"\n🔄 Executing backward pass...")
    time.sleep(DEMO_DELAY)
    total_loss.backward()
    time.sleep(DEMO_DELAY)
    print(f"✅ Backward pass completed!")
    
    # Check gradients
    print(f"\n📊 Gradient statistics:")
    grad_counts = {'actor': 0, 'critic1': 0, 'critic2': 0}
    
    for name, param in actor.named_parameters():
        if param.grad is not None:
            grad_counts['actor'] += 1
            print(f"   - Actor {name}: {param.grad.norm().item():.6f}")
    
    for name, param in critic1.named_parameters():
        if param.grad is not None:
            grad_counts['critic1'] += 1
            print(f"   - Critic1 {name}: {param.grad.norm().item():.6f}")
    
    for name, param in critic2.named_parameters():
        if param.grad is not None:
            grad_counts['critic2'] += 1
            print(f"   - Critic2 {name}: {param.grad.norm().item():.6f}")
    
    print(f"\n📊 Parameters with gradients:")
    for network, count in grad_counts.items():
        print(f"   - {network}: {count} parameters")
    
    return total_loss.item()


def test_target_network_update(sac_networks):
    """Test target network update"""
    print("\n" + "=" * 60)
    print("5. Target Network Update Test")
    print("=" * 60)
    
    critic1, critic2 = sac_networks.get_critics()
    target_critic1, target_critic2 = sac_networks.get_target_critics()
    
    # Get initial target network parameters
    print(f"📊 Checking target network parameters before update...")
    time.sleep(DEMO_DELAY)
    initial_target1_params = [p.clone() for p in target_critic1.parameters()]
    initial_target2_params = [p.clone() for p in target_critic2.parameters()]
    
    # Update target networks
    print(f"🔄 Updating target networks with tau=0.1...")
    time.sleep(DEMO_DELAY)
    sac_networks.update_target_networks(tau=0.1)
    time.sleep(DEMO_DELAY)
    
    # Check if parameters changed
    print(f"📊 Checking target network parameters after update...")
    time.sleep(DEMO_DELAY)
    updated_target1_params = [p for p in target_critic1.parameters()]
    updated_target2_params = [p for p in target_critic2.parameters()]
    
    # Compare parameters
    target1_changed = any(not torch.equal(init, updated) for init, updated in zip(initial_target1_params, updated_target1_params))
    target2_changed = any(not torch.equal(init, updated) for init, updated in zip(initial_target2_params, updated_target2_params))
    
    print(f"   - Target Critic1 parameters changed: {target1_changed}")
    print(f"   - Target Critic2 parameters changed: {target2_changed}")
    print(f"✅ Target network update test completed!")
    
    return target1_changed and target2_changed


def test_different_batch_sizes(sac_networks, device):
    """Test different batch sizes"""
    print("\n" + "=" * 60)
    print("6. Different Batch Sizes Test")
    print("=" * 60)
    
    actor = sac_networks.get_actor()
    critic1, critic2 = sac_networks.get_critics()
    
    batch_sizes = [1, 2, 4, 8]
    seq_len = 5
    state_dim = 23
    action_dim = 7
    
    print(f"📊 Testing different batch sizes...")
    time.sleep(DEMO_DELAY)
    
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, seq_len, state_dim, device=device)
        
        with torch.no_grad():
            # Actor
            policy_mean, policy_log_std, _ = actor(states)
            action, log_prob, _ = actor.get_action(states, deterministic=False)
            
            # Critics
            q1_value, _ = critic1(states, action)
            q2_value, _ = critic2(states, action)
        
        print(f"   - Batch size {batch_size:2d}: States{states.shape} → Policy{policy_mean.shape}, Action{action.shape}, Q1{q1_value.shape}, Q2{q2_value.shape}")
    time.sleep(0.06)


def main():
    """Main test function"""
    print("🚀 Panda SAC Networks Complete Demo")
    print("📝 Testing Actor, Twin Critics, and Target Networks")
    
    try:
        # 1. Create SAC networks
        sac_networks, device = test_sac_networks_creation()
        
        # 2. Test actor forward pass
        states, actions, log_prob = test_actor_forward_pass(sac_networks, device)
        
        # 3. Test critics forward pass
        q1_value, q2_value, target_q1_value, target_q2_value = test_critics_forward_pass(
            sac_networks, states, actions, device
        )
        
        # 4. Test training mode
        total_loss = test_training_mode(sac_networks, states, actions, device)
        
        # 5. Test target network update
        target_update_success = test_target_network_update(sac_networks)
        
        # 6. Test different batch sizes
        test_different_batch_sizes(sac_networks, device)
        
        # Summary
        print("\n" + "=" * 80)
        print("\n🚀 SAC networks are ready for training!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ SAC Networks demo completed successfully!")
    else:
        print("\n❌ SAC Networks demo failed!")
        exit(1)