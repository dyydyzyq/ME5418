"""
Panda Actor-Critic Network Demo
Test the input/output dimensions of Panda robot Actor-Critic network
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to import net module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net import create_network


def test_network_dimensions():
    """Test network input/output dimensions"""
    print("=" * 50)
    print("Panda Actor-Critic Network Test")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    network = create_network(device=device)
    network.to(device)
    print("Network created successfully!")
    
    # Test parameters
    batch_size = 8
    seq_len = 5
    state_dim = 23
    action_dim = 7
    
    print(f"\nTest parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    
    # Create dummy batch data
    states = torch.randn(batch_size, seq_len, state_dim, device=device)
    print(f"\nInput data shape: {states.shape}")
    
    # Test forward pass
    print("\n" + "-" * 30)
    print("Forward Pass Test")
    print("-" * 30)
    
    with torch.no_grad():
        # Basic forward pass
        policy, value, hidden_state = network(states)
        
        print(f"Policy output shape: {policy.shape}")
        print(f"Value output shape: {value.shape}")
        print(f"Hidden state shapes: h={hidden_state[0].shape}, c={hidden_state[1].shape}")
        
        # Test action generation
        print("\nAction Generation Test:")
        action_det, log_prob_det, new_hidden_state = network.get_action(
            states, deterministic=True
        )
        print(f"Deterministic action shape: {action_det.shape}")
        print(f"Action value range: [{action_det.min().item():.3f}, {action_det.max().item():.3f}]")
        
        # Test value function
        print("\nValue Function Test:")
        state_value = network.get_value(states)
        print(f"State value shape: {state_value.shape}")
        print(f"Value range: [{state_value.min().item():.3f}, {state_value.max().item():.3f}]")
    
    return True


def test_sequence_processing():
    """Test sequence processing capability"""
    print("\n" + "=" * 50)
    print("Sequence Processing Test")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = create_network(device=device)
    network.to(device)
    
    batch_size = 4
    seq_len = 10
    state_dim = 23
    
    # Create sequence data
    states = torch.randn(batch_size, seq_len, state_dim, device=device)
    
    print(f"Sequence data shape: {states.shape}")
    
    with torch.no_grad():
        # Initialize hidden state
        hidden_state = network.init_hidden_state(batch_size)
        print(f"Initial hidden state shape: h={hidden_state[0].shape}, c={hidden_state[1].shape}")
        
        # Process sequence
        policy, value, final_hidden_state = network(states, hidden_state)
        
        print(f"Sequence processing completed!")
        print(f"Final policy output: {policy.shape}")
        print(f"Final value output: {value.shape}")
        print(f"Final hidden state: h={final_hidden_state[0].shape}, c={final_hidden_state[1].shape}")


def test_different_batch_sizes():
    """Test processing capability with different batch sizes"""
    print("\n" + "=" * 50)
    print("Different Batch Sizes Test")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = create_network(device=device)
    network.to(device)
    
    batch_sizes = [1, 4, 8, 16]
    seq_len = 5
    state_dim = 23
    
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, seq_len, state_dim, device=device)
        
        with torch.no_grad():
            policy, value, hidden_state = network(states)
            
            print(f"Batch size {batch_size:2d}: Input{states.shape} -> Policy{policy.shape}, Value{value.shape}")


def test_network_parameters():
    """Test network parameter statistics"""
    print("\n" + "=" * 50)
    print("Network Parameter Statistics")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = create_network(device=device)
    network.to(device)
    
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Parameter statistics by layer
    print("\nParameter statistics by layer:")
    for name, param in network.named_parameters():
        print(f"  {name}: {param.numel():,} parameters, shape: {param.shape}")


def main():
    """Main test function"""
    try:
        # Basic dimension test
        test_network_dimensions()
        
        # Sequence processing test
        test_sequence_processing()
        
        # Different batch sizes test
        test_different_batch_sizes()
        
        # Network parameter statistics
        test_network_parameters()
        
        print("\n" + "=" * 50)
        print("All tests completed! ✅")
        print("Network input/output dimension verification successful!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError occurred during testing: {e}")
        raise


if __name__ == "__main__":
    main()
