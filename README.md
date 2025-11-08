# 🦾 ME5418 – Franka Panda Obstacle Simulation & RL Training

This repository provides a **MuJoCo-based simulation** of a Franka Emika Panda manipulator with dynamic obstacles. It includes a custom **Gym-compatible environment**, a self-implemented **Soft Actor–Critic (SAC) / PPO network**, and complete training scripts forming an end-to-end reinforcement learning pipeline.

---

## 🔍 1. Reinforcement Learning Pipeline Setup

This section demonstrates the complete RL workflow — from environment setup to network visualization and training.

```bash
# 1️⃣ Create and activate the environment
mamba env create -f environment-full.yaml

# 2️⃣ Test random actions in the environment
python env/random_action.py           # Run the environment with random actions for debugging

# 3️⃣ Visualize network structure
chmod +x net/.sh
./net/.sh                 # Display the input/output dimensions of PPO & SAC networks

# 4️⃣ Train the agent
# (choose either PPO or SAC)
python learning_agent/train_ppo.py   # Train the agent using PPO
python learning_agent/train_sac.py    # Train the agent using SAC
```

---


## 🔧 2. Full Environment (Optional)

### Full RL YAML

```yaml
name: me5418-full
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  
  - pip:
      - numpy
      - scipy
      - matplotlib
      - tqdm
      - mujoco==3.3.6
      - gymnasium
      - gymnasium-robotics
      - stable-baselines3[extra]
      - imageio
      - imageio-ffmpeg
      - opencv-python
      - torch==2.8.0