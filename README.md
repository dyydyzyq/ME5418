# 🦾 ME5418 – Franka Panda Obstacle Simulation & SAC Network

This repository provides a **MuJoCo-based simulation** of a Franka Emika Panda manipulator with dynamic obstacles, and optionally supports **reinforcement learning (RL) training** using **Stable-Baselines3 (SB3)**.

---

## 🔍 1. Installation

```bash
# Install dependencies and project in editable mode
pip install -e .
```
---
> **This installs all required dependencies (PyTorch, MuJoCo, Gymnasium, NumPy).**
---
## 🗂️ 2. Project Structure

```
ME5418/
├── env/                     # PandaObstacleEnv
│   └── env.py              #environment
|   └── random_action.py    #random_action
├── franka_emika_panda/      # MuJoCo XML models
│   └── scene_withobstacles.xml
├── net/                     # SAC neural network (Actor–Critic)
│   └── net.py               # Network architecture (FeatureExtractor, Actor, Critic)
|   └── demo.py              # Forward-pass test of SAC network (check input/output shapes)
├── train/                   # training scripts
│   └── train.py
├── environment-sim-only.yaml
├── environment-full.yaml
└── README.md
└── setup.py
```
---
## 🚀 3. Quick Start
### 3.1 Simulation Test
Run a simple MuJoCo test to verify setup:
```bash
```bash
python env/random_action.py
```
---
> **A MuJoCo viewer should open and the robot should move randomly.**
> **If so, your environment works correctly ✅**
---
### 3.2 Network Forward Pass Test
You can test the SAC network forward pass (untrained) via:
```bash
```bash
python net/demo.py
```
---
> **This script checks the network architecture and verifies**
> **input/output tensor shapes for FeatureExtractor → Actor → Critic.**
---
### 3.3 Start Reinforcement Learning Training (Optional)
```bash
```bash
# Start training
python train/train.py
```
---

## 📈 4. Visualization

### 4.1 TensorBoard

```bash
pip install tensorboard
tensorboard --logdir tb_logs
```

Then open your browser and visit:

```
http://localhost:6006
```
---
## 🧩 5. About the Network
The **net/ folder** contains the neural network implementation used for Soft Actor-Critic (SAC) training.

🔧 Components
| File | Description |
|------|--------------|
| `net.py` | Defines the SAC networks: **FeatureExtractor**, **Actor**, and **Critic**. |
| `demo.py` | A simple script for **network structure testing** – runs a forward pass of the untrained network to ensure input/output dimensions are correct. |                                                                                                                                                                                                                               |


🧠 Logic Flow

**1**.Environment provides observation (state)
      → sent into FeatureExtractor
      → outputs encoded feature vector.

**2**.Actor head computes policy distribution parameters (μ, σ).

**3**.Critic heads evaluate Q-values.

---
## 🔧 6. Testing tips
After installation, test with:
```bash
python env/random_action.py
```
or run network test:
```bash
```bash
python net/demo.py
```
---
