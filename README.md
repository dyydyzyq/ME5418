# ME5418 – Franka Panda Obstacle Simulation & RL Training

This repository provides a **MuJoCo-based simulation** of a Franka Emika Panda manipulator with dynamic obstacles, and optionally supports **reinforcement learning (RL) training** using **Stable-Baselines3 (SB3)**.

---

## 1. Environment Setup

### 1) Prerequisites
- Install Conda (Anaconda or Miniconda). Recommended: use Miniconda for a
  minimal install or Anaconda if you want a large preinstalled scientific
  stack.

  Quick links and recommended versions
  -  Miniconda (recommended minimal installer): https://docs.conda.io/en/latest/miniconda.html
  - Anaconda (full distribution): https://www.anaconda.com/products/distribution#download-section


### 2) Create the Conda environment from YAML

```powershell
# create a new Conda environment for this project
conda env create -f environment-sim.yaml --name me5418
```

### 3) Activate and verify the environment

```powershell
# Activate Enviornment
conda activate me5418

# Quick checks
python --version
python -c "import numpy as np; print('numpy', np.__version__)"
python -c "import gymnasium; print('gymnasium OK')"
python -c "import mujoco; print('mujoco OK')"
```

## 2. Run the Project

```powershell
python env\random_action.py
```

## 3. Project Structure

```
ME5418/
├── env/                     # PandaObstacleEnv
│   └── env.py
|   └── random_action.py     # Visualize the random action
├── franka_emika_panda/      # MuJoCo XML models
│   └── panda.xml
|   └── scene_withobstacles.xml
├── environment-sim.yaml
└── README.md
```
