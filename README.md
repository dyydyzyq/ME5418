# 🦾 ME5418 – Franka Panda Obstacle Simulation & RL Training

This repository provides a **MuJoCo-based simulation** of a Franka Emika Panda manipulator with dynamic obstacles, and optionally supports **reinforcement learning (RL) training** using **Stable-Baselines3 (SB3)**.

---

## 🔍 1. Environment Setup

```bash
# 1. Create & activate environment
conda create -n me5418 python=3.10
conda activate me5418

# 2. Install basic packages
conda install numpy scipy matplotlib tqdm
```

---

## 🤖 2. Simulation Environment Setup

```bash
# Install MuJoCo and Gymnasium
pip install mujoco==3.3.6 gymnasium

# Check installation
python simulation.py
```

> If a MuJoCo viewer opens and the robot moves, your setup works ✅

---

## 🧠 3. Reinforcement Learning (Optional)

```bash
# Install RL packages
pip install stable-baselines3[extra] gymnasium-robotics

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

### 4.2 Video Recording

```bash
pip install imageio imageio-ffmpeg opencv-python
```

Videos are saved in the `recordings/` folder.

---

## 🗂️ 5. Project Structure

```
ME5418/
├── env/                     # PandaObstacleEnv
│   └── env.py
├── franka_emika_panda/      # MuJoCo XML models
│   └── scene_withobstacles.xml
├── train/                   # training scripts
│   └── train.py
├── models/                  # saved models
├── recordings/              # videos
├── tb_logs/                 # tensorboard logs
├── environment-sim-only.yaml
├── environment-full.yaml
└── README.md
```

## 🔧 6. Full Environment (Optional)

### Simulation-only YAML

```yaml
name: me5418
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - matplotlib
  - tqdm
  - pip:
      - mujoco==3.3.6
      - gymnasium[all]
      - imageio
      - imageio-ffmpeg
      - opencv-python
```

### Full RL YAML

```yaml
name: me5418
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - matplotlib
  - tqdm
  - tensorboard=2.14.*
  - protobuf=3.20.*
  - pip:
      - mujoco==3.3.6
      - gymnasium[all]
      - gymnasium-robotics
      - stable-baselines3[extra]
      - imageio
      - imageio-ffmpeg
      - opencv-python
```
