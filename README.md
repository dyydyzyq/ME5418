# 🦾 ME5418 – Franka Panda Obstacle Simulation & RL Training

This repository provides a **MuJoCo-based simulation** of a Franka Emika Panda manipulator with dynamic obstacles, and optionally supports **reinforcement learning (RL) training** using **Stable-Baselines3 (SB3)**.

---

## 🔍 1. Environment Setup

```bash
# 1. Create & activate environment 

```bash 
conda env create -f environment-sim-only.yaml  #show the random_action
conda env create -f environment-full.yaml  #include the train

```bash
python random_action.py
```


---

> If a MuJoCo viewer opens and the robot moves, your setup works ✅

---

## 🧠 3. Reinforcement Learning (Optional)

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



## 🗂️ 5. Project Structure

```
ME5418/
├── env/                     # PandaObstacleEnv
│   └── env.py              #environment
|   └── random_action.py    #random_action
├── franka_emika_panda/      # MuJoCo XML models
│   └── scene_withobstacles.xml
├── train/                   # training scripts
│   └── train.py
├── environment-sim-only.yaml
├── environment-full.yaml
└── README.md
```

## 🔧 6. Full Environment (Optional)

### Simulation-only YAML

```yaml
name: me5418-sim-only
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - matplotlib
  - tqdms
  - pip:
      - mujoco==3.3.6
      - gymnasium
      - imageio
      - imageio-ffmpeg
      - opencv-python
```

### Full RL YAML

```yaml
name: me5418-full
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - matplotlib
  - tqdm
  - tensorboard=2.14
  - pip:
      - mujoco==3.3.6
      - gymnasium
      - gymnasium-robotics
      - stable-baselines3[extra]
      - imageio
      - imageio-ffmpeg
      - opencv-python
```
