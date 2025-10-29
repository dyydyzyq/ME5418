# Panda SAC 网络架构文档

## 概述

`net.py` 文件实现了完整的SAC（Soft Actor-Critic）网络架构，包括Actor网络、Twin Q网络（Q1和Q2）以及对应的Target网络。

## SAC网络架构

SAC算法需要以下网络组件：

1. **Actor网络**: 输出策略分布（均值和标准差）
2. **Twin Q网络**: Q1(s,a) 和 Q2(s,a) 来减少过估计偏差
3. **Target网络**: 对应的Target Q网络用于计算TD目标

### 架构设计

```
状态输入 → 特征提取器 (FC1 + FC2 + LSTM) → 分支输出
                                    ├── Actor: 策略均值 + 策略标准差
                                    ├── Q1: 特征 + 动作 → Q值
                                    └── Q2: 特征 + 动作 → Q值
```

## 核心类定义

### 1. FeatureExtractor（特征提取器）

**功能**: 共享的特征提取器，所有网络都使用相同的特征提取部分

**架构**:
- FC1: 23 → 64 (特征提取)
- FC2: 64 → 64 (预处理)
- 残差连接: FC1 + FC2
- LSTM: 64 → 64 (序列处理)

**输入**: `[batch_size, seq_len, 23]` 状态序列
**输出**: `[batch_size, 64]` 特征向量

### 2. Actor（Actor网络）

**功能**: 输出高斯策略分布参数

**架构**:
- 共享特征提取器
- 策略均值头: 64 → 7
- 策略标准差头: 64 → 7

**输入**: `[batch_size, seq_len, 23]` 状态序列
**输出**: 
- 策略均值: `[batch_size, 7]`
- 策略标准差: `[batch_size, 7]`
- 隐藏状态: LSTM隐藏状态

### 3. Critic（Critic网络）

**功能**: 评估状态-动作对的Q值

**架构**:
- 共享特征提取器
- Q值头: (64 + 7) → 1

**输入**: 
- 状态: `[batch_size, seq_len, 23]`
- 动作: `[batch_size, 7]`
**输出**: Q值 `[batch_size, 1]`

### 4. SACNetworks（SAC网络集合）

**功能**: 管理所有SAC网络组件

**包含**:
- Actor网络
- Critic1网络（Q1）
- Critic2网络（Q2）
- Target Critic1网络
- Target Critic2网络

## 重要方法

### Actor方法

#### `forward(states, hidden_state=None)`
- **功能**: Actor前向传播
- **输入**: 状态序列和可选隐藏状态
- **输出**: 策略均值、策略标准差、新隐藏状态

#### `get_action(states, hidden_state=None, deterministic=False)`
- **功能**: 生成动作
- **输入**: 状态序列、隐藏状态、是否确定性
- **输出**: 动作、对数概率、新隐藏状态

### Critic方法

#### `forward(states, actions, hidden_state=None)`
- **功能**: Critic前向传播
- **输入**: 状态序列、动作、隐藏状态
- **输出**: Q值、新隐藏状态

### SACNetworks方法

#### `update_target_networks(tau=0.005)`
- **功能**: 软更新Target网络
- **参数**: tau - 软更新系数

#### `get_parameter_count()`
- **功能**: 获取各网络参数数量
- **返回**: 参数字典

## 网络参数统计

### 总参数数量: 117,982

- **Actor**: 39,886 参数
- **Critic1**: 39,048 参数  
- **Critic2**: 39,048 参数
- **总计**: 117,982 参数

### 参数分布

| 网络 | 特征提取器 | 输出头 | 总计 |
|------|------------|--------|------|
| Actor | 32,768 | 7,118 | 39,886 |
| Critic1 | 32,768 | 6,280 | 39,048 |
| Critic2 | 32,768 | 6,280 | 39,048 |

## 使用示例

### 1. 创建SAC网络

```python
from net import create_sac_networks

# 创建完整的SAC网络
sac_networks = create_sac_networks(device="cuda")

# 获取各个网络
actor = sac_networks.get_actor()
critic1, critic2 = sac_networks.get_critics()
target_critic1, target_critic2 = sac_networks.get_target_critics()
```

### 2. Actor使用

```python
# 准备数据
states = torch.randn(4, 5, 23, device="cuda")

# 生成动作
action, log_prob, hidden_state = actor.get_action(states, deterministic=False)

print(f"动作形状: {action.shape}")  # [4, 7]
print(f"对数概率形状: {log_prob.shape}")  # [4, 1]
```

### 3. Critic使用

```python
# 准备数据
states = torch.randn(4, 5, 23, device="cuda")
actions = torch.randn(4, 7, device="cuda")

# 计算Q值
q1_value, _ = critic1(states, actions)
q2_value, _ = critic2(states, actions)

print(f"Q1值形状: {q1_value.shape}")  # [4, 1]
print(f"Q2值形状: {q2_value.shape}")  # [4, 1]
```

### 4. Target网络更新

```python
# 软更新Target网络
sac_networks.update_target_networks(tau=0.005)

# 使用Target网络计算目标值
with torch.no_grad():
    target_q1, _ = target_critic1(states, actions)
    target_q2, _ = target_critic2(states, actions)
```

## SAC算法集成

### 训练循环示例

```python
# 创建SAC网络
sac_networks = create_sac_networks(device="cuda")
actor = sac_networks.get_actor()
critic1, critic2 = sac_networks.get_critics()
target_critic1, target_critic2 = sac_networks.get_target_critics()

# 优化器
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=3e-4)
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=3e-4)

# 训练循环
for episode in range(num_episodes):
    # 收集经验
    states, actions, rewards, next_states = collect_experience()
    
    # Actor损失
    new_actions, log_probs, _ = actor.get_action(states, deterministic=False)
    q1_new = critic1(states, new_actions)[0]
    q2_new = critic2(states, new_actions)[0]
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (log_probs - q_new).mean()
    
    # Critic损失
    with torch.no_grad():
        next_actions, next_log_probs, _ = actor.get_action(next_states, deterministic=False)
        target_q1 = target_critic1(next_states, next_actions)[0]
        target_q2 = target_critic2(next_states, next_actions)[0]
        target_q = torch.min(target_q1, target_q2) - next_log_probs
        target_q = rewards + gamma * target_q
    
    q1_current = critic1(states, actions)[0]
    q2_current = critic2(states, actions)[0]
    critic1_loss = F.mse_loss(q1_current, target_q)
    critic2_loss = F.mse_loss(q2_current, target_q)
    
    # 更新网络
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()
    
    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()
    
    # 更新Target网络
    sac_networks.update_target_networks(tau=0.005)
```

## 关键特性

### 1. **共享特征提取**
- 所有网络使用相同的特征提取器
- 减少参数数量，提高训练效率
- 确保特征一致性

### 2. **Twin Q网络**
- 两个独立的Q网络减少过估计偏差
- 使用min(Q1, Q2)作为目标值
- 提高训练稳定性

### 3. **Target网络**
- 软更新机制（tau=0.005）
- 提高训练稳定性
- 减少目标值波动

### 4. **LSTM序列处理**
- 处理时间依赖关系
- 支持序列决策
- 隐藏状态管理

### 5. **高斯策略**
- 连续动作空间
- Tanh压缩到[-1, 1]范围
- 正确的对数概率计算

## 测试验证

运行 `python net/demo.py` 可以验证：

- ✅ 网络创建成功
- ✅ Actor输出策略分布
- ✅ Twin Critics独立工作
- ✅ Target网络可更新
- ✅ 梯度计算正确
- ✅ 批处理支持

## 总结

符合SAC算法的要求：

1. **Actor网络**: 输出高斯策略分布
2. **Twin Q网络**: Q1和Q2减少过估计偏差
3. **Target网络**: 软更新机制
4. **共享特征**: 提高训练效率
5. **序列处理**: LSTM支持时间依赖

该架构为Panda机械臂的SAC训练提供了完整、高效的网络基础。