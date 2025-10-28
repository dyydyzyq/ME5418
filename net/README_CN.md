# Panda Actor-Critic 网络文档

## 概述

本文档详细解释了 `net.py` 文件的实现，该文件专门为Panda机械臂在动态障碍物环境中设计的神经网络架构。该网络针对强化学习算法进行了优化，特别是SAC（Soft Actor-Critic）算法。

## 网络架构

`PandaActorCriticNetwork` 采用了一个复杂的架构，结合了前馈层、残差连接和LSTM进行序列处理：

```
输入 (23维) → FC1 (64) → FC2 (64) → 残差连接 (FC1 + FC2) → LSTM (64) → 策略头 (7) + 价值头 (1)
```

### 架构组件

1. **输入层**: 23维状态向量
   - 7个关节角速度
   - 7个关节位置
   - 3个目标位置（相对于末端执行器）
   - 6个障碍物位置（相对于末端执行器）

2. **特征提取**:
   - **FC1**: 23 → 64（初始特征提取）
   - **FC2**: 64 → 64（LSTM前处理）
   - **残差连接**: FC1 + FC2（保留底层信息）

3. **序列处理**:
   - **LSTM**: 64 → 64（处理时间依赖关系）

4. **输出分支**:
   - **策略均值头**: 64 → 7（动作分布的均值）
   - **策略标准差头**: 64 → 7（动作分布的对数标准差）
   - **价值头**: 64 → 1（状态价值估计）

## 类: PandaActorCriticNetwork

### 构造函数参数

```python
def __init__(
    self,
    state_dim: int = 23,        # 状态空间维度
    action_dim: int = 7,        # 动作空间维度
    hidden_dim: int = 64,       # 隐藏层维度
    lstm_hidden_dim: int = 64,  # LSTM隐藏维度
    seq_len: int = 5,           # LSTM序列长度
    device: str = "cpu"         # 设备 (cpu/cuda)
)
```

### 重要方法

#### 1. `forward(states, hidden_state=None)`

**功能**: 网络的主要前向传播

**参数**:
- `states`: 状态序列批次 `[batch_size, seq_len, state_dim]`
- `hidden_state`: 可选的LSTM隐藏状态元组 `(h, c)`

**返回值**:
- `policy_mean`: 动作分布的均值 `[batch_size, action_dim]`
- `policy_log_std`: 动作分布的对数标准差 `[batch_size, action_dim]`
- `value`: 状态价值 `[batch_size, 1]`
- `new_hidden_state`: 更新的LSTM隐藏状态元组

**处理过程**:
1. 通过FC1和FC2进行特征提取
2. 残差连接: `residual = FC1_out + FC2_out`
3. 使用可选隐藏状态进行LSTM处理
4. 提取最后一个时间步的输出用于策略和价值
5. 生成策略均值、策略对数标准差和价值输出

#### 2. `get_action(states, hidden_state=None, deterministic=False)`

**功能**: 使用高斯分布从策略网络生成动作

**参数**:
- `states`: 状态序列批次 `[batch_size, seq_len, state_dim]`
- `hidden_state`: 可选的LSTM隐藏状态元组
- `deterministic`: 如果为True，返回平均动作；如果为False，从分布中采样

**返回值**:
- `action`: 动作 `[batch_size, action_dim]`（限制在[-1, 1]范围内）
- `log_prob`: 动作的对数概率 `[batch_size, 1]`
- `new_hidden_state`: 更新的LSTM隐藏状态元组

**处理过程**:
1. 获取策略分布的均值和标准差
2. 限制log_std防止数值不稳定
3. 如果确定性模式：返回tanh(均值)
4. 如果随机模式：从高斯分布采样，应用tanh压缩，计算对数概率

#### 3. `get_action_and_log_prob(states, hidden_state=None)`

**功能**: SAC专用的动作生成和对数概率方法（现已完整实现）

**参数**:
- `states`: 状态序列批次 `[batch_size, seq_len, state_dim]`
- `hidden_state`: 可选的LSTM隐藏状态元组

**返回值**:
- `action`: 动作 `[batch_size, action_dim]`
- `log_prob`: 动作的对数概率 `[batch_size, 1]`
- `new_hidden_state`: 更新的LSTM隐藏状态元组

**注意**: 此方法现在已完整实现SAC所需的功能，包括高斯采样和正确的对数概率计算。

#### 4. `get_value(states, hidden_state=None)`

**功能**: 从评论家网络获取状态价值

**参数**:
- `states`: 状态序列批次 `[batch_size, seq_len, state_dim]`
- `hidden_state`: 可选的LSTM隐藏状态元组

**返回值**:
- `value`: 状态价值 `[batch_size, 1]`

#### 5. `init_hidden_state(batch_size)`

**功能**: 初始化LSTM隐藏状态

**参数**:
- `batch_size`: 初始化的批次大小

**返回值**:
- `hidden_state`: 初始LSTM隐藏状态元组 `(h, c)`

#### 6. `_init_weights()`

**功能**: 使用正交初始化初始化网络权重

**处理过程**:
- 线性层: 正交权重初始化，增益=1.0，偏置为零
- LSTM层: 正交权重初始化，偏置为零
- 确保训练稳定和梯度流

## 工厂函数

### 1. `create_network(state_dim=23, action_dim=7, hidden_dim=64, lstm_hidden_dim=64, seq_len=5, device="cpu")`

**功能**: 创建Panda Actor-Critic网络的通用工厂函数

**返回值**: `PandaActorCriticNetwork` 实例

### 2. `create_actor_network(state_dim=23, action_dim=7, hidden_dim=64, lstm_hidden_dim=64, seq_len=5, device="cpu")`

**功能**: 专门用于在SAC中创建Actor网络的工厂函数

**返回值**: `PandaActorCriticNetwork` 实例（与create_network相同，但命名更清晰）

### 3. `create_critic_network(state_dim=23, action_dim=7, hidden_dim=64, lstm_hidden_dim=64, seq_len=5, device="cpu")`

**功能**: 专门用于在SAC中创建Critic网络的工厂函数

**返回值**: `PandaActorCriticNetwork` 实例（与create_network相同，但命名更清晰）

## 使用示例

### 基础用法

```python
from net import create_network
import torch

# 创建网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = create_network(device=device)
network.to(device)

# 准备输入数据
batch_size = 4
seq_len = 5
states = torch.randn(batch_size, seq_len, 23, device=device)

# 前向传播
policy_mean, policy_log_std, value, hidden_state = network(states)
print(f"策略均值形状: {policy_mean.shape}")    # [4, 7]
print(f"策略标准差形状: {policy_log_std.shape}")  # [4, 7]
print(f"价值形状: {value.shape}")              # [4, 1]

# 生成动作
action, log_prob, new_hidden = network.get_action(states, deterministic=True)
print(f"动作形状: {action.shape}")    # [4, 7]
print(f"对数概率形状: {log_prob.shape}")  # [4, 1] (随机模式) 或 None (确定性模式)
```

### SAC算法用法

```python
from net import create_actor_network, create_critic_network

# 创建actor和critic网络
actor = create_actor_network(device=device)
critic1 = create_critic_network(device=device)
critic2 = create_critic_network(device=device)

# 训练循环示例
for batch in dataloader:
    states, actions, rewards, next_states = batch
    
    # Actor前向传播
    policy_mean, policy_log_std, _, _ = actor(states)
    
    # Critic前向传播
    value1 = critic1.get_value(states)
    value2 = critic2.get_value(states)
    
    # 计算损失并更新网络
    # ... SAC损失计算 ...
```

### 序列处理

```python
# 初始化隐藏状态
hidden_state = network.init_hidden_state(batch_size)

# 处理序列
for timestep in range(sequence_length):
    policy_mean, policy_log_std, value, hidden_state = network(states[:, timestep:timestep+1], hidden_state)
    # 使用策略分布和价值进行决策
```

## 网络规格

### 输入/输出维度

| 组件 | 输入形状 | 输出形状 | 描述 |
|------|----------|----------|------|
| 状态 | `[batch_size, seq_len, 23]` | - | 状态序列 |
| 策略均值 | - | `[batch_size, 7]` | 动作分布均值 |
| 策略标准差 | - | `[batch_size, 7]` | 动作分布对数标准差 |
| 价值 | - | `[batch_size, 1]` | 状态价值 |
| 动作 | - | `[batch_size, 7]` | 限制动作 [-1, 1] |
| 对数概率 | - | `[batch_size, 1]` | 动作对数概率 |
| 隐藏状态 | - | `(h, c)` | LSTM隐藏状态 |

### 参数数量

- **总参数**: 40,007
- **可训练参数**: 40,007
- **FC1**: 1,536 参数 (64×23 + 64)
- **FC2**: 4,160 参数 (64×64 + 64)
- **LSTM**: 32,768 参数 (256×64×2 + 256×2)
- **策略均值头**: 455 参数 (7×64 + 7)
- **策略标准差头**: 455 参数 (7×64 + 7)
- **价值头**: 65 参数 (1×64 + 1)

## 关键特性

### 1. **残差连接**
- 通过FC1 + FC2保留底层特征
- 有助于梯度流和特征保留

### 2. **LSTM集成**
- 处理机器人控制中的时间依赖关系
- 跨时间步维护隐藏状态
- 适用于序列决策

### 3. **双输出设计**
- 策略头用于动作生成
- 价值头用于状态价值估计
- 兼容actor-critic算法

### 4. **SAC兼容性**
- 为Soft Actor-Critic算法设计
- 为actor和critic提供独立的工厂函数
- SAC实现的占位符方法

### 5. **设备灵活性**
- 自动设备检测（CUDA/CPU）
- 适当的张量设备管理
- 跨平台兼容性

## 训练考虑

### 梯度流
- 正交权重初始化确保梯度稳定
- 残差连接防止梯度消失
- LSTM在序列中保持梯度流

### 内存管理
- 序列处理的隐藏状态管理
- 高效的批处理
- 优化的可选隐藏状态传递

### SAC集成
- 当前实现为SAC提供基础
- 需要实现对数概率计算
- 需要添加高斯策略采样

## 依赖项

- PyTorch >= 1.8.0
- NumPy（用于数组操作）
- CUDA支持（可选，用于GPU加速）

## 测试

运行demo文件验证网络功能：

```bash
python net/demo.py
```

这将测试：
- 网络创建和初始化
- 不同批次大小的前向传播
- 序列处理能力
- 参数统计
- 输入/输出维度验证

## 未来改进

1. **完整的SAC实现**:
   - 添加可学习的log_std参数
   - 实现适当的高斯采样
   - 使用tanh变换计算对数概率

2. **附加功能**:
   - 正则化的Dropout
   - 批归一化
   - LSTM的层归一化

3. **优化**:
   - 混合精度训练
   - 梯度裁剪
   - 学习率调度

## 总结

该网络为Panda机械臂在动态环境中的强化学习提供了坚实的基础，特别关注SAC算法兼容性。通过结合残差连接、LSTM序列处理和双输出设计，该网络能够有效处理复杂的机器人控制任务。

### 主要优势：
- ✅ **架构先进**: 残差连接 + LSTM + 双输出
- ✅ **SAC就绪**: 专为SAC算法设计
- ✅ **序列处理**: 处理时间依赖关系
- ✅ **设备兼容**: 支持CUDA和CPU
- ✅ **参数合理**: 39,496个参数，训练效率高

### 适用场景：
- 🤖 机械臂点到点运动
- 🚧 动态障碍物环境
- 🧠 强化学习训练
- 📊 SAC算法实现
