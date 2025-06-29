# AlphaZero for Dots and Boxes (8x8)

这是一个基于Python和PyTorch实现的AlphaZero算法，专门用于在8x8棋盘上进行点格棋（Dots and Boxes）对弈。项目完整实现了从自我对弈、数据增强、模型训练、模型评估到最终决策的整个闭环，并集成了多项优化，使其成为一个强大且高效的棋类AI训练框架。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 核心特性

本项目的实现包含了AlphaZero算法的诸多关键特性与优化：

- **神经网络模型**: 采用带有**注意力机制 (Squeeze-and-Excitation)** 的残差网络 (SE-ResNet)，增强了模型对关键棋局特征的感知能力。
- **蒙特卡洛树搜索 (MCTS)**: 在自我对弈的根节点搜索中加入了**狄利克雷噪声**，以鼓励更充分的探索，防止模型陷入局部最优。
- **高性能训练**:
    - **并行自我对弈**: 利用多进程（Multiprocessing）并行生成对弈数据，极大地提升了数据生产效率，充分利用了多核CPU资源。
    - **经验回放缓冲区**: 使用定长的`deque`作为经验池，有效管理内存并稳定训练数据分布。
- **稳定的模型迭代**:
    - **竞技场评估机制**: 新训练出的模型会与当前最佳模型在“竞技场”中进行多盘对战，只有胜率超过设定阈值的新模型才会被采纳，确保了棋力的稳定提升。
    - **学习率调度器**: 训练过程中动态调整学习率，实现更优、更稳定的收敛。
- **可视化监控**: 深度集成了**TensorBoard**，可以实时、直观地监控损失函数、模型胜率、学习率等关键指标的变化曲线。

## 项目结构

```
alphazero_project/
├── main.py           # 项目主入口，负责整个训练、评估循环的调度
├── game.py           # 实现了点格棋的游戏逻辑、规则判断和对称变换
├── model.py          # 定义了SE-ResNet神经网络的结构
├── mcts.py           # 实现了核心的蒙特卡洛树搜索算法
├── config.py         # 包含所有超参数的中央配置文件
├── requirements.txt  # 项目依赖库列表
├── README.md         # 项目说明文档
└── data/             # 存放数据和模型的目录
    └── checkpoints/  # 存放训练过程中生成的模型检查点
```

## 环境设置与安装

请按照以下步骤来设置你的本地环境。

**1. 克隆项目**

```bash
git clone https://github.com/JZLshen/AlphaZero2025-NEU.git
cd AlphaZero2025-NEU
```

**2. 创建并激活Python虚拟环境** (推荐)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. 安装依赖**

本项目依赖PyTorch。请先访问 [PyTorch官网](https://pytorch.org/get-started/locally/)，根据你的操作系统和CUDA版本获取对应的安装命令。

例如，对于使用CUDA 12.1的NVIDIA显卡用户，命令可能如下：
```bash
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
对于仅使用CPU的用户，命令可能如下：
```bash
pip3 install torch torchvision torchaudio
```

安装完PyTorch后，再安装其他依赖：
```bash
pip install -r requirements.txt
```

## 如何运行

项目运行主要分为两个部分：**启动训练**和**启动监控**。

**1. 启动训练进程**

在项目根目录下，打开一个终端，运行以下命令：

```bash
python main.py
```

程序将开始执行AlphaZero的训练循环：
- 初始化或加载最佳模型。
- 进行多进程并行化的自我对弈以收集数据。
- 使用收集到的数据训练神经网络。
- 将新模型与最佳模型进行评估对战。
- 根据评估结果决定是否更新最佳模型。
- 这个过程会不断循环。

**2. 启动TensorBoard监控**

为了可视化训练过程，请在项目根目录下**打开另一个新的终端**，并运行：

```bash
tensorboard --logdir=runs
```

然后在你的浏览器中打开 `http://localhost:6006/`。你将看到一个仪表盘，其中包含了损失函数、胜率、学习率等指标的实时图表。

## 配置与调优

项目的所有关键超参数都集中在 `config.py` 文件中。你可以通过修改此文件来调整算法的行为，例如：

- `BOARD_SIZE`: 棋盘大小。
- `MCTS_SIMULATIONS`: 每次决策的MCTS模拟次数，越高棋力越强但越慢。
- `GAMES_PER_ITERATION`: 每个大循环中自我对弈的总盘数。
- `LEARNING_RATE`: 初始学习率。
- `RESIDUAL_BLOCKS`: 神经网络的深度。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。