import torch

# --- 游戏参数 ---
BOARD_SIZE = 8

# --- MCTS 参数 ---
MCTS_SIMULATIONS = 5  # 每次移动的MCTS模拟次数
CPUCT = 1.0  # UCT公式中的探索常数
DIRICHLET_ALPHA = 0.3  # 狄利克雷噪声的alpha参数
DIRICHLET_EPSILON = 0.25  # 噪声占的比重

# --- 训练参数 ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_PER_ITERATION = 10  # 每次迭代中，用新数据训练的轮数
REPLAY_BUFFER_MAXLEN = 200000  # 经验回放缓冲区的最大长度

# --- AlphaZero 循环参数 ---
NUM_ITERATIONS = 1000  # 总迭代次数
GAMES_PER_ITERATION = 8  # 每次迭代中，自我对弈的游戏局数 (会被多进程瓜分)
CHECKPOINT_INTERVAL = 10  # 每隔多少次迭代保存一次模型
EVALUATION_GAMES = 20  # 新旧模型评估对战的局数
EVALUATION_THRESHOLD = 0.55  # 新模型取代旧模型的胜率阈值

# --- 学习率调度器参数 ---
LR_DECAY_STEP = 50  # 每隔多少次迭代，学习率衰减一次
LR_DECAY_GAMMA = 0.5  # 每次衰减的乘数因子

# --- 神经网络参数 ---
RESIDUAL_BLOCKS = 5  # 残差块数量
CONV_FILTERS = 128  # 卷积层滤波器数量

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
