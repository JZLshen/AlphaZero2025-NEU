#!/bin/bash

# =================================================================
# AlphaZero 分布式训练启动脚本
# =================================================================

# 设置要使用的GPU数量
# 请确保这个数字小于或等于您服务器上可用的GPU总数
N_GPUS=3

echo "Starting AlphaZero training with DDP on $N_GPUS GPUs."

# 使用 torchrun 来启动分布式任务
# --standalone: 表示在单台机器上运行
# --nproc_per_node: 指定在这台机器上启动多少个进程（即使用多少块GPU）
torchrun --standalone --nproc_per_node=$N_GPUS main.py