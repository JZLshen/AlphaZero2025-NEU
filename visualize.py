import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
from tqdm import tqdm

from config import *
from game import DotsAndBoxesGame
from model import DotsAndBoxesNet
from mcts import MCTS

# --- 可视化帮助函数 ---

def plot_board(game, board, filename=None, pi=None, title=""):
    """
    绘制棋盘状态，可以选择性地叠加策略热力图。
    """
    h_lines, v_lines, box_owners = board
    n = game.n
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_facecolor('lightgray')

    # 绘制点
    for r in range(n):
        for c in range(n):
            ax.plot(c, r, 'ko', markersize=10) # 'ko' is black circle

    # 绘制已有的边
    for r in range(n):
        for c in range(n - 1):
            if h_lines[r, c]:
                ax.plot([c, c + 1], [r, r], 'k-', linewidth=4)
    for r in range(n - 1):
        for c in range(n):
            if v_lines[r, c]:
                ax.plot([c, c], [r, r + 1], 'k-', linewidth=4)

    # 绘制策略热力图 (如果提供了pi)
    if pi is not None:
        max_pi = max(pi)
        norm = mcolors.Normalize(vmin=0, vmax=max_pi if max_pi > 0 else 1)
        cmap = plt.cm.viridis

        num_h_lines = game.h_lines_shape[0] * game.h_lines_shape[1]
        for action, prob in enumerate(pi):
            if prob > 0:
                color = cmap(norm(prob))
                if action < num_h_lines:
                    r, c = action // game.h_lines_shape[1], action % game.h_lines_shape[1]
                    ax.plot([c, c + 1], [r, r], color=color, linewidth=2, alpha=0.8)
                else:
                    idx = action - num_h_lines
                    r, c = idx // game.v_lines_shape[1], idx % game.v_lines_shape[1]
                    ax.plot([c, c], [r, r + 1], color=color, linewidth=2, alpha=0.8)

    # 标记被占领的格子
    for r in range(n - 1):
        for c in range(n - 1):
            owner = box_owners[r, c]
            if owner != 0:
                player_char = "P1" if owner == 1 else "P2"
                player_color = "blue" if owner == 1 else "red"
                ax.text(c + 0.5, r + 0.5, player_char,
                        ha='center', va='center', fontsize=20, color=player_color, weight='bold')

    ax.set_xticks(np.arange(-1, n + 1, 1))
    ax.set_yticks(np.arange(-1, n + 1, 1))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=16)
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=100)
        plt.close(fig)
    else:
        plt.show()

# --- 主逻辑 ---

def visualize_policy(game, nnet):
    """生成一张AI决策热力图"""
    print("Generating policy heatmap...")
    board = game.get_init_board()
    # 模拟一个中局局面
    board, _ = game.get_next_state(board, 1, 30)
    board, _ = game.get_next_state(board, -1, 31)
    board, _ = game.get_next_state(board, 1, 38)
    board, _ = game.get_next_state(board, -1, 46)

    mcts = MCTS(game, nnet)
    pi = mcts.get_action_prob(board, temp=0, add_exploration_noise=False)

    plot_board(game, board, filename="policy_heatmap.png", pi=pi, title="AI Decision Heatmap (Policy)")
    print("Policy heatmap saved to policy_heatmap.png")


def create_game_gif(game, nnet1, nnet2):
    """生成一个AI对战的GIF动画"""
    print("Generating gameplay GIF animation...")
    frames = []
    board = game.get_init_board()
    current_player = 1
    players = {1: nnet1, -1: nnet2}
    move_count = 0

    while game.get_game_ended(board, 1) == 0:
        move_count += 1
        player_name = "P1 (Blue)" if current_player == 1 else "P2 (Red)"
        
        # 绘制当前帧
        frame_filename = f"frames/frame_{move_count:03d}.png"
        plot_board(game, board, filename=frame_filename, title=f"Move {move_count}: {player_name}'s Turn")
        frames.append(imageio.imread(frame_filename))
        
        # AI决策
        nnet = players[current_player]
        mcts = MCTS(game, nnet)
        pi = mcts.get_action_prob(board, temp=0, add_exploration_noise=False)
        action = np.argmax(pi)
        
        board, current_player = game.get_next_state(board, current_player, action)
        if board is None:
            print("Game ended due to invalid move.")
            break
    
    # 绘制最后一帧
    frame_filename = f"frames/frame_{move_count+1:03d}.png"
    p1_score = np.sum(board[2] == 1)
    p2_score = np.sum(board[2] == -1)
    winner_text = f"Game Over! P1:{p1_score} vs P2:{p2_score}"
    plot_board(game, board, filename=frame_filename, title=winner_text)
    frames.append(imageio.imread(frame_filename))
    
    # 合成GIF
    gif_path = "game_animation.gif"
    imageio.mimsave(gif_path, frames, duration=1.0) # duration是每帧的秒数
    print(f"Gameplay animation saved to {gif_path}")

    # 清理临时图片文件
    for f in os.listdir("frames"):
        os.remove(os.path.join("frames", f))
    os.rmdir("frames")


if __name__ == "__main__":
    # --- 加载模型 ---
    model_path = "data/checkpoints/best_ddp.pth.tar"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train a model first.")
    else:
        game = DotsAndBoxesGame()
        device = torch.device('cpu')
        
        print(f"Loading model from {model_path}...")
        nnet = DotsAndBoxesNet(game).to(device)
        nnet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        nnet.eval()
        
        # --- 生成可视化 ---
        
        # 1. 生成决策热力图
        visualize_policy(game, nnet)
        
        # 2. 生成GIF动画 (AI vs AI)
        if not os.path.exists("frames"):
            os.makedirs("frames")
        # 我们可以让同一个AI互相对战
        create_game_gif(game, nnet, nnet)