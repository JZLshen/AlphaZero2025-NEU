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

def plot_board(game, board, filename=None, pi=None, title="", line_owners=None):
    """
    绘制棋盘状态，可以选择性地叠加策略热力图。
    如果提供了 line_owners，则根据玩家绘制不同颜色的边线。
    """
    h_lines, v_lines, box_owners = board
    n = game.n
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_facecolor('lightgray')

    p1_color = "blue"
    p2_color = "red"

    # 绘制已占用的边线
    if line_owners:
        h_line_owners, v_line_owners = line_owners
        for r in range(n):
            for c in range(n - 1):
                owner = h_line_owners[r, c]
                if owner != 0:
                    color = p1_color if owner == 1 else p2_color
                    ax.plot([c, c + 1], [r, r], color=color, linewidth=4, zorder=1)
        for r in range(n - 1):
            for c in range(n):
                owner = v_line_owners[r, c]
                if owner != 0:
                    color = p1_color if owner == 1 else p2_color
                    ax.plot([c, c], [r, r + 1], color=color, linewidth=4, zorder=1)
    else:
        for r in range(n):
            for c in range(n - 1):
                if h_lines[r, c]:
                    ax.plot([c, c + 1], [r, r], 'k-', linewidth=4, zorder=1)
        for r in range(n - 1):
            for c in range(n):
                if v_lines[r, c]:
                    ax.plot([c, c], [r, r + 1], 'k-', linewidth=4, zorder=1)

    # 绘制策略热力图
    if pi is not None:
        pi_cpu = pi.cpu().numpy() if isinstance(pi, torch.Tensor) else np.array(pi)
        max_pi = max(pi_cpu) if len(pi_cpu) > 0 else 1
        norm = mcolors.Normalize(vmin=0, vmax=max_pi if max_pi > 1e-6 else 1)
        cmap = plt.cm.viridis

        num_h_lines = game.h_lines_shape[0] * game.h_lines_shape[1]
        for action, prob in enumerate(pi_cpu):
            if prob > 1e-4:
                color = cmap(norm(prob))
                if action < num_h_lines:
                    r, c = action // game.h_lines_shape[1], action % game.h_lines_shape[1]
                    if not (line_owners and line_owners[0][r, c] != 0):
                        ax.plot([c, c + 1], [r, r], color=color, linewidth=4, alpha=0.7, zorder=2)
                else:
                    idx = action - num_h_lines
                    r, c = idx // game.v_lines_shape[1], idx % game.v_lines_shape[1]
                    if not (line_owners and line_owners[1][r, c] != 0):
                         ax.plot([c, c], [r, r + 1], color=color, linewidth=4, alpha=0.7, zorder=2)

    # 绘制棋盘格点
    for r in range(n):
        for c in range(n):
            ax.plot(c, r, 'ko', markersize=10, zorder=3)

    # 标记被占领的格子
    for r in range(n - 1):
        for c in range(n - 1):
            owner = box_owners[r, c]
            if owner != 0:
                player_char = "P1" if owner == 1 else "P2"
                player_color = "white"
                bbox = dict(boxstyle="round,pad=0.3", fc=(p1_color if owner==1 else p2_color), ec="black", lw=2, alpha=0.9)
                ax.text(c + 0.5, r + 0.5, player_char,
                            ha='center', va='center', fontsize=20, color=player_color, weight='bold', zorder=4, bbox=bbox)
    
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
    """生成一张AI决策热力图。"""
    print("Generating policy heatmap...")
    board = game.get_init_board()
    # 模拟一个中局局面
    board, _ = game.get_next_state(board, 1, 30)
    board, _ = game.get_next_state(board, -1, 31)
    
    mcts = MCTS(game, nnet)
    pi = mcts.get_action_prob(board, temp=0.2, add_exploration_noise=False) # 使用 temp=0.2 获得更丰富的热力图
        
    plot_board(game, board, filename="policy_heatmap.png", pi=pi, title="AI Decision Heatmap (Policy)")
    print("Policy heatmap saved to policy_heatmap.png")


def create_game_gif(game, nnet1, nnet2):
    """生成一个AI对战的GIF动画。"""
    print("Generating gameplay GIF animation...")
    frames = []
    board = game.get_init_board()
    current_player = 1
    device = next(nnet1.parameters()).device
    players = {1: nnet1.to(device), -1: nnet2.to(device)}
    move_count = 0

    h_line_owners = np.zeros_like(board[0])
    v_line_owners = np.zeros_like(board[1])
    line_owners = (h_line_owners, v_line_owners)
    num_h_lines = game.h_lines_shape[0] * game.h_lines_shape[1]

    os.makedirs("frames", exist_ok=True)

    pbar = tqdm(total=game.get_action_size())
    while game.get_game_ended(board, 1) == 0:
        pbar.update(1)
        move_count += 1
        player_name = "P1 (Blue)" if current_player == 1 else "P2 (Red)"
        
        frame_filename = f"frames/frame_{move_count:03d}.png"
        plot_board(game, board, filename=frame_filename, title=f"Move {move_count}: {player_name}'s Turn", line_owners=line_owners)
        frames.append(imageio.imread(frame_filename))
        
        nnet = players[current_player]
        mcts = MCTS(game, nnet)
        pi = mcts.get_action_prob(board, temp=0, add_exploration_noise=False)
        action = np.argmax(pi.cpu().numpy() if isinstance(pi, torch.Tensor) else pi)
        
        # 记录边线归属
        if action < num_h_lines:
            r, c = action // game.h_lines_shape[1], action % game.h_lines_shape[1]
            h_line_owners[r, c] = current_player
        else:
            idx = action - num_h_lines
            r, c = idx // game.v_lines_shape[1], idx % game.v_lines_shape[1]
            v_line_owners[r, c] = current_player

        board, current_player = game.get_next_state(board, current_player, action)
        if board is None:
            print("Game ended due to invalid move.")
            break
    pbar.close()
    
    # 绘制并保存最后一帧
    frame_filename = f"frames/frame_{move_count+1:03d}.png"
    p1_score = np.sum(board[2] == 1)
    p2_score = np.sum(board[2] == -1)
    winner_text = f"Game Over! P1:{p1_score} vs P2:{p2_score}"
    plot_board(game, board, filename=frame_filename, title=winner_text, line_owners=line_owners)
    frames.append(imageio.imread(frame_filename))
    
    # 合成并保存GIF
    gif_path = "game_animation.gif"
    imageio.mimsave(gif_path, frames, duration=1.0)
    print(f"Gameplay animation saved to {gif_path}")

    # 清理临时帧文件
    for f in os.listdir("frames"):
        os.remove(os.path.join("frames", f))
    os.rmdir("frames")


if __name__ == "__main__":
    model_path = "data/checkpoints/best_ddp.pth.tar"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train a model first.")
    else:
        game = DotsAndBoxesGame()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 加载并设置模型 ---
        print(f"Using device: {device}")
        print(f"Loading model from {model_path}...")
        
        nnet = DotsAndBoxesNet(game).to(device)
        nnet.load_state_dict(torch.load(model_path, map_location=device))
        nnet.eval()
        
        # --- 运行可视化任务 ---
        visualize_policy(game, nnet)
        create_game_gif(game, nnet, nnet)