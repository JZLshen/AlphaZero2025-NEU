import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config import *
from game import DotsAndBoxesGame
from model import DotsAndBoxesNet
from mcts import MCTS

# --- 可视化对战主类 ---

class VisualGame:
    def __init__(self, game, nnet, mcts):
        self.game = game
        self.nnet = nnet
        self.mcts = mcts
        self.board = self.game.get_init_board()
        self.current_player = 1 # 1 for Human, -1 for AI

        # 初始化绘图窗口
        self.fig, self.ax = plt.subplots(1, figsize=(8, 8))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_plot()

    def update_plot(self):
        """重绘整个棋盘"""
        self.ax.clear()
        self.ax.set_facecolor('#EAEAF2') # 设置背景色
        
        h_lines, v_lines, box_owners = self.board
        n = self.game.n

        # 绘制点
        for r in range(n):
            for c in range(n):
                self.ax.plot(c, r, 'ko', markersize=12, zorder=5)

        # 绘制已有的边
        for r in range(n):
            for c in range(n - 1):
                if h_lines[r, c]: self.ax.plot([c, c + 1], [r, r], 'k-', linewidth=5, zorder=10)
        for r in range(n - 1):
            for c in range(n):
                if v_lines[r, c]: self.ax.plot([c, c], [r, r + 1], 'k-', linewidth=5, zorder=10)

        # 绘制被占领的格子
        for r in range(n - 1):
            for c in range(n - 1):
                if box_owners[r, c] != 0:
                    color = '#4C72B0' if box_owners[r, c] == 1 else '#C44E52'
                    rect = Rectangle((c, r), 1, 1, facecolor=color, alpha=0.6, zorder=1)
                    self.ax.add_patch(rect)

        # 设置标题和坐标轴
        title = "Your Turn (Player 1 - Blue)" if self.current_player == 1 else "AI's Turn (Player 2 - Red)"
        self.ax.set_title(title, fontsize=16, pad=20)
        self.ax.set_xticks(np.arange(-1, n + 1, 1))
        self.ax.set_yticks(np.arange(-1, n + 1, 1))
        self.ax.set_xlim(-0.5, n - 0.5)
        self.ax.set_ylim(-0.5, n - 0.5)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.invert_yaxis()
        plt.grid(True, linestyle='--', color='gray', alpha=0.5)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax or self.current_player != 1:
            return

        x, y = event.xdata, event.ydata
        action = self.get_action_from_click(x, y)

        if action is not None and self.game.get_valid_moves(self.board)[action] == 1:
            self.make_move(action)
        else:
            print("Invalid click or move. Please click closer to an empty line.")

    def get_action_from_click(self, x, y):
        """根据点击坐标计算是哪条边"""
        # (四舍五入到最近的点)
        c, r = round(x), round(y)
        dx, dy = abs(x - c), abs(y - r)
        
        num_h_lines = self.game.h_lines_shape[0] * self.game.h_lines_shape[1]

        if dx > dy: # 更接近水平边
            if c >= self.game.n - 1: c = self.game.n - 2
            action = r * (self.game.n - 1) + c
        else: # 更接近垂直边
            if r >= self.game.n - 1: r = self.game.n - 2
            action = num_h_lines + r * self.game.n + c
        
        if 0 <= action < self.game.get_action_size():
            return action
        return None

    def make_move(self, action):
        """执行一步移动并更新状态"""
        self.board, self.current_player = self.game.get_next_state(self.board, self.current_player, action)
        self.update_plot()
        self.check_game_state()

    def ai_turn(self):
        """执行AI的回合"""
        while self.current_player == -1 and self.game.get_game_ended(self.board, 1) == 0:
            self.ax.set_title("AI is thinking...", fontsize=16, pad=20)
            self.fig.canvas.draw_idle()
            plt.pause(0.1) # 留出时间给UI刷新

            canonical_board = self.game.get_canonical_form(self.board, self.current_player)
            pi = self.mcts.get_action_prob(canonical_board, temp=0, add_exploration_noise=False)
            action = np.argmax(pi)
            
            print(f"AI Chose Move: {action}")
            self.make_move(action)

    def check_game_state(self):
        """检查游戏状态，如果是AI回合则触发AI，如果结束则显示结果"""
        winner = self.game.get_game_ended(self.board, 1)
        if winner != 0:
            p1_score = np.sum(self.board[2] == 1)
            p2_score = np.sum(self.board[2] == -1)
            title = f"Game Over! P1(Blue): {p1_score} vs P2(Red): {p2_score}"
            if winner > 0: title += " - You Won!"
            elif winner < 0: title += " - AI Won!"
            else: title += " - It's a Draw!"
            
            self.ax.set_title(title, fontsize=16, pad=20)
            self.fig.canvas.draw_idle()
        elif self.current_player == -1:
            # 使用一个短暂的延迟来启动AI回合，让UI有时间响应
            self.fig.canvas.start_event_loop(timeout=0.1)
            self.ai_turn()

    def start(self):
        self.check_game_state()
        plt.show()


if __name__ == "__main__":
    model_path = "data/checkpoints/best_ddp.pth.tar"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train a model first.")
    else:
        game = DotsAndBoxesGame()
        device = torch.device('cpu')
        
        print("Loading trained model...")
        nnet = DotsAndBoxesNet(game).to(device)
        nnet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        nnet.eval()
        
        mcts = MCTS(game, nnet)
        
        print("Starting visual game. Click on an empty line to make a move.")
        visual_game = VisualGame(game, nnet, mcts)
        visual_game.start()