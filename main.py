import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time

from config import *
from game import DotsAndBoxesGame
from model import GomokuNet
from mcts import MCTS


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def push(self, data):
        self.buffer.extend(data)

    def sample(self, batch_size):
        sample_ids = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in sample_ids]

    def __len__(self):
        return len(self.buffer)


def self_play_worker(model_path, num_games, replay_buffer_queue):
    game = DotsAndBoxesGame()
    nnet = GomokuNet(game).to(DEVICE)
    if os.path.exists(model_path):
        nnet.load_state_dict(torch.load(model_path, map_location=DEVICE))
    nnet.eval()

    local_buffer = []
    for _ in range(num_games):
        train_examples = []
        board = game.get_init_board()
        current_player = 1
        episode_step = 0
        while True:
            episode_step += 1
            canonical_board = game.get_canonical_form(board, current_player)
            mcts = MCTS(game, nnet)
            temp = 1 if episode_step < 15 else 0
            pi = mcts.get_action_prob(
                canonical_board, temp=temp, add_exploration_noise=True
            )

            sym = game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append(
                    [game.string_representation(b), current_player, p, None]
                )

            action = np.random.choice(len(pi), p=pi)
            board, current_player = game.get_next_state(board, current_player, action)
            game_result = game.get_game_ended(board, 1)

            if game_result != 0:
                final_data = [
                    (x[0], x[2], game_result if x[1] == 1 else -game_result)
                    for x in train_examples
                ]
                local_buffer.extend(final_data)
                break
    replay_buffer_queue.put(local_buffer)


def train(nnet, optimizer, replay_buffer, writer, global_step):
    nnet.train()
    game = DotsAndBoxesGame()
    n = game.n

    for _ in range(EPOCHS_PER_ITERATION):
        examples = replay_buffer.sample(BATCH_SIZE)
        boards_bytes, pis, vs = list(zip(*examples))

        board_tensors = []
        h_size = n * (n - 1)
        v_size = (n - 1) * n

        for b_bytes in boards_bytes:
            h_lines = np.frombuffer(b_bytes[:h_size], dtype=np.int8).reshape(n, n - 1)
            v_lines = np.frombuffer(
                b_bytes[h_size : h_size + v_size], dtype=np.int8
            ).reshape(n - 1, n)
            box_owners = np.frombuffer(
                b_bytes[h_size + v_size :], dtype=np.int8
            ).reshape(n - 1, n - 1)

            h_padded = np.pad(h_lines, ((0, 0), (0, 1)), "constant")
            v_padded = np.pad(v_lines, ((0, 1), (0, 0)), "constant")
            box_padded = np.pad(box_owners, ((0, 1), (0, 1)), "constant")

            board_tensor = torch.tensor(
                np.array([h_padded, v_padded, box_padded]), dtype=torch.float32
            )
            board_tensors.append(board_tensor)

        boards_tensor_batch = torch.stack(board_tensors)
        target_pis = torch.FloatTensor(np.array(pis))
        target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

        out_pi, out_v = nnet(boards_tensor_batch)

        l_pi = -torch.sum(target_pis.to(DEVICE) * out_pi) / target_pis.size()[0]
        l_v = (
            torch.sum((target_vs.to(DEVICE).view(-1) - out_v.view(-1)) ** 2)
            / target_vs.size()[0]
        )
        total_loss = l_pi + l_v

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if global_step % 10 == 0:
            writer.add_scalar("Loss/total", total_loss.item(), global_step)
            writer.add_scalar("Loss/policy", l_pi.item(), global_step)
            writer.add_scalar("Loss/value", l_v.item(), global_step)
        global_step += 1
    return global_step


def play_game(game, nnet1, nnet2):
    players = {1: nnet1, -1: nnet2}
    board = game.get_init_board()
    current_player = 1
    while game.get_game_ended(board, 1) == 0:
        nnet = players[current_player]
        canonical_board = game.get_canonical_form(board, current_player)
        mcts = MCTS(game, nnet)
        action_probs = mcts.get_action_prob(
            canonical_board, temp=0, add_exploration_noise=False
        )
        action = np.argmax(action_probs)
        board, current_player = game.get_next_state(board, current_player, action)
    return game.get_game_ended(board, 1)


def pit(game, nnet1, nnet2, num_games):
    p1_wins, p2_wins, draws = 0, 0, 0
    for _ in tqdm(range(num_games // 2), desc="Arena (P1 starts)"):
        winner = play_game(game, nnet1, nnet2)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1

    for _ in tqdm(range(num_games // 2), desc="Arena (P2 starts)"):
        winner = play_game(game, nnet2, nnet1)
        if winner == -1:
            p1_wins += 1
        elif winner == 1:
            p2_wins += 1
        else:
            draws += 1
    return p1_wins, p2_wins, draws


def main():
    print(f"Using device: {DEVICE}")
    game = DotsAndBoxesGame()
    replay_buffer = ReplayBuffer(maxlen=REPLAY_BUFFER_MAXLEN)
    writer = SummaryWriter(f"runs/dots_and_boxes_experiment_{int(time.time())}")
    global_step = 0

    net = GomokuNet(game).to(DEVICE)
    best_net = GomokuNet(game).to(DEVICE)

    if not os.path.exists("data/checkpoints"):
        os.makedirs("data/checkpoints")

    best_model_path = "data/checkpoints/best.pth.tar"
    if os.path.exists(best_model_path):
        print("Loading best model...")
        net.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("No best model found, saving initial model.")
        torch.save(net.state_dict(), best_model_path)

    best_net.load_state_dict(net.state_dict())

    optimizer = optim.Adam(
        net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    for i in range(1, NUM_ITERATIONS + 1):
        print(f"--- Iteration {i} ---")

        print("Starting self-play...")
        best_net.load_state_dict(torch.load(best_model_path))
        replay_buffer_queue = mp.Queue()
        num_processes = mp.cpu_count()
        games_per_process = (GAMES_PER_ITERATION + num_processes - 1) // num_processes

        processes = []
        for _ in range(num_processes):
            p = mp.Process(
                target=self_play_worker,
                args=(best_model_path, games_per_process, replay_buffer_queue),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        for _ in range(num_processes):
            replay_buffer.push(replay_buffer_queue.get())

        if len(replay_buffer) < BATCH_SIZE:
            print(
                f"Buffer size {len(replay_buffer)} is not enough for training. Skipping."
            )
            continue

        print("Training neural network...")
        global_step = train(net, optimizer, replay_buffer, writer, global_step)

        scheduler.step()

        print("Pitting new model against best model...")
        net.eval()
        best_net.eval()
        p1_wins, p2_wins, draws = pit(game, net, best_net, EVALUATION_GAMES)
        win_rate = (p1_wins + 0.5 * draws) / (p1_wins + p2_wins + draws + 1e-8)
        print(f"New Model Win Rate: {win_rate:.3f} ({p1_wins}-{p2_wins}-{draws})")

        writer.add_scalar("WinRate/vs_best", win_rate, i)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], i)
        writer.add_scalar("ReplayBuffer/size", len(replay_buffer), i)

        if win_rate > EVALUATION_THRESHOLD:
            print("New model is better! Updating best model.")
            torch.save(net.state_dict(), best_model_path)
            best_net.load_state_dict(net.state_dict())
        else:
            print("New model is not better. Reverting to the best model.")
            net.load_state_dict(best_net.state_dict())

        if i % CHECKPOINT_INTERVAL == 0:
            torch.save(net.state_dict(), f"data/checkpoints/checkpoint_{i}.pth.tar")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
