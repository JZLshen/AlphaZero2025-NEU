import os
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from config import *
from game import DotsAndBoxesGame
from model import DotsAndBoxesNet
from mcts import MCTS


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def push(self, data):
        self.buffer.extend(data)

    def __len__(self):
        return len(self.buffer)


class ReplayDataset(Dataset):
    def __init__(self, buffer_list):
        self.examples = buffer_list

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def self_play_for_rank(rank, games_per_process, model_path):
    device = torch.device(f"cuda:{rank}")
    game = DotsAndBoxesGame()
    nnet = DotsAndBoxesNet(game).to(device)
    if os.path.exists(model_path):
        nnet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    nnet.eval()

    local_examples = []
    
    iterable_range = range(games_per_process)
    if rank == 0:
        iterable_range = tqdm(iterable_range, desc=f"Self-Play on Rank {rank}")

    for _ in iterable_range:
        train_examples = []
        board = game.get_init_board()
        current_player = 1
        while True:
            canonical_board = game.get_canonical_form(board, current_player)
            mcts = MCTS(game, nnet)
            pi = mcts.get_action_prob(canonical_board, temp=1, add_exploration_noise=True)
            
            sym = game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([game.string_representation(b), current_player, p, None])
            
            action = np.random.choice(len(pi), p=pi)
            next_board, next_player = game.get_next_state(board, current_player, action)

            if next_board is None:
                if rank == 0: print(f"Warning: MCTS chose an invalid move {action}. Ending game.")
                break
            
            board, current_player = next_board, next_player
            game_result = game.get_game_ended(board, 1)

            if game_result != 0:
                final_data = [(x[0], np.array(x[2], dtype=np.float32), game_result if x[1] == 1 else -game_result) for x in train_examples]
                local_examples.extend(final_data)
                break
    return local_examples


def train(rank, model, optimizer, replay_buffer, writer, current_iter, global_step_counter):
    model.train()
    device = torch.device(f"cuda:{rank}")
    game = DotsAndBoxesGame()
    n = game.n
    
    dataset = ReplayDataset(list(replay_buffer.buffer))
    # DistributedSampler会确保每个进程拿到不同的数据子集
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True, num_workers=0)

    if rank == 0:
        print(f"Training on local data (size: {len(dataset)}), total data across GPUs is larger.")
    
    for epoch in range(EPOCHS_PER_ITERATION):
        sampler.set_epoch(current_iter * EPOCHS_PER_ITERATION + epoch)
        
        pbar = dataloader
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Iter {current_iter} - Epoch {epoch+1}/{EPOCHS_PER_ITERATION}")

        for boards_bytes, pis, vs in pbar:
            h_size, v_size = n * (n - 1), (n - 1) * n
            board_tensors = []
            for b_bytes in boards_bytes:
                h_lines = np.frombuffer(b_bytes[:h_size], dtype=np.int8).reshape(n, n-1)
                v_lines = np.frombuffer(b_bytes[h_size:h_size+v_size], dtype=np.int8).reshape(n-1, n)
                box_owners = np.frombuffer(b_bytes[h_size+v_size:], dtype=np.int8).reshape(n-1, n-1)
                h_padded = np.pad(h_lines, ((0, 0), (0, 1)), "constant")
                v_padded = np.pad(v_lines, ((0, 1), (0, 0)), "constant")
                box_padded = np.pad(box_owners, ((0, 1), (0, 1)), "constant")
                board_tensors.append(torch.tensor(np.array([h_padded, v_padded, box_padded]), dtype=torch.float32))
            
            boards_tensor_batch = torch.stack(board_tensors).to(device)
            target_pis = torch.FloatTensor(np.array(pis)).to(device)
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float32)).to(device)

            out_pi, out_v = model(boards_tensor_batch)

            l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
            l_v = torch.sum((target_vs.view(-1) - out_v.view(-1)) ** 2) / target_vs.size(0)
            total_loss = l_pi + l_v

            optimizer.zero_grad()
            total_loss.backward() # DDP会自动同步梯度
            optimizer.step()
            
            if rank == 0:
                if global_step_counter[0] % 10 == 0:
                    writer.add_scalar("Loss/total", total_loss.item(), global_step_counter[0])
                    writer.add_scalar("Loss/policy", l_pi.item(), global_step_counter[0])
                    writer.add_scalar("Loss/value", l_v.item(), global_step_counter[0])
                global_step_counter[0] += 1


def play_game(game, nnet1, nnet2, device):
    players = {1: nnet1, -1: nnet2}
    board = game.get_init_board()
    current_player = 1
    while game.get_game_ended(board, 1) == 0:
        # 在评估时，模型应该在单个设备上，并且处于eval模式
        nnet = players[current_player].to(device)
        nnet.eval()
        canonical_board = game.get_canonical_form(board, current_player)
        mcts = MCTS(game, nnet)
        action_probs = mcts.get_action_prob(canonical_board, temp=0, add_exploration_noise=False)
        action = np.argmax(action_probs)
        next_board, next_player = game.get_next_state(board, current_player, action)
        if next_board is None:
            return -current_player 
        board, current_player = next_board, next_player
    return game.get_game_ended(board, 1)


def pit(game, new_net, best_net, num_games, device):
    p1_wins, p2_wins, draws = 0, 0, 0
    # 评估时，只在主进程显示进度条
    iterable = range(num_games)
    if dist.get_rank() == 0:
        iterable = tqdm(iterable, desc="Pitting Arena")
        
    for i in iterable:
        # 为了公平，轮流执先
        winner = play_game(game, new_net, best_net, device) if i % 2 == 0 else play_game(game, best_net, new_net, device)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1
    return p1_wins, p2_wins, draws


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    writer = None
    if rank == 0:
        print(f"Starting DDP with {world_size} GPUs.")
        if not os.path.exists("data/checkpoints"):
            os.makedirs("data/checkpoints")
        writer = SummaryWriter(f"runs/dots_and_boxes_ddp_{int(time.time())}")

    global_step_counter = [0]
    
    game = DotsAndBoxesGame()
    model = DotsAndBoxesNet(game).to(rank)
    # 修正：根据PyTorch的建议，移除了 find_unused_parameters=True
    model = DDP(model, device_ids=[rank])
    
    best_model_path = "data/checkpoints/best_ddp.pth.tar"
    if os.path.exists(best_model_path):
        if rank == 0: print("Loading best model...")
        map_location = {'cuda:0': f'cuda:{rank}'}
        model.module.load_state_dict(torch.load(best_model_path, map_location=map_location, weights_only=True))
    elif rank == 0:
        print("No best model found, saving initial model.")
        torch.save(model.module.state_dict(), best_model_path)
    
    dist.barrier() 

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    replay_buffer = ReplayBuffer(maxlen=REPLAY_BUFFER_MAXLEN)

    for i in range(1, NUM_ITERATIONS + 1):
        if rank == 0: print(f"--- Iteration {i} --- \nStarting self-play...")
        
        model.eval()
        games_per_rank = (GAMES_PER_ITERATION + world_size - 1) // world_size
        new_examples = self_play_for_rank(rank, games_per_rank, best_model_path)
        replay_buffer.push(new_examples)

        dist.barrier() 
        
        # --- 修正的数据同步逻辑 ---
        # 1. 每个进程创建一个包含其本地buffer长度的张量
        local_buffer_size = torch.tensor([len(replay_buffer)], dtype=torch.long, device=rank)
        
        # 2. 使用 all_reduce 将所有进程的buffer长度相加。完成后，所有进程的 local_buffer_size 值都等于全局总和
        dist.all_reduce(local_buffer_size, op=dist.ReduceOp.SUM)
        
        total_buffer_size = local_buffer_size.item()
        
        if rank == 0: writer.add_scalar("ReplayBuffer/size", total_buffer_size, i)

        if total_buffer_size < BATCH_SIZE:
            if rank == 0: print(f"Buffer size {total_buffer_size} is not enough for training. Skipping.")
            continue

        if rank == 0: print("Training neural network...")
        train(rank, model, optimizer, replay_buffer, writer, i, global_step_counter)

        scheduler.step()
        
        # 评估和保存只在主进程（rank 0）上进行
        if rank == 0:
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], i)
            
            print("Pitting new model against best model...")
            # 创建一个独立的 best_net 用于评估，加载CPU上的权重
            best_net = DotsAndBoxesNet(game)
            best_net.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=True))
            # .module 才能访问到DDP包装下的原始模型
            new_net = model.module
            
            p1_wins, p2_wins, draws = pit(game, new_net, best_net, EVALUATION_GAMES, torch.device(f'cuda:{rank}'))
            win_rate = (p1_wins + 0.5 * draws) / (p1_wins + p2_wins + draws + 1e-8)
            
            print(f"New Model Win Rate vs Best: {win_rate:.3f} ({p1_wins}-{p2_wins}-{draws})")
            writer.add_scalar("WinRate/vs_best", win_rate, i)

            if win_rate > EVALUATION_THRESHOLD:
                print("New model is better! Updating best model.")
                torch.save(new_net.state_dict(), best_model_path)
        
        # 等待主进程完成评估和保存，再开始下一轮迭代
        dist.barrier()

    if rank == 0 and writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()