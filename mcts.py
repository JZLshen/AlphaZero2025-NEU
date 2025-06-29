import numpy as np
import math
from config import *


class MCTS:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}

    def get_action_prob(self, canonical_board, temp=1, add_exploration_noise=True):
        s = self.game.string_representation(canonical_board)

        # 首次访问或需要加噪声时，先跑一次search来获取Ps[s]
        if s not in self.Ps or add_exploration_noise:
            self.search(canonical_board, add_exploration_noise=add_exploration_noise)
        else:
            for _ in range(MCTS_SIMULATIONS):
                self.search(canonical_board, add_exploration_noise=False)

        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        if temp == 0:
            best_a = np.argmax(counts)
            probs = np.zeros_like(counts, dtype=float)
            probs[best_a] = 1.0
            return probs

        counts = [c ** (1.0 / temp) for c in counts]
        total_counts = float(sum(counts))
        if total_counts == 0:
            valid_moves = self.game.get_valid_moves(canonical_board)
            return valid_moves / np.sum(valid_moves)
        probs = [x / total_counts for x in counts]
        return probs

    def search(self, canonical_board, add_exploration_noise=False):
        s = self.game.string_representation(canonical_board)

        ended = self.game.get_game_ended(canonical_board, 1)
        if ended != 0:
            return -ended

        if s not in self.Ps:
            pi, v = self.nnet(self.nnet.to_input_tensor(canonical_board))
            pi = torch.exp(pi).data.cpu().numpy()[0]
            v = v.data.cpu().numpy()[0][0]

            valid_moves = self.game.get_valid_moves(canonical_board)
            pi *= valid_moves
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                pi = valid_moves / np.sum(valid_moves)

            self.Ps[s] = pi

            if add_exploration_noise:
                noise = np.random.dirichlet(
                    [DIRICHLET_ALPHA] * self.game.get_action_size()
                )
                self.Ps[s] = (1 - DIRICHLET_EPSILON) * self.Ps[
                    s
                ] + DIRICHLET_EPSILON * noise

            self.Ns[s] = 0
            return -v

        valid_moves = self.game.get_valid_moves(canonical_board)
        cur_best = -float("inf")
        best_act = -1

        for a in range(self.game.get_action_size()):
            if valid_moves[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + CPUCT * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, _ = self.game.get_next_state(canonical_board, 1, a)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
