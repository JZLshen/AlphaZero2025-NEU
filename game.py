import numpy as np


class DotsAndBoxesGame:
    def __init__(self, n=8):
        """
        n: a grid of n x n dots.
        This creates a board with (n-1) x (n-1) possible boxes.
        """
        self.n = n
        self.num_boxes = (n - 1) * (n - 1)

        # Horizontal lines: n rows of (n-1) lines
        self.h_lines_shape = (n, n - 1)
        # Vertical lines: (n-1) rows of n lines
        self.v_lines_shape = (n - 1, n)

        # Total actions = total lines
        self.action_size = n * (n - 1) + (n - 1) * n  # 56 + 56 = 112 for n=8

    def get_init_board(self):
        """Returns a representation of the initial board state."""
        h_lines = np.zeros(self.h_lines_shape, dtype=np.int8)
        v_lines = np.zeros(self.v_lines_shape, dtype=np.int8)
        # box_owners: stores which player owns which box. 0 = no one.
        box_owners = np.zeros((self.n - 1, self.n - 1), dtype=np.int8)
        return h_lines, v_lines, box_owners

    def get_board_size(self):
        return (self.n, self.n)

    def get_action_size(self):
        return self.action_size

    def get_next_state(self, board, player, action):
        """
        Executes a move and returns the next state.
        If a box is completed, the same player moves again.
        """
        h_lines, v_lines, box_owners = board
        h_lines, v_lines, box_owners = (
            np.copy(h_lines),
            np.copy(v_lines),
            np.copy(box_owners),
        )

        num_h_lines = self.h_lines_shape[0] * self.h_lines_shape[1]

        if action < num_h_lines:  # It's a horizontal line
            r, c = action // self.h_lines_shape[1], action % self.h_lines_shape[1]
            if h_lines[r, c] == 1:
                return None, None  # Invalid move
            h_lines[r, c] = 1
        else:  # It's a vertical line
            idx = action - num_h_lines
            r, c = idx // self.v_lines_shape[1], idx % self.v_lines_shape[1]
            if v_lines[r, c] == 1:
                return None, None  # Invalid move
            v_lines[r, c] = 1

        boxes_scored = 0
        # Check for completed boxes
        for r in range(self.n - 1):
            for c in range(self.n - 1):
                if box_owners[r, c] == 0:  # If box is not already owned
                    if (
                        h_lines[r, c]
                        and h_lines[r + 1, c]
                        and v_lines[r, c]
                        and v_lines[r, c + 1]
                    ):
                        box_owners[r, c] = player
                        boxes_scored += 1

        next_player = player if boxes_scored > 0 else -player

        return (h_lines, v_lines, box_owners), next_player

    def get_valid_moves(self, board):
        h_lines, v_lines, _ = board
        h_valid = (h_lines == 0).flatten()
        v_valid = (v_lines == 0).flatten()
        return np.concatenate((h_valid, v_valid)).astype(np.uint8)

    def get_game_ended(self, board, player):
        """
        Game ends when all boxes are filled.
        Returns score difference from the perspective of `player`.
        """
        _, _, box_owners = board
        if np.all(box_owners != 0):
            my_score = np.sum(box_owners == player)
            opponent_score = np.sum(box_owners == -player)
            if my_score > opponent_score:
                return 1
            elif my_score < opponent_score:
                return -1
            else:  # Draw
                return 1e-4
        return 0  # Game not ended

    def get_canonical_form(self, board, player):
        h_lines, v_lines, box_owners = board
        # The canonical form has the current player as 1 in the box_owners array
        return (h_lines, v_lines, box_owners * player)

    def get_symmetries(self, board, pi):
        """
        Dots and Boxes has the same 8 symmetries as other board games.
        This is complex to implement correctly for this board representation and
        can be skipped for a first working version if needed.
        For now, we return just the original board and policy.
        """
        # TODO: A proper implementation should rotate and flip h_lines, v_lines, and pi.
        # This is non-trivial. For example, rotating h_lines gives v_lines.
        return [(board, pi)]

    def string_representation(self, board):
        """A unique string representation for the board state."""
        h_lines, v_lines, box_owners = board
        return h_lines.tobytes() + v_lines.tobytes() + box_owners.tobytes()
