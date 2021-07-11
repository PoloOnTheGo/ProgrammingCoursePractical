import math
import numpy as np

from agents.common import BoardPiece, PlayerAction, check_end_state, get_valid_actions, GameState, PLAYER1, PLAYER2


class State(object):
    def __init__(self, board: np.ndarray, player: BoardPiece = None, value: float = 0.0, visits: int = 0,
                 action: PlayerAction = 0, parent=None):
        self.board = board.copy()
        self.children = {}
        self.value = value
        self.visits = visits
        self.player = player
        self.action = action
        self.parent = parent
        self.is_terminal = not(check_end_state(self.board, PLAYER1) == GameState.STILL_PLAYING) \
                           or not(check_end_state(self.board, PLAYER2) == GameState.STILL_PLAYING)

    def add_child(self, child):
        self.children[child.action] = child

    def update_state(self, value: float) -> None:
        self.visits += 1
        self.value += value

    def is_leaf_node(self) -> bool:
        valid_actions = get_valid_actions(self.board)
        return len(self.children.values()) != len(valid_actions)

    def get_ucb1_value(self, s_p, exploration_constant):
        w_i = self.value
        s_i = self.visits
        return w_i / s_i + exploration_constant * math.sqrt(math.log(s_p) / s_i)

    def get_best_move(self, exploration_constant):
        # define best score & best moves
        best_score = float('-inf')
        best_child = None

        # loop over child nodes
        for child_node in self.children.values():
            w_i = child_node.value
            s_i = child_node.visits
            s_p = self.visits
            if s_i == 0:
                best_child = child_node
            else:
                move_score = w_i if exploration_constant == 0 else child_node.get_ucb1_value(s_p, exploration_constant)
                if move_score > best_score:
                    best_score = move_score
                    best_child = child_node

        return best_child
