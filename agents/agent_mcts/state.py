import math
import numpy as np

from agents.common import BoardPiece, PlayerAction, check_end_state, get_valid_actions, GameState, PLAYER1, PLAYER2


class State(object):  # Remark: call this class Node, because that's what it describes
    # Remark: class and method docstrings missing
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
        # Remark: don't wrap lines using the backslash


    def add_child(self, child):
        self.children[child.action] = child
        # Remark: you could set the child's parent here as well, i.e. child.parent = self

    def update_state(self, value: float) -> None:
        self.visits += 1
        self.value += value

    def is_leaf_node(self) -> bool:
        # Remark: You should rename this method to reflect that you mean a leaf node for the selection (because it might
        #  not actually be a leaf node of the search tree, and to distinguish from leaf nodes of the game tree).
        #  Maybe something like stop_selection() or terminate_selection()?
        # Remark: Also, you could store the valid actions in the node itself, that way you don't have to compute them
        #  again at every step of the selection.
        valid_actions = get_valid_actions(self.board)
        return len(self.children.values()) != len(valid_actions)

    def get_ucb1_value(self, s_p, exploration_constant):
        w_i = self.value
        s_i = self.visits
        # Remark: you can make the exploration constant a class variable, because you're using the same one all the
        #  time. And s_p is actually self.parent.visits, no need to pass it in.
        return w_i / s_i + exploration_constant * math.sqrt(math.log(s_p) / s_i)

    def get_best_move(self, exploration_constant):
        # Remark: call this method differently, something like 'get_max_ucb1_move'. get_best_move sounds too much like
        #  the purpose of the method is to get the best move to play.
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
