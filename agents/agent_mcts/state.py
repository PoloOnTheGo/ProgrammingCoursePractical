import copy
import math

import numpy as np

from agents.common import BoardPiece, PlayerAction, check_end_state, get_valid_actions, apply_player_action, \
    get_opponent, GameState, NO_PLAYER, PLAYER1, PLAYER2


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
        # self.is_fully_expanded = self.is_terminal

    def add_child(self, child):
        self.children.append(copy.deepcopy(child))

    def update_state(self, value: float) -> None:
        self.visits += 1
        self.value += value

    def is_leaf_node(self) -> bool:
        valid_actions = get_valid_actions(self.board)
        return len(self.children.values()) != len(valid_actions)

    def simulate(self, init_state):
        board = self.get_board()
        player = self.get_player()
        value = 0

        if init_state == self:
            return value
        while True:
            valid_actions = get_valid_actions(board)
            selected_action = np.random.choice(valid_actions)
            board = apply_player_action(board, selected_action, player, True)
            if (state_status := check_end_state(board, player)) == GameState.STILL_PLAYING:
                player = get_opponent(player)
            else:
                break

        if state_status == GameState.IS_WIN and self.player == player:
            value = 1
        return value

    def get_state_with_given_board(self, board: np.ndarray):
        if (self.board == board).all():
            return self

        matched_child = None
        for child in self.children:
            matched_child = child.get_state_with_given_board(board)
            if matched_child is not None:
                break
        return matched_child

    def back_propagation(self, v: float, current_state):
        while True:
            if current_state is None:
                break
            current_state_board = current_state.get_board()
            current_state_in_init_state = self.get_state_with_given_board(current_state_board)
            current_state_in_init_state.update_state(v)
            current_state = current_state_in_init_state.get_parent()
            v = not v

    def get_child_with_max_ucb1(self):
        max_ucb1 = - np.inf
        best_child_idx = -1
        for i, child in enumerate(self.children):
            no_of_visits = child.get_visits()
            if no_of_visits == 0:
                best_child_idx = i
                break
            else:
                current_child_ucb1_value = child.get_value() + 2 * np.sqrt(np.log(self.visits)/child.get_visits())
                if current_child_ucb1_value > max_ucb1:
                    max_ucb1 = current_child_ucb1_value
                    best_child_idx = i
        chosen_child: State = self.children[best_child_idx]
        return chosen_child

    def expand(self):
        board = self.get_board()
        valid_actions = get_valid_actions(board)
        self.remove_action_already_present_as_child(valid_actions)
        action = np.random.choice(valid_actions)

        player = get_opponent(self.get_player())
        child_board = apply_player_action(board, action, player, True)
        child = State(child_board, player, parent=self, action=action)
        self.add_child(child)
        return child

    def remove_action_already_present_as_child(self, valid_actions: []):
        for child in self.children:
            valid_actions.remove(child.get_action())

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
                move_score = w_i / s_i + exploration_constant * math.sqrt(math.log(s_p) / s_i)
                if move_score > best_score:
                    best_score = move_score
                    best_child = child_node

        return best_child
