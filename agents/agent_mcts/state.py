import copy

import numpy as np

from agents.common import BoardPiece, PlayerAction, check_end_state, get_valid_actions, apply_player_action, \
    get_next_player, GameState


class State(object):
    def __init__(self, board: np.ndarray, player: BoardPiece, value: float = 0.0, visits: int = 0,
                 action: PlayerAction = 0, parent=None):
        self.board = board.copy()
        self.children = []
        self.value = value
        self.visits = visits
        self.player = player
        self.action = action
        self.parent = parent

    def get_board(self) -> np.ndarray:
        return self.board

    def get_children(self) -> []:
        return self.children

    def get_value(self) -> float:
        return self.value

    def get_visits(self) -> int:
        return self.visits

    def get_player(self) -> BoardPiece:
        return self.player

    def get_action(self) -> PlayerAction:
        return self.action

    def get_parent(self):
        return self.parent

    def set_visits(self, visits: int) -> None:
        self.visits = visits

    def set_value(self, value: float) -> None:
        self.value = value

    def add_child(self, child):
        self.children.append(copy.deepcopy(child))

    def update_state(self, value: float) -> None:
        self.visits = self.visits + 1
        self.value = value

    def is_leaf_node(self) -> bool:
        return len(self.children) == 0

    def simulate(self):
        board = self.get_board()
        player = self.get_player()
        value = 0

        while (state_status := check_end_state(board, player)) == GameState.STILL_PLAYING:
            valid_actions = get_valid_actions(board)
            selected_action = np.random.choice(valid_actions)
            board = apply_player_action(board, selected_action, player, True)
            player = get_next_player(player)

        if state_status == GameState.IS_WIN:
            if self.player == player:
                value = 1
            else:
                value = -1
        elif state_status == GameState.IS_DRAW:
            value = 0.5
        return value

    def back_propagation(self, v: float):
        current_state = self
        while current_state is not None:
            current_state.update_state(v)
            current_state = current_state.get_parent()

    def get_child_with_max_ucb1(self):
        max_ucb1 = - np.inf
        best_child_idx = -1
        for i, child in enumerate(self.children):
            no_of_visits = child.get_visits()
            if no_of_visits == 0:
                best_child_idx = i
            else:
                current_child_ucb1_value = child.get_value() + np.sqrt(np.log(self.visits)/child.get_visits())
                if current_child_ucb1_value > max_ucb1:
                    max_ucb1 = current_child_ucb1_value
                    best_child_idx = i
        chosen_child: State = self.children[best_child_idx]
        return chosen_child

    def expand(self):
        board = self.get_board()
        valid_actions = get_valid_actions(board)
        random_two_actions = np.random.choice(valid_actions, 2)
        next_player = get_next_player(self.get_player())
        for action in random_two_actions:
            board = apply_player_action(board, action, next_player, True)
            child = State(board, next_player, parent=self)
            self.add_child(child)
        return self.get_children()[0]
