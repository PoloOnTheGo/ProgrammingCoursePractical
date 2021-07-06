from typing import Optional, Tuple

import numpy as np
from agents.agent_mcts import State

from agents.common import BoardPiece, SavedState, PlayerAction


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
        PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column and return it as `action`
    no_of_iterations = 35
    action = mcts(board, no_of_iterations, player)
    return action, saved_state


def mcts(board: np.ndarray, no_of_iterations: int, player: BoardPiece) -> PlayerAction:
    init_state = State(board.copy())
    for _ in range(no_of_iterations):
        tree_traversal(init_state, player)
    chosen_child: State = init_state.get_child_with_max_ucb1()
    return chosen_child.get_action()


def tree_traversal(initial_state: State, player: BoardPiece):
    current = initial_state
    while True:
        if current.is_leaf_node():
            if current.get_visits() != 0:
                current = current.expand()

            v = current.simulate(initial_state)
            initial_state.back_propagation(v, current)
            break
        else:
            current = current.get_child_with_max_ucb1()
