from typing import Optional, Tuple

import numpy as np

from agents.common import BoardPiece, SavedState, PlayerAction


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
        PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column and return it as `action`
    no_of_iterations = 100
    action = mcts(board, no_of_iterations, player)
    return action, saved_state


def mcts(board, no_of_iterations, player) -> PlayerAction:
    init_state = State(board.copy(), player)
    for _ in range(no_of_iterations):
        tree_traversal(init_state)
    chosen_child: State = init_state.get_child_with_max_ucb1()
    return chosen_child.get_action()


def tree_traversal(initial_state: State):
    current = initial_state
    while True:
        if current.is_leaf_node():
            if current.get_visits() != 0:
                current = current.expand()

            v = current.simulate()
            current.back_propagation(v)
            break
        else:
            current = current.get_child_with_max_ucb1()
