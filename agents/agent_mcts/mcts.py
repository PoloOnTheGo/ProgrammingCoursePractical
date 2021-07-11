from typing import Optional, Tuple

from agents.common import BoardPiece, SavedState, PlayerAction
import numpy as np
from agents.agent_mcts import State


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
        PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column and return it as `action`
    no_of_iterations = 2000
    action = mcts(board, no_of_iterations, player)
    return action, saved_state


def mcts(board: np.ndarray, no_of_iterations: int, player: BoardPiece) -> PlayerAction:
    """
    the Monte-Carlo Tree Search Algo wrapper, that creates the tree with root with initial board, traverse the tree and
    finally chooses the best action as per the no of wins
    :param board:            np.ndarray
                             Current board represented by array for game state
    :param no_of_iterations: int
                             Number of iterations for the Monte Carlo Algorithm
    :param player:           BoardPiece
                             Current player taking the turn
    :return:                 PlayerAction
                             Chosen best action the player should take
    """
    init_state = State(board.copy(), player=player)
    tree_traversal(no_of_iterations, init_state, player)
    chosen_child: State = init_state.get_best_move(0)
    return PlayerAction(chosen_child.action)


def tree_traversal(no_of_iterations: int, initial_node: State, player: BoardPiece):
    """
    traverse the tree for the given number of iterations to populate the search tree with wins and visit suggestions
    :param no_of_iterations: int
                             Number of iterations for the Monte Carlo Algorithm
    :param initial_node:     State
                             State object for the current game board acting the root of the search tree
    :param player:           BoardPiece
                             Current player taking the turn
    """
    for _ in range(no_of_iterations):
        current_node = initial_node.select_leaf_node()

        if current_node.visits != 0:
            current_node = current_node.expand()
        v = current_node.rollout(initial_node, player)
        current_node.backpropagate(v)
