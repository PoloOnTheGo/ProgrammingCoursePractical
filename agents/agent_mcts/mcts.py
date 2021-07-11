from copy import copy
from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, get_valid_actions, apply_player_action, get_opponent, \
    check_end_state, GameState

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
        current_node = select_leaf_node(initial_node)

        if current_node.visits != 0:
            current_node = expand(current_node)
        v = rollout(current_node, initial_node, player)
        backpropagate(current_node, v)


def select_leaf_node(node) -> State:
    """
    checks if the given node is a leaf node i.e. whether it is not fully expanded(there are available actions still left
    to explore) or continue to move down the tree until it finds a leaf node
    :param node: State
                 current node to be checked
    :return:     State
                 Leaf node (not fully expanded tree node)
    """
    if node.is_leaf_node():
        return node
    best_node = node.get_best_move(exploration_constant=2)
    return select_leaf_node(best_node)


def calculate_value(rolledout_node: State, terminal_node: State, init_player: BoardPiece):
    """
    Calculate the value of the node according to the WIN of the game current player. If PLAYER1 wins so each visited
    PLAYER2 node's win count is incremented. This flip is due to the fact that each node’s statistics are used for its
    parent node’s choice, not its own

    :param rolledout_node:   State
                             the node that is simulated
    :param terminal_node:    State
                             the terminal or end state
    :param init_player:      BoardPiece
                             Current game player taking the turn(initial player)
    """
    value = 0
    if check_end_state(terminal_node.board, init_player) == GameState.IS_WIN \
            and rolledout_node.player == get_opponent(init_player):
        value = 1
    return value


def rollout(rolledout_node: State, init_node: State, init_player: BoardPiece):
    """
    Calculate and return the value of the node according to the WIN of the game current player by simulating randomly until it
    reaches terminal state. Returns 0 when it is the root node, since it does not need simulation

    :param rolledout_node:   State
                             the node that is simulated
    :param init_node:        State
                             initial node
    :param init_player:      BoardPiece
                             Current game player taking the turn(initial player)
    :return:                 Float
                             simulation value according to the WIN of the game current player
    """
    value = 0
    if rolledout_node == init_node:
        return value
    current_node = copy(rolledout_node)

    while True:
        board = current_node.board
        current_player = current_node.player
        if current_node.is_terminal:
            value = calculate_value(rolledout_node, current_node, init_player)
            break
        valid_actions = get_valid_actions(board)
        selected_action = np.random.choice(valid_actions)
        board = apply_player_action(board, selected_action, current_player, True)
        current_node = State(board=board, player=get_opponent(current_player))
    return value


def backpropagate(current_node, v):
    """
    Backpropagate the simulated value and the visit till the root node and

    :param current_node:     State
                             the node from where the backpropagation starts
    :param v:                int
                             simulation value
    """
    # update nodes's up to root node
    while current_node is not None:
        # update node's visits
        current_node.update_state(v)
        # set node to parent
        current_node = current_node.parent
        v = int(not v)


def expand(node):
    """
    Expand the node with a randomly chosen child
    :param node: State
                 The node where a new randomly chosen child is added
    :return:     State
                 created child node
    """
    board = node.board
    valid_actions = get_valid_actions(board)
    valid_actions = remove_action_already_present_as_child(node, valid_actions)

    action = np.random.choice(valid_actions)
    player = node.player
    child_board = apply_player_action(board, action, player, True)
    child = State(child_board, get_opponent(player), parent=node, action=action)
    node.children[action] = child
    return child


def remove_action_already_present_as_child(node, valid_actions):
    """
    Remove the valid action options if that is already played
    :param node: State
                 The node whose children are checked
    :param valid_actions: array
                 array of valid actions
    :return:     array
                 valid actions except action_already_present_as_children
    """
    children_actions = [*node.children]
    return np.setdiff1d(valid_actions, children_actions)
