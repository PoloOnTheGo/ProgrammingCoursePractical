import time
from copy import copy
from typing import Optional, Tuple

import numpy as np
from agents.agent_mcts import State

from agents.common import BoardPiece, SavedState, PlayerAction, get_valid_actions, apply_player_action, get_opponent, \
    check_end_state, GameState


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
        PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column and return it as `action`
    no_of_iterations = 2000
    action = mcts(board, no_of_iterations, player)
    return action, saved_state


def mcts(board: np.ndarray, no_of_iterations: int, player: BoardPiece) -> PlayerAction:
    init_state = State(board.copy(), player=player)
    tree_traversal(no_of_iterations, init_state, player)
    for child in init_state.children.values():
        print("Action: " + str(child.action) + ", Wins:" + str(child.value) + " , Visits:", str(child.visits)
              + ", Value:" + (str(child.value/child.visits)))
    chosen_child: State = init_state.get_best_move(0)
    return chosen_child.action


def tree_traversal(no_of_iterations:int, initial_node: State, player: BoardPiece):
    for _ in range(no_of_iterations):
        current_node = select_leaf_node(initial_node)

        if current_node.visits != 0:
            current_node = expand(current_node)
        v = rollout(current_node, initial_node, player)
        backpropagate(current_node, v)


def select_leaf_node(node):
    if node.is_leaf_node():
        return node
    best_node = node.get_best_move(exploration_constant=2)
    return select_leaf_node(best_node)


def calculate_value(rolledout_node: State, terminal_node: State, init_player: BoardPiece):
    value = 0
    if check_end_state(terminal_node.board, init_player) == GameState.IS_WIN \
            and rolledout_node.player == get_opponent(init_player):
        value = 1
    return value


def rollout(rolledout_node: State, init_node: State, init_player: BoardPiece):
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
        current_node = State(board=board, player= get_opponent(current_player))
    return value


def backpropagate(current_node, v):
    # update nodes's up to root node
    while current_node is not None:
        # update node's visits
        current_node.update_state(v)
        # set node to parent
        current_node = current_node.parent
        v = int(not v)


def expand(node):
    board = node.board
    valid_actions = get_valid_actions(board)
    # no_of_possible_children = len(valid_actions)
    valid_actions = remove_action_already_present_as_child(node, valid_actions)

    action = np.random.choice(valid_actions)
    player = node.player
    child_board = apply_player_action(board, action, player, True)
    child = State(child_board, get_opponent(player), parent=node, action=action)
    node.children[action] = child
    # if no_of_possible_children == len(node.children):
    #     node.is_fully_expanded = True
    return child
    # debugging
    # print('Should not get here!!!')


def remove_action_already_present_as_child(node, valid_actions):
    children_actions = [*node.children]
    return np.setdiff1d(valid_actions, children_actions)


