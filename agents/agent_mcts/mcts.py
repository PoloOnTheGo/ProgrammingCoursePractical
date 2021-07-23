from copy import copy
from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, get_valid_actions, apply_player_action, get_opponent, \
    check_end_state, GameState

import numpy as np
from agents.agent_mcts import State


# Remark: Overall, great work! I think you implemented the MCTS algorithm very cleanly and your coding style is very
#  clean and clear. My only complaint is that you could have used OOP a bit more exhaustively, for example in the ucb1
#  function. But also in mcts.py: you are requiring redundant parameters for some functions, where you could get the same
#  information from node objects you also pass in. Actually, most of the functions below basically only operate on node
#  objects, so you could make them methods as well (or put another 'MCTS tree' class on top of the 'State' class).

# Remark: Btw, I think the performance of your agent could be improved substantially by handling terminal nodes in the
#  search tree better. Right now there is nothing in the selection/expansion process that would be affected by
#  wins/draws, i.e. you're actually selecting through nodes with a win and keep expanding them. This is especially
#  problematic if there is a loss, because future children might still contain a win for the init_player, which you
#  would detect in the calculate_value function. This way, you would overestimate the value of these nodes. And of
#  course you would create nodes and run calculations you don't need.


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
        PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column and return it as `action`
    # Remark: docstring missing
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
                             # Remark: isn't this stored in the initial_node?
    """
    for _ in range(no_of_iterations):
        current_node = select_leaf_node(initial_node)

        if current_node.visits != 0:
            # Remark: this step is unnecessary, since every rollout starts at a node which has just been expanded
            #  -> as soon as a node is added to the search tree, it gets visited as well.
            # Remark: Sorry, I think I only got why this is necessary in the rollout function. But there's no reason why
            #  the root node shouldn't be expanded in the first iteration as well (if you don't the result will never
            #  matter for the move selection, because no child of the root node will be updated)
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
    # Remark: I'm not a fan of return statements like that. I think it's better to do only one step per line in general,
    #  because it's much easier to follow the logic that way. I actually missed that you use this function recursively
    #  at first. And you should mention this recursiveness in the docstring as well.
    #  But that's very much a personal preference I think, but I wanted to alert you that there's some caution
    #  to be applied here. You can certainly afford an extra line here though, since you did things quite elegantly in
    #  general :).


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
    # Remark: you're missing the returned variable in the docstring, and you should explain that the value you're
    #  calculating is in reference to the rolledout_node, not in absolute terms
    #  Having said that, I think it's more intuitive to simulate a game and store the absolute result (i.e. winner), and
    #  calculate the respective value only where you need it (i.e. in the backpropagation step). But what you're doing
    #  is equivalent.
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
                             # Remark: that's a bit unclear, do you mean the root node of the search tree?
    :param init_player:      BoardPiece
                             Current game player taking the turn(initial player)
    :return:                 Float
                             simulation value according to the WIN of the game current player
    """
    value = 0
    if rolledout_node == init_node:
        # Remark: that should never happen, because on the first iteration you expand the root node and start the
        #  rollout from there. See the comment in tree_traversal
        return value
    current_node = copy(rolledout_node)

    while True:
        board = current_node.board
        current_player = current_node.player
        if current_node.is_terminal:
            value = calculate_value(rolledout_node, current_node, init_player)
            # Remark: since you're checking each node for a win upon creation (to get the is_terminal attribute), you
            #  could also store the winner there directly without the need for any additional calculation. This would
            #  save you one application of check_end_state, and you would only have to compare players.
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
    # Remark: you might run into problems if the node is fully extended already. I think you catch that by comparing the
    #  number of children and valid moves, but that's something to be aware of (and write in the docstring :)).
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
                 rray of valid actions
    :return:     array
                 valid actions except action_already_present_as_children
    """
    children_actions = [*node.children]
    return np.setdiff1d(valid_actions, children_actions)
