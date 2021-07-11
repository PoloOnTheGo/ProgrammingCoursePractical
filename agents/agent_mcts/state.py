import math
from copy import copy

import numpy as np

from agents.common import BoardPiece, PlayerAction, check_end_state, get_valid_actions, GameState, PLAYER1, PLAYER2, \
    apply_player_action, get_opponent


class State(object):
    def __init__(self, board: np.ndarray, player: BoardPiece = None, value: float = 0.0, visits: int = 0,
                 action: PlayerAction = 0, parent=None):
        """
        Constructor of the State object
        :param board:  np.ndarray
                       board of the state
        :param player: BoardPiece
                       the player playing on the board
        :param value:  float
                       number of wins for the game board player
        :param visits: int
                       number of visit of the node int the Monte carlo tree
        :param action: PlayerAction
                       action taken to create this board state
        :param parent: State
                       parent of this node
        """
        self.board = board.copy()
        self.children = {}
        self.value = value
        self.visits = visits
        self.player = player
        self.action = action
        self.parent = parent
        self.is_terminal = not(check_end_state(self.board, PLAYER1) == GameState.STILL_PLAYING) \
                           or not(check_end_state(self.board, PLAYER2) == GameState.STILL_PLAYING)

    def add_child(self, child) -> None:
        """
        adds a child (with action as the key) to the self node
        :param self:  State
                      node to which child is added
        :param child: State
                      child to be added
        """
        self.children[child.action] = child

    def update_state(self, value: float) -> None:
        """
        update a node by incrementing the visit by one and the value by v as input
        :param self:  State
                      node to be updated
        :param value: float
                      value of the node
        """
        self.visits += 1
        self.value += value

    def is_leaf_node(self) -> bool:
        """
        check if it is a leaf node( i.e of not fully expanded)
        :param self:  State
                      node to be checked
        :return:      bool
                      True or False whether it is leaf node
        """
        valid_actions = get_valid_actions(self.board)
        return len(self.children.values()) != len(valid_actions)

    def get_ucb1_value(self, s_p, exploration_constant):
        """
        calculate and return the ucb1 value of a node
        :param self:                    State
                                        node of which ucb1 value is calculated
        :param exploration_constant:    int
                                        exploration_constant
        :return:      float
                      ucb1 value
        """
        w_i = self.value
        s_i = self.visits
        return w_i / s_i + exploration_constant * math.sqrt(math.log(s_p) / s_i)

    def get_best_move(self, exploration_constant):
        """
        Fetch the best child node according to ucb1 value (when exploration_constant !=0) or according to the wins
        :param self:                    State
                                        node whose children are compared by ucb1 value or wins
        :param exploration_constant:    int
                                        decides whether to use ucb1 value or just the wins
        :return:                        State
                                        best child
        """
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

    def select_leaf_node(self):
        """
        checks if the given node is a leaf node i.e. whether it is not fully expanded(there are available actions still
        left to explore) or continue to move down the tree until it finds a leaf node
        :param self: State
                     current node to be checked
        :return:     State
                     Leaf node (not fully expanded tree node)
        """
        if self.is_leaf_node():
            return self
        best_node = self.get_best_move(exploration_constant=2)
        return best_node.select_leaf_node()

    def remove_action_already_present_as_child(self, valid_actions):
        """
        Remove the valid action options if that is already played
        :param self: State
                     The node whose children are checked
        :param valid_actions: array
                     array of valid actions
        :return:     array
                     valid actions except action_already_present_as_children
        """
        children_actions = [*self.children]
        return np.setdiff1d(valid_actions, children_actions)

    def expand(self):
        """
        Expand the node with a randomly chosen child
        :param self: State
                     The node where a new randomly chosen child is added
        :return:     State
                     created child node
        """
        board = self.board
        valid_actions = get_valid_actions(board)
        valid_actions = self.remove_action_already_present_as_child(valid_actions)

        action = np.random.choice(valid_actions)
        player = self.player
        child_board = apply_player_action(board, action, player, True)
        child = State(child_board, get_opponent(player), parent=self, action=action)
        self.add_child(child)
        return child

    def backpropagate(self, v):
        """
        Backpropagate the simulated value and the visit till the root node and

        :param self:             State
                                 the node from where the backpropagation starts
        :param v:                int
                                 simulation value
        """
        # update nodes's up to root node
        current_node = self
        while current_node is not None:
            # update node's visits
            current_node.update_state(v)
            # set node to parent
            current_node = current_node.parent
            v = int(not v)

    def rollout(self, init_node, init_player: BoardPiece):
        """
        Calculate and return the value of the node according to the WIN of the game current player by simulating
        randomly until it reaches terminal state. Returns 0 when it is the root node, since it does not need simulation.

        :param self:             State
                                 the node that is simulated
        :param init_node:        State
                                 initial node
        :param init_player:      BoardPiece
                                 Current game player taking the turn(initial player)
        :return:                 Float
                                 simulation value according to the WIN of the game current player
        """
        value = 0
        if self == init_node:
            return value
        current_node = copy(self)

        while True:
            board = current_node.board
            current_player = current_node.player
            if current_node.is_terminal:
                value = self.calculate_value(current_node, init_player)
                break
            valid_actions = get_valid_actions(board)
            selected_action = np.random.choice(valid_actions)
            board = apply_player_action(board, selected_action, current_player, True)
            current_node = State(board=board, player=get_opponent(current_player))
        return value

    def calculate_value(self, terminal_node, init_player: BoardPiece):
        """
        Calculate the value of the node according to the WIN of the game current player. If PLAYER1 wins so each visited
        PLAYER2 node's win count is incremented. This flip is due to the fact that each node’s statistics are used for
        it's parent node’s choice, not its own

        :param self:             State
                                 the node that is simulated
        :param terminal_node:    State
                                 the terminal or end state
        :param init_player:      BoardPiece
                                 Current game player taking the turn(initial player)
        """
        value = 0
        if check_end_state(terminal_node.board, init_player) == GameState.IS_WIN \
                and self.player == get_opponent(init_player):
            value = 1
        return value
