import numpy as np

from agents.agent_mcts import generate_move, State
from agents.agent_mcts.mcts import tree_traversal, select_leaf_node, expand, rollout, backpropagate
from agents.common import string_to_board, PLAYER1, PLAYER1_PRINT, PLAYER2_PRINT, PlayerAction, PLAYER2


def test_generate_mcts_move():
    """
    First checking whether the agent can return an action.
    Then it asserts the agent will producing valid move.
    Next, it will test if the agent can produce a winning move
    given a board state
    """
    print("\n 1st Test : ---------------------")
    # positive test - winning move
    from agents.common import string_to_board
    board_str = """|==============| 
|              |
|              |
|              |
|        X O   |
|    O X X O   |
|X O X O O X   |
|==============|
|0 1 2 3 4 5 6 |"""
    print(board_str)
    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 5)

    print("\n 2nd Test : ---------------------")

    board_str = """|==============| 
|              |
|              |
|              |
|        X O   |
|    O O X O   |
|X O X O X X   |
|==============|
|0 1 2 3 4 5 6 |"""

    print(board_str)
    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 4)

    print("\n 3rd Test : ---------------------")

    board_str = """|==============|
|              |
|              |
|              |
|              |
|    X X       |
|    X O O O   |
|==============|
|0 1 2 3 4 5 6 |"""

    print(board_str)
    board = string_to_board(board_str)
    action, _ = generate_move(board, PLAYER1, None)
    assert (action == 6)
    # Remark: I like those tests, maybe add a case where a direct win by the opponent needs to be blocked


def test_tree_traversal():
    """
        Testing the tree_traversal by analysing the wins, visits and the values
    """
    # Remark: this test is too messy. I get what you're doing here, but the analysis with all these printouts should be
    #  kept elsewhere.
    player = PLAYER1
    no_of_iteration = 2000

    print("\n Case 1: Analysis of immediate win(diagonal win) for player: ---------------------")
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player)
    print(board_str)
    print('Player playing as : ' + str(PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT))
    print('No of Iterations : ' + str(no_of_iteration))
    tree_traversal(no_of_iteration, init_state, player)
    assert (child_traversal_max_wins(init_state) == 5)

    # -------------------------------------------------------------------------

    print("\n Case 2: Analysis of immediate win(vertical win) for player: ---------------------")
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O O X O   |\n|X O X O X X   |\n|==============|\n|0 1 2 3 4 5 6 |"
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player)
    print(board_str)
    print('Player playing as : ' + str(PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT))
    print('No of Iterations : ' + str(no_of_iteration))
    tree_traversal(no_of_iteration, init_state, player)
    assert (child_traversal_max_wins(init_state) == 4)

    # -------------------------------------------------------------------------

    print("\n Case 3: Analysis of immediate win(vertical win) for opponent: ---------------------")
    board_str = "|==============|\n|              |\n|              |\n|              |\n|              |" \
                "\n|    X X       |\n|    X O O O   |\n|==============|\n|0 1 2 3 4 5 6 |"
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player)
    print(board_str)
    print('Player playing as : ' + str(PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT))
    print('No of Iterations : ' + str(no_of_iteration))
    tree_traversal(no_of_iteration, init_state, player)
    assert (child_traversal_max_wins(init_state) == 6)
    # Remark:There are more things that could be tested, e.g., whether you actually simulated 2000 iterations (can be
    #  checked at every level of the search tree, and if the stats are consistent (i.e. if the number of wins/games
    #  of any node is the sum of all wins/games of its children etc.)


def test_select_leaf_node():
    # Case 1: Initial state is leaf node: ---------------------"
    player = PLAYER1
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|  X O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_1_board = string_to_board(board_str)
    child_1 = State(child_1_board, player=PLAYER2, action=PlayerAction(1))
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|X   O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_2_board = string_to_board(board_str)
    child_2 = State(child_2_board, player=PLAYER2, action=PlayerAction(0))

    init_state.add_child(child_1)
    init_state.add_child(child_2)
    leaf_node = select_leaf_node(init_state)
    assert(leaf_node == init_state)

    # Case 2: Child is leaf node: ---------------------"
    player = PLAYER1
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player, visits=2000)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|  X O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_1_board = string_to_board(board_str)
    child_1 = State(child_1_board, player=PLAYER2, action=PlayerAction(1), value=182.0, visits=204)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|X   O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_2_board = string_to_board(board_str)
    child_2 = State(child_2_board, player=PLAYER2, action=PlayerAction(0), value=206.0, visits=226)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|      X X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_3_board = string_to_board(board_str)
    child_3 = State(child_3_board, player=PLAYER2, action=PlayerAction(3), value=342.0, visits=348)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|    X   X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_4_board = string_to_board(board_str)
    child_4 = State(child_4_board, player=PLAYER2, action=PlayerAction(2), value=249.0, visits=265)
    board_str = "|==============|\n|              |\n|              |\n|        X     |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_5_board = string_to_board(board_str)
    child_5 = State(child_5_board, player=PLAYER2, action=PlayerAction(4), value=342.0, visits=348)
    board_str = "|==============|\n|              |\n|              |\n|          X   |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_6_board = string_to_board(board_str)
    child_6 = State(child_6_board, player=PLAYER2, action=PlayerAction(5), value=393.0, visits=393)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X X |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_7_board = string_to_board(board_str)
    child_7 = State(child_7_board, player=PLAYER2, action=PlayerAction(6), value=194.0, visits=215)

    init_state.add_child(child_1)
    init_state.add_child(child_2)
    init_state.add_child(child_3)
    init_state.add_child(child_4)
    init_state.add_child(child_5)
    init_state.add_child(child_6)
    init_state.add_child(child_7)
    leaf_node = select_leaf_node(init_state)
    assert (leaf_node == child_7)


def test_expand():
    player = PLAYER1
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player, visits=1)
    child = expand(init_state)

    assert(len(init_state.children) == 1)

    children = init_state.children.values()
    children_iterator = iter(children)
    first_child = next(children_iterator)

    assert(first_child == child)
    # Remark: a case with only one child is a special case, I think it would be important to check whether everything
    #  still works if there are children already before expanding



def test_rollout():
    # Immediate win -----------------------------
    player = PLAYER1
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player, visits=1)

    board_str = "|==============|\n|              |\n|              |\n|          X   |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    current_node_board = string_to_board(board_str)
    current_node = State(current_node_board, player=PLAYER2, action=PlayerAction(5))

    v = rollout(current_node, init_state, player)
    assert(v == 1)

    # Immediate loss -----------------------------
    player = PLAYER1
    board_str = "|==============|\n|              |\n|              |\n|              |\n|              |" \
                "\n|    X X       |\n|    X O O O   |\n|==============|\n|0 1 2 3 4 5 6 |"
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player, visits=1)

    board_str = "|==============|\n|              |\n|              |\n|              |\n|              |" \
                "\n|    X X X     |\n|    X O O O   |\n|==============|\n|0 1 2 3 4 5 6 |"
    current_node_board = string_to_board(board_str)
    current_node = State(current_node_board, player=PLAYER2, action=PlayerAction(5))

    v = rollout(current_node, init_state, player)
    assert (v == 0)
    # Remark: a random game is hard to test of course, and I think what you're doing here is fine because it's a quite easy case.
    #         Just for the sake of illustration (and I know that it's a bit overkill here, but maybe you'll find it useful):
    #         I think this is a case where TDD proper would lead to code that is easier to test (I'm assuming you didn't write
    #         tests first for this function, sorry if I'm wrong - but if you did, this could be an even more useful tangent).
    #         If you would write tests first, you would be confronted with the question what the random game is supposed to do,
    #         of course,  and - and this is the crucial point - how you would know that your random gane is doing what it should.
    #         So you could, for example, check whether the player changes on every move or if the board changes on every move
    #         and whether it changes in the right way (only one column etc).
    #         Once you've written a test for that, your implementation would have to follow this demand. Overall, your code
    #         would thus be super easy to test and you could do more than just checking whether the output is True or False
    #         (which you would know anyway because using the function wouldn't be possible in this case).


def test_back_propagate():
    player = PLAYER1
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|    O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    board = string_to_board(board_str)
    init_state = State(board.copy(), player=player, visits=1)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|  X O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_1_board = string_to_board(board_str)
    child_1 = State(child_1_board, player=PLAYER2, action=PlayerAction(1), parent=init_state)
    board_str = "|==============|\n|              |\n|              |\n|              |\n|        X O   |" \
                "\n|X   O X X O   |\n|X O X O O X   |\n|==============|\n|0 1 2 3 4 5 6 |"""
    child_2_board = string_to_board(board_str)
    child_2 = State(child_2_board, player=PLAYER2, action=PlayerAction(0), parent=init_state)

    init_state.add_child(child_1)
    init_state.add_child(child_2)

    backpropagate(child_1, 1)
    assert(child_1.value == 1.0)
    assert(child_1.visits == 1)
    assert(child_2.visits == 0)
    assert(child_2.value == 0.0)
    assert(init_state.value == 0.0)
    assert(init_state.visits == 2)


def child_traversal_max_wins(init_state: State) -> PlayerAction:
    children_wins = np.zeros(len(init_state.children.values()))
    for child in init_state.children.values():
        children_wins[child.action] = child.value
        print("Action: " + str(child.action) + ", Wins:" + str(child.value) + " , Visits:", str(child.visits)
              + ", Value:" + (str(child.value / child.visits)))

    return PlayerAction(np.argmax(children_wins))
    # Remark: that's not much of a test
