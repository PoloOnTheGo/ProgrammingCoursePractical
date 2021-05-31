import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2
from agents.common import initialize_game_state
import timeit


def test_initialize_game_state():
    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.common import pretty_print_board

    int_board = initialize_game_state()
    int_board[0, 0] = PLAYER1
    int_board[0, 1] = PLAYER2
    print(int_board)
    ret = pretty_print_board(int_board)

    assert isinstance(ret, str)
    assert ret == """|==============|
|              |
|              |
|              |
|              |
|              |
|X O           |
|==============|
|0 1 2 3 4 5 6 |"""


def test_string_to_board():
    from agents.common import string_to_board
    board_str = """|==============| 
|              |
|              |
|              |
|              |
|      X       |
|X O X O       |
|==============|
|0 1 2 3 4 5 6 |"""

    ret = string_to_board(board_str)
    int_board = initialize_game_state()
    int_board[0, 0] = PLAYER1
    int_board[0, 1] = PLAYER2
    int_board[0, 2] = PLAYER1
    int_board[0, 3] = PLAYER2
    int_board[1, 3] = PLAYER1
    assert isinstance(ret, np.ndarray)
    assert (ret.dtype == BoardPiece)
    assert ((int_board == ret).all())


def test_apply_player_action():
    from agents.common import apply_player_action

    int_board = initialize_game_state()
    int_board[0, 0] = PLAYER1
    int_board[0, 1] = PLAYER2
    int_board[0, 2] = PLAYER1
    int_board[0, 3] = PLAYER2
    int_board[1, 3] = PLAYER1

    ret = apply_player_action(int_board, 0, PLAYER1)

    assert (ret.dtype == BoardPiece)
    assert ((int_board == ret).all())

    ret = apply_player_action(int_board.copy(), 0, PLAYER1, True)
    assert (ret.dtype == BoardPiece)
    assert ((int_board != ret).any())


def test_connected_four():
    from agents.common import connected_four

    int_board = initialize_game_state()
    int_board[5, 0] = PLAYER1
    int_board[5, 1] = PLAYER1
    int_board[5, 2] = PLAYER1
    int_board[5, 3] = PLAYER1

    int_board[4, 0] = PLAYER2
    int_board[4, 1] = PLAYER2
    int_board[4, 2] = PLAYER2

    assert (connected_four(int_board, PLAYER1))

    int_board = initialize_game_state()
    int_board[2, 4] = PLAYER2
    int_board[3, 4] = PLAYER2
    int_board[4, 4] = PLAYER2
    int_board[5, 4] = PLAYER2

    int_board[3, 5] = PLAYER1
    int_board[4, 5] = PLAYER1
    int_board[5, 5] = PLAYER1

    assert (connected_four(int_board, PLAYER2))

    int_board = initialize_game_state()
    int_board[5, 0] = PLAYER1
    int_board[4, 1] = PLAYER1
    int_board[3, 2] = PLAYER1
    int_board[2, 3] = PLAYER1

    int_board[5, 6] = PLAYER2
    int_board[5, 5] = PLAYER2
    int_board[5, 4] = PLAYER2

    assert (connected_four(int_board, PLAYER1))

    assert (connected_four(int_board.T, PLAYER1))

    int_board = initialize_game_state()
    int_board[0, 1] = PLAYER1
    int_board[1, 2] = PLAYER1
    int_board[2, 3] = PLAYER1
    int_board[3, 4] = PLAYER1

    int_board[5, 6] = PLAYER2
    int_board[5, 5] = PLAYER2
    int_board[5, 4] = PLAYER2

    assert (connected_four(int_board, PLAYER1, 2))

    assert (connected_four(int_board.T, PLAYER1))

    number = 10 ** 4

    res = timeit.timeit("connected_four_iter(board, player)",
                        setup="connected_four_iter(board, player)",
                        number=number,
                        globals=dict(connected_four_iter=connected_four,
                                     board=int_board,
                                     player=BoardPiece(1)))
    print(f"Python iteration-based: {res / number * 1e6 : .1f} us per call")


def test_check_end_state():
    pass