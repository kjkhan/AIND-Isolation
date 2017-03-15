"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import isolation

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            #_, move = self.minimax(game, self.search_depth)
            _, move = self.alphabeta(game, self.search_depth)
            return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Instead of duplicating the code with minor changes for maximizing vs
        # minimizing player, the min_max function is set here with best score
        # intialized to its repective limit
        if maximizing_player:
            max_min_fn = max
            best_score = float("-inf")
        else:
            max_min_fn = min
            best_score = float("+inf")

        # get a listing of possible moves
        legal_moves = game.get_legal_moves()

        # check to see if a terminal leaf is reached
        if not legal_moves:
            return best_score, (-1,-1)

        # At depth of 1, the bottom of the depth-limited search has been reached
        if depth == 1:

            # create a tuple listing of scores and moves, e.g. [(4.0, (2,3))]
            scored_moves = [(self.score(game.forecast_move(m), self), m) for m in legal_moves]

            # pass back up the max or min scores of the children
            return max_min_fn(scored_moves)

        else:

            best_move = None

            for move in legal_moves:
                # apply the next move to a copy of the game board
                new_game = game.forecast_move(move)

                # recursively call function unless terminal state or depth 1 reached
                score, _ = self.minimax(new_game, depth-1, not maximizing_player)

                if max_min_fn(score, best_score) == score:
                    best_score = score
                    best_move = move

            return best_score, best_move
  

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # set intial values for best score and move depending on max or min level
        if maximizing_player:
            best_score = float("-inf")
            best_move = (-1,-1)
        else:
            best_score = float("+inf")
            best_move = (-1,-1)

        # get a listing of possible moves
        legal_moves = game.get_legal_moves()

        # check to see if there are no more moves, indicating a terminal leaf
        if not legal_moves:
            return best_score, best_move 

        # iterate through legal moves, return early if remaining moves could be pruned
        for move in legal_moves:

            # at the bottom leaf
            if depth == 1:

                # calcuate score for this move
                score = self.score(game.forecast_move(move), self)

            # at a higher tier leaf; copy board and use recursion to move one level lower
            else:
                # create a new game board to next move
                new_game = game.forecast_move(move)

                # go down one more leaf using recursion, return score of this leaf
                score, _ = self.alphabeta(new_game, depth-1, alpha, beta, not maximizing_player)

            # in a maximizing layer
            if maximizing_player:
        
                # save the highest score and move from this leaf
                if score > best_score:
                    best_score = score
                    best_move = move

                # if higher or equal to beta, return early (pruning remaining iterations in this leaf)
                if best_score >= beta:
                    return best_score, best_move

                # adjust the value used for the next minimizing recursion
                alpha = max(alpha, best_score)
            
            # in a minimizing layer
            else:
                
                # save the lowest score and move from this leaf
                if score < best_score:
                    best_score = score
                    best_move = move

                # if lower or equal to alpha, return early (pruning remaining iterations in this leaf)
                if best_score <= alpha:
                    return best_score, best_move

                # adjust the value used for the next maximizing recursion
                beta = min(beta, best_score)

        return best_score, best_move