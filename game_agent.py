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

class TimeExpiring(Exception):
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

    # select which heuristic to use here, current choices:
    #    warnsdorf, modified_warnsdorf, improved_keepclose
    HEURISTIC = 'improved_keepclose'

    # return with respective score if game is over
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # WARNSDORF:  Uses the principle behind Warnsdorf's Rule for the knight tour
    # problem, which chooses a move with the fewest onward moves to conserve 
    # high-onward-move choices for later in the game when these choices are
    # diminished.  To do this, the heuristic takes the number of legal moves
    # currently available and multiplies it by -1.0; this way, the higher the
    # value, the fewer moves available.  It ignores the other player with one
    # exception: if only one move is available, avoid this move if the opponent
    # could also move there to sidestep from being blocked in.
    if HEURISTIC == "warnsdorf":
        own_moves = game.get_legal_moves(player)
        opp_moves = game.get_legal_moves(game.get_opponent(player))

        score = -1.0 * float(len(own_moves))

        if len(own_moves) == 1:
            if own_moves in opp_moves:
                score = -10.0

        return float(score)

    #  MODIFIED WARNSDORF:  A variation of the Warnsdorf heuristic above that
    # now also deducts the opponent's legal moves from the score.  This way,
    # some of the adversarial nature of isolation is incorporated into original
    # rule intended for non-adversarial "knight's tour."
    elif HEURISTIC == "modified_warnsdorf":
        own_moves = game.get_legal_moves(player)
        opp_moves = game.get_legal_moves(game.get_opponent(player))

        score = -1.0 * float(len(own_moves))

        if len(own_moves) == 1:
            if own_moves in opp_moves:
                score = -10.0

        return float(score - len(opp_moves))

    # IMPROVED KEEPCLOSE:  This is a variation of the "improved" heuristic that
    # now also minimizes the distance from the opponent, in essence,  keeping
    # the enemy close. The distance is calculated using an approximation of the
    # Pythogream Theorem and is divided by 2.0 to lower its impact on the final
    # score. (An optimal distance factor can likely be found with more study)
    elif HEURISTIC == "improved_keepclose":
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

        c_own, r_own = game.get_player_location(player)
        c_opp, r_opp = game.get_player_location(game.get_opponent(player))

        # get distance using hypothenuse approximation function 
        length = abs(c_opp-c_own)
        height = abs(r_opp-r_own)
        distance = max(length, height) + 0.5*min(length, height)

        return float(own_moves - opp_moves - distance/2.0)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. 

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
        self.TIMER_MARGIN = 10

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

        # return immediately if there are no legal moves
        if not legal_moves:
            return (-1,-1)

        # select initial move; a corner position is chosen to minimize the number
        # of options of legal moves from the start.  This follows Warnsdorf's
        # Rule, which seeks out moves with the fewest onward moves to save
        # higher-onward-moves spots for later in the game 
        if game.move_count <= 1:

            move = (game.width-1, game.height-1)
            if move in legal_moves:
                return move
            else:
                return (0,0)

        # set default move, to be updated in try/except block
        move = (-1,-1)

        try:
            #  call minimax or alphabeta with class defined search depth
            if self.method == "minimax":
                method = self.minimax
            elif self.method == "alphabeta":
                method = self.alphabeta
            else:
                raise NotImplementedError
                
            # use iterative deepening if iterative flag set
            # start search at depth 1 and continue until time is about expire
            if self.iterative:

                # set initial depth
                depth = 1

                # loop until time runs low
                while self.time_left() > self.TIMER_MARGIN:

                    # get score and move for this depth
                    score, move = method(game, depth)

                    # if terminal move reached, break out of the loop
                    if score == float("inf") or score == float("-inf"):
                        break
                    
                    depth += 1

            # use standard depth-first search to traverse tree
            else:
                _, move = method(game, self.search_depth)

        # minimax and alphabet throws this exception if it is in the middle of a
        # search and time begins to run out.  The last good move is saved in 
        # "move" from the while loop above, which is then returned  
        except TimeExpiring:
            return move

        except Timeout:
            pass

        # Return the best move from the last completed search iteration
        return move

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

        # if running low on time, raise flag to break loop
        if self.time_left() < self.TIMER_MARGIN:
            raise TimeExpiring()

        # at lowest depth (terminal condition), end recursion, and return score
        if depth == 0:
            return self.score(game, self), game.get_player_location(game.active_player)

        # at a higher depth, recursively iterate though legal moves
        else:
            # set initial values for max and min terminal leafs
            best_move = (-1,-1)
            if maximizing_player:
                best_score = float("-inf")
            else:
                best_score = float("+inf")

            # get a listing of possible moves
            legal_moves = game.get_legal_moves()

            # check to see if a terminal leaf is reached
            if not legal_moves:
                return best_score, best_move

            for move in legal_moves:

                # copy game board with new move, use recursion to go down level
                new_game = game.forecast_move(move)
                score, m = self.minimax(new_game, depth-1, not maximizing_player)

                if m == (-2,-2):
                    return score, move

                # save best score and move
                if (maximizing_player and score > best_score) \
                   or (not maximizing_player and score < best_score):
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

        # if running low on time, raise flag to break loop
        if self.time_left() < self.TIMER_MARGIN:
            raise TimeExpiring()

        # at lowest depth (terminal condition), end recursion, and return score
        if depth == 0:
            return self.score(game, self), game.get_player_location(game.active_player)

        # at a higher depth, recursively iterate though legal moves
        else:
            # set initial values for max and min terminal leafs
            best_move = (-1,-1) 
            if maximizing_player:
                best_score = float("-inf") 
            else:
                best_score = float("+inf") 

            # get a listing of possible moves
            legal_moves = game.get_legal_moves()

            # check for terminal condition: no more moves; return limit values
            if not legal_moves:
                return best_score, best_move

            for move in legal_moves:

                # copy game board with new move, use recursion to go down level
                new_game = game.forecast_move(move)
                score, _ = self.alphabeta(new_game, depth-1, alpha, beta, not maximizing_player)

                if maximizing_player:

                    # save best score and move
                    if score > best_score:
                        best_score = score
                        best_move = move

                    # if better than beta, return early (pruning other moves)
                    if best_score >= beta:
                        return best_score, best_move

                    # adjust the value used for the next minimizing recursion
                    alpha = max(alpha, best_score)
                
                else:

                    # save best score and move
                    if score < best_score:
                        best_score = score
                        best_move = move

                    # if better than alpha, return early (pruning other moves)
                    if best_score <= alpha:
                        return best_score, best_move

                    # adjust the value used for the next maximizing recursion
                    beta = min(beta, best_score)

            return best_score, best_move