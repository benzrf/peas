"""
    Module implements a checkers game with alphabeta search and input for a heuristic
"""

### IMPORTS ###
import random
import copy

import numpy as np

### SHORTCUTS ###
inf = float('inf')


### CONSTANTS ###

EMPTY = 0
WHITE = 1
BLACK = 2
MAN   = 4
KING  = 8
FREE  = 16 

NUMBERING = np.array([[ 4,  0,  3,  0,  2,  0,  1,  0],
                      [ 0,  8,  0,  7,  0,  6,  0,  5],
                      [12,  0, 11,  0, 10,  0,  9,  0],
                      [ 0, 16,  0, 15,  0, 14,  0, 13],
                      [20,  0, 19,  0, 18,  0, 17,  0],
                      [ 0, 24,  0, 23,  0, 22,  0, 21],
                      [28,  0, 27,  0, 26,  0, 25,  0],
                      [ 0, 32,  0, 31,  0, 30,  0, 29]])

CENTER = [(2,2), (2,4), (3,3), (3,5), (4,2), (4,4), (5,3), (5,5)]
EDGE = [(0,0), (0,2), (0,4), (0,6), (1,7), (2,0), (3,7), (4,0), (5,7), (6,0), (7,1), (7,3), (7,5), (7,7)]
SAFEEDGE = [(0,6), (1,7), (6,0), (7,1)]
                 
INVNUM = dict([(n, tuple(a[0] for a in np.nonzero(NUMBERING == n))) for n in range(1, NUMBERING.max() + 1)])


### FUNCTIONS ###

def alphabeta(node, heuristic, player_max=True, depth=4, alpha=-inf, beta=inf):
    """ Performs alphabeta search.
        From wikipedia pseudocode.
    """
    if depth == 0 or node.game_over():
        return heuristic(node)
    if player_max:
        for move in node.moves():
            alpha = max(alpha, alphabeta(node.copy_and_play(move), heuristic, not player_max, depth-1, alpha, beta))
            if beta <= alpha:
                break # Beta cutoff
        return alpha
    else:
        for move in node.moves():
            beta = min(beta, alphabeta(node.copy_and_play(move), heuristic, not player_max, depth-1, alpha, beta))     
            if beta <= alpha:
                break # Alpha cutoff
        return beta

def gamefitness(game):
    """ Returns the fitness of
        the black player. (according to {gauci2008case}) """
    counts = np.bincount(game.board.flat)
    return (100 + 2 * counts[BLACK|MAN] + 3 * counts[BLACK|KING] + 
            2 * (12 - counts[WHITE|MAN] + 3 * (12 - counts[WHITE|KING])))

### CLASSES ###

class CheckersTask(object):
    """ Represents a checkers game played by an evolved phenotype against
        a fixed opponent.
    """
    def __init__(self):
        pass

    def evaluate(self, network):
        # Setup
        game = Checkers()
        player = HeuristicOpponent(NetworkHeuristic(network))
        opponent = HeuristicOpponent(SimpleHeuristic())
        # Play the game
        fitness = [gamefitness(game)] * 100
        current, next = player, opponent
        i = 0
        while not game.game_over():
            i += 1
            move = current.pickmove(game)
            game.play(move)
            current, next = next, current
            fitness.pop(0)
            fitness.append(gamefitness(game))
        print "Game finished in %d turns." % (i,)
        if game.to_move == WHITE:
            fitness.append(30000)
            print "BLACK won."
        score = sum(fitness)
        return {'fitness':score}

    def solve(self, network):
        return self.evaluate(network)['fitness'] > 30000

    def visualize(self, network):
        pass

        
class HeuristicOpponent(object):
    """ Opponent that utilizes a heuristic combined with alphabeta search
        to decide on a move.
    """
    def __init__(self, heuristic, search_depth=4):
        self.search_depth = search_depth
        self.heuristic = heuristic
    
    def pickmove(self, board):
        player_max = board.to_move == BLACK
        bestmove = None
        bestval = -inf if player_max else inf
        for move in board.moves():
            val = alphabeta(board.copy_and_play(move), self.heuristic.evaluate, depth=self.search_depth, player_max=player_max)
            if player_max and val > bestval or not player_max and val < bestval:
                bestval = val
                bestmove = move
        print bestval, bestmove
        return bestmove

class SimpleHeuristic(object):
    """ Simple piece/position counting heuristic, adapted from simplech
    """
    def evaluate(self, game):
        if game.game_over():
            return -5000 if game.to_move == BLACK else 5000
        board = game.board
        counts = np.bincount(board.flat)

        turn = 2;     # color to move gets +turn
        brv = 3;      # multiplier for back rank
        kcv = 5;      # multiplier for kings in center
        mcv = 1;      # multiplier for men in center
        mev = 1;      # multiplier for men on edge
        kev = 5;      # multiplier for kings on edge
        cramp = 5;    # multiplier for cramp
        opening = -2; # multipliers for tempo
        midgame = -1;
        endgame = 2;
        intactdoublecorner = 3;

        nwm = counts[WHITE|MAN]
        nwk = counts[WHITE|KING]
        nbm = counts[BLACK|MAN]
        nbk = counts[BLACK|KING]

        vb = (100 * nbm + 130 * nbk)
        vw = (100 * nwm + 130 * nwk)
        
        val = (vb - vw) + (250 * (vb-vw))/(vb+vw); #favor exchanges if in material plus

        nm = nwm + nbm
        nk = nwk + nbk

        val += turn if game.to_move == BLACK else -turn

        if board[4][0] == (BLACK|MAN) and board[5][1] == (WHITE|MAN):
            val += cramp
        if board[3][7] == (WHITE|MAN) and board[2][6] == (BLACK|MAN):
            val -= cramp

        # Back rank guard
        code = 0
        if (board[0][0] & MAN): code += 1
        if (board[0][2] & MAN): code += 2
        if (board[0][4] & MAN): code += 4
        if (board[0][6] & MAN): code += 8
        if code == 1:
            backrankb = -1
        elif code in (0, 3, 9):
            backrankb = 0
        elif code in (2, 4, 5, 7, 8):
            backrankb = 1
        elif code in (6, 12, 13):
            backrankb = 2
        elif code == 11:
            backrankb = 4
        elif code == 10:
            backrankb = 7
        elif code == 15:
            backrankb = 8
        elif code == 14:
            backrankb = 9
        
        code = 0
        if (board[7][1] & MAN): code += 8
        if (board[7][3] & MAN): code += 4
        if (board[7][5] & MAN): code += 2
        if (board[7][7] & MAN): code += 1

        if code == 1:
            backrankw = -1
        elif code in (0, 3, 9):
            backrankw = 0
        elif code in (2, 4, 5, 7, 8):
            backrankw = 1
        elif code in (6, 12, 13):
            backrankw = 2
        elif code == 11:
            backrankw = 4
        elif code == 10:
            backrankw = 7
        elif code == 15:
            backrankw = 8
        elif code == 14:
            backrankw = 9
        
        val += brv * (backrankb - backrankw)

        if board[0][6] == BLACK|MAN and (board[1][5] == BLACK|MAN or board[1][7] == BLACK|MAN):
            val += intactdoublecorner
        if board[7][1] == WHITE|MAN and (board[6][0] == WHITE|MAN or board[6][2] == WHITE|MAN):
            val -= intactdoublecorner

        bm = bk = wm = wk = 0
        for pos in CENTER:
            if board[pos] == BLACK|MAN: bm += 1
            elif board[pos] == BLACK|KING: bk += 1
            elif board[pos] == WHITE|MAN: wm += 1
            elif board[pos] == WHITE|KING: wk += 1

        val += (bm - wm) * mcv
        val += (bk - wk) * kcv

        bm = bk = wm = wk = 0
        for pos in EDGE:
            if board[pos] == BLACK|MAN: bm += 1
            elif board[pos] == BLACK|KING: bk += 1
            elif board[pos] == WHITE|MAN: wm += 1
            elif board[pos] == WHITE|KING: wk += 1

        val += (bm - wm) * mev
        val += (bk - wk) * kev

        tempo = 0
        for i in xrange(8):
            for j in xrange(8):
                if board[i,j] == BLACK|MAN:
                    tempo += i
                elif board[i,j] == WHITE|MAN:
                    tempo -= 7-i
        if nm >= 16:
            val += opening * tempo
        if 12 <= nm <= 15:
            val += midgame * tempo
        if nm < 9:
            val += endgame * tempo

        for pos in SAFEEDGE:
            if nbk + nbm > nwk + nwm and nwk < 3:
                if board[pos] == (WHITE|KING):
                    val -= 15
            
            if nwk + nwm > nbk + nbm and nbk < 3:
                if board[pos] == (BLACK|KING):
                    val += 15

        return val

class NetworkHeuristic(object):
    """ Heuristic based on feeding the board state to a neural network
    """
    def __init__(self, network):
        self.network = network

    def evaluate(self, board):
        if game.game_over():
            return -5000 if game.to_move == BLACK else 5000

        net_inputs = ((board.board == BLACK | MAN) * 0.5 +
                      (board.board == WHITE | MAN) * -0.5 +
                      (board.board == BLACK | KING) * 0.75 +
                      (board.board == WHITE | KING) * -0.75)
        # Feed twice to propagate:
        value = self.network.feed(net_inputs, add_bias=False)
        value = self.network.feed(net_inputs, add_bias=False)
        value = self.network.feed(net_inputs, add_bias=False)
        # print value
        return value[-1]

class RandomOpponent(object):
    """ An opponent that plays random moves """
    def pickmove(self, board):
        return random.choice(list(board.moves()))

class Checkers(object):
    """ Represents the checkers game(state)
    """

    def __init__(self):
        """ Initialize the game board. """
        self.board = NUMBERING.copy() #: The board state
        self.to_move = BLACK          #: Whose move it is

        tiles = self.board > 0
        self.board[tiles] = EMPTY
        self.board[:3,:] = BLACK | MAN
        # self.board[3, :] = WHITE | MAN
        self.board[5:,:] = WHITE | MAN
        self.board[np.logical_not(tiles)] = FREE
        
        self._moves = None
                       
    def moves(self):
        """ Return a list of possible moves. """
        if self._moves is not None:
            for move in self._moves:
                yield move
            return
        captures_possible = False
        pieces = []
        # Check for possible captures first:
        for n, (y,x) in INVNUM.iteritems():
            piece = self.board[y, x]
            if piece & self.to_move:
                pieces.append((n, (y, x), piece))
                for m in self.captures((y, x), piece, self.board):
                    if len(m) > 1:
                        captures_possible = True
                        yield m
        # Otherwise check for normal moves:
        if not captures_possible:
            for (n, (y, x), piece) in pieces:
                # MAN moves
                if piece & MAN:
                    nextrow = y + 1 if self.to_move == BLACK else y - 1
                    if 0 <= nextrow < 8:
                        if x - 1 >= 0 and self.board[nextrow, x - 1] == EMPTY:
                            yield (n, NUMBERING[nextrow, x - 1])
                        if x + 1 < 8 and self.board[nextrow, x + 1] == EMPTY:
                            yield (n, NUMBERING[nextrow, x + 1])
                # KING moves
                else:
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:                
                            dist = 1
                            while True:
                                tx, ty = x + (dist + 1) * dx, y + (dist + 1) * dy # Target square
                                if not ((0 <= tx < 8 and 0 <= ty < 8) and self.board[ty, tx] == EMPTY):
                                    break
                                else:
                                    yield (n, NUMBERING[ty, tx])
                                dist += 1

    
    def captures(self, (py, px), piece, board, captured=[], start=None):
        """ Return a list of possible capture moves for given piece. """
        if start is None:
            start = (py, px)
        # Look for capture moves
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                dist = 1 
                while True:
                    jx, jy = px + dist * dx, py + dist * dy # Jumped square
                    # Check if piece at jx, jy:
                    if not (0 <= jx < 8 and 0 <= jy < 8):
                        break
                    if board[jy, jx] != EMPTY:
                        tx, ty = px + (dist + 1) * dx, py + (dist + 1) * dy # Target square
                        # Check if it can be captured:
                        if ((0 <= tx < 8 and 0 <= ty < 8) and
                            ((ty, tx) == start or board[ty, tx] == EMPTY) and
                            (jy, jx) not in captured and
                            ((piece & WHITE) and (board[jy, jx] & BLACK) or
                             (piece & BLACK) and (board[jy, jx] & WHITE))
                            ):
                            # Normal pieces cannot continue capturing after reaching last row
                            if not piece & KING and (piece & WHITE and ty == 0 or piece & BLACK and ty == 7):
                                yield (NUMBERING[py, px], NUMBERING[ty, tx])
                            else:
                                for sequence in self.captures((ty, tx), piece, board, captured + [(jy, jx)], start):
                                    yield (NUMBERING[py, px],) + sequence
                        break
                    else:
                        if piece & MAN:
                            break
                    dist += 1
        yield (NUMBERING[py, px],)
                        
        
    def play(self, move):
        """ Play the given move on the board. """
        positions = [INVNUM[p] for p in move]
        (ly, lx) = positions[0]
        # Check for captures
        for (py, px) in positions[1:]:
            ydir = 1 if py > ly else -1
            xdir = 1 if px > lx else -1
            for y, x in zip(xrange(ly + ydir, py, ydir),xrange(lx + xdir, px, xdir)):
                self.board[y,x] = EMPTY
            (ly, lx) = (py, px)
        # Move the piece
        (ly, lx) = positions[0]
        (py, px) = positions[-1]
        piece = self.board[ly, lx]
        self.board[ly, lx] = EMPTY
        # Check if the piece needs to be crowned
        if piece & BLACK and py == 7 or piece & WHITE and py == 0:
            piece = piece ^ MAN | KING
        self.board[py, px] = piece

        self.to_move = WHITE if self.to_move == BLACK else BLACK
        # Cached moveset is invalidated.
        self._moves = None
        return self

    def copy_and_play(self, move):
        return self.copy().play(move)
        
    def game_over(self):
        """ Whether the game is over. """
        for move in self.moves():
            # If the iterator returns any moves at all, the game is not over.
            return False
        # Otherwise it is.
        return True

        
    def winner(self):
        """ Returns board score. """
        if not self.game_over():
            return 0.0
        else:
            return 1.0 if self.to_move == WHITE else -1.0

    def copy(self):
        new = copy.copy(self)               # Copy all.
        new.board = self.board.copy()       # Copy the board explicitly
        new._moves = copy.copy(self._moves) # Shallow copy is enough.
        return new

    def __str__(self):
        s = np.array([l for l in "-    wb  WB      "])
        s = s[self.board]
        if self.to_move == BLACK:
            s[0,7] = 'v'
        else:
            s[7,0] = '^'
        s = '\n'.join(' '.join(l) for l in s)
        return s
    
### PROCEDURE ###

if __name__ == '__main__':
    scores = []
    for i in range(5):
        game = Checkers()
        player = RandomOpponent()
        opponent = HeuristicOpponent(SimpleHeuristic())
        # Play the game
        current, next = player, opponent
        i = 0
        while not game.game_over():
            i += 1
            move = current.pickmove(game)
            game.play(move)
            current, next = next, current
        scores.append(gamefitness(game))
    print 'Score', scores