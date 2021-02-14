import numpy as np

from scipy import signal
from abc import abstractmethod, ABCMeta


class FiniteStateMachine:
    def __init__(self, table, init_state=0):
        self._table = table
        self.state = init_state
    
    def update(self, color):
        new_color, direction, self.state = self._table[self.state][int(color)]
        return new_color, direction


class AbstractAgent:
    def __init__(self):
        self._delta_rotation = {
            'N': (0, -1),
            'S': (0, 1),
            'W': (-1, 0),
            'E': (1, 0)
        }

        self._delta_translation = {
            'F': (1, 0),
            'B': (-1, 0),
            'L': (0, 1),
            'R': (0, -1),
            'N': (0, 0)
        }

        self._directions = 'NESW'   # North, East, South, West
        self._rotations = 'LNRU'    # Left, None, Right, U-turn
        self._translations = 'FBLR'  # Forward, backward, left, right

        self.direction = 'N'        

        self.state_machine = FiniteStateMachine(
            self.get_rule()
        )

        self.dimx = None
        self.dimy = None
    
    def init(self, dimx, dimy, x=None, y=None):
        self.dimx = dimx
        self.dimy = dimy

        self.x = x if x else dimx//2
        self.y = y if y else dimy//2
    
    @abstractmethod
    def get_rule(self):
        pass

    def update(self, grid):
        fsm_out = self.run_fsm(grid)
        self.step(grid, fsm_out)

    def run_fsm(self, grid):
        color = grid.state[self.x, self.y]
        new_color, fsm_out = self.state_machine.update(color)
        grid.state[self.x, self.y] = new_color
        return fsm_out
    
    @abstractmethod
    def step(self, grid, fsm_out):
        pass

    def rotate(self, how):
        rotation = self._rotations.index(how)
        direction = self._directions.index(self.direction)
        direction = (direction + (rotation - 1) + 4) % 4
        self.direction = self._directions[direction]
    
    def move(self, how='F'):
        deltar_x, deltar_y = self._delta_rotation[self.direction]
        deltam_x, deltam_y = self._delta_translation[how]
        self.x = (self.x + deltam_x * deltar_x + deltam_y * deltar_y + self.dimx) % self.dimx
        self.y = (self.y - deltam_y * deltar_x + deltam_x * deltar_y + self.dimy) % self.dimy


class AbstractAbsoluteAgent(AbstractAgent):
    def step(self, grid, fsm_out):
        self.rotate('N')
        self.move(fsm_out)

class AbstractRelativeAgent(AbstractAgent):
    def step(self, grid, fsm_out):
        self.rotate(fsm_out)
        self.move('F')


class LangdonsAnt(AbstractRelativeAgent):
    def get_rule(self):
        return [
            [[1, 'L', 0], [0, 'R', 0]]
        ]


class CoiledRope(AbstractRelativeAgent):
    def get_rule(self):
        return [
            [[1, 'N', 1], [1, 'L', 0]],
            [[1, 'R', 1], [0, 'N', 0]]
        ]


class ComputerArt(AbstractRelativeAgent):
    def get_rule(self):
        return [
            [[1, 'L', 0], [1, 'R', 1]],
            [[0, 'R', 0], [0, 'L', 1]]
        ]


class Fibonacci(AbstractRelativeAgent):
    def get_rule(self):
        return [
            [[1, 'L', 1], [1, 'L', 1]],
            [[1, 'R', 1], [0, 'N', 0]]
        ]


class WormTrails(AbstractRelativeAgent):
    def get_rule(self):
        return [
            [[1, 'R', 1], [1, 'L', 1]],
            [[1, 'R', 1], [0, 'R', 0]]
        ]


class StripedSpiral(AbstractRelativeAgent):
    def get_rule(self):
        return [
            [[0, 'R', 1], [0, 'L', 0]],
            [[1, 'L', 1], [0, 'R', 0]]
        ]


class RandomlyInitializedAgent(AbstractRelativeAgent):
    def __init__(self, n_states=2):
        self.n_states = n_states
        super(RandomRuleAgent, self).__init__()

    def get_rule(self):
        colors = np.random.randint(2, size=(self.n_states, 2))
        directions = np.random.randint(4, size=(self.n_states, 2))
        states = np.random.randint(self.n_states, size=(self.n_states, 2))

        rule = []
        for i, (color, direction, state) in enumerate(zip(colors, directions, states)):
            color_rule = []
            for j in range(2):
                while color[j] == j and state[j] == i:
                    state[j], color[j] = np.random.randint(self.n_states), np.random.randint(2)
                color_rule.append(
                    [color[j], self._rotations[direction[j]], state[j]],
                )
            rule.append(color_rule)
        
        return rule
