import numpy as np

from abc import ABCMeta, abstractmethod
from scipy import signal


class AbstractGrid:
    def __init__(self, n_states=2):
        self.dimx = None
        self.dimy = None
        self.n_states = n_states

    def init(self, dimx, dimy):
        self.dimx = dimx
        self.dimy = dimy
        self.state = self.get_initial_state()

    def update(self):
        pass

    @abstractmethod
    def get_initial_state(self):
        pass


class KernelBasedAbstractGrid(AbstractGrid):
    def __init__(self, n_states=2):
        super(KernelBasedAbstractGrid, self).__init__(n_states)
        self.kernel = self.get_kernel()
        self.rule = self.get_rule()
    
    def update(self):
        state_kernel_out = self.apply_state_kernel()
        if self.n_states > 2:
            alive_kernel_out = self.apply_alive_kernel()
        else: 
            alive_kernel_out = state_kernel_out
        self.state = self.apply_rule(alive_kernel_out, state_kernel_out)
    
    def apply_state_kernel(self):
        return signal.convolve2d(
            self.state, self.kernel, boundary='wrap', mode='same'
        )
    
    def apply_alive_kernel(self):
        return signal.convolve2d(
            np.where(self.state > 0, 1, 0), self.kernel, boundary='wrap', mode='same'
        )

    def apply_rule(self, alive_kernel_out, state_kernel_out):
        birth = np.isin(alive_kernel_out, self.rule['birth'])
        survive = np.isin(alive_kernel_out, self.rule['survive'])
        alive = self.state > 0
        return np.select(
            [~alive & birth, survive & alive],
            [np.rint(state_kernel_out/alive_kernel_out), self.state],
            default=0
        )
    
    @abstractmethod
    def get_kernel(self):
        pass

    @abstractmethod
    def get_rule(self):
        pass


class BlankGrid(AbstractGrid):
    def get_initial_state(self):
        return np.zeros(shape=(self.dimx, self.dimy))


class RandomlyInitializedGrid(AbstractGrid):
    def get_initial_state(self):
        nonzero = np.random.randint(2, size=(self.dimx, self.dimy))
        state = np.random.randint(1, self.n_states, size=(self.dimx, self.dimy))
        return np.where(nonzero == 1, state, 0)


class ConwaysGameOfLifeGrid(KernelBasedAbstractGrid, RandomInitializedGrid):
    def get_kernel(self):
        return np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

    def get_rule(self):
        return {
            'birth': [3],
            'survive': [2, 3]
        }

