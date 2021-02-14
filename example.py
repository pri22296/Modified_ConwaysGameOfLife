from automata.agent import *
from automata.grid import *
from automata.automata import CellularAutomata

ca = CellularAutomata(
    100, 100, grid=ConwaysGameOfLifeGrid(n_states=2),
    #agents=[FourStateLangdonsAnt() for i in range(1)]
)

ca.show(8)
