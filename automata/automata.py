"""Modified version of Conway's Game of Life

Usually S[t] = CGOL(S[t-1]) where CGOL is the standard function of Conway's game of life,
and S is a binary matrix denoting if a cell is alive or dead.

Here it is modified to S[t] = Min(1, CGOL(S[t-1]) + L[t-1]) * (I - D[t-1])
where L and D are two matrices which denote Lifezone and a Deadzone.

All cells in a deadzone always remain dead and all cells in a alive zone always remain alive.
The matrices L, D can also change every iteration. In this implementation it's static but there is
no reason it can't change with time.
"""

import abc
import imageio
import cv2
import numpy as np
import scipy.ndimage as ndi

import pygame
import numpy as np

from .grid import BlankGrid
from .agent import AbstractAgent


class CellularAutomata(metaclass=abc.ABCMeta):
    def __init__(self, dimx, dimy, grid=None, agents=None):
        self.dimx = dimx
        self.dimy = dimy

        self.grid = grid if grid else BlankGrid()
        self.agents = agents if agents else []

        if isinstance(self.agents, AbstractAgent):
            self.agents = [self.agents]
        
        self.grid.init(dimx, dimy)
        for agent in self.agents:
            agent.init(dimx, dimy)
        
        self._color_space = self._get_colors()
    
    def update(self):
        self.grid.update()
        for agent in self.agents:
            agent.update(self.grid)

    def get_image_binary_matrix(self, path, thresh=128):
        # Use this to get an image as initial state, or a lifezone or a deadzone.
        im = imageio.imread(path, as_gray=True)
        im = cv2.resize(im, (self.dimx, self.dimy))
        im = np.where(im>thresh if thresh else np.mean(im), 1, 0)
        return im
    
    def _get_colors(self):
        import colorsys
        colors=[[0, 0, 0]]
        for i in np.arange(0., 360., 360. / (self.grid.n_states - 1)):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append([int(255*rgb[0]), int(255*rgb[1]), int(255*rgb[2])])
        return colors

    def make_img(self, cellsize):
        state = ndi.zoom(self.grid.state, cellsize, order=0)
        state = np.expand_dims(state, axis=-1).astype('uint8')

        colors = self._color_space

        r = np.select(
            [state == i for i in range(1, self.grid.n_states)],
            [colors[i][0] for i in range(1, self.grid.n_states)]
        )
        g = np.select(
            [state == i for i in range(1, self.grid.n_states)],
            [colors[i][1] for i in range(1, self.grid.n_states)]
        )
        b = np.select(
            [state == i for i in range(1, self.grid.n_states)],
            [colors[i][2] for i in range(1, self.grid.n_states)]
        )

        img = np.concatenate([r, g, b], axis=-1)
        return img

    def show(self, cellsize):
        pygame.init()
        surface = pygame.display.set_mode((self.dimx*cellsize, self.dimy*cellsize))
        pygame.display.set_caption("John Conway's Game of Life")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            pygame.display.update()
            pygame.surfarray.blit_array(surface, self.make_img(cellsize))

            self.update()
