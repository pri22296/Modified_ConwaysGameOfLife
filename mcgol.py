"""Modified version of Conway's Game of Life

Usually S[t] = CGOL(S[t-1]) where CGOL is the standard function of Conway's game of life,
and S is a binary matrix denoting if a cell is alive or dead.

Here it is modified to S[t] = Min(1, CGOL(S[t-1]) + L[t-1]) * (I - D[t-1])
where L and D are two matrices which denote Lifezone and a Deadzone.

All cells in a deadzone always remain dead and all cells in a alive zone always remain alive.
The matrices L, D can also change every iteration. In this implementation it's static but there is
no reason it can't change with time.
"""

import imageio
import cv2
import numpy as np
import scipy.ndimage as ndi

from scipy import signal


import pygame
import numpy as np

def get_kernel():
    gol_kernel = np.ones((3, 3))
    gol_kernel[1][1] = 0
    return gol_kernel

def next_state(x):
    gol_kernel = get_kernel()
    num_neighbours = signal.convolve2d(x, gol_kernel, boundary='wrap', mode='same')
    return np.where((num_neighbours==3) | (np.isin(num_neighbours, [2,3]) & (x == 1)), 1, 0)

def get_image_binary_matrix(path, dimx, dimy, thresh=128):
    # Use this to get an image as initial state, or a lifezone or a deadzone.
    im = imageio.imread(path, as_gray=True)
    im = cv2.resize(im, (dimx, dimy))
    im = np.where(im>thresh if thresh else np.mean(im), 1, 0)
    return im

def get_deadzone(dimx, dimy):
    # The state is always dead in this zone.
    return np.zeros(shape=(dimx, dimy))

def get_lifezone(dimx, dimy):
    # The state is always alive in this zone.
    return np.zeros(shape=(dimx, dimy))

def get_initial_state(dimx, dimy):
    return np.random.randint(2, size=(dimx,dimy))

def make_img(state, cellsize):
    state = ndi.zoom(state, cellsize, order=0)
    state = np.expand_dims(state, axis=-1)
    
    img = np.concatenate([state, state, state], axis=-1)
    return (img*255).astype('uint8')

def main(dimx, dimy, cellsize):
    pygame.init()
    surface = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("John Conway's Game of Life")
    
    deadzone = get_deadzone(dimx, dimy)
    lifezone = get_lifezone(dimx, dimy)
    state = get_initial_state(dimx, dimy)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        pygame.display.update()
        pygame.surfarray.blit_array(surface, make_img(state, cellsize))
        state = np.minimum(next_state(state) + lifezone, 1)
        state = state * (1 - deadzone)

        
if __name__ == '__main__':
    main(100, 100, 8)
