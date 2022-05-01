import numpy as np
import time

import matplotlib

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
from numpy import array
from matplotlib import animation
from numpy.random import random, normal

from explauto_master.explauto import SensorimotorModel
from explauto_master.explauto.utils import rand_bounds, prop_choice

from exploration_algorithms.environment import *
from exploration_algorithms.learning_module import LearningModule, HierarchicalLearningModule

grid_size = 10


def compute_explo(data, mins, maxs, gs=100):
    n = len(mins)
    if len(data) == 0:
        return 0
    else:
        assert len(data[0]) == n
        epss = (maxs - mins) / gs
        grid = np.zeros([gs] * n)
        for i in range(len(data)):
            idxs = np.array((data[i] - mins) / epss, dtype=int)
            idxs[idxs>=gs] = gs-1
            idxs[idxs<0] = 0
            grid[tuple(idxs)] = grid[tuple(idxs)] + 1
        grid[grid > 1] = 1
        return np.sum(grid)


def display_movement(fig, ax, environment, iterations=50):

    lines, = ax.plot([], [], lw=3)
    background = None

    def animate(i):
        fig.canvas.restore_region(background)
        lines = environment.env.plot_update(ax, i)
        for line in lines:
            ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)
        return lines,

    def init():
        fig.canvas.draw()
        ax.set_aspect('equal')
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        lines.set_data([], [])
        return lines,

    fig.canvas.draw()

    background = fig.canvas.copy_from_bbox(ax.bbox)
    ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=iterations)
    return ani
