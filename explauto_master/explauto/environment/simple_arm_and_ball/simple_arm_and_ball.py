import numpy as np
import random

from ..environment import Environment
from ...utils import bounds_min_max


class Ball(object):

    def __init__(self, s_mins, s_maxs, diameter):
        self.x_min = 0.0
        self.x_max = s_maxs[0]
        self.y_min = s_mins[1]
        self.y_max = s_maxs[1]
        #self.xcoor = random.uniform(self.x_min, self.x_max)
        #self.ycoor = random.uniform(self.y_min, self.y_max)
        self.xcoor = 0.5
        self.ycoor = 0.5
        self.diameter = diameter

    def set_ball(self, x, y):
        self.xcoor = x
        self.ycoor = y

    def is_touched(self, position):
        x, y = position
        if x >= (self.xcoor - (self.diameter/2.)) and x <= (self.xcoor + (self.diameter/2.)):
            if y >= (self.ycoor - (self.diameter/2.)) and y <= (self.ycoor + (self.diameter/2.)):
                return True

        return False

    def update_position(self):
        #x_new = self.xcoor + random.random()
        #y_new = self.ycoor + random.random()
        x_new = self.xcoor + 0.05
        y_new = self.ycoor - 0.05
        #y_new = self.ycoor + 0.25
        if x_new >= self.x_max:
           # x_new  = x_new - (self.x_max - self.x_min)
            x_new = 0.05
            y_new = 0.95
        if y_new <= self.y_min:
            x_new = 0.05
            y_new = 0.95
            # y_new = y_new - (self.y_max - self.y_min)
        self.xcoor = x_new
        self.ycoor = y_new
        print(x_new)
        print(y_new)
        return 1

    def update_position_advanced(self, hand):
        x_hand, y_hand = hand

        x_new = self.xcoor - ( x_hand - self.xcoor)
        y_new = self.ycoor - ( y_hand - self.ycoor)

        if x_new >= self.x_max:
            x_new = self.x_max

        if x_new <= self.x_min:
            x_new = self.x_min

        if y_new >= self.y_max:
            y_new = self.y_max

        if y_new <= self.y_min:
            y_new = self.y_min

        self.xcoor = x_new
        self.ycoor = y_new
        print(x_new)
        print(y_new)
        return 1



def forward(angles, lengths):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: a tuple (x, y) of the end-effector position

    .. warning:: angles and lengths should be the same size.
    """
    x, y = joint_positions(angles, lengths)
    return x[-1], y[-1]


def joint_positions(angles, lengths, unit='rad'):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: x positions of each joint, y positions of each joints, except the first one wich is fixed at (0, 0)

    .. warning:: angles and lengths should be the same size.
    """
    if len(angles) != len(lengths):
        raise ValueError('angles and lengths must be the same size!')

    if unit == 'rad':
        a = np.array(angles)
    elif unit == 'std':
        a = np.pi * np.array(angles)
    else:
        raise NotImplementedError
     
    a = np.cumsum(a)
    return np.cumsum(np.cos(a)*lengths), np.cumsum(np.sin(a)*lengths)


def lengths(n_dofs, ratio):
    l = np.ones(n_dofs)
    for i in range(1, n_dofs):
        l[i] = l[i-1] / ratio
    return l / sum(l)

class SimpleArmEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length_ratio, noise, diameter):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length_ratio = length_ratio
        self.noise = noise
        self.diameter = diameter
        self.hits = 0

        # generate initial ball position randomly
        self.ball = Ball(s_mins, s_maxs, diameter)

        self.lengths = lengths(self.conf.m_ndims, self.length_ratio)

    def get_ball(self):
        return self.ball

    def compute_motor_command(self, joint_pos_ag):
        return bounds_min_max(joint_pos_ag, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, joint_pos_env):
        hand_pos = np.array(forward(joint_pos_env, self.lengths))
        hand_pos += self.noise * np.random.randn(*hand_pos.shape)
        if self.ball.is_touched(hand_pos):
            self.hits = self.hits + self.ball.update_position_advanced(hand_pos)
        ball_pos = np.array([self.ball.xcoor, self.ball.ycoor])
        s = np.concatenate([hand_pos, ball_pos])
        return s


    def plot(self, ax, m, s, **kwargs_plot):
        self.plot_arm(ax, m, **kwargs_plot)
        self.plot_ball(ax, s, **kwargs_plot)

    def plot_arm(self, ax, m, **kwargs_plot):
        x, y = joint_positions(m, self.lengths)
        x = np.hstack((0., x))
        y = np.hstack((0., y))
        ax.plot(x, y, 'grey', lw=2, **kwargs_plot)
        ax.plot(x[0], y[0], 'ok', ms=6)
        ax.plot(x[-1], y[-1], 'sk', ms=6)
        ax.axis([self.conf.s_mins[0], self.conf.s_maxs[0], self.conf.s_mins[1], self.conf.s_maxs[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def plot_ball(self, ax, s, **kwargs_plot):
        x_center = s[2]
        y_center = s[3]
        x = np.linspace (x_center -(self.ball.diameter/2.), x_center +(self.ball.diameter/2.), 1000)
        y = np.sqrt(-x**2+(self.ball.diameter/2.)**2)
        ax.plot(x, y_center + y, 'b')
        ax.plot(x, y_center - y, 'b')
