# coding=utf8

import numpy as np

from matplotlib.patches import Circle, Rectangle
from explauto_master.explauto.environment.dynamic_environment import DynamicEnvironment
from explauto_master.explauto.environment.modular_environment import FlatEnvironment, HierarchicalEnvironment

from explauto_master.explauto.utils import bounds_min_max
from explauto_master.explauto.environment.environment import Environment
from explauto_master.explauto.environment.simple_arm.simple_arm import joint_positions

def arm_lengths(n_joints):
    if n_joints == 3:
        return [0.5, 0.3, 0.2]
    elif n_joints == 7:
        return [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
    else:
        return [1./n_joints] * n_joints
    

class Arm(Environment):
    use_process = True
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 lengths, angle_shift, rest_state, id):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.lengths = lengths
        self.angle_shift = angle_shift
        self.rest_state = rest_state
        self.logs = []
        self.id = id
        self.reset()
        
    def reset(self):
        self.lines = None
        self.holding = 0

    def compute_motor_command(self, m):
        return bounds_min_max(m[0:3], self.conf.m_mins, self.conf.m_maxs)


    def get_angle(self, a):
        if a > 0:
            a_mod = a % 2
        else:
            a_mod = a % -2

        if a_mod > 1:
            return a_mod - 2
        elif a_mod < -1:
            return  a_mod + 2
        else:
            return a_mod


    def compute_sensori_effect(self, m):
        a = self.angle_shift + np.cumsum(np.array(m))
        a_pi = np.pi * a
        hand_pos = np.array([np.sum(np.cos(a_pi) * self.lengths), np.sum(np.sin(a_pi) * self.lengths)])
        angle = np.mod(a[-1] + 1, 2) - 1
        joint_angles = [self.get_angle(a_i) * np.pi for a_i in a]
        self.logs.append( list(joint_angles) + [hand_pos[0], hand_pos[1], angle, self.holding])#(m[0:3])
        return  list(joint_angles) + list([hand_pos[0], hand_pos[1], angle, self.holding])

    def print_log(self):
        print("Arm:")
        print(self.logs)
        print(len(self.logs))

    def plot(self, ax, i, **kwargs_plot):
        m = self.logs[i]
        angles = np.array(m[0:3])
        angles[0] += self.angle_shift
        x, y = joint_positions(angles, self.lengths, 'std')
        x = np.array([0.] + list(x))
        y = np.array([0.] + list(y))
        l = []
        l += ax.plot(x, y, 'grey', lw=4, animated=True, **kwargs_plot)
        l += ax.plot(0., 0., 'sk', ms=8, animated=True, **kwargs_plot)
        for i in range(len(self.lengths) - 1):
            l += ax.plot(x[i + 1], y[i + 1], 'ok', ms=8, animated=True, **kwargs_plot)
        l += ax.plot(x[-1], y[-1], 'or', ms=8, animated=True, **kwargs_plot)
        self.lines = l
        return l

    def plot_update(self, ax, i, **kwargs_plot):
        if self.lines is None:
            self.plot(ax, 0, **kwargs_plot)
        m = self.logs[i]
        angles = np.array(m[0:3])
        angles[0] += self.angle_shift
        x, y = joint_positions(angles, self.lengths, 'std')
        x = np.array([0.] + list(x))
        y = np.array([0.] + list(y))
        l = []
        l += [[x, y]]
        l += [[x[0], y[0]]]
        for i in range(len(self.lengths)-1):
            l += [[x[i+1], y[i+1]]]
        l += [[x[-1], y[-1]]]
        for (line, data) in zip(self.lines, l):
            line.set_data(data[0], data[1])
        return self.lines

class Stick(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length, handle_tol, rest_state, type, color, id):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length = length
        self.handle_tol = handle_tol
        self.handle_tol_sq = handle_tol * handle_tol
        self.rest_state = rest_state
        self.type = type
        self.id = id
        self.end_color=color
        self.logs = []

        self.reset()

    def reset(self):
        self.lines = None
        self.held = False
        self.holding = 0
        self.handle_pos = np.array(self.rest_state[0:2])
        self.angle = self.rest_state[2]
        self.compute_end_pos()

    def compute_end_pos(self):
        a = np.pi * self.angle
        self.end_pos = [self.handle_pos[0] + np.cos(a) * self.length,
                        self.handle_pos[1] + np.sin(a) * self.length]

    def compute_motor_command(self, m):
        return m

    def compute_sensori_effect(self, m):
        m = m[3:]
        hand_pos = m[0:2]
        hand_angle = m[2]
        arm_holding = m[3]

        if not self.held:
            if arm_holding == 0:
                if (hand_pos[0] - self.handle_pos[0]) ** 2. + (hand_pos[1] - self.handle_pos[1]) ** 2. < self.handle_tol_sq:
                    self.handle_pos = hand_pos
                    self.angle = hand_angle
                    #self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                    self.compute_end_pos()
                    self.held = True
        else:
            assert(arm_holding == self.id) #TODO drum kümmern

            self.handle_pos = hand_pos
            self.angle = hand_angle
            # self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
            self.compute_end_pos()

        self.logs.append([self.handle_pos,
                          self.angle,
                          self.end_pos,
                          self.held])
        # print "Tool hand_pos:", hand_pos, "hand_angle:", hand_angle, "gripper_change:", gripper_change, "self.handle_pos:", self.handle_pos, "self.angle:", self.angle, "self.held:", self.held
        return list(self.end_pos) + [self.holding]  + [self.held] # Tool pos

    def print_log(self):
        print("Stick:")
        print(self.logs)
        print(len(self.logs))

    def plot(self, ax, i, **kwargs_plot):
        handle_pos = self.logs[i][0]
        end_pos = self.logs[i][2]
        l = []
        l += ax.plot([handle_pos[0], end_pos[0]], [handle_pos[1], end_pos[1]], '-', color="g", lw=6, animated=True,
                     **kwargs_plot)
        l += ax.plot(handle_pos[0], handle_pos[1], 'o', color="r", ms=12, animated=True, **kwargs_plot)
        l += ax.plot(end_pos[0], end_pos[1], 'o', color=self.end_color, ms=12, animated=True, **kwargs_plot)
        self.lines = l
        return l

    def plot_update(self, ax, i, **kwargs_plot):
        if self.lines is None:
            self.plot(ax, 0, **kwargs_plot)
        handle_pos = self.logs[i][0]
        end_pos = self.logs[i][2]

        l = [[[handle_pos[0], end_pos[0]], [handle_pos[1], end_pos[1]]]]
        l += [[handle_pos[0], handle_pos[1]]]
        l += [[end_pos[0], end_pos[1]]]
        for (line, data) in zip(self.lines, l):
            line.set_data(data[0], data[1])
        return self.lines

class Toy(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 size, initial_position, id):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.size = size
        self.size_sq = size * size
        self.initial_position = initial_position
        self.logs = []
        self.reset()
        self.input = {'Arm': 0, 'Magnet Stick': 1, 'Gripper Stick': 2, 'Magnet 1': 3, 'Magnet 2': 4, 'Magnet 3': 5}
        self.reverse_input = {0:'Arm', 1:'Magnet Stick', 2:'Gripper Stick', 3:'Magnet 1', 4:'Magnet 2', 5:'Magnet 3'}
        self.id = id


    def reset(self):
        self.move = {'Arm': False, 'Gripper Stick': False, 'Magnet Stick': False, 'Magnet 1': False, 'Magnet 2': False, 'Magnet 3': False}
        self.circle = None
        self.pos = np.array(self.initial_position)
        #self.logs = []

    def compute_motor_command(self, m):
        return m

    def compute_magnet_hold_pos(self, angle, magnet_pos, size):
        a = np.pi * angle
        pos = [magnet_pos[0] + np.cos(a) * size, magnet_pos[1] + np.sin(a) * size]
        return pos

    def compute_sensori_effect(self, m_orig):
        m = np.array(m_orig).copy()
        m =list(m)
        arm_holding = int(m.pop(11))
        mstick_holding = int(m.pop(8))
        angle = m.pop(5)
        m = m[3:]
        m = np.array([m]).reshape((-1, 2))

        if self.move['Arm'] or ((m[self.input['Arm'], 0] - self.pos[0]) ** 2 + (m[self.input['Arm'], 1] - self.pos[1]) ** 2 < self.size_sq):
            if arm_holding == self.input['Arm'] or arm_holding == self.id:
                self.pos = m[self.input['Arm'], 0:2]
                self.move['Arm'] = 1
                arm_holding = self.id
        if self.move['Magnet Stick'] or ((m[self.input['Magnet Stick'], 0] - self.pos[0]) ** 2 + (m[self.input['Magnet Stick'], 1] - self.pos[1]) ** 2 < self.size_sq):
            if arm_holding == self.input['Magnet Stick']:
                if mstick_holding == 0 or mstick_holding==self.id:
                    self.pos = self.compute_magnet_hold_pos(angle, m[self.input['Magnet Stick'], 0:2], 2*self.size)
                    self.move['Magnet Stick'] = 1
                    mstick_holding = self.id
        if self.move['Gripper Stick'] or ((m[self.input['Gripper Stick'], 0] - self.pos[0]) ** 2 + (m[self.input['Gripper Stick'], 1] - self.pos[1]) ** 2 < self.size_sq):
            if arm_holding == self.input['Gripper Stick']:
                self.pos = m[self.input['Gripper Stick'], 0:2]
                self.move['Gripper Stick'] = 1
        if self.move['Magnet 1'] or ((m[self.input['Magnet 1'], 0] - self.pos[0]) ** 2 + (m[self.input['Magnet 1'], 1] - self.pos[1]) ** 2 < self.size_sq):
            if arm_holding == self.input['Magnet 1'] or mstick_holding == self.input['Magnet 1']:
                self.pos = self.compute_magnet_hold_pos(angle, m[self.input['Magnet 1'], 0:2], 1.5*self.size)      #m[self.input['Magnet 1'], 0:2]
                self.move['Magnet 1'] = 1
        if self.move['Magnet 2'] or ((m[self.input['Magnet 2'], 0] - self.pos[0]) ** 2 + (m[self.input['Magnet 2'], 1] - self.pos[1]) ** 2 < self.size_sq):
            if arm_holding == self.input['Magnet 2'] or mstick_holding == self.input['Magnet 2']:
                self.pos = self.compute_magnet_hold_pos(angle, m[self.input['Magnet 2'], 0:2], 1.5*self.size)      # m[self.input['Magnet 2'], 0:2]
                self.move['Magnet 2'] = 1
        if self.move['Magnet 3'] or ((m[self.input['Magnet 3'], 0] - self.pos[0]) ** 2 + (m[self.input['Magnet 3'], 1] - self.pos[1]) ** 2 < self.size_sq):
            if arm_holding == self.input['Magnet 3'] or mstick_holding == self.input['Magnet 3']:
                self.pos = self.compute_magnet_hold_pos(angle, m[self.input['Magnet 3'], 0:2], 1.5*self.size)      # m[self.input['Magnet 3'], 0:2]
                self.move['Magnet 3'] = 1
        # TODO für meherere Magnete müsste man hier arm_holding/mstick_holding bedingung anpassen

        if ((0. - self.pos[0]) ** 2 + (1.1 - self.pos[1]) ** 2 < self.size_sq): # Box
            self.pos = [0., 1.1]
            if arm_holding == self.id:
                arm_holding = 0
            if mstick_holding == self.id:
                mstick_holding = 0
            for key in self.move:
                self.move[key] = False


        self.logs.append([self.pos,
                          self.move])
        return list(self.pos) + [arm_holding, mstick_holding]

    def print_log(self):
        print("Toy:")
        print(self.logs)
        print(len(self.logs))

    def plot(self, ax, i, **kwargs_plot):
        #self.logs = self.logs[-50:]
        pos = self.logs[i][0]
        self.circle = Circle((pos[0], pos[1]), self.size, fc='c', animated=True, **kwargs_plot)
        ax.add_patch(self.circle)
        return [self.circle]

    def plot_update(self, ax, i, **kwargs_plot):
        if self.circle is None:
            self.plot(ax, 0, **kwargs_plot)
        #self.logs = self.logs[-50:]
        pos = self.logs[i][0]
        self.circle.center = tuple(pos)
        return [self.circle]

class Magnet(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 size, color, initial_position, id):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.size = size
        self.color = color
        self.size_sq = size * size
        self.initial_position = initial_position
        self.logs = []
        self.input = {'Arm': 0, 'Magnet Stick': 1}
        self.id = id
        self.reset()

    def reset(self):
        self.rect = None
        self.pos = np.array(self.initial_position)
        #self.logs = []

    def print_log(self):
        print("Magnet:")
        print(self.logs)
        print(len(self.logs))

    def compute_motor_command(self, m):
        return m

    def compute_magnet_hold_pos(self, angle, magnet_pos):
        a = np.pi * angle
        pos = [magnet_pos[0] + np.cos(a) * self.size, magnet_pos[1] + np.sin(a) * self.size]
        return pos

    def compute_sensori_effect(self, m_orig):
        m = np.copy(np.array(m_orig))
        m = list(m)
        arm_holding = int(m.pop(6))
        mstick_holding = int(m.pop(5))
        angle = m.pop(2)
        m = np.array([m]).reshape((-1, 2))
        if arm_holding==self.id or (arm_holding==0 and (m[self.input['Arm'], 0] - self.pos[0]) ** 2 + (m[self.input['Arm'], 1] - self.pos[1]) ** 2 < self.size_sq):
            self.pos = m[self.input['Arm']][0:2]
            arm_holding=self.id
        elif mstick_holding==self.id or (mstick_holding==0 and  (m[self.input['Magnet Stick'], 0] - self.pos[0]) ** 2 + (m[self.input['Magnet Stick'], 1] - self.pos[1]) ** 2 < self.size_sq):
            self.pos = self.compute_magnet_hold_pos(angle, m[self.input['Magnet Stick']][0:2]) #m[self.input['Magnet Stick']][0:2]
            mstick_holding = self.id
            # TODO Magnetic Stick kann jetzt nur einen Magneten/objekt auf einmal halten:
        self.logs.append([self.pos])
        return list(self.pos) + [arm_holding, mstick_holding]

    def plot(self, ax, i, **kwargs_plot):
        #self.logs = self.logs[-50:]
        pos = self.logs[i][0]
        x_bottom = pos[0] - self.size/2
        y_bottom = pos[1] - self.size/2
        self.rect = Rectangle((x_bottom, y_bottom), self.size, self.size,fc=self.color, animated=True, **kwargs_plot)
        ax.add_patch(self.rect)
        return [self.rect]

    def plot_update(self, ax, i, **kwargs_plot):
        if self.rect is None:
            self.plot(ax, 0, **kwargs_plot)
        #print("Magnet plot update")
        #print(self.logs)
        #print(len(self.logs))
        #print(self.logs[-50:])
        #self.logs = self.logs[-50:]
        pos = self.logs[i][0]
        x_bottom = pos[0] - self.size/2
        y_bottom = pos[1] - self.size/2
        self.rect.xy = (x_bottom, y_bottom)
        return [self.rect]

class Box(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 size, position):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.size = size
        self.size_sq = size * size
        self.position = position
        self.logs = []
        self.reset()

    def reset(self):
        self.filled = False
        self.circle = None
        #self.logs = []

    def compute_motor_command(self, m):
        return m

    def compute_sensori_effect(self, m):
        if ((m[0] - self.position[0]) ** 2 + (m[1] - self.position[1]) ** 2 < self.size_sq):
            self.filled = True
        return [int(self.filled)]

    def print_log(self):
        print("Box:")
        print(self.logs)
        print(len(self.logs))

    def plot(self, ax, i, **kwargs_plot):
        self.circle = Circle((self.position[0], self.position[1]), self.size, fc='gray', animated=True, **kwargs_plot)
        ax.add_patch(self.circle)
        return [self.circle]

    def plot_update(self, ax, i, **kwargs_plot):
        if self.circle is None:
            self.plot(ax, 0, **kwargs_plot)
        self.circle.center = tuple(self.position)
        return [self.circle]


class My_Very_Own_Complex_Environment_and_Joints(DynamicEnvironment):
    def __init__(self,
                 n_joints=3,
                 n_dmp_basis=3,
                 goal_size=1.5,
                 stick_handle_tol=0.05,
                 ball_size=0.1,
                 stick_length=0.3, toy_position=[-0.2, 0.2], toy_class=Toy):

        arm_config = dict(
            m_mins=[-2.] * n_joints,
            m_maxs=[2.] * n_joints,
            s_mins=[-goal_size] * (4 + n_joints),
            s_maxs=[goal_size] * (4 + n_joints),
            lengths=arm_lengths(n_joints),
            angle_shift=0.5,
            rest_state=[0.] * n_joints,
            id=0)

        magnetic_stick_config = dict(
            m_mins=[-goal_size, -goal_size, -goal_size, -goal_size] + [-goal_size, -goal_size, -goal_size],  # Hand pos + arm angle + joint angles
            m_maxs=[goal_size, goal_size, goal_size, goal_size] + [-goal_size, -goal_size, -goal_size],
            s_mins=[-goal_size, -goal_size, -goal_size, -goal_size],  # Tool pos
            s_maxs=[goal_size, -goal_size, -goal_size, -goal_size],
            type='magnetic',
            color='m',
            length=stick_length,
            handle_tol=stick_handle_tol,
            rest_state=[0.6, 0.0, 0.25],
            id=1)

        gripper_stick_config = dict(
            m_mins=[-goal_size, -goal_size, -goal_size, -goal_size] + [-goal_size, -goal_size, -goal_size],  # Hand pos + arm angle + joint angles
            m_maxs=[goal_size, goal_size, goal_size, goal_size] + [-goal_size, -goal_size, -goal_size],
            s_mins=[-goal_size, -goal_size, -goal_size, -goal_size],  # Tool pos
            s_maxs=[goal_size, -goal_size, -goal_size, -goal_size],
            type='gripper',
            color='b',
            length=stick_length,
            handle_tol=stick_handle_tol,
            rest_state=[-0.5, -0.4, 0.75],
            id=2)

        sticks_config = dict(
            s_mins = list([-goal_size] * 8),
            s_maxs = list([goal_size] * 8),
            envs_cls = [Stick, Stick],
            envs_cfg = [magnetic_stick_config, gripper_stick_config],
            combined_s = lambda s: s)

        arm_sticks_config = dict(
            m_mins=list([-2.] * n_joints * 2),  # 3DOF + gripper
            m_maxs=list([2.] * n_joints * 2),
            s_mins=list([-goal_size] * 12),
            s_maxs=list([goal_size] * 12),
            top_env_cls=FlatEnvironment,
            lower_env_cls=Arm,
            top_env_cfg=sticks_config,
            lower_env_cfg=arm_config,
            fun_m_lower=lambda m: m,
            fun_s_lower=lambda m, s: s+s,
            fun_s_top=lambda m, s_lower, s: s_lower[0:6] + s[0:3] + s[4:6] + arm_holding_stick(s_lower, s))

        def arm_holding_stick(arm_pos, stick_pos): #es könnte immer noch theoretisch sein, dass beide gleichzeitig aufgehoben werden
            if stick_pos[3]:
                return [1]
            elif stick_pos[7]:
                return [2]
            else:
                return [arm_pos[6]]



        magnet1_config = dict(
            m_mins=[goal_size] * 7,
            m_maxs=[goal_size] * 7,
            s_mins=[goal_size] * 4,
            s_maxs=[goal_size] * 4,
            size=2*ball_size,
            initial_position=[0.5, -0.5],
            color="m",
            id=3)

        magnet2_config = dict(
            m_mins=[-goal_size] * 7,
            m_maxs=[goal_size] * 7,
            s_mins=[-goal_size] * 4,
            s_maxs=[goal_size] * 4,
            size=2*ball_size,
            initial_position=[-0.6, 0.7], #0.6
            color="m",
            id=4)

        magnet3_config = dict(
            m_mins=[-goal_size] * 7,
            m_maxs=[goal_size] * 7,
            s_mins=[-goal_size] * 4,
            s_maxs=[goal_size] * 4,
            size=2*ball_size,
            initial_position=[0.3, 1.],
            color="m",
            id=5)

        box_config = dict(
            m_mins=[-goal_size] * 4,
            m_maxs=[goal_size] * 4,
            s_mins=[0.] * 1,
            s_maxs=[1.] * 1,
            size=ball_size,
            position=[0., 1.1])

        toy_config = dict(
            m_mins=[-goal_size] * 18,
            m_maxs=[goal_size] * 18,
            s_mins=[-goal_size] * 4,
            s_maxs=[goal_size] * 4,
            size=ball_size,
            initial_position= toy_position,
            id=6)

        box_toy_config = dict(
            m_mins=[-goal_size] * 18,
            m_maxs=[goal_size] * 18,
            s_mins=[-goal_size] * 5,
            s_maxs=[goal_size] * 5,
            top_env_cls=Box,
            lower_env_cls=toy_class,
            top_env_cfg=box_config,
            lower_env_cfg=toy_config,
            fun_m_lower=lambda m: m,
            fun_s_lower=lambda m, s: s,
            fun_s_top=lambda m, s_lower, s: s_lower + s)


        magnets_config = dict(
            s_mins = list([-goal_size] * 12),
            s_maxs = list([goal_size] * 12),
            envs_cls = [Magnet, Magnet, Magnet],
            envs_cfg = [magnet1_config, magnet2_config, magnet3_config],
            combined_s = lambda s: s)


        arm_sticks_magnets_config = dict(
            m_mins=[-1.] * (n_joints),
            m_maxs=[1.] * (n_joints),
            s_mins=[-goal_size] * 18,
            s_maxs=[goal_size] * 18,
            top_env_cls=FlatEnvironment,
            lower_env_cls=HierarchicalEnvironment,
            top_env_cfg=magnets_config,
            lower_env_cfg=arm_sticks_config,
            fun_m_lower=lambda m: m + m,
            fun_s_lower=lambda m, s: s[3:9] + [s[-1]] + s[3:9] + [s[-1]] + s[3:9] + [s[-1]],
            fun_s_top=lambda m, s_lower, s: holding_update(s_lower,  s))

        def holding_update(s_lower, s): #s_lower: handpos, angle, mstickpos, m_holding, stick2pos, arm_holding,
            #s:  m1_pos, holding,  m2_pos, holding,  m3_pos, holding
            if s[2]==3:
                s_lower[11]=3
            elif s[6]==4:
                s_lower[11]=4
            elif s[10]==5:
                s_lower[11]=5
            if s[3]==3:
                s_lower[8]=3
            elif s[7]==4:
                s_lower[8]=4
            elif s[11]==5:
                s_lower[8]=5
            ret = s_lower + s[0:2] + s[4:6] + s[8:10]
            return ret

        all_config = dict(
            m_mins=[-1.] * (n_joints),
            m_maxs=[1.] * (n_joints),
            s_mins=[-goal_size] * 18,
            s_maxs=[goal_size] * 18,
            top_env_cls=HierarchicalEnvironment,
            lower_env_cls=HierarchicalEnvironment,
            top_env_cfg=box_toy_config,
            lower_env_cfg=arm_sticks_magnets_config,
            fun_m_lower=lambda m: m,
            fun_s_lower=lambda m, s: s,
            fun_s_top=lambda m, s_lower, s: s_top(s_lower, s))

        def s_top(s_lower, s):
            s_lower[11]=s[2]
            s_lower[8]=s[3]
            arm_update = s_lower.pop(11)
            mstick_update = s_lower.pop(8)
            s_lower.pop(5)
            self.update_arm_stick(arm_update, mstick_update)
            return s_lower + s[0:2] + [s[4]]

        dynamic_environment_config = dict(
            env_cfg=all_config,
            env_cls=HierarchicalEnvironment,
            m_mins=[-0.5] * n_dmp_basis * (n_joints),
            m_maxs=[0.5] * n_dmp_basis * (n_joints),
            s_mins=[-goal_size] * n_dmp_basis * 18,
            s_maxs=[goal_size] * n_dmp_basis * 18,
            n_bfs=n_dmp_basis,
            move_steps=50,
            n_dynamic_motor_dims=n_joints,
            n_dynamic_sensori_dims=8,
            sensori_traj_type="samples",
            max_params=1000)
        DynamicEnvironment.__init__(self, **dynamic_environment_config)

    def random_motor(self, it=1):
        m = self.random_motors(n=it)
        if it is 1:
            return m[0]
        else:
            return m


