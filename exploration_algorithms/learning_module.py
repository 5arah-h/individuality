import numpy as np
import random

from numpy import array, hstack
from scipy.spatial import distance

from explauto_master.explauto.agent import Agent
from explauto_master.explauto.utils import rand_bounds
from explauto_master.explauto.utils.config import make_configuration
from explauto_master.explauto.exceptions import ExplautoBootstrapError
from explauto_master.explauto.sensorimotor_model.non_parametric import NonParametric

from exploration_algorithms.interest_model import MiscRandomInterest, competence_dist


class LearningModule(Agent):
    def __init__(self, mid, m_space, s_space, env_conf):


        explo_noise = 0.05


        self.conf = make_configuration(env_conf.m_mins[m_space], 
                                       env_conf.m_maxs[m_space], 
                                       array(list(env_conf.m_mins[m_space]) + list(env_conf.s_mins))[s_space],
                                       array(list(env_conf.m_maxs[m_space]) + list(env_conf.s_maxs))[s_space])
        
        self.im_dims = self.conf.s_dims
        
        self.mid = mid
        self.m_space = m_space
        self.s_space = s_space
        self.motor_babbling_n_iter = 10
        
        self.s = None
        self.last_interest = 0
        

        im_cls, kwargs = (MiscRandomInterest, {
                          'competence_measure': competence_dist,
                           'win_size': 1000,
                           'competence_mode': 'knn',
                           'k': 20,
                           'progress_mode': 'local'})
        
        self.im = im_cls(self.conf, self.im_dims, **kwargs)
        
        sm_cls, kwargs = (NonParametric, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':explo_noise})
        self.sm = sm_cls(self.conf, **kwargs)
        
        Agent.__init__(self, self.conf, self.sm, self.im)
        
        
    def motor_babbling(self, n=1): 
        if n == 1:
            return rand_bounds(self.conf.m_bounds)[0]
        else:
            return rand_bounds(self.conf.m_bounds, n)
        
    def goal_babbling(self):
        s = rand_bounds(self.conf.s_bounds)[0]
        m = self.sm.infer(self.conf.s_dims, self.conf.m_dims, s)
        return m
            
    def get_m(self, ms): return array(ms[self.m_space])
    def get_s(self, ms): return array(ms[self.s_space])
        
    def set_one_m(self, ms, m):
        """ Set motor dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['m']] = m
        
    def set_m(self, ms, m):
        """ Set motor dimensions used by module on one ms
        """
        self.set_one_m(ms, m)
        if self.mconf['operator'] == "seq":
            return [array(ms), array(ms)]
        elif self.mconf['operator'] == "par":
            return ms
        else:
            raise NotImplementedError
    
    def set_s(self, ms, s):
        """ Set sensory dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['s']] = s
        return ms          
    
    def inverse(self, s):
        m = self.infer(self.conf.s_dims, self.conf.m_dims, s, pref='')
        return self.motor_primitive(m)
        
    def infer(self, expl_dims, inf_dims, x, pref='', n=1, explore=True):
        try:
            if self.n_bootstrap > 0:
                self.n_bootstrap -= 1
                raise ExplautoBootstrapError
            mode = "explore" if explore else "exploit"
            if n == 1:
                self.sensorimotor_model.mode = mode
                m = self.sensorimotor_model.infer(expl_dims, inf_dims, x.flatten())
            else:
                self.sensorimotor_model.mode = mode
                m = []
                for _ in range(n):
                    m.append(self.sensorimotor_model.infer(expl_dims, inf_dims, x.flatten()))
            self.emit(pref + 'inference' + '_' + self.mid, m)
        except ExplautoBootstrapError:
            if n == 1:
                m = rand_bounds(self.conf.bounds[:, inf_dims]).flatten()
            else:
                m = rand_bounds(self.conf.bounds[:, inf_dims], n)
        return m
            
    def produce(self, n=1):
        if self.t < self.motor_babbling_n_iter:
            self.m = self.motor_babbling(n)
            self.s = np.zeros(len(self.s_space))
            self.x = np.zeros(len(self.expl_dims))
        else:
            self.x = self.choose()
            self.y = self.infer(self.expl_dims, self.inf_dims, self.x, n=n)
            #self.m, self.s = self.extract_ms(self.x, self.y)
            self.m, sg = self.y, self.x#self.extract_ms(self.x, self.y)
            #self.m = self.motor_primitive(self.m)
            
            self.s = sg
            #self.emit('movement' + '_' + self.mid, self.m)          
        return self.m        
    
    def update_sm(self, m, s): 
        self.sensorimotor_model.update(m, s)   
        self.t += 1 
    
    def update_im(self, m, s):
        if self.t >= self.motor_babbling_n_iter:
            return self.interest_model.update(hstack((m, self.s)), hstack((m, s)))
        
    def competence(self): return self.interest_model.competence()
    def interest(self): return self.interest_model.interest()

    def perceive(self, m, s, has_control = True):
        self.update_sm(m, s)
        if has_control:
            self.last_interest = self.update_im(m, s)


class HierarchicalLearningModule(LearningModule):
    def __init__(self, mid, m_space, s_space, env_conf, input_ids, interest_win_size=1000):
        self.input_models = input_ids
        self.count_input_models = {}
        for inp in self.input_models:
            self.count_input_models[inp] = 0
        explo_noise = 0.05
        self.conf = make_configuration(array(list(env_conf.m_mins) + list(env_conf.s_mins))[m_space],
                                       array(list(env_conf.m_maxs) + list(env_conf.s_maxs))[m_space],
                                       array(list(env_conf.m_mins) + list(env_conf.s_mins))[s_space],
                                       array(list(env_conf.m_maxs) + list(env_conf.s_maxs))[s_space])

        self.im_dims = self.conf.s_dims
        self.mid = mid
        self.m_space = m_space
        self.s_space = s_space
        self.motor_babbling_n_iter = 10
        self.s = None
        self.last_interest = 0

        im_cls, kwargs = (MiscRandomInterest, {
            'competence_measure': competence_dist,
            'win_size': interest_win_size,
            'competence_mode': 'knn',
            'k': 20,
            'progress_mode': 'local'})

        self.im = im_cls(self.conf, self.im_dims, **kwargs)
        sm_cls, kwargs = (NonParametric, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio': explo_noise})
        self.sm = sm_cls(self.conf, **kwargs)
        Agent.__init__(self, self.conf, self.sm, self.im)

    def set_all_learning_modules(self, l_modules):
        self.learning_modules = l_modules

    def infer(self, expl_dims, inf_dims, x, pref='', n=1, explore=True):
        m = LearningModule.infer(self, expl_dims, inf_dims, x, pref=pref, n=n, explore=explore)
        if self.input_models:
            nearest = [0], 1000000
            if n == 1:
                for input in self.input_models:
                    # Idee 1: erst backward prediction -> darauf forward prediction -> dann Nähe
                    m_inp = self.learning_modules[input].sm.infer(self.learning_modules[input].expl_dims, self.learning_modules[input].inf_dims, m)
                    m_pred = self.learning_modules[input].sm.infer(self.learning_modules[input].inf_dims, self.learning_modules[input].expl_dims, m_inp)
                    #if not explore:
                        #print(self)
                        #print(m_inp)
                        #print(m_pred)
                    if distance.euclidean(m, m_pred) < nearest[1]:
                        nearest = [input], distance.euclidean(m, m_pred)
                    elif distance.euclidean(m, m_pred) == nearest[1]:
                        nearest[0].append(input)
                    #if not explore:
                        #print("nearest", nearest)
                nearest_mod = random.choice(nearest[0])
                self.count_input_models[nearest_mod] += 1
                _, global_m = self.learning_modules[nearest_mod].infer(self.learning_modules[nearest_mod].expl_dims, self.learning_modules[nearest_mod].inf_dims, m, pref=pref, explore=explore)
                #if not explore:
                    #print(global_m)
            else:
                global_m = []
                for _ in range(n):
                    for input in self.input_models:
                        # Idee 1: erst backward prediction -> darauf forward prediction -> dann Nähe
                        m_inp = self.learning_modules[input].sm.infer(self.learning_modules[input].expl_dims,
                                                                      self.learning_modules[input].inf_dims, m)
                        m_pred = self.learning_modules[input].sm.infer(self.learning_modules[input].inf_dims,
                                                                       self.learning_modules[input].expl_dims, m_inp)
                        if distance.euclidean(m, m_pred) < nearest[1]:
                            nearest = [input], distance.euclidean(m, m_pred)
                        elif distance.euclidean(m, m_pred) == nearest[1]:
                            nearest[0].append(input)
                    nearest_mod = random.choice(nearest[0])
                    self.count_input_models[nearest_mod] += 1
                    _, g_m = self.learning_modules[nearest_mod].infer(self.learning_modules[nearest_mod].expl_dims,
                                                                          self.learning_modules[nearest_mod].inf_dims, m, pref=pref, explore=explore)
                    global_m.append(g_m)
            return m, global_m
        return m, m     # first belongs to current model, second to arm

    def produce(self, n=1):
        if self.t < self.motor_babbling_n_iter:
            self.m = self.motor_babbling(n)
            self.s = np.zeros(len(self.s_space))
            self.x = np.zeros(len(self.expl_dims))
        else:
            self.x = self.choose()  # Goal
            # TODO self.m ist global, self.y ist lokal, macht das Sinn? -> eigentlich wird nur self.x benutzt
            self.m, self.y = self.infer(self.expl_dims, self.inf_dims, self.x, n=n)
            # self.m, self.s = self.extract_ms(self.x, self.y)
            sg = self.x
            # self.m, sg = self.y, self.x#self.extract_ms(self.x, self.y)
            # self.m = self.motor_primitive(self.m)

            self.s = sg
            # self.emit('movement' + '_' + self.mid, self.m)
        return self.m

class HierarchicalLearningModuleCHC(HierarchicalLearningModule):

    def __init__(self, mid, m_space, s_space, env_conf, input_ids, interest_win_size=1000):
        HierarchicalLearningModule.__init__(self, mid, m_space, s_space, env_conf, input_ids, interest_win_size=interest_win_size)

    def infer(self, expl_dims, inf_dims, x, pref='', n=1, explore=True):
        m = LearningModule.infer(self, expl_dims, inf_dims, x, pref=pref, n=n, explore=explore)
        if self.input_models:
            nearest = [0], 1000000
            if n == 1:
                interests = [self.learning_modules[input].interest() for input in self.input_models]
                high_inter = [ip for i, ip in enumerate(self.input_models) if interests[i]==max(interests)]
                model = random.choice(high_inter)
                #self.count_input_models[model] += 1
                _, global_m = self.learning_modules[model].infer(self.learning_modules[model].expl_dims, self.learning_modules[model].inf_dims, m, pref=pref, explore=explore)
                #if not explore:
                #print(global_m)
            else:
                global_m = []
                for _ in range(n):
                    interests = [self.learning_modules[input].interest() for input in self.input_models]
                    high_inter = [ip for i, ip in enumerate(self.input_models) if interests[i]==max(interests)]
                    model = random.choice(high_inter)
                    #self.count_input_models[model] += 1
                    _, g_m = self.learning_modules[model].infer(self.learning_modules[model].expl_dims,
                                                        self.learning_modules[model].inf_dims, m, pref=pref, explore=explore)
                    global_m.append(g_m)
            return m, global_m
        return m, m     # first belongs to current model, second to arm





