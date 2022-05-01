# coding=utf8
from BROJA_2PID.BROJA_2PID import pid

from exploration_algorithms.utils import *
from explauto_master.explauto.models.dataset import *

path = "./test_m2_korrig/"
iterations = 300000
save_iterations = 20000
intervall_measures = 5000
motor_babbling_iterations = 10
noise = False
env = My_Very_Own_Complex_Environment_and_Joints


print("iterations: ", iterations)
print("saving iterations: ", save_iterations)
print("saving to: ", path)
print("measuring intervall: ", intervall_measures)
print("noise: ", False)
print("environment:", env)

goal_size = 2.
environment = env(n_joints=3,  # Number of joints
                  n_dmp_basis=3,  # Number of basis per joint
                  goal_size=goal_size,  # Size of goal space
                  stick_handle_tol=0.05, # Maximal distance to grab the stick with the hand
                  stick_length=0.30,  # Stick length,
                  ball_size=0.10, # Maximal distance to grab the ball with the stick
                  )

evolution_organismal = []
evolution_colonial = []
evolution_env_det = []
evolution_env_coding = []
evolution_shared = []
evolution_unique_sn = []
evolution_unique_en = []
evolution_comp = []
evolution_prim_feas = []
evolution_dual_feas = []
evolution_gap = []

# Define motor and sensory spaces:
m_ndims = environment.conf.m_ndims  # number of motor parameters
m_space = list(range(m_ndims))
s_joints = list(range(m_ndims, m_ndims + 9))
s_hand = list(range(m_ndims + 9, m_ndims + 15))
s_stick1 = list(range(m_ndims + 15, m_ndims + 21))
s_stick2 = list(range(m_ndims + 21, m_ndims + 27))
s_m1 = list(range(m_ndims + 27, m_ndims + 33))
s_m2 = list(range(m_ndims + 33, m_ndims + 39))
s_m3 = list(range(m_ndims + 39, m_ndims + 45))
s_toy = list(range(m_ndims + 45, m_ndims + 51))
s_box = list(range(m_ndims + 51, m_ndims + 54))
s_ndims = 54 - 6

min_m = -3.
max_m = 3.
min_s = - goal_size
max_s = goal_size

learning_modules = {}
learning_modules['mod1'] = HierarchicalLearningModule("mod1", m_space, s_hand, environment.conf, [], 200)
learning_modules['mod2'] = HierarchicalLearningModule("mod2", s_hand, s_stick1, environment.conf, ['mod1'], 200)
learning_modules['mod3'] = HierarchicalLearningModule("mod3", s_hand, s_stick2, environment.conf, ['mod1'], 200)
learning_modules['mod4'] = HierarchicalLearningModule("mod4", s_hand, s_m1, environment.conf, ['mod1'], 200)
learning_modules['mod5'] = HierarchicalLearningModule("mod5", s_hand, s_m2, environment.conf, ['mod1'], 200)
learning_modules['mod6'] = HierarchicalLearningModule("mod6", s_hand, s_m3, environment.conf, ['mod1'], 200)
learning_modules['mod7'] = HierarchicalLearningModule("mod7", s_hand, s_toy, environment.conf, ['mod1'], 200)
learning_modules['mod8'] = HierarchicalLearningModule("mod8", s_stick1, s_m1, environment.conf, ['mod2'], 200)
learning_modules['mod9'] = HierarchicalLearningModule("mod9", s_stick1, s_m2, environment.conf, ['mod2'], 200)
learning_modules['mod10'] = HierarchicalLearningModule("mod10", s_stick1, s_m3, environment.conf, ['mod2'], 200)
learning_modules['mod11'] = HierarchicalLearningModule("mod11", s_stick1, s_toy, environment.conf, ['mod2'], 200)
learning_modules['mod12'] = HierarchicalLearningModule("mod12", s_stick2, s_toy, environment.conf, ['mod3'], 200)
learning_modules['mod13'] = HierarchicalLearningModule("mod13", s_m1, s_toy, environment.conf, ['mod4', 'mod8'], 200)
learning_modules['mod14'] = HierarchicalLearningModule("mod14", s_m2, s_toy, environment.conf, ['mod5', 'mod9'], 200)
learning_modules['mod15'] = HierarchicalLearningModule("mod15", s_m3, s_toy, environment.conf, ['mod6', 'mod10'], 200)
learning_modules['mod16'] = HierarchicalLearningModule("mod16", s_toy, s_box, environment.conf,
                                                       ['mod7', 'mod11', 'mod12', 'mod13', 'mod14', 'mod15'], 200)

for module in learning_modules.values():
    module.set_all_learning_modules(learning_modules)
    module.motor_babbling_n_iter = motor_babbling_iterations
    iteration = 0
    while not module.sm.bootstrapped_s or iteration < motor_babbling_iterations:
        m = environment.random_motor()
        s = environment.update(m)
        iteration += 1
        module.update_sm(module.get_m(array(list(m) + list(s))), module.get_s(array(list(m) + list(s))))

'''
data, mins, maxs are tuples
'''
def compute_probability(data, mins, maxs, gs):
    assert len(data) == len(mins)
    assert len(data) == len(maxs)
    assert len(data) == len(gs)
    for i in range(len(data)):
        if i != 0:
            assert len(data[i]) == len(data[0])
    states = np.zeros(tuple([gs[i] ** len(d[0]) for i, d in enumerate(data)]))
    for i, _ in enumerate(data[0]):
        idxs = [get_bin(d[i], np.array([mins[d_i]] * len(d[0])), np.array([maxs[d_i]] * len(d[0])), gs[d_i]) for d_i, d
                in enumerate(data)]
        states[tuple(idxs)] = states[tuple(idxs)] + 1
    return states / len(data[0])

def get_dictionary(probabilities):
    return {(float(i), float(j), float(k)): float(probabilities[i][j][k]) for i, prob_i in enumerate(probabilities) for j, prob_j in enumerate(prob_i) for k, _ in enumerate(prob_j)}


def get_bin(sample, mins, maxs, gs):
    assert len(sample) == len(mins)
    assert len(sample) == len(maxs)
    epss = (maxs - mins) / gs
    idxs = np.array((sample - mins) / epss, dtype=int)
    idxs[idxs >= gs] = gs - 1
    idxs[idxs < 0] = 0
    flat_i = 0
    for i, ind in enumerate(idxs):
        flat_i += ind * gs ** (len(idxs) - i - 1)
    return flat_i


def compute_individuality(s_n_values, s__n_values, e_n_values, min_s, max_s, min_e, max_e):

    obj__n_values = np.array(s__n_values)[:,-2:]
    s_rest__n_values = np.array(s__n_values)[:,:-2]

    obj_n_values = np.array(s_n_values)[:,-2:]
    s_rest_n_values = np.array(s_n_values)[:,:-2]

    p__s_s_e = compute_probability((obj__n_values, s_rest__n_values, obj_n_values, s_rest_n_values, e_n_values),
                                   (min_e, min_s, min_e, min_s, min_e), (max_e, max_s, max_e, max_s, max_e), (2, 3, 2, 3, 2))
    p__s_s_e = np.reshape(p__s_s_e, (27 * 4, 27 * 4, p__s_s_e.shape[-1]))

    prob_dict = get_dictionary(p__s_s_e)
    pid_data = pid(prob_dict)
    organismal = pid_data['SI'] + pid_data['UIY']
    colonial =  pid_data['CI'] + pid_data['UIY']
    environmental_determined = pid_data['CI'] + pid_data['UIZ']
    env_coding = pid_data['SI'] - pid_data['CI']
    return organismal, colonial, environmental_determined, env_coding, pid_data

m_old = None
s_old = None
history_env = Dataset(5, 11)
history_system = Dataset(5, 5)
for step in range(iterations):
    if (step + 1) % 500 == 0:
        print("Iteration:", step + 1)

    if (step + 1) % intervall_measures == 0:
        s_ns = list(history_env.iter_x())  # s_n
        e_ns = list(history_env.iter_y())  # e_n
        s__ns = list(history_system.iter_y())  # s_n+1

        organismal, colonial, env_det, env_coding, pid_data = compute_individuality(s_ns, s__ns, e_ns, min_s=min_m,
                                                                                    max_s=max_m, min_e=min_s,
                                                                                    max_e=max_s)
        print("iteration", step + 1)
        print("organismal", organismal)
        print("colonial", colonial)
        print("env_det", env_det)
        msg = """
        Shared information: {SI}
        Unique information in S_n: {UIY}
        Unique information in E_n: {UIZ}
        Synergistic information: {CI}
        """
        print(msg.format(**pid_data))
        evolution_organismal.append(organismal)
        evolution_colonial.append(colonial)
        evolution_env_det.append(env_det)
        evolution_env_coding.append(env_coding)
        evolution_shared.append(pid_data['SI'])
        evolution_unique_sn.append(pid_data['UIY'])
        evolution_unique_en.append(pid_data['UIZ'])
        evolution_comp.append(pid_data['CI'])
        evolution_prim_feas.append(pid_data['Num_err'][0])
        evolution_dual_feas.append(pid_data['Num_err'][1])
        evolution_gap.append(pid_data['Num_err'][2])

        history_env = Dataset(5, 11)
        history_system = Dataset(5, 5)

    if (step + 1) % save_iterations == 0:
        with open(path + 'organismal_' + str(step + 1) + '.npy', 'wb') as f:
            np.save(f, np.array(evolution_organismal))
        with open(path + 'colonial_' + str(step + 1) + '.npy', 'wb') as f:
            np.save(f, np.array(evolution_colonial))
        with open(path + 'environmental_determination_' + str(step + 1) + '.npy', 'wb') as f:
            np.save(f, np.array(evolution_env_det))
        with open(path+'environmental_coding_'+str(step+1)+'.npy', 'wb') as f:
            np.save(f, np.array(evolution_env_coding))
        with open(path+'shared_information_'+str(step+1)+'.npy', 'wb') as f:
            np.save(f, np.array(evolution_shared))
        with open(path + 'unique_information_sn_' + str(step + 1) + '.npy', 'wb') as f:
            np.save(f, np.array(evolution_unique_sn))
        with open(path+'complementary_information_'+str(step+1)+'.npy', 'wb') as f:
            np.save(f, np.array(evolution_comp))
        with open(path + 'unique_information_en_' + str(step + 1) + '.npy', 'wb') as f:
            np.save(f, np.array(evolution_unique_en))
        with open(path + 'primal_feasibility_' + str(step + 1) + '.npy', 'wb') as f:
            np.save(f, np.array(evolution_prim_feas))
        with open(path+'dual_feasibility_'+str(step+1)+'.npy', 'wb') as f:
            np.save(f, np.array(evolution_dual_feas))
        with open(path + 'duality_gap_' + str(step + 1) + '.npy', 'wb') as f:
            np.save(f, np.array(evolution_gap))

    # Compute the interest of modules
    interests = [learning_modules[mid].interest() for mid in learning_modules.keys()]
    i = prop_choice(interests, eps=0.2)
    babbling_module = list(learning_modules.values())[i]

    # The babbling module picks a random goal in its sensory space
    m_list = babbling_module.produce(n=0)
    _, m = babbling_module.infer(babbling_module.expl_dims, babbling_module.inf_dims, babbling_module.x, n=1,
                                 explore=True)
    s = environment.update(m)

    if noise:
        s_entropy = s
    else:
        _, m_entropy = babbling_module.infer(babbling_module.expl_dims, babbling_module.inf_dims, babbling_module.x,
                                             n=1,
                                             explore=False)
        s_entropy = environment.update(m_entropy)

    m_first = [s_entropy[0], s_entropy[3], s_entropy[6], s_entropy[33], s_entropy[36]]
    m_middle = [s_entropy[1], s_entropy[4], s_entropy[7], s_entropy[34], s_entropy[37]]
    m_last = [s_entropy[2], s_entropy[5], s_entropy[8], s_entropy[35], s_entropy[38]]

    s_first = [s_entropy[15], s_entropy[18],
               s_entropy[21], s_entropy[24],
               s_entropy[27], s_entropy[30],

               s_entropy[39], s_entropy[42],
               s_entropy[45], s_entropy[48],
               s_entropy[51]]

    s_middle = [s_entropy[16], s_entropy[19],
                s_entropy[22], s_entropy[25],
                s_entropy[28], s_entropy[31],

                s_entropy[40], s_entropy[43],
                s_entropy[46], s_entropy[49],
                s_entropy[52]]

    s_last = [s_entropy[17], s_entropy[20],
              s_entropy[23], s_entropy[26],
              s_entropy[29], s_entropy[32],

              s_entropy[41], s_entropy[44],
              s_entropy[47], s_entropy[50],
              s_entropy[53]]

    if m_first is not None:
        history_env.add_xy(m_first, s_first)
        history_system.add_xy(m_first, m_last)


    # Update the interest of the babbling module:
    babbling_module.update_im(babbling_module.get_m(array(list(m) + list(s))),
                              babbling_module.get_s(array(list(m) + list(s))))
    # Update each sensorimotor models
    for mid in learning_modules.keys():
        learning_modules[mid].update_sm(learning_modules[mid].get_m(array(list(m) + list(s))),
                                        learning_modules[mid].get_s(array(list(m) + list(s))))
