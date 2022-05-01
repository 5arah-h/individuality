# coding=utf8
from BROJA_2PID.BROJA_2PID import pid

from exploration_algorithms.utils import *
from explauto_master.explauto.models.dataset import *
from explauto_master.explauto.utils.config import make_configuration

path = "./test_rgb/"
iterations = 300000
save_iterations = 20000
intervall_measures = 10000
motor_babbling_iterations = 10
noise = False

print("iterations: ", iterations)
print("saving iterations: ", save_iterations)
print("saving to: ", path)
print("measuring intervall: ", intervall_measures)
print("noise:", noise)



goal_size = 2.
environment = My_Very_Own_Complex_Environment_and_Joints(n_joints=3,  # Number of joints
                                                         n_dmp_basis=3,  # Number of basis per joint
                                                         goal_size=goal_size,  # Size of goal space
                                                         stick_handle_tol=0.05, # Maximal distance to grab the stick with the hand
                                                         stick_length=0.30, # Stick length,
                                                         ball_size=0.10 # Maximal distance to grab the ball with the stick
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
s_all = list(range(m_ndims + 9, m_ndims + 54))

min_m = -3.
max_m = 3.
min_s = - goal_size
max_s = goal_size


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

def get_dictionary(probabilities):
    return {(float(i), float(j), float(k)): float(probabilities[i][j][k]) for i, prob_i in enumerate(probabilities) for j, prob_j in enumerate(prob_i) for k, _ in enumerate(prob_j)}


def compute_individuality(s_n_values, s__n_values, e_n_values, min_s, max_s, min_e, max_e):

    p__s_s_e = compute_probability((s__n_values, s_n_values, e_n_values), (min_s, min_s, min_e), (max_s, max_s, max_e),
                                   (3, 3, 2))
    prob_dict = get_dictionary(p__s_s_e)

    pid_data = pid(prob_dict)
    organismal = pid_data['SI'] + pid_data['UIY']
    colonial = pid_data['CI'] + pid_data['UIY']
    environmental_determined = pid_data['CI'] + pid_data['UIZ']
    env_coding = pid_data['SI'] - pid_data['CI']
    return organismal, colonial, environmental_determined, env_coding, pid_data


m_old = None
s_old = None
history_env = Dataset(3, 13)
history_system = Dataset(3, 3)

conf = make_configuration(environment.conf.m_mins[m_space],
                          environment.conf.m_maxs[m_space],
                          array(list(environment.conf.m_mins[m_space]) + list(environment.conf.s_mins))[s_all],
                          array(list(environment.conf.m_maxs[m_space]) + list(environment.conf.s_maxs))[s_all])

# Initialization of the sensorimotor model
sm_cls, kwargs = (NonParametric, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.05})
sm_model = sm_cls(conf, **kwargs)

iteration = 0
while not sm_model.bootstrapped_s or iteration < motor_babbling_iterations:
    m = environment.random_motor()
    s = environment.update(m)
    iteration += 1
    sm_model.update(m, s[9:])

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

        history_env = Dataset(3, 13)
        history_system = Dataset(3, 3)

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


    s_goal = rand_bounds(environment.conf.s_bounds)[0]
    sm_model.mode = "explore"
    m = sm_model.infer(sm_model.conf.s_dims, sm_model.conf.m_dims, s_goal[9:])
    s = environment.update(m)

    if noise:
        m_timestep = m
        s_timestep = s
    else:
        sm_model.mode = "exploit"
        m_timestep = sm_model.infer(sm_model.conf.s_dims, sm_model.conf.m_dims, s_goal[9:])
        s_timestep = environment.update(m)


    m_timestep = s[6:9]
    s_timestep = np.append(
        np.append(np.append(np.append(np.append(np.append(s[19:21], s[25:27]), s[31:33]), s[37:39]), s[43:45]),
                  s[49:51]), [s[53]])

    if m_old is not None:
        history_env.add_xy(m_old, s_old)
        history_system.add_xy(m_old, m_timestep)
    m_old = m_timestep
    s_old = s_timestep

    sm_model.update(m, s[9:])
