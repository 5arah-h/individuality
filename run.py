import sys
from explauto_master.utils import *
import pickle

condition = sys.argv[1]
trial = int(sys.argv[2])
iterations = int(sys.argv[3])

log_dir = './logs/'
filename = condition + str(trial) + '.pickle'
    
    
if condition == 'rmb':
    res = random_motor_babbling(trial, iterations)
elif condition == 'rgb':
    res = random_goal_babbling(trial, iterations)
elif condition == 'amb':
    res = active_model_babbling(trial, iterations)
    
print(condition, trial, iterations, "result:", res)

with open(log_dir + filename, 'w') as f:
    pickle.dump(res, f)