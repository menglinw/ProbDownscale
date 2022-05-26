from utils.define_functions import run_metadownscale
import sys

# task dimension 3*3
task_dim = 3

# proportion of test data
test_proportion = 0.3

# number of lagging steps
n_lag = 30

# number of components of MDN
components=50

# save path
save_path = sys.argv[1]
target_var = sys.argv[2]
if sys.argv[3] == 'prob':
    prob = True
else:
    prob = False

meta_downscaler = run_metadownscale(task_dim, test_proportion, n_lag, components, save_path, target_var)
meta_downscaler.meta_train(3, 10, 0.0005, prob)