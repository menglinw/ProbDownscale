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
# .py save_path data_part use_beta use_meta
save_path = sys.argv[1]
target_var = 'TOTEXTTAU'
data_part = sys.argv[2]
if sys.argv[3]=='beta':
    use_beta = True
else:
    use_beta = False

use_meta = True if sys.argv[4] == 'meta' else False
meta_downscaler = run_metadownscale(task_dim, test_proportion, n_lag, components, save_path, target_var, data_part,
                                    use_beta, use_meta)
if use_meta:
    meta_downscaler.meta_train(5, 5, 0.0005, prob=True)
    meta_downscaler.meta_train(5, 5, 0.001, prob=False)
    #meta_downscaler.downscale(50, prob_use_meta=True, reg_use_meta=True)
else:
    pass
    #meta_downscaler.downscale(50, prob_use_meta=False, reg_use_meta=False)
