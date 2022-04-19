import numpy as np
import netCDF4 as nc
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import probdownscale.utils.data_processing as data_processing
import geopandas as gpd
import sys
import os
import copy
import pandas as pd
import h5py

class TaskExtractor():
    def __init__(self, data, n_task, test_proportion):
        self.data = data
        self.n_task = n_task
        self.test_proportion = test_proportion

    def draw_sample(self):
        pass

