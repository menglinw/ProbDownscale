import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# import probdownscale.utils.data_processing as data_processing
from itertools import product
from random import sample


class TaskExtractor():
    def __init__(self, data, lats_lons, task_dim, test_proportion, n_lag):
        # initialize all arguments
        # high and low resolution data
        self.h_data, self.l_data = data
        self.h_data = ((self.h_data - self.h_data.min())/(self.h_data.max()- self.h_data.min()))
        self.l_data = ((self.l_data - self.l_data.min())/(self.l_data.max() - self.l_data.min()))
        # high and low resolution latitude and longitude
        self.h_lats, self.h_lons, self.l_lats, self.l_lons = lats_lons
        # task dimension
        if isinstance(task_dim, int):
            self.task_dim = [task_dim, task_dim]
        elif isinstance(task_dim, list) and len(task_dim)==2:
            self.task_dim = task_dim
        else:
            raise ValueError
        # test proportion within each data
        self.test_proportion = test_proportion
        # number of lagging days
        self.n_lag = int(n_lag)

        # all available topleft index of tasks
        avlb_lats = self.h_lats[:len(self.h_lats)-(self.task_dim[0]-1)]
        avlb_lons = self.h_lons[:len(self.h_lons)-(self.task_dim[1]-1)]
        self.avlb_location = list(product(avlb_lats, avlb_lons))
        # collect all topleft index we have seen
        self.seen_location = dict()

    def get_seen(self):
        return self.seen_location

    def _get_random_topleft_index(self, record=True):
        '''
        get a random topleft index of a task
        :param record: True or False, record the task index to seen tasks
        :return:
        '''
        sample_index = sample(self.avlb_location, 1)[0]
        if record:
            self.seen_location.setdefault(sample_index, 0)
            self.seen_location[sample_index] += 1
        return sample_index

    def _get_one_random_task(self, is_random=True, record=True, lat_lon=None, is_seq=True, use_all_data=False,
                             return_init=False):
        if is_random:
            # get random topleft index
            topleft_location = self._get_random_topleft_index(record=record)
        else:
            topleft_location = lat_lon
        # get high resolution data
        lat_index = list(self.h_lats).index(topleft_location[0])
        lon_index = list(self.h_lons).index(topleft_location[1])
        h_data = self.h_data[:, lat_index:(lat_index+self.task_dim[0]), lon_index:(lon_index+self.task_dim[1])]

        # get low resolution data
        l_data = np.zeros_like(h_data)
        for i, lat_idx in enumerate(range(lat_index, lat_index+self.task_dim[0])):
            for j, lon_idx in enumerate(range(lon_index, (lon_index+self.task_dim[1]))):
                lat = self.h_lats[lat_idx]
                lon = self.h_lons[lon_idx]
                l_lat_idx = np.argmin(np.abs(self.l_lats - lat))
                l_lon_idx = np.argmin(np.abs(self.l_lons - lon))
                l_data[:, i, j] = self.l_data[:, l_lat_idx, l_lon_idx]

        avlb_y = list(range(h_data.shape[0]))[self.n_lag:]
        # train test split
        if use_all_data:
            train_y_day = avlb_y
            test_y_day = []
        else:
            if is_seq:
                test_y_day = avlb_y[-int(len(avlb_y) * self.test_proportion):]
                train_y_day = list(set(avlb_y).difference(set(test_y_day)))
            else:
                test_y_day = sample(avlb_y, int(len(avlb_y) * self.test_proportion))
                train_y_day = list(set(avlb_y).difference(set(test_y_day)))


        train_y_day = sample(train_y_day, len(train_y_day))
        #print('Test Y Index:', test_y_day)
        #print('Train Y Index:', train_y_day)
        # flatten the data
        # output dim (time, channel, rows, cols)
        if self.task_dim == [1, 1]:
            train_y = np.squeeze(h_data[train_y_day], (-1, -2))
            test_y = np.squeeze(h_data[test_y_day], (-1, -2))
        else:
            train_y = h_data[train_y_day]
            test_y = h_data[test_y_day]

        train_x_1, train_x_2, train_x_3 = self._get_inputX(train_y_day, h_data, l_data)
        test_x_1, test_x_2, test_x_3 = self._get_inputX(test_y_day, h_data, l_data)
        init_1 = h_data[-self.n_lag:]
        if self.task_dim != [1, 1]:
            init_1 = np.expand_dims(init_1, [0, -1])
        else:
            init_1 = np.expand_dims(init_1, 0)
            init_1 = np.squeeze(init_1, -1)
        init_3 = np.remainder(np.array([h_data.shape[0]]), 365)
        if return_init:
            return [train_x_1, train_x_2, train_x_3], train_y, [test_x_1, test_x_2, test_x_3], test_y, \
                   topleft_location, [init_1, init_3]
        else:
            return [train_x_1, train_x_2, train_x_3], train_y, [test_x_1, test_x_2, test_x_3], test_y, topleft_location


    def _get_inputX(self, y_index, h_data, l_data):
        # input 1: HR temporal input
        # input 2: LR image input
        # input 3: day of the year
        train_x_1 = np.zeros((len(y_index), self.n_lag, self.task_dim[0], self.task_dim[1], 1))
        for i, indx in enumerate(y_index):
            train_x_1[i, :, :, :, :] = np.expand_dims(h_data[int(indx-self.n_lag):int(indx)], -1)
        train_x_2 = l_data[y_index]
        train_x_3 = np.remainder(np.array(y_index), 365)
        if self.task_dim==[1, 1]:
            return np.squeeze(train_x_1, (-1, -2)), np.squeeze(train_x_2, -1), train_x_3
        else:
            return train_x_1, train_x_2, train_x_3

    def get_random_tasks(self, n_task=None, record=True, locations=None):
        if not locations:
            train_x, train_y, test_x, test_y, locations= [], [], [], [], []
            for _ in range(n_task):
                x1, y1, x2, y2, location = self._get_one_random_task(record=record)
                train_x.append(x1)
                train_y.append(y1)
                test_x.append(x2)
                test_y.append(y2)
                locations.append(location)
        else:
            train_x, train_y, test_x, test_y = [], [], [], []
            for location in locations:
                x1, y1, x2, y2, _ = self._get_one_random_task(record=record, is_random=False, lat_lon=location)
                train_x.append(x1)
                train_y.append(y1)
                test_x.append(x2)
                test_y.append(y2)
        return train_x, train_y, test_x, test_y, locations

    def get_grid_locations(self):
        # all available topleft index of tasks
        avlb_lats = self.h_lats[:len(self.h_lats)-(self.task_dim[0]-1)]
        avlb_lons = self.h_lons[:len(self.h_lons)-(self.task_dim[1]-1)]
        avlb_lats = [avlb_lats[i*self.task_dim[0]] for i in range(int(len(avlb_lats)/self.task_dim[0]))]
        avlb_lons = [avlb_lons[i*self.task_dim[1]] for i in range(int(len(avlb_lons)/self.task_dim[1]))]
        return list(product(avlb_lats, avlb_lons))





