
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np
from random import sample
from tensorflow_probability import distributions as tfd
from itertools import product


class Downscaler():
    def __init__(self, meta_learner, components, l_data):
        self.meta_learner = meta_learner
        self.meta_weights = meta_learner.meta_model.get_weights()
        self.meta_model = meta_learner.meta_model
        self.loss = meta_learner.loss
        self.task_extractor = meta_learner.task_extractor
        self.components = components
        self.task_dim = self.task_extractor.task_dim
        self.n_lag = self.task_extractor.n_lag
        self.l_data = l_data

    def _creat_task_l_data(self, topleft_location):
        l_data = np.zeros((self.l_data.shape[0], self.task_dim[0], self.task_dim[1]))
        lat_index = list(self.task_extractor.h_lats).index(topleft_location[0])
        lon_index = list(self.task_extractor.h_lons).index(topleft_location[1])
        for i, lat_idx in enumerate(range(lat_index, lat_index+self.task_dim[0])):
            for j, lon_idx in enumerate(range(lon_index, (lon_index+self.task_dim[1]))):
                lat = self.task_extractor.h_lats[lat_idx]
                lon = self.task_extractor.h_lons[lon_idx]
                l_lat_idx = np.argmin(np.abs(self.task_extractor.l_lats - lat))
                l_lon_idx = np.argmin(np.abs(self.task_extractor.l_lons - lon))
                l_data[:, i, j] = self.l_data[:, l_lat_idx, l_lon_idx]
        return l_data

    def downscale(self, fine_tune_epochs, optimizer):
        # TODO: need to debug
        # initialize the meta model
        self.meta_model.compile(optimizer=optimizer, loss=self.loss)
        self.meta_model.set_weights(self.meta_weights)

        # iter over task to downscale
        downscaled_data = np.zeros((self.l_data.shape[0]-1,
                                    self.task_extractor.h_data.shape[1],
                                    self.task_extractor.h_data.shape[2]))

        lats = [self.task_extractor.h_lats[int(i*self.task_dim[0])] for i in range(int(len(self.task_extractor.h_lats)//self.task_dim[0]))]

        lons = [self.task_extractor.h_lons[int(i * self.task_dim[1])] for i in range(int(len(self.task_extractor.h_lons) // self.task_dim[1]))]
        locations = list(product(lats, lons))
        for location in locations:
            # initial the meta model
            self.meta_model.set_weights(self.meta_weights)
            temp = self._task_downscaler(location, epochs=fine_tune_epochs)
            lat_index = list(self.task_extractor.h_lats).index(location[0])
            lon_index = list(self.task_extractor.h_lons).index(location[1])
            downscaled_data[:, lat_index:(lat_index+self.task_dim[0]), lon_index:(lon_index+self.task_dim[1])] = temp
        return downscaled_data

    def _task_downscaler(self, location, epochs):
        ## TODO: need to debug
        train_x, train_y, _, _, location, init = self.task_extractor._get_one_random_task(is_random=False, record=False,
                                                                                    lat_lon=location, use_all_data=True,
                                                                                    return_init=True)
        l_data = self._creat_task_l_data(location)

        # initialize with meta weights
        self.meta_model.set_weights(self.meta_weights)

        # fine tune the meta model with task data
        # the meta_model should be compiled first
        self.meta_model.fit(train_x, train_y, epochs=epochs)

        # predict several steps
        pred = self.sequential_predict(self.meta_model, init, l_data, self.l_data.shape[0]-1)
        return pred

    def slice_parameter_vectors(self, parameter_vector, no_parameters):
        return [parameter_vector[:, i *self.components: (i + 1)*self.components] for i in range(no_parameters)]

    def sequential_predict(self, model, init_data, l_data, predict_steps):
        # TODO: need to debug
        init_1, init_3 = init_data
        init_2 = np.expand_dims(l_data[0], 0)
        for i in range(predict_steps):
            #print('Input 1 shape:', init_1[:, -self.n_lag:].shape)
            #print('Input 2 shape:', init_2.shape)
            #print('Input 3 shape:', init_3.shape)
            y_hat = model.predict([init_1[:, -self.n_lag:], init_2, init_3])
            alpha, mu, sigma = self.slice_parameter_vectors(y_hat, 3)
            MDN_Yhat = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.Gamma(
                    concentration=mu, rate=sigma))

            Yhat = MDN_Yhat.sample(np.prod(self.task_dim)).numpy().reshape(self.task_dim)
            Yhat = np.expand_dims(Yhat, [0,1, -1])
            init_1 = np.concatenate([init_1, Yhat], axis=1)
            init_2 = np.expand_dims(l_data[i+1], 0)
            init_3 = np.remainder(init_3+1, 365)
        
        return init_1[0, self.n_lag:, 0, :, :]

