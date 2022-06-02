
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np
from random import sample
from tensorflow_probability import distributions as tfd
from itertools import product


class Downscaler():
    def __init__(self, meta_learner_prob, meta_learner_reg, components, l_data):
        self.meta_learner_prob = meta_learner_prob
        self.meta_weights_prob = meta_learner_prob.meta_model.get_weights()
        self.meta_model_prob = meta_learner_prob.meta_model
        self.loss_prob = meta_learner_prob.loss


        self.meta_learner_reg = meta_learner_reg
        self.meta_weights_reg = meta_learner_reg.meta_model.get_weights()
        self.meta_model_reg = meta_learner_reg.meta_model
        self.loss_reg = meta_learner_reg.loss

        self.task_extractor = meta_learner_prob.task_extractor
        self.components = components
        self.task_dim = self.task_extractor.task_dim
        self.n_lag = self.task_extractor.n_lag
        self.l_data = l_data
        self.reg_epochs=[]
        self.prob_epochs=[]

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

    def downscale(self, fine_tune_epochs, optimizer, callbacks_prob=None, callbacks_reg=None):
        # TODO: need to debug
        # initialize the meta model
        self.meta_model_prob.compile(optimizer=optimizer, loss=self.loss_prob)
        self.meta_model_reg.compile(optimizer=optimizer, loss=self.loss_reg)


        # iter over task to downscale
        downscaled_data = np.zeros((self.l_data.shape[0],
                                        self.task_extractor.h_data.shape[1],
                                        self.task_extractor.h_data.shape[2]))

        lats = [self.task_extractor.h_lats[int(i*self.task_dim[0])] for i in range(int(len(self.task_extractor.h_lats)//self.task_dim[0]))]

        lons = [self.task_extractor.h_lons[int(i * self.task_dim[1])] for i in range(int(len(self.task_extractor.h_lons) // self.task_dim[1]))]
        locations = list(product(lats, lons))
        for location in locations:
            # initial the meta model
            self.meta_model_prob.set_weights(self.meta_weights_prob)
            self.meta_model_reg.set_weights(self.meta_weights_reg)
            temp = self._task_downscaler(location, epochs=fine_tune_epochs,
                                         callbacks_prob=callbacks_prob, callbacks_reg=callbacks_reg)
            lat_index = list(self.task_extractor.h_lats).index(location[0])
            lon_index = list(self.task_extractor.h_lons).index(location[1])
            if self.task_dim != [1, 1]:
                downscaled_data[:, lat_index:(lat_index+self.task_dim[0]), lon_index:(lon_index+self.task_dim[1])] = temp
            else:
                downscaled_data[:, lat_index, lon_index] = temp
        epochs_data = np.array([self.prob_epochs, self.reg_epochs])
        return downscaled_data, epochs_data

    def _task_downscaler(self, location, epochs, callbacks_prob=None, callbacks_reg=None):
        train_x, train_y, _, _, location, init = self.task_extractor._get_one_random_task(is_random=False, record=False,
                                                                                    lat_lon=location, use_all_data=True,
                                                                                    return_init=True)
        l_data = self._creat_task_l_data(location)

        # fine tune the meta model with task data
        # the meta_model should be compiled first
        self.meta_model_prob.fit(train_x, train_y, epochs=epochs, validation_split=0.2, callbacks=callbacks_prob)
        self.meta_model_reg.fit(train_x, train_y, epochs=epochs, validation_split=0.2, callbacks=callbacks_reg)
        if callbacks_prob:
            self.prob_epochs.append(callbacks_prob[1].stopped_epoch)
        if callbacks_reg:
            self.reg_epochs.append(callbacks_reg.stopped_epoch)
        # predict several steps
        pred = self.sequential_predict(init, l_data, self.l_data.shape[0])
        return pred

    def slice_parameter_vectors(self, parameter_vector):
        alphas = parameter_vector[:, :self.components]
        mus = parameter_vector[:, self.components:(self.components * (self.task_dim[0] * self.task_dim[1] + 1))]
        sigmas = parameter_vector[:, (self.components * (self.task_dim[0] * self.task_dim[1] + 1)):]
        return alphas, mus, sigmas

    def sequential_predict(self, init_data, l_data, predict_steps):
        # TODO: need to debug
        init_1, init_3 = init_data
        #print('shape 1:', init_1.shape)
        #init_2 = np.expand_dims(l_data[0], 0)
        #print('shape 2:', init_2.shape)
        #print('shape 3:', init_3.shape)
        if predict_steps > l_data.shape[0]:
            predict_steps = l_data.shape[0]
        for i in range(predict_steps):
            #print('Input 1 shape:', init_1[:, -self.n_lag:].shape)
            #print('Input 2 shape:', init_2.shape)
            #print('Input 3 shape:', init_3.shape)
            init_2 = np.expand_dims(l_data[i], 0)
            # predict with prob model
            y_hat = self.meta_model_prob.predict([init_1[:, -self.n_lag:], init_2, init_3])
            alphas, mus, sigmas = self.slice_parameter_vectors(y_hat)
            if self.task_dim != [1, 1]:
                mus = tf.reshape(mus, (tf.shape(mus)[0], self.components, self.task_dim[0], self.task_dim[1]))
                sigmas = tf.reshape(sigmas, (tf.shape(sigmas)[0], self.components, self.task_dim[0], self.task_dim[1]))
                MDN_Yhat = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=alphas),
                    components_distribution=tfd.Independent(tfd.Gamma(concentration=mus, rate=sigmas),
                                                            reinterpreted_batch_ndims=2))
            else:
                MDN_Yhat = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=alphas),
                    components_distribution=tfd.Gamma(concentration=mus, rate=sigmas))

            yhat = MDN_Yhat.sample().numpy()
            Yhat_prob = yhat

            # predict with regular model
            y_hat = self.meta_model_reg.predict([init_1[:, -self.n_lag:], init_2, init_3])
            Yhat_reg = y_hat

            # combine
            Yhat = (Yhat_prob + Yhat_reg)/2
            Yhat = np.expand_dims(Yhat, [0, -1])
            #print('init_1:', init_1.shape)
            #print('Y hat:', Yhat.shape)
            init_1 = np.concatenate([init_1, Yhat], axis=1)
            init_3 = np.remainder(init_3 + 1, 365)
        if self.task_dim != [1, 1]:
            return init_1[0, self.n_lag:, 0, :, :]
        else:
            return init_1[0, self.n_lag:, 0]

