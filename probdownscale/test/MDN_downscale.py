import time
import numpy as np
import netCDF4 as nc
import probdownscale.utils.data_processing as data_processing
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Dropout, LSTM
from tensorflow.keras.models import Model
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import geopandas as gpd
import sys
import os
import copy
import pandas as pd
import h5py


class country_model():
    def __init__(self, file_path_g_05, file_path_g_06, file_path_m, file_path_ele, file_path_country,
                 data_season: str, test_proportion, with_trans, with_merra, forward, n_lag,
                 target_variable):
        '''

        :param file_path_g_05:
        :param file_path_g_06:
        :param file_path_m:
        :param file_path_ele:
        :param file_path_country:
        :param data_season: string, season of the data used to model (1,2,3,4, all)
        :param test_season: the data used to train are divided into 10% + 3*30%, test_season is the number of 30% used as test and validation
        :param with_trans: indicator of with transformation
        :param with_merra: indicator of with merra data
        :param forward: indicator of direction (forward or backward)
        :param n_lag: number of lagging day used in the model
        :param target_variable: target variable to model
        '''
        # define path and target variable
        self.with_trans = True if with_trans == 'with_trans' else False
        self.forward = True if forward == 'forward' else False
        self.with_merra = True if with_merra == 'with_merra' else False
        self.test_proportion = test_proportion
        self.n_lag = int(n_lag)
        self.target_var = target_variable
        self.data_season = data_season
        # self.file_path_g_05 = file_path_g_05
        # self.file_path_g_06 = file_path_g_06
        # self.file_path_m = file_path_m
        # self.file_path_ele = file_path_ele

        # output the setting for checking purpose
        print('with merra: ', self.with_merra)
        print('With trans:', self.with_trans)
        print('Forward: ', self.forward)
        print('Data season: ', self.data_season)
        print('Test proportion: ', self.test_proportion)
        print('Number of lag day: ', self.n_lag)
        print('Target variable: ', self.target_var)

        # define shape of target area
        if len(file_path_country) == 1:
            self.country_shape = gpd.read_file(file_path_country[0])
        else:
            self.country_shape = gpd.read_file(file_path_country[0])
            for country_path in file_path_country[1:]:
                self.country_shape = pd.concat([self.country_shape, gpd.read_file(country_path)])

        # define transfered model
        if self.with_trans:
            within_merra_model = keras.models.load_model(r'within_merra_model.h5', compile=False)
            self.merra_model = Model(within_merra_model.input, within_merra_model.output)
            self.merra_model.trainable = False

        # read g5nr data
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        self.g_data = np.concatenate((self._data_g5nr_to_array(g05_data), self._data_g5nr_to_array(g06_data)), axis=0)
        self.g_data = np.log(self.g_data)
        self.g_data = (self.g_data - self.g_data.min()) / (np.quantile(self.g_data, 0.95) - self.g_data.min())
        # read merra data as nc
        m_data = nc.Dataset(file_path_m)
        self.m_data = m_data.variables[self.target_var][range(1826, 1826 + 730), :, :]
        self.m_data = np.log(self.m_data)
        self.m_data = (self.m_data - self.m_data.min()) / (np.quantile(self.m_data, 0.95) - self.m_data.min())
        # self.m_data = m_data.variables[self.target_var][range(1826, 1826+730), :, :]

        # define lat&lon of MERRA, G5NR and mete
        self.M_lons = m_data.variables['lon'][:]
        # self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        self.M_lats = m_data.variables['lat'][:]
        # self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
        self.G_lons = g05_data.variables['lon'][:]
        # self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
        self.G_lats = g05_data.variables['lat'][:]

        # read elevation data as array
        self.ele_data = np.load(file_path_ele)
        self.ele_data, sub_lats, sub_lons = data_processing.country_filter(self.ele_data, self.G_lats,
                                                                           self.G_lons, self.country_shape)
        # save lats and lons after clipping
        np.save('subimg_lats', sub_lats)
        np.save('subimg_lons', sub_lons)

        # define NN model
        self.model = self.define_model()

    def _data_g5nr_to_array(self, nc_data, time_start=0, time_length=365):
        time_interval = range(time_start, time_start + time_length)
        out = nc_data.variables[self.target_var][:][time_interval, :, :]
        return out

    def _train_test_split(self):
        '''
        self.g_data: ndarray
        self.mete_data: dict
        self.m_data: nc
        '''
        # all index after are based on base_index, so that wouldn't worry about direction
        if self.forward:
            base_index = np.arange(self.g_data.shape[0])
        else:
            base_index = np.arange(self.g_data.shape[0] - 1, -1, -1)

        # get index of data, season or all data
        def season_to_index(season, days_of_season, overlap_day):
            start1 = max(sum(days_of_season[:max(season - 1, 0)]) - overlap_day, 0)
            end1 = sum(days_of_season[:min(season, 4)])
            start2 = 365 + sum(days_of_season[:max(season - 1, 0)]) - overlap_day
            end2 = 365 + end1
            return [np.arange(start1, end1), np.arange(start2, end2)]

        days_of_season = [91, 91, 91, 92]
        if self.data_season == 'all':
            season_index = [np.arange(len(base_index))]
        else:
            if int(self.data_season) in [1, 2, 3, 4]:
                season_index = season_to_index(int(self.data_season), days_of_season, overlap_day=45)
            else:
                raise ValueError

        # get test index, test and validation share the same size
        n = sum([len(i) for i in season_index])
        n_test = int((n - self.n_lag * len(season_index)) * self.test_proportion)
        n_valid = n_test
        if len(season_index) == 1:
            test_y_index = season_index[0][-n_test:]
            valid_y_index = season_index[0][-(n_test + n_valid):-n_test]
            train_y_index = season_index[0][self.n_lag:-(n_test + n_valid)]
        else:
            test_y_index = season_index[1][-n_test:]
            valid_y_index = season_index[0][-(n_test + n_valid):-n_test]
            train_y_index = [season_index[0][self.n_lag:-(n_test + n_valid)],
                             season_index[0][-n_test:],
                             season_index[1][self.n_lag:-n_test]]

        def get_x_index(y_index, n_lag):
            xm_index = y_index
            xg_index = np.zeros((len(y_index), n_lag))
            xg_index.astype('int64')
            for i in range(len(y_index)):
                for j in range(n_lag):
                    xg_index[i, j] = int(y_index[i] - j - 1)
            return xg_index, xm_index

        # get x index, including xg_index, xm_index
        test_xg_index, test_xm_index = get_x_index(test_y_index, self.n_lag)
        valid_xg_index, valid_xm_index = get_x_index(valid_y_index, self.n_lag)
        if self.data_season == 'all':
            train_xg_index, train_xm_index = get_x_index(train_y_index, self.n_lag)
        else:
            train_xg_index, train_xm_index = [], []
            for subindex in train_y_index:
                sub_xg_index, sub_xm_index = get_x_index(subindex, self.n_lag)
                train_xg_index.append(sub_xg_index)
                train_xm_index.append(sub_xm_index)

        # from mask index to real index
        def map_to_baseindex(target_index, b_index):
            '''map 2d array to base index'''
            out = np.zeros_like(target_index)
            out.astype('int64')
            for i in range(target_index.shape[0]):
                for j in range(target_index.shape[1]):
                    out[i, j] = b_index[int(target_index[i, j])]
            return out

        self.test_y_index = base_index[test_y_index]
        self.test_xg_index = map_to_baseindex(test_xg_index, base_index)
        self.test_xm_index = base_index[test_xm_index]

        self.valid_y_index = base_index[valid_y_index]
        self.valid_xg_index = map_to_baseindex(valid_xg_index, base_index)
        self.valid_xm_index = base_index[valid_xm_index]

        if self.data_season == 'all':
            self.train_xg_index = map_to_baseindex(train_xg_index, base_index)
            self.train_y_index = base_index[train_y_index]
            self.train_xm_index = base_index[train_xm_index]
        else:
            self.train_y_index = []
            self.train_xg_index = []
            self.train_xm_index = []
            for i in range(len(train_y_index)):
                self.train_y_index.append(base_index[train_y_index[i]])
                self.train_xm_index.append(base_index[train_xm_index[i]])
                self.train_xg_index.append(map_to_baseindex(train_xg_index[i], base_index))

    def _flatten(self, xg_index, xm_index, y_index, is_perm=True, clean_nan=True):
        """

        Parameters
        ----------
        xg_index
        xm_index
        y_index
        is_perm
        clean_nan

        Returns
        -------
        flattened data: m_AOD, lat, lon, day of year, elev, lag AOD
        """
        sample_data, _, _ = data_processing.country_filter(self.g_data[0, :, :], self.G_lats,
                                                           self.G_lons, self.country_shape)
        single_img_size = np.prod(sample_data.shape)
        self.single_img_size = single_img_size
        # define x, y shape
        nvar_x = 4 + self.n_lag + 1 if self.with_merra else 4 + self.n_lag
        x = np.zeros((len(y_index) * single_img_size, nvar_x))
        y = np.zeros((len(y_index) * single_img_size, 1))

        # iter to process data
        for i in range(len(y_index)):
            # process y
            outy, _, _ = data_processing.country_filter(self.g_data[y_index[i]], self.G_lats, self.G_lons,
                                                        self.country_shape)
            y[i * single_img_size:(i + 1) * single_img_size] = outy.reshape((np.prod(outy.shape), 1))

            # process x
            x_list = xg_index[i]
            image, lats, lons = data_processing.country_filter(self.g_data[int(x_list[0])], self.G_lats,
                                                               self.G_lons, self.country_shape)
            outx_table = data_processing.image_to_table(image,
                                                        (lats - lats.min()) / (lats.max() - lats.min()),
                                                        (lons - lons.min()) / (lons.max() - lons.min()),
                                                        ((x_list[0] % 365) + 1) / 365, self.ele_data)
            for x_day in x_list[1:]:
                image, lats, lons = data_processing.country_filter(self.g_data[int(x_day)], self.G_lats,
                                                                   self.G_lons, self.country_shape)
                outx_table = np.concatenate((outx_table, image.reshape((np.prod(image.shape), 1))), 1)

            if self.with_merra:
                xm_img = data_processing.resolution_downward(self.m_data[xm_index[i], :, :], self.M_lats, self.M_lons,
                                                             self.G_lats, self.G_lons)
                xm_img, _, _ = data_processing.country_filter(xm_img, self.G_lats, self.G_lons, self.country_shape)
                outx_table = np.concatenate((xm_img.reshape((np.prod(xm_img.shape), 1)), outx_table), 1)
            x[i * single_img_size:(i + 1) * single_img_size] = outx_table

        if clean_nan:
            if is_perm:
                perm = np.random.permutation(x.shape[0])
                x = x[perm]
                y = y[perm]
            return x[~np.isnan(x[:, 0])], y[~np.isnan(y[:, 0])]
        else:
            return x, y, [len(y_index), sample_data.shape[0], sample_data.shape[1]]

    def _process(self):
        if 'data.h5' in os.listdir():
            print('start processing')
            self._train_test_split()
            print('-----------finished train test split-----------')
            h5f = h5py.File('data.h5', 'r')
            self.train_x = np.array(h5f['train_x'])
            self.train_y = np.array(h5f['train_y'])
            self.test_x = np.array(h5f['test_x'])
            self.test_y = np.array(h5f['test_y'])
            self.valid_x = np.array(h5f['valid_x'])
            self.valid_y = np.array(h5f['valid_y'])
            self.shape = np.load('test_shape.npy')
            self.single_img_size = self.shape[1] * self.shape[2]
            print('-----------finished read data process-----------')
        else:
            # processing data
            print('start processing')
            self._train_test_split()
            print('-----------finished train test split-----------')

            # h5f = h5py.File('data.h5', 'w')
            # process test data
            self.test_x, self.test_y, self.shape = self._flatten(self.test_xg_index, self.test_xm_index,
                                                                 self.test_y_index, is_perm=True, clean_nan=False)
            # h5f.create_dataset('test_x', data=self.test_x)
            # h5f.create_dataset('test_y', data=self.test_y)
            np.save('test_shape', self.shape)
            print('-----------finished test set-------------')
            # process validation data
            self.valid_x, self.valid_y = self._flatten(self.valid_xg_index, self.valid_xm_index,
                                                       self.valid_y_index, is_perm=True, clean_nan=True)
            # h5f.create_dataset('valid_x', data=self.valid_x)
            # h5f.create_dataset('valid_y', data=self.valid_y)
            print('-----------finished valid set-------------')
            # process training data
            if self.data_season == 'all':
                self.train_x, self.train_y = self._flatten(self.train_xg_index, self.train_xm_index,
                                                           self.train_y_index, is_perm=True, clean_nan=True)
            else:
                nvar_x = 4 + self.n_lag + 1 if self.with_merra else 4 + self.n_lag
                self.train_x = np.zeros((0, nvar_x))
                self.train_y = np.zeros((0, 1))
                for i in range(len(self.train_y_index)):
                    train_x, train_y = self._flatten(self.train_xg_index[i], self.train_xm_index[i],
                                                     self.train_y_index[i], is_perm=True, clean_nan=True)
                    self.train_x = np.concatenate([self.train_x, train_x], 0)
                    self.train_y = np.concatenate([self.train_y, train_y], 0)
            # h5f.create_dataset('train_x', data=self.train_x)
            # h5f.create_dataset('train_y', data=self.train_y)
            # h5f.close()
            print('-----------finished train set-------------')

    def train(self, epochs=100):
        self._process()
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='final_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
            )
        ]
        # '''
        history = self.model.fit([self.train_x[:, 5:].reshape((self.train_x.shape[0], 1, self.n_lag)),
                                  self.train_x[:, :5]], self.train_y, epochs=epochs,
                                 validation_data=([self.valid_x[:, 5:].reshape((self.valid_x.shape[0], 1, self.n_lag)),
                                                   self.valid_x[:, :5]], self.valid_y),
                                 batch_size=1000,
                                 callbacks=callbacks_list
                                 )
        fig = plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label="train loss")
        plt.plot(history.history['val_loss'], label='val loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss_curve.jpg')

    def define_model(self):
        def unit_layer(nodes, input, with_dropout=False, activation='LeakyReLU'):
            if activation == 'LeakyReLU':
                x = layers.Dense(nodes, kernel_initializer="he_normal", use_bias=True)(input)
                x = BatchNormalization()(x)
                if with_dropout:
                    x = Dropout(0.5)(x)
                x = LeakyReLU(alpha=0.1)(x)
            else:
                x = layers.Dense(nodes, kernel_initializer="he_normal", activation=activation, use_bias=True)(input)
                x = BatchNormalization()(x)
                if with_dropout:
                    x = Dropout(0.5)(x)
            return x

        def temporal_block(input):
            x = LSTM(8, return_sequences=True, activation=LeakyReLU(), use_bias=True, kernel_initializer="he_normal")(
                input)
            x = LSTM(16, return_sequences=True, activation=LeakyReLU(), use_bias=True, kernel_initializer="he_normal")(
                x)
            x = LSTM(32, activation=LeakyReLU(), use_bias=True, kernel_initializer="he_normal")(x)
            x = unit_layer(16, x)
            # x = Reshape((1, 16))(x)
            # x = Lambda(lambda y: K.batch_flatten(y))(x)
            return x

        def process_block(input):
            x = unit_layer(8, input)
            x = unit_layer(8, x)
            x = unit_layer(16, x)
            x = unit_layer(16, x)
            x = unit_layer(16, x)
            return x

        def nnelu(input):
            """ Computes the Non-Negative Exponential Linear Unit
            """
            return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

        components = 100
        no_parameters = 3
        def slice_parameter_vectors(parameter_vector):
            """ Returns an unpacked list of paramter vectors.
            """
            return [parameter_vector[:, i * components:(i + 1) * components] for i in range(no_parameters)]

        def gnll_loss(y, parameter_vector):
            """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
            """
            alpha, mu, sigma = slice_parameter_vectors(parameter_vector)  # Unpack parameter vectors

            gm = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.Normal(
                    loc=mu,
                    scale=sigma))

            log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

            return -tf.reduce_mean(log_likelihood, axis=-1)

        tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

        input1 = Input(shape=(1, self.n_lag))  # input of AOD
        input2 = Input(shape=(4 + 1,
                              )) if self.with_merra else Input(
            shape=(4 + self.n_lag))  # input of lat, lon, day, elev, merra AOD

        if self.with_trans:
            encode1 = self.merra_model(input1)
            encode1 = unit_layer(16, encode1)
            encode2 = process_block(input2)
            encode3 = temporal_block(input1)
            encode4 = temporal_block(input1)
            X = layers.Concatenate(axis=1)([encode1, encode2, encode3, encode4])
        else:
            encode1 = temporal_block(input1)
            encode2 = process_block(input2)
            encode3 = temporal_block(input1)
            X = layers.Concatenate(axis=1)([encode1, encode2, encode3])
        X1 = unit_layer(64, X)
        X2 = unit_layer(128, X1)
        X = unit_layer(256, X2)
        # X = unit_layer(512, X)
        # X = unit_layer(256, X)

        X = unit_layer(128, X)
        X = layers.Concatenate(axis=1)([X, X2])
        X = unit_layer(64, X)
        X = layers.Concatenate(axis=1)([X, X1])
        X = unit_layer(32, X)
        X = unit_layer(16, X)
        X = unit_layer(8, X)
        # X = layers.Dense(1, activation=lambda x: keras.activations.relu(x, max_value=6))(X)
        # X = layers.Dense(1, activation='relu')(X)
        alphas = layers.Dense(components, activation="softmax", name="alphas")(X)
        mus = layers.Dense(components, name="mus")(X)
        sigmas = layers.Dense(components, activation="nnelu", name="sigmas")(X)
        output = layers.Concatenate()([alphas, mus, sigmas])
        model = Model([input1, input2], output)
        model.compile(optimizer='adam', loss=gnll_loss)
        return model

    def _evaluate(self, pred_data, true_data):
        if pred_data.shape != true_data.shape:
            print('Please check data consistency!')
            raise ValueError
        RMSE_out = np.zeros(pred_data.shape[1:])
        R2_out = np.zeros(pred_data.shape[1:])
        for i in range(pred_data.shape[1]):
            for j in range(pred_data.shape[2]):
                if np.isnan(true_data[0, i, j]):
                    RMSE_out[i, j] = float("nan")
                    R2_out[i, j] = float("nan")
                else:
                    RMSE_out[i, j] = np.square(pred_data[:, i, j] - true_data[:, i, j]).mean()
                    R2_out[i, j], _ = nc_process.rsquared(pred_data[:, i, j], true_data[:, i, j])
        return RMSE_out, R2_out

    def predict(self):
        # normal predict
        h5f = h5py.File('pred_result.h5', 'w')
        test_x, test_y = self.test_x, self.test_y
        shape = [len(self.test_y_index), self.shape[1], self.shape[2]]
        temp = test_x[~np.isnan(test_x[:, 0])]
        n_pred_y = self.model.predict([temp[:, 5:].reshape((temp.shape[0], 1, self.n_lag)),
                                       temp[:, :5]])
        template_y = copy.deepcopy(test_y)
        template_y[~np.isnan(template_y[:, 0])] = n_pred_y
        n_pred_y = copy.deepcopy(template_y)
        # RMSE_mat, R2_mat = self._evaluate(n_pred_y.reshape(shape), test_y.reshape(shape))
        # np.save('RMSE_mat', RMSE_mat)
        # np.save('R2_mat', R2_mat)
        # n_pred_y.reshape(shape).dump('normal_pred_y')
        h5f.create_dataset('normal_pred_y', data=n_pred_y.reshape(shape))
        h5f.create_dataset('true_y', data=test_y.reshape(shape))

        # sequential predict
        lag_g_columns = list(range(5, 5 + self.n_lag))
        y_hat = test_x[:self.single_img_size, lag_g_columns].reshape((self.single_img_size, len(lag_g_columns)))
        filter = ~np.isnan(y_hat[:, 0])
        y_hat = y_hat[filter]
        n_withoutnan = y_hat.shape[0]
        y_pred = np.zeros((len(self.test_y_index) * n_withoutnan, 1))
        for i in range(len(self.test_y_index)):
            step_pred = self.model.predict([y_hat.reshape((n_withoutnan, 1, len(lag_g_columns))),
                                            test_x[i * self.single_img_size:(1 + i) * self.single_img_size, :5][
                                                filter]])
            y_pred[i * y_hat.shape[0]:(1 + i) * y_hat.shape[0]] = step_pred
            y_hat[:, 1:] = y_hat[:, :-1]
            y_hat[:, 0] = step_pred[:, 0]
        template_y[~np.isnan(template_y[:, 0])] = y_pred
        # seq_RMSE_mat, seq_R2_mat = self._evaluate(template_y.reshape(shape), test_y.reshape(shape))
        # np.save('seq_RMSE_mat', seq_RMSE_mat)
        # np.save('seq_R2_mat', seq_R2_mat)
        # template_y.reshape(shape).dump('pred_y')
        # test_y.reshape(shape).dump('true_y')
        h5f.create_dataset('sequential_pred_y', data=template_y.reshape(shape))
        h5f.close()


if __name__ == "__main__":
    start = time.time()
    file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_ele = '/project/mereditf_284/menglin/Downscale_data/ELEV/elevation_data.npy'
    file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp']
    test_proportion = 0.1

    # input argument: data_season, with_trans, with_merra, forward, target_var, country code
    data_season = sys.argv[1]
    with_trans = sys.argv[2]
    with_merra = sys.argv[3]
    forward = sys.argv[4]
    target_var = sys.argv[5]
    country_code = sys.argv[6]  # 1: AFG, 2: ARE + QAT, 3: IRQ_KWT, 4: SAU, 5: ALL
    if country_code == '5':
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']
    elif country_code == '4':
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']
    elif country_code == '3':
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp']
    elif country_code == '2':
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp']
    elif country_code == '1':
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp']

    # merra_var = ['BCEXTTAU', 'DUEXTTAU', 'OCEXTTAU', 'SUEXTTAU', 'TOTEXTTAU', 'BCSMASS', 'DUSMASS25', 'DUSMASS',
    #             'OCSMASS', 'SO4SMASS', 'SSSMASS']
    model = country_model(file_path_g_05, file_path_g_06, file_path_m, file_path_ele, file_path_country,
                          data_season, test_proportion, with_trans, with_merra, forward, n_lag=25,
                          target_variable=target_var)
    model.train(epochs=100)
    print('Train time:', (time.time() - start) / 60, 'mins')
    model.predict()
    print('Duriation:', (time.time() - start) / 60, 'mins')
