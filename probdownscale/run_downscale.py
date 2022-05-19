import os
import sys
from MetaTrain import MetaSGD
from TaskExtractor import TaskExtractor
from Downscaler import Downscaler
import math
import numpy as np
import netCDF4 as nc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from matplotlib import pyplot as plt
from math import exp, sqrt, log
import time

file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
target_var = 'BCSMASS'

# read data
g05_data = nc.Dataset(file_path_g_05)
g06_data = nc.Dataset(file_path_g_06)
m_data_nc = nc.Dataset(file_path_m)

# define lat&lon of MERRA, G5NR and mete
M_lons = m_data_nc.variables['lon'][:30]
# self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
M_lats = m_data_nc.variables['lat'][:30]
# self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
G_lons = g05_data.variables['lon'][:50]
# self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
G_lats = g05_data.variables['lat'][:50]

# extract target data
g_data = np.concatenate((g05_data.variables[target_var][:, :50, :50], g06_data.variables[target_var][:, :50, :50]), axis=0)
m_data = m_data_nc.variables[target_var][5*365:7*365, :30, :30]

# split data into traing and test
train_g_data, test_g_data = g_data[:657], g_data[657:]
train_m_data, test_m_data = m_data[:657], m_data[657:]
data = [train_g_data, train_m_data]
lats_lons = [G_lats, G_lons, M_lats, M_lons]
# task dimension 5 * 5
task_dim = 5

# proportion of test data
test_proportion = 0.3

# number of lagging steps
n_lag = 15

# number of components in Mixture Density Network
components = 500

# define necessary tool functions
def nnelu(input):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


tf.keras.utils.get_custom_objects().update({'nnelu': layers.Activation(nnelu)})


def slice_parameter_vectors(parameter_vector):
    alphas = parameter_vector[:, :components]
    mus = parameter_vector[:, components:(components * (task_dim * task_dim + 1))]
    sigmas = parameter_vector[:, (components * (task_dim * task_dim + 1)):]
    return alphas, mus, sigmas


def res_loss(y, parameter_vector):
    alphas, mus, sigmas = slice_parameter_vectors(parameter_vector)
    mus = tf.reshape(mus, (tf.shape(mus)[0], components, task_dim, task_dim))
    sigmas = tf.reshape(sigmas, (tf.shape(sigmas)[0], components, task_dim, task_dim))
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alphas),
        components_distribution=tfd.Independent(tfd.Gamma(concentration=mus, rate=sigmas), reinterpreted_batch_ndims=2))
    log_likelihood = tf.clip_by_value(gm.log_prob(tf.cast(y, tf.float32)), clip_value_min=-10000, clip_value_max=0)
    return -tf.reduce_mean(log_likelihood)


def plot_history(history, title):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(title)


def model_generator(num_components, n_lag, task_dim, prob=True):
    input1 = layers.Input(shape=(n_lag, task_dim, task_dim, 1), dtype='float32')
    input1 = layers.BatchNormalization()(input1)
    input2 = layers.Input(shape=(task_dim, task_dim, 1), dtype='float32')
    input2 = layers.BatchNormalization()(input2)
    input3 = layers.Input(shape=(1,), dtype='float32')
    input3 = layers.BatchNormalization()(input3)

    X = layers.ConvLSTM2D(filters=50, kernel_size=(3, 3), activation='tanh', padding='same', return_sequences=True)(
        input1)
    X = layers.ConvLSTM2D(filters=50, kernel_size=(3, 3), activation='tanh', return_sequences=True)(X)
    X = layers.ConvLSTM2D(filters=50, kernel_size=(1, 1), activation='tanh')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(128)(X)
    X = layers.LeakyReLU(alpha=0.05)(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(128)(X)
    X = layers.LeakyReLU(alpha=0.05)(X)

    X1 = layers.Conv2D(20, (3, 3), activation='relu')(input2)
    X1 = layers.Flatten()(X1)
    X2 = layers.BatchNormalization()(input3)
    X = layers.Concatenate()([X, X1, X2])

    X1 = layers.Dense(128)(X)
    X1 = layers.LeakyReLU(alpha=0.05)(X1)
    X1 = layers.BatchNormalization()(X1)
    X1 = layers.Dense(128)(X)
    X1 = layers.LeakyReLU(alpha=0.05)(X1)
    X1 = layers.BatchNormalization()(X1)
    X1 = layers.Dense(128)(X)
    X1 = layers.LeakyReLU(alpha=0.05)(X1)
    X1 = layers.BatchNormalization()(X1)
    X1 = layers.Dense(128)(X)
    X1 = layers.LeakyReLU(alpha=0.05)(X1)
    X1 = layers.BatchNormalization()(X1)
    X1 = layers.Dense(128)(X)
    X1 = layers.LeakyReLU(alpha=0.05)(X1)
    X1 = layers.BatchNormalization()(X1)
    X1 = layers.Dense(128)(X)
    X1 = layers.LeakyReLU(alpha=0.05)(X1)
    X1 = layers.BatchNormalization()(X1)

    alphas1 = layers.Dense(num_components, activation="softmax")(X1)
    mus1 = layers.Dense(num_components * task_dim * task_dim, activation='nnelu')(X1)
    sigmas1 = layers.Dense(num_components * task_dim * task_dim, activation='nnelu')(X1)
    output1 = layers.Concatenate()([alphas1, mus1, sigmas1])

    X2 = layers.Dense(128)(X)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)
    X2 = layers.Dense(128)(X2)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)
    X2 = layers.Dense(128)(X2)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)
    X2 = layers.Dense(128)(X2)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)
    X2 = layers.Dense(128)(X2)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)
    X2 = layers.Dense(128)(X2)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)
    X2 = layers.Dense(128)(X2)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)

    output2 = layers.Dense(task_dim * task_dim, activation='nnelu')(X2)
    output2 = layers.Reshape((task_dim, task_dim))(output2)

    if prob:
        model = Model([input1, input2, input3], output1)
    else:
        model = Model([input1, input2, input3], output2)
    return model
start = time.time()
meta_model = model_generator(components, n_lag, task_dim, prob=True)

# define TaskExtractor

taskextractor_meta = TaskExtractor(data, lats_lons, task_dim, test_proportion, n_lag)

# define meta learner
meta_optimizer = tf.keras.optimizers.Adam(0.001)
inner_step = 1
inner_optimizer = tf.keras.optimizers.Adam(0.001)

meta_learner = MetaSGD(meta_model, res_loss,  meta_optimizer, inner_step, inner_optimizer, taskextractor_meta, meta_lr=0.001)
meta_learner.load_meta_weights('../../Results/meta_weights_wb_prob')

def scheduler(epoch, lr):
    if epoch <= 10:
        return 0.00001
    elif epoch <= 20 and epoch > 10:
        return 0.000001
    else:
        return 0.0000001
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#meta_learner = MetaSGD(fine_tune_model, res_loss,  meta_optimizer, inner_step, inner_optimizer, taskextractor_meta, meta_lr=0.002)
downscaler = Downscaler(meta_learner, components, test_m_data)
optimizer = tf.keras.optimizers.Adam()
downscaled_data = downscaler.downscale(20, optimizer, prob=True, callbacks=callback)
np.save(r'../../Results/downscaled_data_prob', downscaled_data)
print('Prob Downscale Time:', (time.time() - start)/60, 'mins')

start = time.time()
meta_model = model_generator(components, n_lag, task_dim, prob=False)

# define TaskExtractor

taskextractor_meta = TaskExtractor(data, lats_lons, task_dim, test_proportion, n_lag)

# define meta learner
meta_optimizer = tf.keras.optimizers.Adam(0.001)
inner_step = 1
inner_optimizer = tf.keras.optimizers.Adam(0.001)

meta_learner = MetaSGD(meta_model, tf.keras.losses.MeanAbsoluteError(),  meta_optimizer, inner_step, inner_optimizer,
                       taskextractor_meta, meta_lr=0.001)
meta_learner.load_meta_weights('../../Results/meta_weights_wob')


def scheduler(epoch, lr):
    if epoch <= 10:
        return 0.0001
    elif epoch <= 15 and epoch > 10:
        return 0.0005
    else:
        return 0.00001
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#meta_learner = MetaSGD(fine_tune_model, res_loss,  meta_optimizer, inner_step, inner_optimizer, taskextractor_meta, meta_lr=0.002)
downscaler = Downscaler(meta_learner, components, test_m_data)
optimizer = tf.keras.optimizers.Adam()
downscaled_data = downscaler.downscale(20, optimizer, prob=False, callbacks=callback)
np.save(r'../../Results/downscaled_data_det', downscaled_data)
print('Det Downscale Time:', (time.time() - start)/60, 'mins')