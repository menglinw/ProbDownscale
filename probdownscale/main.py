
import os
import sys
from MetaTrain import MetaSGD
from TaskExtractor import TaskExtractor
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

# define beta function
def covariance_function(h, phi=0.5):
    return exp(-h/phi)

def distance_function(loc1, loc2):
    return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def beta_function(meta_rate, batch_locations, seen_locations, covariance_function, distance_function, decay_rate=1):
    # decay_rate: 1-0, larger -> faster decay
    batch_size = len(batch_locations)
    seen_size = len(seen_locations.items())
    if seen_size == 0:
        return meta_rate
    temp = 0
    for b_loc in batch_locations:
        for s_loc, n in seen_locations.items():
            cov = covariance_function(distance_function(b_loc, s_loc))
            temp += cov * (1 + log(n))
    mean_cov = temp/(batch_size*sum(list(seen_locations.values())))
    cov_factor = -log(mean_cov)
    bsize_factor = exp((batch_size/seen_size)**decay_rate) - 1
    print('mean cov:', mean_cov)
    print('covariance factor:', cov_factor)
    print('batch size factor:', bsize_factor)
    lr = meta_rate*bsize_factor*cov_factor
    return lr


def meta_compare(data, lats_lons, task_dim, test_proportion, n_lag, meta_lr, loss, prob=True, n_epochs=5, batch_size=20):
    # Prob Model Training
    prob_meta_model = model_generator(components, n_lag, task_dim, prob=prob)

    # define TaskExtractor

    taskextractor_meta = TaskExtractor(data, lats_lons, task_dim, test_proportion, n_lag)

    # define meta learner
    meta_optimizer = tf.keras.optimizers.Adam(meta_lr)
    inner_step = 1
    inner_optimizer = tf.keras.optimizers.Adam(meta_lr)

    meta_learner = MetaSGD(prob_meta_model, loss,  meta_optimizer, inner_step, inner_optimizer, taskextractor_meta,
                           meta_lr=meta_lr)

    # meta train with beta
    meta_beta_history = meta_learner.meta_fit(n_epochs, batch_size=batch_size, basic_train=True, bootstrap_train=True,
                                              use_test_for_meta=True, randomize=True, beta_function=beta_function,
                                              covariance_function=covariance_function,
                                              distance_function=distance_function)
    if prob:
        meta_learner.save_meta_weights(r"../../Results/meta_weights_wb_prob")
    else:
        meta_learner.save_meta_weights(r"../../Results/meta_weights_wb")


    prob_meta_model_wob = model_generator(components, n_lag, task_dim, prob=prob)
    inner_optimizer_wob = tf.keras.optimizers.Adam(meta_lr)
    meta_optimizer_wob = tf.keras.optimizers.Adam(meta_lr)
    meta_learner_wob = MetaSGD(prob_meta_model_wob, loss,  meta_optimizer_wob, inner_step, inner_optimizer_wob, taskextractor_meta)
    # meta train without beta
    meta_history_wob = meta_learner_wob.meta_fit(n_epochs, batch_size=batch_size, basic_train=True,
                                                 bootstrap_train=True, use_test_for_meta=True, randomize=True)
    if prob:
        meta_learner_wob.save_meta_weights(r"../../Results/meta_weights_wob_prob")
    else:
        meta_learner_wob.save_meta_weights(r"../../Results/meta_weights_wob")

    plt.plot(meta_history_wob[0], "-b", label="without beta loss")
    plt.plot(meta_beta_history[0], "-r", label="with beta loss")
    plt.plot(meta_history_wob[1], "--b", label="without beta val loss")
    plt.plot(meta_beta_history[1], "--r", label="with beta val loss")
    plt.legend(loc="upper left")
    plt.title('Meta Training History Compare')
    plt.show()
    if prob:
        plt.savefig('../../Results/Meta_train_prob_compare_prob.jpg')
    else:
        plt.savefig('../../Results/Meta_train_prob_compare.jpg')

print('Now doing prob meta training')
start = time.time()
meta_compare(data, lats_lons, task_dim, test_proportion, n_lag, meta_lr=0.0005, loss=res_loss, prob=True, n_epochs=10, batch_size=10)
print('Prob Meta Training:', (time.time() - start)/60, ' mins')

print('Now doing meta training')
start = time.time()
meta_compare(data, lats_lons, task_dim, test_proportion, n_lag, meta_lr=0.005, loss=tf.keras.losses.MeanAbsoluteError(),
             prob=False, n_epochs=10, batch_size=10)
print('Meta Training:', (time.time() - start)/60, ' mins')
