import os
import sys
from MetaTrain import MetaSGD
from TaskExtractor import TaskExtractor
from Downscaler import Downscaler
import utils.data_processing as data_processing
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
import geopandas as gpd
import pandas as pd

file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
target_var = 'TOTEXTTAU'

file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']

# task dimension 3*3
task_dim = 3

# proportion of test data
test_proportion = 0.3

# number of lagging steps
n_lag = 20

# number of components of MDN
components=500

# save path
save_path = sys.argv[1]
# read data
g05_data = nc.Dataset(file_path_g_05)
g06_data = nc.Dataset(file_path_g_06)
m_data_nc = nc.Dataset(file_path_m)

# define lat&lon of MERRA, G5NR and mete
M_lons = m_data_nc.variables['lon'][:]
# self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
M_lats = m_data_nc.variables['lat'][:]
# self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
G_lons = g05_data.variables['lon'][:]
# self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
G_lats = g05_data.variables['lat'][:]

# extract target data
g_data = np.concatenate((g05_data.variables[target_var], g06_data.variables[target_var]), axis=0)
m_data = m_data_nc.variables[target_var][5*365:7*365, :, :]


def normalize(data):
    return (data - data.min())/(data.max() - data.min())


g_data = normalize(g_data)*100
m_data = normalize(m_data)*100

country_shape = gpd.read_file(file_path_country[0])
for country_path in file_path_country[1:]:
    country_shape = pd.concat([country_shape, gpd.read_file(country_path)])

latmin, lonmin, latmax, lonmax =country_shape.total_bounds
latmin_ind = np.argmin(np.abs(G_lats - latmin))
latmax_ind = np.argmin(np.abs(G_lats - latmax))
lonmin_ind = np.argmin(np.abs(G_lons - lonmin))
lonmax_ind = np.argmin(np.abs(G_lons - lonmax))
# 123 * 207
g_data = g_data[:, latmin_ind-1:latmax_ind+1, lonmin_ind:lonmax_ind+2]

G_lats = G_lats[latmin_ind-1:latmax_ind+1]
G_lons = G_lons[lonmin_ind:lonmax_ind+2]

# take a subset
#g_data = g_data[:, :30, :30]
#G_lats = G_lats[:30]
#G_lons = G_lons[:30]

# split data into traing and test
train_g_data, test_g_data = g_data[:657], g_data[657:]
train_m_data, test_m_data = m_data[:657], m_data[657:]
data = [train_g_data, train_m_data]
lats_lons = [G_lats, G_lons, M_lats, M_lons]


# define necessary tool functions
# number of components in Mixture Density Network

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


def model_generator(num_components, n_lag, task_dim, prob=True):
    input1 = layers.Input(shape=(n_lag, task_dim, task_dim, 1), dtype='float32')
    input1 = layers.BatchNormalization()(input1)
    input2 = layers.Input(shape=(task_dim, task_dim, 1), dtype='float32')
    input2 = layers.BatchNormalization()(input2)
    input3 = layers.Input(shape=(1,), dtype='float32')
    input3 = layers.BatchNormalization()(input3)

    X = layers.ConvLSTM2D(filters=50, kernel_size=(2, 2), activation=layers.LeakyReLU(), padding='same',
                          return_sequences=True)(input1)
    X = layers.ConvLSTM2D(filters=50, kernel_size=(2, 2), activation=layers.LeakyReLU(), return_sequences=True)(X)
    X = layers.ConvLSTM2D(filters=50, kernel_size=(1, 1), activation=layers.LeakyReLU())(X)
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
    for nodes in [64, 128, 256, 512, 256, 128, 64, 32, 16, 8]:
        X1 = layers.Dense(nodes)(X1)
        X1 = layers.LeakyReLU(alpha=0.05)(X1)
        X1 = layers.BatchNormalization()(X1)

    alphas1 = layers.Dense(num_components, activation="softmax")(X1)
    mus1 = layers.Dense(num_components * task_dim * task_dim, activation='nnelu')(X1)
    sigmas1 = layers.Dense(num_components * task_dim * task_dim, activation='nnelu')(X1)
    output1 = layers.Concatenate()([alphas1, mus1, sigmas1])

    X2 = layers.Dense(128)(X)
    X2 = layers.LeakyReLU(alpha=0.05)(X2)
    X2 = layers.BatchNormalization()(X2)

    for nodes in [64, 128, 256, 128, 64, 32, 16, 8]:
        X2 = layers.Dense(nodes)(X2)
        X2 = layers.LeakyReLU(alpha=0.05)(X2)
        X2 = layers.BatchNormalization()(X2)

    output2 = layers.Dense(task_dim * task_dim, activation='relu')(X2)
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

def beta_function(meta_rate, batch_locations, seen_locations, covariance_function, distance_function):
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
    bsize_factor = exp((batch_size/seen_size)**0.5) - 1
    print('mean cov:', mean_cov)
    print('covariance factor:', cov_factor)
    print('batch size factor:', bsize_factor)
    lr = meta_rate*bsize_factor*cov_factor
    return lr


def meta_compare(data, lats_lons, task_dim, test_proportion, n_lag, meta_lr, loss, beta_function, covariance_function,
                 distance_function, save_path, n_epochs=5, batch_size=10):
    # Prob Model Training
    prob_meta_model = model_generator(components, n_lag, task_dim)

    # define TaskExtractor

    taskextractor_meta = TaskExtractor(data, lats_lons, task_dim, test_proportion, n_lag)

    # define meta learner
    meta_optimizer = tf.keras.optimizers.Adam(meta_lr)
    inner_step = 1
    inner_optimizer = tf.keras.optimizers.Adam(meta_lr)

    meta_learner = MetaSGD(prob_meta_model, loss,  meta_optimizer, inner_step, inner_optimizer, taskextractor_meta,
                           meta_lr=meta_lr)

    # meta train with beta
    meta_beta_history = meta_learner.meta_fit(n_epochs, batch_size=batch_size, basic_train=True, bootstrap_train=False,
                                              randomize=True, beta_function=beta_function,
                                              covariance_function=covariance_function,
                                              distance_function=distance_function)
    # save weights and history
    meta_learner.save_meta_weights(os.path.join(save_path, "meta_weights_wb_prob"))
    np.save(os.path.join(save_path, 'meta_history_wb'), np.array(meta_beta_history))

    # meta train without beta
    prob_meta_model_wob = model_generator(components, n_lag, task_dim)
    inner_optimizer_wob = tf.keras.optimizers.Adam(meta_lr)
    meta_optimizer_wob = tf.keras.optimizers.Adam(meta_lr)
    meta_learner_wob = MetaSGD(prob_meta_model_wob, loss,  meta_optimizer_wob, inner_step, inner_optimizer_wob, taskextractor_meta)
    meta_history_wob = meta_learner_wob.meta_fit(n_epochs, batch_size=batch_size, basic_train=True,
                                                 bootstrap_train=False, randomize=True)
    # save weights and history
    meta_learner_wob.save_meta_weights(os.path.join(save_path, "meta_weights_wob_prob"))
    np.save(os.path.join(save_path, 'meta_history_wob'), np.array(meta_history_wob))
    # save history plot
    plt.figure()
    plt.plot(meta_history_wob[0], "-b", label="without beta loss")
    plt.plot(meta_beta_history[0], "-r", label="with beta loss")
    plt.plot(meta_history_wob[1], "--b", label="without beta val loss")
    plt.plot(meta_beta_history[1], "--r", label="with beta val loss")
    plt.legend(loc="upper left")
    plt.title('Meta Training History')
    plt.show()
    plt.savefig(os.path.join(save_path, 'Meta_train_prob_compare_prob.jpg'))

    return meta_learner

print('Now doing prob meta training')
start = time.time()
meta_learner = meta_compare(data, lats_lons, task_dim, test_proportion, n_lag, meta_lr=0.0005, loss=res_loss, beta_function=beta_function,
             covariance_function=covariance_function, distance_function=distance_function, save_path=save_path, n_epochs=10, batch_size=15)
print('Prob Meta Training:', (time.time() - start)/60, ' mins')

'''
print('Now doing downscaling')
start = time.time()
downscaler = Downscaler(meta_learner, components, test_m_data)
optimizer = tf.keras.optimizers.Adam()
def scheduler(epoch, lr):
    if epoch <= 10:
        return 0.001
    elif epoch <= 30 and epoch > 10:
        return 0.0005
    else:
        return 0.00001
lr_scheduler= tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
downscaled_data = downscaler.downscale(100, optimizer, prob=True, callbacks=[lr_scheduler, early_stopping])
np.save(os.path.join(save_path, 'downscaled_data'), downscaled_data)
print('Downscale Time:', (time.time() - start)/60, 'mins')

def image_evaluate(pred_data, true_data):
    if pred_data.shape != true_data.shape:
        print('Please check data consistency!')
        raise ValueError
    length = np.prod(pred_data.shape[1:])
    r2_list = np.zeros(pred_data.shape[0])
    rmse_list = np.zeros(pred_data.shape[0])
    filter = ~np.isnan(pred_data[0].reshape(length))
    for i in range(pred_data.shape[0]):
        r2_list[i],_ = data_processing.rsquared(pred_data[i].reshape(length)[filter], true_data[i].reshape(length)[filter])
        rmse_list[i] = np.nanmean(np.square(pred_data[i] - true_data[i]))
    return rmse_list, r2_list

rmse_list, r2_list = image_evaluate(downscaled_data, test_g_data)
plt.figure()
#plt.plot(rmse_list/100, "-b",label='RMSE')
plt.plot(r2_list, "-r",label='R2')
plt.legend()
plt.savefig(os.path.join(save_path, 'downscale_R2.jpg'))
'''