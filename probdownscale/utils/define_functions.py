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


def normalize(data):
    return (data - data.min())/(data.max() - data.min())

def nnelu(input):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

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

    X1 = layers.Conv2D(20, (2, 2), activation='relu')(input2)
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