
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np


class MetaSGD:
    def __init__(self, target_model, target_loss, meta_step, meta_optimizer, inner_step, inner_optimizer, task_extractor):
        '''
        :param target_model:
        :param target_loss:
        :param meta_step:
        :param meta_optimizer:
        :param inner_step:
        :param inner_optimizer:
        :param batch_size:
        :param data:
        '''

        self.meta_model = target_model
        self.loss = target_loss
        '''
        self.alpha = []
        initializer = tf.random.Generator.from_seed(123)
        for tensor in target_model.trainable_variables:
            init = tf.Variable(initializer.normal(shape=tf.shape(tensor), mean=0, stddev=0.01))
            print(tf.shape(init))

            self.alpha.append(init)
        '''
        self.meta_step = meta_step
        self.meta_optimizer = meta_optimizer
        self.inner_step = inner_step
        self.inner_optimizer = inner_optimizer
        self.task_extractor = task_extractor

    def _train_on_batch(self, batch_size):
        """

        :param train_data: training data for one batch, including meta_train_X, meta_train_Y, meta_test_X, meta_test_Y
        :param inner_optimizer: optimizer for inner loop
        :param inner_step: steps of learning for inner loop
        :param outer_optimizer: optimizer for outer loop, if None, then not update
        :return: batch query loss
        """
        # define lists to collect loss and weights
        batch_loss = []
        task_weights = []

        # save the weight as initialization
        meta_weights = self.meta_model.get_weights()
        #print(self.alpha)
        #self.inner_optimizer.learning_rate.assign(1)
        meta_train_X, meta_train_Y, meta_test_X, meta_test_Y, locations = self.task_extractor.get_random_tasks(batch_size)
        for train_X, train_Y in zip(meta_train_X, meta_train_Y):

            # load the initialization
            self.meta_model.set_weights(meta_weights)
            # update several inner_step
            for _ in range(self.inner_step):
                with tf.GradientTape() as tape:
                    Y_hat = self.meta_model(train_X, training=True)
                    loss = self.loss(train_Y, Y_hat)
                    loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                # TODO check this alpha work or not
                # element wise multiply alpha and gradients
                '''
                grads2 = []
                for grad, al in zip(grads, self.alpha):
                    grads2.append(tf.math.multiply(grad, al))
                
                #grads = tf.math.multiply(grads, self.alpha)
                '''
                self.inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # save the weights after inner loop
            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as para_tape: # ValueError: No gradients provided for any variable:
            # calculate the test loss
            for i, (test_X, test_Y) in enumerate(zip(meta_test_X, meta_test_Y)):
                # load the trained weight after inner loop
                self.meta_model.set_weights(task_weights[i])

                Y_hat = self.meta_model(test_X, training=True)
                loss = self.loss(test_Y, Y_hat)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)
            mean_loss = tf.reduce_mean(batch_loss)
            grads_para = para_tape.gradient(mean_loss, self.meta_model.trainable_variables)
            # grads_alpha = alpha_tape.gradient(mean_loss, self.alpha)
        # start meta learning step: using the initialization weight
        self.meta_model.set_weights(meta_weights)
        print('origional lr:', self.meta_optimizer.learning_rate)
        if self.meta_optimizer:
            self.meta_optimizer.learning_rate.assign(0.0123)
            print('updated lr:', self.meta_optimizer.learning_rate)
            self.meta_optimizer.apply_gradients(zip(grads_para, self.meta_model.trainable_variables))
            # self.meta_optimizer.apply_gradients(zip(grads_alpha, self.alpha))
        #print(self.alpha)
        return mean_loss
