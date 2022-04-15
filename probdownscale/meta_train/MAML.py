
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np


class MAML:
    def __init__(self, target_model, target_loss):
        '''
        :param target_model: target model
        :param loss: the loss of target model
        '''
        self.meta_model = target_model
        self.loss = target_loss
        self.alpha = tf.Variable(tf.random_normal_initializer(shape=target_model.get_weights().shape,

                                                              dtype=tf.float64))

    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):
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
        alpha = self.alpha

        meta_train_X, meta_train_Y, meta_test_X, meta_test_Y = train_data
        for train_X, train_Y in zip(meta_train_X, meta_train_Y):

            # load the initialization
            self.meta_model.set_weights(meta_weights)
            # update several inner_step
            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    Y_hat = self.meta_model(train_X, training=True)
                    loss = self.loss(train_Y, Y_hat)
                    loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                # TODO check this alpha work or not
                # element wise multiply alpha and gradients
                grads = tf.math.multiply(grads, alpha)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # save the weights after inner loop
            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as tape:
            tape.watch(alpha)
            tape.watch(self.meta_model.trainable_variables)
            # calculate the test loss
            for i, (test_X, test_Y) in enumerate(zip(meta_test_X, meta_test_Y)):
                # load the trained weight after inner loop
                self.meta_model.set_weights(task_weights[i])

                Y_hat = self.meta_model(test_X, training=True)
                loss = self.loss(test_Y, Y_hat)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)
            mean_loss = tf.reduce_mean(batch_loss)

        # start meta learning step: using the initialization weight
        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, [self.meta_model.trainable_variables, alpha])
            outer_optimizer.apply_gradients(zip(grads[0], self.meta_model.trainable_variables))
            outer_optimizer.apply_gradients(zip(grads[1], alpha))
            self.alpha = alpha
        return mean_loss
