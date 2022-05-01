
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np
from random import sample

class MetaSGD:
    def __init__(self, target_model, target_loss, meta_optimizer, inner_step, inner_optimizer, task_extractor,
                 meta_loss=None):
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
        self.meta_loss = meta_loss
        '''
        self.alpha = []
        initializer = tf.random.Generator.from_seed(123)
        for tensor in target_model.trainable_variables:
            init = tf.Variable(initializer.normal(shape=tf.shape(tensor), mean=0, stddev=0.01))
            print(tf.shape(init))

            self.alpha.append(init)
        '''
        self.meta_optimizer = meta_optimizer
        self.inner_step = inner_step
        self.inner_optimizer = inner_optimizer
        self.task_extractor = task_extractor
        self.history = []

    def _train_on_batch(self, batch_size=None, inner_rate_f=None, meta_rate_f=None, use_test_for_meta=True, locations=None):
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
        if inner_rate_f:
            inner_rate = self.inner_optimizer.learning_rate.numpy()
            self.inner_optimizer.learning_rate.assign(inner_rate_f(inner_rate, batch_size, self.inner_step))
        #self.inner_optimizer.learning_rate.assign(1)
        if locations:
            meta_train_X, meta_train_Y, meta_test_X, meta_test_Y, locations = self.task_extractor.get_random_tasks(locations=locations)
        else:
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
                # TODO check this alpha work or not, not working, give up
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
            if use_test_for_meta:
                for i, (test_X, test_Y) in enumerate(zip(meta_test_X, meta_test_Y)):
                    # load the trained weight after inner loop
                    self.meta_model.set_weights(task_weights[i])

                    Y_hat = self.meta_model(test_X, training=True)
                    if self.meta_loss:
                        loss = self.meta_loss(test_Y, Y_hat)
                    else:
                        loss = self.loss(test_Y, Y_hat)
                    loss = tf.reduce_mean(loss)
                    batch_loss.append(loss)
            else:
                for i, (train_X, train_Y) in enumerate(zip(meta_train_X, meta_train_Y)):
                    # load the trained weight after inner loop
                    self.meta_model.set_weights(task_weights[i])

                    Y_hat = self.meta_model(train_X, training=True)
                    if self.meta_loss:
                        loss = self.meta_loss(train_Y, Y_hat)
                    else:
                        loss = self.loss(train_Y, Y_hat)
                    loss = tf.reduce_mean(loss)
                    batch_loss.append(loss)
            mean_loss = tf.reduce_mean(batch_loss)
            grads_para = para_tape.gradient(mean_loss, self.meta_model.trainable_variables)
            # grads_alpha = alpha_tape.gradient(mean_loss, self.alpha)
        # start meta learning step: using the initialization weight
        self.meta_model.set_weights(meta_weights)
        #print('origional lr:', self.meta_optimizer.learning_rate)
        if self.meta_optimizer:
            # TODO: develop beta function
            #self.meta_optimizer.learning_rate.assign(0.000001)
            print('Meta lr:', self.meta_optimizer.learning_rate)
            self.meta_optimizer.apply_gradients(zip(grads_para, self.meta_model.trainable_variables))
            # self.meta_optimizer.apply_gradients(zip(grads_alpha, self.alpha))
            # print(batch_loss)
        #print(self.alpha)
        return mean_loss.numpy()

    def meta_fit(self, epochs, batch_size, basic_train=True, bootstrap_train=True, bootstrap_step=10, use_test_for_meta=True, randomize=True):
        # train over all task that cover the study domain
        all_tasks = self.task_extractor.get_grid_locations()
        if randomize:
            all_tasks = sample(all_tasks, len(all_tasks))

        for i in range(epochs):
            if basic_train:
                for step in range((len(all_tasks)//batch_size)):
                    locations = all_tasks[step*batch_size:(step+1)*batch_size]
                    loss = self._train_on_batch(locations=locations, use_test_for_meta=use_test_for_meta)
                    self.history.append(loss)
                    print('Epoch:', i+1, '/', epochs, ' Basic training step: ', step+1, '/', len(all_tasks)//batch_size, 'loss: ', loss)
            if bootstrap_train:
                for step in range(bootstrap_step):
                    loss = self._train_on_batch(batch_size=batch_size, use_test_for_meta=use_test_for_meta)
                    self.history.append(loss)
                    print('Epoch:', i+1, '/', epochs, 'Bootstrap training step:', step+1, '/', bootstrap_step, 'loss: ', loss)
        return self.history

    def downscale(self):
        pass

    def fine_tune(self):
        pass