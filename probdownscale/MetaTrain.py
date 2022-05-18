
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np
from random import sample


class MetaSGD:
    def __init__(self, target_model, target_loss, meta_optimizer, inner_step, inner_optimizer, task_extractor,
                meta_lr=0.005):
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
        self.meta_optimizer = meta_optimizer
        self.inner_step = inner_step
        self.inner_optimizer = inner_optimizer
        self.task_extractor = task_extractor
        self.history = []
        self.seen_locations = dict()
        self.meta_lr = meta_lr
        self.beta_history = []
        self.val_history = []

    def _train_on_batch(self, batch_size=None, inner_rate_f=None, beta_function=None,
                        locations=None, covariance_function=None, distance_function=None):
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
            meta_train_X, meta_train_Y, meta_test_X, meta_test_Y, locations = self.task_extractor.get_random_tasks(locations=locations, record=True)
        else:
            meta_train_X, meta_train_Y, meta_test_X, meta_test_Y, locations = self.task_extractor.get_random_tasks(batch_size, record=True)

        # add location to the seen dictionary
        for loc in locations:
            self.seen_locations.setdefault(loc, 0)
            self.seen_locations[loc] += 1

        # if beta function is given, update learning rate with beta function
        if beta_function:
            lr = beta_function(meta_rate=self.meta_lr, batch_locations=locations,
                               seen_locations=self.seen_locations,
                               covariance_function=covariance_function,
                               distance_function=distance_function)
            self.beta_history.append(lr)
            self.meta_optimizer.learning_rate.assign(lr)
            self.inner_optimizer.learning_rate.assign(lr)

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
            for i, (train_X, train_Y) in enumerate(zip(meta_train_X, meta_train_Y)):
                # load the trained weight after inner loop
                self.meta_model.set_weights(task_weights[i])

                Y_hat = self.meta_model(train_X, training=True)
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

            print('Meta lr:', self.meta_optimizer.learning_rate)
            print('Base lr:', self.inner_optimizer.learning_rate)
            self.meta_optimizer.apply_gradients(zip(grads_para, self.meta_model.trainable_variables))
            # validation loss
            val_loss = []
            for i, (test_X, test_Y) in enumerate(zip(meta_test_X, meta_test_Y)):

                Y_hat = self.meta_model(test_X)
                loss = self.loss(test_Y, Y_hat)
                loss = tf.reduce_mean(loss)
                val_loss.append(loss)
            val_loss = tf.reduce_mean(val_loss)
            # self.meta_optimizer.apply_gradients(zip(grads_alpha, self.alpha))
            # print(batch_loss)
        #print(self.alpha)
        return mean_loss.numpy(), val_loss.numpy()

    def meta_fit(self, epochs, batch_size, basic_train=True, bootstrap_train=True, bootstrap_step=10,
                randomize=True, beta_function=None, covariance_function=None,
                 distance_function=None):
        best_loss = float('inf')
        best_weights = self.meta_model.get_weights()
        # train over all task that cover the study domain
        all_tasks = self.task_extractor.get_grid_locations()


        for i in range(epochs):
            if randomize:
                all_tasks = sample(all_tasks, len(all_tasks))

            if basic_train:
                for step in range((len(all_tasks)//batch_size)):
                    locations = all_tasks[step*batch_size:(step+1)*batch_size]
                    loss, val_loss = self._train_on_batch(locations=locations, 
                                                beta_function=beta_function, covariance_function=covariance_function,
                                                distance_function=distance_function
                                                )
                    self.history.append(loss)
                    self.val_history.append(val_loss)
                    print('Epoch:', i+1, '/', epochs, ' Basic training step: ', step+1, '/', len(all_tasks)//batch_size, 
                          'loss: ', loss, 'val loss: ', val_loss)
            if bootstrap_train:
                for step in range(bootstrap_step):
                    loss, val_loss = self._train_on_batch(batch_size=batch_size, 
                                                beta_function=beta_function, covariance_function=covariance_function,
                                                distance_function=distance_function)
                    self.history.append(loss)
                    self.val_history.append(val_loss)
                    if val_loss < best_loss:
                        best_weights = self.meta_model.get_weights()
                    print('Epoch:', i+1, '/', epochs, 'Bootstrap training step:', step+1, '/', bootstrap_step, 'loss: ',
                          loss, 'val loss: ', val_loss)
            self.meta_model.set_weights(best_weights)
        return self.history, self.val_history, self.beta_history

    def save_meta_weights(self, weights_name):
        self.meta_model.save_weights(weights_name)

    def load_meta_weights(self, weight_path):
        self.meta_model.load_weights(weight_path)

