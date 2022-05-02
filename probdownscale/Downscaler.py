
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np
from random import sample


class Downscaler():
    def __init__(self, meta_learner):
        self.meta_learner = meta_learner
        self.meta_weights = meta_learner.meta_model.get_weights()
        self.meta_model = meta_learner.meta_model
        self.loss = meta_learner.loss
        self.task_extractor = meta_learner.task_extractor



    def downscale(self):
        # TODO: need to compile meta model
        pass

    def _task_downscaler(self, location, epochs, sequential_steps):
        train_x, train_y, _, _, location, init = self.task_extractor._get_one_random_task(is_random=False, record=False,
                                                                                    lat_lon=location, use_all_data=True,
                                                                                    return_init=True)
        # initialize with meta weights
        self.meta_model.set_weights(self.meta_weights)

        # fine tune the meta model with task data
        # the meta_model should be compiled first
        self.meta_model.fit(train_x, train_y, epochs=epochs)

        # predict several step


    def sequential_predict(self, model, init_data, predict_steps, is_prob=False):
        n_lag = len(init_data)
        for i in range(predict_steps):
            temp = np.zeros((1, 1, n_lag))
            temp[0, 0] = init_data[-n_lag:]
            y_hat = model.predict(temp)
            if is_prob:
                alpha_pred, mu_pred = slice_parameter_vectors(y_hat)
                MDN_Yhat = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=alpha_pred),
                    components_distribution=tfd.Exponential(
                        rate=mu_pred))
                init_data.append(np.asarray(MDN_Yhat.sample())[0])
            else:
                init_data.append(y_hat[0, 0])
            # print(y_hat[0,0])

        return init_data[n_lag:]

